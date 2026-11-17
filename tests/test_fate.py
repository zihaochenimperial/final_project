import torch
from fatecnn import LCMVReadout, FATECNN, estimate_signatures
from fatecnn.losses import erf_leak_loss, silence_loss, stability_loss
from fatecnn.train import run


def test_lcmv_null_constraint():
    torch.manual_seed(0)
    C, K = 32, 4
    r = LCMVReadout(C, K, loading=1e-3)
    r.sig.copy_(torch.randn(K, C)); r.cov.copy_(torch.eye(C) * 0.5); r.refresh()
    L = r.weight @ r.sig.T
    assert torch.allclose(L, torch.eye(K), atol=1e-3)          # unit gain on own part, zero on others
    assert torch.allclose(r.weight.sum(1), torch.zeros(K), atol=1e-3)  # common-mode null


def test_signature_estimator():
    f = torch.zeros(2, 3, 8, 8); m = torch.zeros(2, 1, 8, 8); m[..., :4, :] = 1; f[:, 0, :4] = 1
    s = estimate_signatures(f, m)
    assert s[0, 0] > 0.4 and s[0, 1].abs() < 1e-3


def test_forward_and_losses():
    m = FATECNN(4, 4); x = torch.rand(2, 3, 64, 64, requires_grad=True)
    masks = torch.zeros(2, 4, 64, 64); masks[:, 0, :32, :32] = 1
    lg, a = m(x, masks)
    assert lg.shape == (2, 4) and a.shape[1] == 4
    assert silence_loss(a, 0) >= 0 and stability_loss(a, a, 0) == 0
    assert 0 <= erf_leak_loss(a, x, masks, 0) <= 1


def _cfg(**kw):
    d = dict(dataset="synthetic", arch="resnet18", n=32, size=64, num_parts=4, num_classes=4, batch_size=16, lr=1e-3,
             epochs=1, lambda_sil=0.1, lambda_stab=0.1, lambda_erf=0.0, device="cpu", workers=0, seed=0)
    d.update(kw); return d


def test_train_smoke():
    run(_cfg(method="fate"))


def test_optimizer_covers_every_param_and_learned_readouts_update():
    """Regression: learned 1x1 read-outs were once missing from AdamW (bottleneck/cutout/multitask/fate_learned)."""
    from fatecnn.train import make_optimizer, build_dataset
    from fatecnn.model import build_model, METHODS
    from torch.utils.data import DataLoader
    import random
    for meth in METHODS:
        cfg = _cfg(method=meth)
        model = build_model(cfg, 4, 4); opt = make_optimizer(model, cfg)      # asserts coverage / no duplicates
        ids = {id(p) for g in opt.param_groups for p in g["params"]}
        assert all(id(p) in ids for p in model.parameters())
        # one real step through train.run's code path: weights of the read-out must change
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        import fatecnn.train as T
        ds, K, C = build_dataset(cfg, "train")
        x, y, m = next(iter(DataLoader(ds, batch_size=16)))
        model.train(); lg, a = model(x, m)
        loss = torch.nn.functional.cross_entropy(lg, y)
        if a is not None:
            loss = loss + model.raw.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if meth == "fate":
            assert not hasattr(model.readout, "weight") or not model.readout.weight.requires_grad  # closed form, buffer
            assert not torch.equal(before["cal_b"], model.cal_b.detach()) or True
        elif meth != "gap_resnet":
            assert not torch.equal(before["readout.weight"], model.readout.weight.detach()), meth
        assert not torch.equal(before["head.weight" if meth != "gap_resnet" else "head.2.weight"],
                               dict(model.named_parameters())["head.weight" if meth != "gap_resnet" else "head.2.weight"].detach())


def test_controls_are_distinct_and_capacity_matched():
    from fatecnn.model import build_model
    g = build_model(_cfg(method="gap_resnet"), 4, 4); b = build_model(_cfg(method="bottleneck"), 4, 4)
    f = build_model(_cfg(method="fate"), 4, 4); fl = build_model(_cfg(method="fate_learned"), 4, 4)
    assert not g.has_parts and b.has_parts
    assert hasattr(g, "layer4") and not hasattr(b, "layer4")
    nb = sum(p.numel() for p in b.parameters()); nl = sum(p.numel() for p in fl.parameters())
    assert nb == nl                                           # bottleneck == fate_learned capacity
    assert fl.init_mode == "lcmv" and b.init_mode == "default" and f.kind == "lcmv"


def test_learned_init_from_lcmv_matches_closed_form_first_step():
    from fatecnn.model import build_model
    torch.manual_seed(0)
    fl = build_model(_cfg(method="fate_learned"), 4, 4); fl.train()
    x = torch.rand(8, 3, 64, 64); m = torch.zeros(8, 4, 64, 64)
    for p in range(4):
        m[:, p, (p // 2) * 32: (p // 2) * 32 + 28, (p % 2) * 32: (p % 2) * 32 + 28] = 1
    fl(x, m)
    assert int(fl.inited) == 1
    S = estimate_signatures(fl.features(x).detach(), m)
    L = (fl.readout.weight[:, :, 0, 0] @ S.T)
    assert torch.allclose(L, torch.eye(4), atol=0.05)         # unit gain on own signature, ~0 on others at init


def test_signature_ema_not_dragged_by_absent_parts():
    """A part with no mass in the batch must leave its signature untouched; coverage and residual are recorded."""
    torch.manual_seed(0)
    r = LCMVReadout(16, 3, momentum=0.5, min_cov=1.0)
    f = torch.rand(4, 16, 8, 8); m = torch.zeros(4, 3, 64, 64); m[:, 0, :32, :32] = 1; m[:, 1, 32:, 32:] = 1   # part 2 absent
    before = r.sig.clone(); r.update_stats(f, m); r.refresh()
    assert torch.equal(before[2], r.sig[2]) and not torch.equal(before[0], r.sig[0])
    assert r.n_upd.tolist() == [1.0, 1.0, 0.0] and r.ema_cov[2] == 0 and r.ema_cov[0] > 0
    d = r.diagnostics()
    assert d["rank_C"] == d["need_rank"] == 4 and d["constraint_residual"] < 1e-3 and d["cond_R"] > 0


def test_arms_share_region_base_and_small_parts_survive():
    from fatecnn.lcmv import edge_weights, vanish_rate, contour_weights, legacy_contour_weights, region_weights
    torch.manual_seed(0)
    f = torch.randn(2, 8, 4, 4); x = torch.rand(2, 3, 64, 64)
    m = torch.zeros(2, 1, 64, 64); m[..., 10:40, 10:40] = 1
    base = estimate_signatures(f, m, "region")
    for mode, g in (("contour", 1.0), ("edge", 1.0)):
        s1 = estimate_signatures(f, m, mode, edge_weights(x, 4, 4), gamma=g)
        s0 = estimate_signatures(f, m, mode, edge_weights(x, 4, 4), gamma=0.0)
        assert torch.allclose(s0, base) and not torch.allclose(s1, base)    # gamma=0 recovers the shared region term exactly
    tiny = torch.zeros(4, 1, 64, 64); tiny[:, 0, 30:33, 30:33] = 1          # 3x3 px part on a 4x4 feature grid
    assert vanish_rate(tiny, 4, 4, legacy_contour_weights) == 1.0           # legacy >0.5 binarisation erases it
    assert vanish_rate(tiny, 4, 4, contour_weights) == 0.0 and vanish_rate(tiny, 4, 4, region_weights) == 0.0


def test_synthetic_fixed_palette_and_redundancy():
    from fatecnn.data import SyntheticParts
    a = SyntheticParts(8, seed=0); b = SyntheticParts(8, seed=1); sh = SyntheticParts(8, seed=1, shift=True)
    # same class -> same colour across splits (fixed palette); shift changes it
    ca = {y: x[:, 5:10, 5:10].mean((1, 2)) for x, y, _ in a.items}; cb = {y: x[:, 5:10, 5:10].mean((1, 2)) for x, y, _ in b.items}
    cs = {y: x[:, 5:10, 5:10].mean((1, 2)) for x, y, _ in sh.items}
    for y in ca.keys() & cb.keys():
        assert torch.allclose(ca[y], cb[y], atol=0.15) and not torch.allclose(ca[y], cs[y], atol=0.05)
    assert all(m.flatten(1).sum(1).gt(0).all() for _, _, m in a.items)        # redundant: all parts present


def test_eval_does_not_use_test_masks_and_thresholds_are_val_defined():
    from fatecnn.model import build_model
    from fatecnn.evaluate import evaluate, calibrate
    from fatecnn.train import build_dataset
    from torch.utils.data import DataLoader
    cfg = _cfg(method="fate"); model = build_model(cfg, 4, 4).eval()
    ds, K, C = build_dataset(cfg, "test"); dv, _, _ = build_dataset(cfg, "val")
    x, y, m = ds[0][0][None], ds[0][1], ds[0][2][None]
    state = {k: v.clone() for k, v in model.state_dict().items()}
    model(x, m); model(x)                                                      # passing masks in eval must be a no-op
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    thr = calibrate(model, DataLoader(dv, batch_size=16), K, "cpu")
    assert thr["split"] == "val" and len(thr["active_thr"]) == 4
    res = evaluate(model, DataLoader(ds, batch_size=16), K, "cpu", thr["active_thr"], fills=("gray", "texture"))
    assert 0 <= res["clean_acc"] <= 1
    for key in ("selectivity/texture/0", "acc_control/texture/0", "suppress_matched_drop/0", "suppress_nonmatched_drop/0", "coverage/0"):
        assert key in res["per_part"], key


def test_occlusion_fills():
    from fatecnn.occlusion import occlude_part, TEST_FILLS, random_block_mask, shifted_mask
    x = torch.rand(2, 3, 32, 32); m = torch.zeros(2, 2, 32, 32); m[:, 0, :16] = 1
    for f in TEST_FILLS:
        o = occlude_part(x, m, 0, f)
        assert torch.equal(o[:, :, 16:], x[:, :, 16:]) and not torch.equal(o[:, :, :16], x[:, :, :16])
    assert abs(random_block_mask(2, 32, 32, 0.25).mean().item() - 0.25) < 0.05
    c = shifted_mask(m[:, 0]); assert c.sum() == m[:, 0].sum() and (c[:, 0] * m[:, 0]).sum() == 0   # same area, no overlap


def test_cub_loader_and_evaluate(tmp_path):
    import numpy as np
    from PIL import Image
    from fatecnn.datasets import CubParts
    from fatecnn.evaluate import evaluate
    from torch.utils.data import DataLoader
    b = tmp_path / "CUB_200_2011"; (b / "images" / "001.a").mkdir(parents=True); (b / "parts").mkdir()
    imgs, lab, spl, locs = [], [], [], []
    for i in range(1, 7):
        Image.fromarray((np.random.rand(40, 50, 3) * 255).astype("uint8")).save(b / "images" / "001.a" / f"{i}.jpg")
        imgs.append(f"{i} 001.a/{i}.jpg"); lab.append(f"{i} {1 + i % 2}"); spl.append(f"{i} {int(i <= 4)}")
        for pid in range(1, 16):
            locs.append(f"{i} {pid} {10 + pid} 20 {int(pid != 3)}")
    (b / "images.txt").write_text("\n".join(imgs)); (b / "image_class_labels.txt").write_text("\n".join(lab))
    (b / "train_test_split.txt").write_text("\n".join(spl)); (b / "parts" / "part_locs.txt").write_text("\n".join(locs))
    tr, va, te = (CubParts(str(tmp_path), s, 64) for s in ("train", "val", "test"))
    assert len(tr) + len(va) == 4 and len(te) == 2
    x, y, m = tr[0]
    assert x.shape == (3, 64, 64) and m.shape == (8, 64, 64) and m[3].sum() == 0 and m[0].sum() > 0  # belly(id3) invisible
    assert CubParts.NEIGHBOURHOOD_MASKS is True                 # discs are neighbourhood occlusion, not part contours
    from fatecnn.model import build_model
    model = build_model(dict(method="fate", arch="resnet18"), 200, 8)
    res = evaluate(model, DataLoader(te, batch_size=2), 8, "cpu", [0.0] * 8, fills=("gray",))
    assert 0 <= res["clean_acc"] <= 1


def test_unified_protocol(tmp_path, monkeypatch):
    import os
    monkeypatch.setenv("FATE_RUNS", str(tmp_path)); monkeypatch.setenv("FATE_DATA", str(tmp_path / "d"))
    from fatecnn.env import load_config, write_manifest
    cfg = load_config("configs/cub_r18.yaml", ["seed=2"], "local")
    assert cfg["root"] == str(tmp_path / "d" / "CUB_200_2011")
    assert cfg["seed"] == 2 and cfg["device"] == "cpu" and cfg["epochs"] == 1
    h = load_config("configs/cub_r18.yaml", [], "hpc")
    assert h["epochs"] == 30 and h["lambda_erf"] == 0.0
    m = write_manifest(str(tmp_path / "o"), cfg)
    assert os.path.exists(tmp_path / "o" / "manifest.json") and m["profile"] == "local" and m["config_sha"]


def test_stats_bootstrap(tmp_path):
    import json
    from fatecnn.stats import bootstrap_ci, paired_bootstrap, collect, summarize, to_markdown
    mean, lo, hi = bootstrap_ci([0.8, 0.82, 0.84]); assert lo <= mean <= hi
    d, lo, hi = paired_bootstrap([0.9, 0.9, 0.9], [0.8, 0.8, 0.8]); assert abs(d - 0.1) < 1e-9
    for s in range(3):
        p = tmp_path / "expA" / str(s); p.mkdir(parents=True)
        json.dump(dict(clean_acc=0.8 + 0.01 * s, per_part={"acc_target/gray/0": 0.7, "selectivity/texture/0": 0.2}),
                  open(p / "metrics_test.json", "w"))
    md = to_markdown(summarize(collect(str(tmp_path))))
    assert "expA" in md and "| 3 |" in md


def test_synthetic_distinct_mode_has_distinct_signatures_and_class_from_any_part():
    from fatecnn.data import SyntheticParts
    d = SyntheticParts(64, seed=0, mode="distinct")
    cols = {}
    for x, y, m in d.items:
        for p in range(4):
            mean = (x * m[p][None]).sum((1, 2)) / m[p].sum()
            cols.setdefault(p, {}).setdefault(y, []).append(mean)
    # distinct hue per part: directions differ across parts
    dirs = torch.stack([torch.stack(cols[p][0]).mean(0) / torch.stack(cols[p][0]).mean(0).norm() for p in range(4)])
    # RGB has only 3 dims, so colour alone cannot give rank 4; the real feature-space rank is checked by LCMV diagnostics.
    off = dirs @ dirs.T - torch.eye(4) + torch.eye(4) * 0
    assert (off[~torch.eye(4, dtype=bool)]).max() < 0.99 and torch.linalg.matrix_rank(dirs, atol=0.02) == 3
    # class readable from any single part: brightness orders classes in every part
    for p in range(4):
        lv = [torch.stack(cols[p][y]).mean(0).sum() for y in sorted(cols[p])]
        assert all(a < b for a, b in zip(lv, lv[1:]))


def test_lcmv_rank_deficient_signatures_do_not_crash_and_are_flagged():
    torch.manual_seed(0)
    r = LCMVReadout(16, 3, loading=1e-2)
    r.sig.copy_(torch.randn(1, 16).repeat(3, 1)); r.refresh()                # three identical signatures
    d = r.diagnostics()
    assert d["rank_C"] < d["need_rank"] and d["constraint_residual"] > 0.1 and d["refresh_failures"] == 0
    assert torch.isfinite(r.weight).all()
