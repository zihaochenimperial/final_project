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
    assert s[0, 0] > 0.9 and s[0, 1].abs() < 1e-3


def test_forward_and_losses():
    m = FATECNN(4, 4); x = torch.rand(2, 3, 64, 64, requires_grad=True)
    masks = torch.zeros(2, 4, 64, 64); masks[:, 0, :32, :32] = 1
    lg, a = m(x, masks)
    assert lg.shape == (2, 4) and a.shape[1] == 4
    assert silence_loss(a, 0) >= 0 and stability_loss(a, a, 0) == 0
    assert 0 <= erf_leak_loss(a, x, masks, 0) <= 1


def test_train_smoke():
    run(dict(seed=0, arch="resnet18", n=32, size=64, num_parts=4, num_classes=4, batch_size=16,
             lr=1e-3, epochs=1, lambda_sil=1.0, lambda_stab=1.0, lambda_erf=0.0))


def test_occlusion_fills():
    from fatecnn.occlusion import occlude_part, TEST_FILLS, random_block_mask
    x = torch.rand(2, 3, 32, 32); m = torch.zeros(2, 2, 32, 32); m[:, 0, :16] = 1
    for f in TEST_FILLS:
        o = occlude_part(x, m, 0, f)
        assert torch.equal(o[:, :, 16:], x[:, :, 16:]) and not torch.equal(o[:, :, :16], x[:, :, :16])
    assert abs(random_block_mask(2, 32, 32, 0.25).mean().item() - 0.25) < 0.05


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
    model = FATECNN(200, 8)
    res = evaluate(model, DataLoader(te, batch_size=2), 8, "cpu", areas=(0.1,))
    assert 0 <= res["clean_acc"] <= 1 and "texture" in res["silence"]
