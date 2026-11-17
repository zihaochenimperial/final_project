import argparse, os, random
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .data import SyntheticParts
from .losses import fate_loss, silence_loss, stability_loss, erf_leak_loss
from .model import build_model
from .occlusion import TRAIN_FILLS, occlude_part
from . import metrics
from .env import load_config, paths, write_manifest


def build_dataset(cfg, split):
    name = cfg.get("dataset", "synthetic")
    if name == "synthetic":
        from .data import SPLIT_SEED
        n = cfg["n"] if split == "train" else cfg.get("n_eval", cfg["n"])
        ds = SyntheticParts(n, cfg["size"], cfg["num_parts"], cfg["num_classes"], seed=SPLIT_SEED[split],
                            mode=cfg.get("synthetic_mode", "redundant"), shift=(split == "test_shift"))
        return ds, cfg["num_parts"], cfg["num_classes"]
    from .datasets import CubParts, FolderParts
    if name == "cub":
        ds = CubParts(cfg["root"], split, cfg["size"], flip=(split == "train"))
        return ds, ds.K, 200
    if name == "folder":
        ds = FolderParts(cfg["root"], split, cfg["size"])
        return ds, cfg["num_parts"], cfg["num_classes"]
    raise ValueError(name)


BACKBONE_PREFIXES = ("stem", "body", "layer4")


def make_optimizer(model, cfg):
    """Every trainable parameter must be in exactly one group (guards against silently frozen read-outs)."""
    bb = [p for n, p in model.named_parameters() if n.startswith(BACKBONE_PREFIXES)]
    rest = [p for n, p in model.named_parameters() if not n.startswith(BACKBONE_PREFIXES)]
    groups = [{"params": bb, "lr": cfg["lr"] * cfg.get("backbone_lr_mult", 1.0)}, {"params": rest, "lr": cfg["lr"]}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=cfg.get("wd", 1e-4))
    seen = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(seen) == len(set(seen)) == sum(1 for p in model.parameters() if p.requires_grad), "optimizer misses/duplicates params"
    return opt


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def run(cfg):
    seed_all(cfg.get("seed", 0))
    dev = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    tr, K, C = build_dataset(cfg, "train")
    dl = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("workers", 0), drop_last=False)
    va, _, _ = build_dataset(cfg, "val")
    vdl = DataLoader(va, batch_size=cfg["batch_size"], num_workers=cfg.get("workers", 0))
    model = build_model(cfg, C, K).to(dev)
    method = cfg.get("method", "fate")
    opt = make_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    out = cfg.get("out")
    if out:
        os.makedirs(out, exist_ok=True); yaml.safe_dump(cfg, open(os.path.join(out, "config.yaml"), "w"))
        write_manifest(out, cfg)
    step = 0
    for ep in range(cfg["epochs"]):
        model.train()
        warm = min(1.0, (ep + 1) / max(1.0, cfg.get("warmup_frac", 0.3) * cfg["epochs"]))  # ramp erasure losses
        for x, y, m in dl:
            x, y, m = x.to(dev), y.to(dev), m.to(dev)
            p = random.randrange(K)
            use_erf = method.startswith("fate") and cfg["lambda_erf"] > 0 and step % cfg.get("erf_every", 4) == 0
            x.requires_grad_(use_erf)
            logits, a = model(x, m)
            ce = F.cross_entropy(logits, y)
            if not getattr(model, "has_parts", True):
                opt.zero_grad(); ce.backward(); opt.step(); step += 1; loss = ce; continue
            if method in ("gap_resnet", "bottleneck"):
                loss = ce
            elif method == "multitask":  # same part supervision, no null / silence constraint
                tgt = F.interpolate(m, size=a.shape[2:], mode="area")
                loss = ce + cfg.get("lambda_seg", 1.0) * F.binary_cross_entropy_with_logits(model.raw, tgt)
            else:
                fill = random.choice(TRAIN_FILLS)
                xo = occlude_part(x.detach(), m, p, fill)
                if method == "cutout":  # same occlusion amount, classification-only
                    loss = ce + F.cross_entropy(model(xo, m, update=False)[0], y)
                else:                   # fate / fate_learned
                    _, ae = model(xo, m, update=False)
                    has = m[:, p].flatten(1).sum(1) > 0   # erasing an absent part is a no-op: exclude
                    if has.any():
                        erf = erf_leak_loss(a, x, m, p) if use_erf else None
                        loss = fate_loss(ce, silence_loss(ae[has], p), stability_loss(ae[has], a[has], p), erf,
                                         warm * cfg["lambda_sil"], warm * cfg["lambda_stab"], warm * cfg["lambda_erf"])
                    else:
                        loss = ce
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
        sched.step()
        model.eval(); c = n = 0; sr = 0.0
        with torch.no_grad():
            for x, y, m in vdl:
                x, y, m = x.to(dev), y.to(dev), m.to(dev)
                lg, a = model(x); c += (lg.argmax(1) == y).sum().item(); n += len(y)
                if a is not None:
                    _, ae = model(occlude_part(x, m, 0, "gray")); sr += metrics.silence_rate(a, ae, 0) * len(y)
        print(f"ep{ep} loss={loss.item():.3f} val_acc={c/n:.3f} val_silence(p0)={sr/n:.3f}", flush=True)
        if out:
            torch.save(model.state_dict(), os.path.join(out, "ckpt.pt"))
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/synthetic.yaml")
    ap.add_argument("--profile", default=None, help="local|gpu|hpc (default: auto-detect / $FATE_PROFILE)")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides, e.g. seed=1")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set, a.profile)
    if cfg.get("out") and not os.path.isabs(cfg["out"]):  # runs always land under $FATE_RUNS
        cfg["out"] = os.path.join(paths()["FATE_RUNS"], cfg["out"])
    run(cfg)
