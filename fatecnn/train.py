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
        ds = SyntheticParts(cfg["n"], cfg["size"], cfg["num_parts"], cfg["num_classes"], seed=0 if split == "train" else 1)
        return ds, cfg["num_parts"], cfg["num_classes"]
    from .datasets import CubParts, FolderParts
    if name == "cub":
        ds = CubParts(cfg["root"], split, cfg["size"], flip=(split == "train"))
        return ds, ds.K, 200
    if name == "folder":
        ds = FolderParts(cfg["root"], split, cfg["size"])
        return ds, cfg["num_parts"], cfg["num_classes"]
    raise ValueError(name)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def run(cfg):
    seed_all(cfg.get("seed", 0))
    dev = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    tr, K, C = build_dataset(cfg, "train")
    dl = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("workers", 0), drop_last=False)
    va, _, _ = build_dataset(cfg, "val" if cfg.get("dataset") != "synthetic" else "test")
    vdl = DataLoader(va, batch_size=cfg["batch_size"], num_workers=cfg.get("workers", 0))
    model = build_model(cfg, C, K).to(dev)
    method = cfg.get("method", "fate")
    backbone = list(model.stem.parameters()) + list(model.body.parameters())
    opt = torch.optim.AdamW([{"params": backbone, "lr": cfg["lr"] * cfg.get("backbone_lr_mult", 1.0)},
                             {"params": model.head.parameters(), "lr": cfg["lr"]}], weight_decay=cfg.get("wd", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    out = cfg.get("out")
    if out:
        os.makedirs(out, exist_ok=True); yaml.safe_dump(cfg, open(os.path.join(out, "config.yaml"), "w"))
        write_manifest(out, cfg)
    step = 0
    for ep in range(cfg["epochs"]):
        model.train()
        for x, y, m in dl:
            x, y, m = x.to(dev), y.to(dev), m.to(dev)
            p = random.randrange(K)
            use_erf = method.startswith("fate") and cfg["lambda_erf"] > 0 and step % cfg.get("erf_every", 4) == 0
            x.requires_grad_(use_erf)
            logits, a = model(x, m)
            ce = F.cross_entropy(logits, y)
            if method == "plain":
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
                    erf = erf_leak_loss(a, x, m, p) if use_erf else None
                    loss = fate_loss(ce, silence_loss(ae, p), stability_loss(ae, a, p), erf,
                                     cfg["lambda_sil"], cfg["lambda_stab"], cfg["lambda_erf"])
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
        sched.step()
        model.eval(); c = n = 0; sr = 0.0
        with torch.no_grad():
            for x, y, m in vdl:
                x, y, m = x.to(dev), y.to(dev), m.to(dev)
                lg, a = model(x); _, ae = model(occlude_part(x, m, 0, "gray"))
                c += (lg.argmax(1) == y).sum().item(); n += len(y); sr += metrics.silence_rate(a, ae, 0) * len(y)
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
