import argparse, os, random
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .data import SyntheticParts
from .losses import fate_loss, silence_loss, stability_loss, erf_leak_loss
from .model import FATECNN
from .occlusion import TRAIN_FILLS, occlude_part
from . import metrics


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
    model = FATECNN(C, K, cfg["arch"], pretrained=cfg.get("pretrained", False)).to(dev)
    backbone = list(model.stem.parameters()) + list(model.body.parameters())
    opt = torch.optim.AdamW([{"params": backbone, "lr": cfg["lr"] * cfg.get("backbone_lr_mult", 1.0)},
                             {"params": model.head.parameters(), "lr": cfg["lr"]}], weight_decay=cfg.get("wd", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    out = cfg.get("out")
    if out:
        os.makedirs(out, exist_ok=True); yaml.safe_dump(cfg, open(os.path.join(out, "config.yaml"), "w"))
    step = 0
    for ep in range(cfg["epochs"]):
        model.train()
        for x, y, m in dl:
            x, y, m = x.to(dev), y.to(dev), m.to(dev)
            p = random.randrange(K)
            use_erf = cfg["lambda_erf"] > 0 and step % cfg.get("erf_every", 4) == 0
            x.requires_grad_(use_erf)
            logits, a = model(x, m)
            _, ae = model(occlude_part(x.detach(), m, p, random.choice(TRAIN_FILLS)), m, update=False)
            erf = erf_leak_loss(a, x, m, p) if use_erf else None
            loss = fate_loss(F.cross_entropy(logits, y), silence_loss(ae, p), stability_loss(ae, a, p), erf,
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
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides, e.g. seed=1 out=runs/x/1")
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config))
    for kv in a.set:
        k, v = kv.split("=", 1); cfg[k] = yaml.safe_load(v)
    run(cfg)
