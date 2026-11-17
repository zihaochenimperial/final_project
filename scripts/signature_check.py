"""Gate G0 (docs/07): signature identifiability BEFORE any training. Runs on frozen-backbone features.
Reports part co-occurrence, rank/conditioning of C=[s_1..s_K,1], and the closed-form constraint residual.
  python scripts/signature_check.py --config configs/rival10_r18.yaml --split train [--batches 20]
Synthetic demo: python scripts/signature_check.py --synthetic distinct|redundant|missing   (random-init backbone)"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), ".."))
import argparse, json
import torch
from torch.utils.data import DataLoader
from fatecnn.env import load_config
from fatecnn.lcmv import LCMVReadout
from fatecnn.model import build_model
from fatecnn.train import build_dataset

ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--split", default="train")
ap.add_argument("--synthetic"); ap.add_argument("--batches", type=int, default=20); ap.add_argument("--sig_mode", default="region")
a = ap.parse_args()
if a.synthetic:
    cfg = dict(dataset="synthetic", synthetic_mode=a.synthetic, arch="resnet18", n=256, size=64, num_parts=4, num_classes=4,
               batch_size=32, method="fate", pretrained=False)
else:
    cfg = load_config(a.config)
ds, K, C = build_dataset(cfg, a.split)
cfg["method"] = "fate"; cfg["sig_mode"] = a.sig_mode
torch.manual_seed(0); model = build_model(cfg, C, K).eval()
r = LCMVReadout(model.readout.C, K, momentum=0.2, sig_mode=a.sig_mode, min_cov=1.0)
vis_all = []
with torch.no_grad():
    for i, (x, y, m) in enumerate(DataLoader(ds, batch_size=cfg.get("batch_size", 32), shuffle=True)):
        if i >= a.batches: break
        r.update_stats(model.features(x), m, x); vis_all.append(m.flatten(2).sum(2) > 0)
r.refresh()
V = torch.cat(vis_all).float()
cooc = (V.T @ V) / V.sum(0)[:, None].clamp_min(1)           # P(q visible | p visible)
out = dict(visible_rate=V.mean(0).tolist(), cooccurrence_min=float(cooc.min()), diagnostics=r.diagnostics())
print(json.dumps(out, indent=1))
d = out["diagnostics"]; ok = d["rank_C"] == d["need_rank"] and d["constraint_residual"] < 1e-3
print("G0", "PASS" if ok else "FAIL (rank-deficient or inconsistent constraints; LCMV arm invalid on this data)")
