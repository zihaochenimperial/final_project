"""Evaluate: clean acc, per-part occluded acc, silence rate, other-filter stability, random-block acc.
Usage: python -m fatecnn.evaluate --run runs/<name>/<seed> [--split test]"""
import argparse, json, os
import torch
import yaml
from torch.utils.data import DataLoader

from . import metrics
from .model import FATECNN
from .occlusion import TEST_FILLS, occlude_mask, occlude_part, random_block_mask
from .train import build_dataset


@torch.no_grad()
def evaluate(model, dl, K, device, areas=(0.1, 0.2, 0.3)):
    model.eval()
    out = {"clean_acc": 0.0, "part_occ_acc": {}, "silence": {}, "stability": {}, "block_acc": {}}
    n = 0
    acc = lambda lg, y: (lg.argmax(1) == y).float().sum().item()
    acc_part = {(f, p): 0.0 for f in TEST_FILLS for p in range(K)}
    sil = {(f, p): 0.0 for f in TEST_FILLS for p in range(K)}; stab = dict(sil); cnt = {p: 0 for p in range(K)}
    blk = {(f, a): 0.0 for f in TEST_FILLS for a in areas}
    for x, y, m in dl:
        x, y, m = x.to(device), y.to(device), m.to(device); B = x.shape[0]; n += B
        lg, a = model(x); out["clean_acc"] += acc(lg, y)
        for p in range(K):
            has = m[:, p].flatten(1).sum(1) > 0  # only samples where part p is visible
            if not has.any():
                continue
            cnt[p] += int(has.sum())
            for f in TEST_FILLS:
                lge, ae = model(occlude_part(x, m, p, f))
                acc_part[(f, p)] += acc(lge[has], y[has])
                sil[(f, p)] += metrics.silence_rate(a[has], ae[has], p) * int(has.sum())
                stab[(f, p)] += metrics.other_stability(a[has], ae[has], p) * int(has.sum())
        for f in TEST_FILLS:
            for ar in areas:
                bm = random_block_mask(B, *x.shape[2:], ar).to(device)
                blk[(f, ar)] += acc(model(occlude_mask(x, bm, f))[0], y)
    out["clean_acc"] /= n
    for f in TEST_FILLS:
        out["part_occ_acc"][f] = {p: acc_part[(f, p)] / max(cnt[p], 1) for p in range(K)}
        out["silence"][f] = {p: sil[(f, p)] / max(cnt[p], 1) for p in range(K)}
        out["stability"][f] = {p: stab[(f, p)] / max(cnt[p], 1) for p in range(K)}
        out["block_acc"][f] = {str(ar): blk[(f, ar)] / n for ar in areas}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--split", default="test")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(a.run, "config.yaml")))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds, K, C = build_dataset(cfg, a.split)
    model = FATECNN(C, K, cfg["arch"]).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.run, "ckpt.pt"), map_location=dev))
    res = evaluate(model, DataLoader(ds, batch_size=cfg["batch_size"]), K, dev)
    json.dump(res, open(os.path.join(a.run, f"metrics_{a.split}.json"), "w"), indent=1)
    print(json.dumps(res["clean_acc"]))
