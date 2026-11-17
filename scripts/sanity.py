"""Mechanism sanity on synthetic data (CPU). NOT a result: go/no-go on the plumbing and early diagnostics only.
Fixed settings by design (size 64, loading 1e-2, amax pooling, lr 3e-4, 20 epochs): do not change several at once.
python scripts/sanity.py [out.md] [mode=distinct|redundant|missing] [seeds=0,1]"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), ".."))
import contextlib, io, sys
import numpy as np
from torch.utils.data import DataLoader
from fatecnn.train import run, build_dataset
from fatecnn.evaluate import evaluate, calibrate
from fatecnn.stats import scalar_metrics

out = sys.argv[1] if len(sys.argv) > 1 else "sanity.md"
mode = sys.argv[2] if len(sys.argv) > 2 else "redundant"
seeds = [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "0,1").split(",")]
base = dict(dataset="synthetic", synthetic_mode=mode, arch="resnet18", n=512, n_eval=128, size=64, num_parts=4, num_classes=4,
            batch_size=32, lr=3e-4, epochs=20, lambda_sil=0.1, lambda_stab=0.1, lambda_erf=0.0, device="cpu", workers=0)
ARMS = {"gap_resnet": dict(method="gap_resnet"), "bottleneck": dict(method="bottleneck"), "cutout": dict(method="cutout"),
        "multitask": dict(method="multitask"), "fate_learned": dict(method="fate_learned"), "fate_lcmv": dict(method="fate")}
cols = ["clean_acc", "acc_target/texture", "acc_control/texture", "silence/texture", "silence_control/texture", "silence_selectivity/texture",
        "other_unconnected/texture", "coverage", "suppress_matched_drop", "suppress_nonmatched_drop"]
f = lambda v: "-" if v != v else f"{v:.3f}"
with open(out, "a") as fh:
    fh.write(f"\n## mode={mode} seeds={seeds}  (test and colour-shift test; thresholds from val)\n\n")
    fh.write("| arm | seed | split | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 3) + "\n"); fh.flush()
    for name, ov in ARMS.items():
        for sd in seeds:
            cfg = {**base, **ov, "seed": sd}
            with contextlib.redirect_stdout(io.StringIO()):
                model = run(cfg)
            ds_v, K, C = build_dataset(cfg, "val")
            thr = calibrate(model, DataLoader(ds_v, batch_size=64), K, "cpu")["active_thr"] if model.has_parts else [0.0] * K
            for split in ("test", "test_shift"):
                ds, _, _ = build_dataset(cfg, split)
                r = scalar_metrics(evaluate(model, DataLoader(ds, batch_size=64), K, "cpu", thr))
                fh.write(f"| {name} | {sd} | {split} | " + " | ".join(f(r[c]) for c in cols) + " |\n"); fh.flush()
            if hasattr(getattr(model, "readout", None), "diagnostics"):
                fh.write(f"|  | {sd} | lcmv_diag | {model.readout.diagnostics()} |\n"); fh.flush()
