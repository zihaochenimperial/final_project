"""Mechanism sanity check on synthetic data (CPU, minutes): does FATE silence erased parts better than baselines?
python scripts/sanity.py [epochs] [n]   -> prints a table. Not a result; only a go/no-go before spending GPU time."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), ".."))
import io, contextlib, sys
import numpy as np
from torch.utils.data import DataLoader
from fatecnn.train import run, build_dataset
from fatecnn.evaluate import evaluate

ep = int(sys.argv[1]) if len(sys.argv) > 1 else 12
n = int(sys.argv[2]) if len(sys.argv) > 2 else 512
base = dict(dataset="synthetic", arch="resnet18", n=n, size=64, num_parts=4, num_classes=4, batch_size=32, lr=1e-3,
            epochs=ep, lambda_sil=1.0, lambda_stab=1.0, lambda_erf=0.0, device="cpu", workers=0, seed=0)
print(f"{'method':<22}{'clean':>7}{'occ(gray)':>10}{'occ(tex)':>10}{'sil(gray)':>10}{'sil(tex)':>10}{'stab':>7}")
for name, ov in {"plain": dict(method="plain"), "cutout": dict(method="cutout"), "multitask": dict(method="multitask"),
                 "fate_learned": dict(method="fate_learned"), "fate_region": dict(method="fate", sig_mode="region"),
                 "fate_contour": dict(method="fate", sig_mode="contour")}.items():
    cfg = {**base, **ov}
    with contextlib.redirect_stdout(io.StringIO()):
        model = run(cfg)
    ds, K, C = build_dataset(cfg, "test")
    r = evaluate(model, DataLoader(ds, batch_size=64), K, "cpu", areas=(0.2,))
    m = lambda d: np.mean(list(d.values()))
    print(f"{name:<22}{r['clean_acc']:>7.3f}{m(r['part_occ_acc']['gray']):>10.3f}{m(r['part_occ_acc']['texture']):>10.3f}"
          f"{m(r['silence']['gray']):>10.3f}{m(r['silence']['texture']):>10.3f}{m(r['stability']['gray']):>7.3f}", flush=True)
