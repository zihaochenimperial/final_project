"""Single-factor follow-ups on the synthetic 'distinct' mode (one change at a time, everything else = scripts/sanity.py).
python scripts/single_factor.py out.md"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), ".."))
import contextlib, io, sys
from torch.utils.data import DataLoader
from fatecnn.train import run, build_dataset
from fatecnn.evaluate import evaluate, calibrate
from fatecnn.stats import scalar_metrics

base = dict(dataset="synthetic", synthetic_mode="distinct", arch="resnet18", n=512, n_eval=128, size=64, num_parts=4,
            num_classes=4, batch_size=32, lr=3e-4, epochs=20, lambda_sil=0.1, lambda_stab=0.1, lambda_erf=0.0,
            device="cpu", workers=0, seed=0)
RUNS = {"fate_learned | readout_init=default": dict(method="fate_learned", readout_init="default"),
        "fate_lcmv | loading=1.0": dict(method="fate", loading=1.0),
        "fate_learned | init=lcmv, loading=1.0": dict(method="fate_learned", loading=1.0)}
f = lambda v: "-" if v != v else f"{v:.3f}"
with open(sys.argv[1], "a") as fh:
    fh.write("\n## single-factor follow-ups (distinct mode, seed 0, test split)\n\n| run | clean | acc_target/tex | silence/tex | silence_ctrl/tex | silence_sel/tex | lcmv_diag |\n|---|---|---|---|---|---|---|\n")
    for name, ov in RUNS.items():
        cfg = {**base, **ov}
        with contextlib.redirect_stdout(io.StringIO()):
            m = run(cfg)
        dv, K, C = build_dataset(cfg, "val"); thr = calibrate(m, DataLoader(dv, batch_size=64), K, "cpu")["active_thr"]
        ds, _, _ = build_dataset(cfg, "test"); r = scalar_metrics(evaluate(m, DataLoader(ds, batch_size=64), K, "cpu", thr))
        dg = m.readout.diagnostics() if hasattr(m.readout, "diagnostics") else "-"
        if dg != "-": dg = {k: round(v, 3) for k, v in dg.items() if k in ("rank_C", "cond_G", "constraint_residual", "w_norm_max")}
        fh.write(f"| {name} | {f(r['clean_acc'])} | {f(r['acc_target/texture'])} | {f(r['silence/texture'])} | {f(r['silence_control/texture'])} | {f(r['silence_selectivity/texture'])} | {dg} |\n"); fh.flush()
