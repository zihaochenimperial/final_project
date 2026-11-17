"""Experiment matrix (docs/02_protocol.md). Single source of truth for what gets run.
  python scripts/experiments.py list                 # names
  python scripts/experiments.py jobs [cfg] > hpc/jobs.txt   # one shell command per (experiment, seed)
"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), ".."))
import sys

SEEDS = (0, 1, 2)
# name -> overrides on the base config
MATRIX = {
    # main method and the 3-arm signature comparison (H4)
    "fate_region": dict(method="fate", sig_mode="region"),
    "fate_contour": dict(method="fate", sig_mode="contour"),
    "fate_edge": dict(method="fate", sig_mode="edge"),
    # baselines under identical parameters/budget
    "plain": dict(method="plain"),
    "cutout": dict(method="cutout"),
    "multitask": dict(method="multitask"),
    # ablations: LCMV vs learned readout (H2), erasure losses (H3), ERF
    "abl_learned_readout": dict(method="fate_learned"),
    "abl_no_erf": dict(method="fate", lambda_erf=0.0),
    "abl_no_stab": dict(method="fate", lambda_stab=0.0),
    "abl_no_sil": dict(method="fate", lambda_sil=0.0),
}


def jobs(cfg="configs/cub_r18.yaml"):
    base = _o.path.splitext(_o.path.basename(cfg))[0]
    for name, ov in MATRIX.items():
        for s in SEEDS:
            sets = " ".join(f"{k}={v}" for k, v in ov.items())
            out = f"{base}/{name}/{s}"
            yield (f"python -m fatecnn.train --config {cfg} --set seed={s} {sets} out={out} && "
                   f"python -m fatecnn.evaluate --run {out} --split test")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        print("\n".join(MATRIX))
    elif cmd == "jobs":
        print("\n".join(jobs(*sys.argv[2:3])))
