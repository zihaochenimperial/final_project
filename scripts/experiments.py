"""Experiment matrix (docs/02_protocol.md). Single source of truth for what gets run.
  python scripts/experiments.py list                 # names
  python scripts/experiments.py jobs [cfg] > hpc/jobs.txt   # one shell command per (experiment, seed)
"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), ".."))
import sys

SEEDS = (0, 1, 2)
# STAGE 1 = minimal mechanism test (docs/07_acceptance.md): R18, lambda_erf=0, no R50, no ERF regulariser.
# name -> overrides on the base config
STAGE1 = {
    "gap_resnet": dict(method="gap_resnet"),                       # standard GAP+FC ResNet
    "bottleneck": dict(method="bottleneck"),                       # capacity-matched K-channel bottleneck, CE only
    "cutout": dict(method="cutout"),                               # + same-amount part occlusion
    "multitask": dict(method="multitask"),                         # + part-mask supervision
    "fate_learned": dict(method="fate_learned"),                   # learned read-out (LCMV-init) + erasure losses
    "fate_lcmv_region": dict(method="fate", sig_mode="region"),    # closed-form LCMV read-out + erasure losses
    # same read-out family, loss ablations (silence / stability)
    "fate_learned_no_sil": dict(method="fate_learned", lambda_sil=0.0),
    "fate_learned_no_stab": dict(method="fate_learned", lambda_stab=0.0),
    "fate_lcmv_no_sil": dict(method="fate", lambda_sil=0.0),
    "fate_lcmv_no_stab": dict(method="fate", lambda_stab=0.0),
}
# DEFERRED until STAGE 1 passes: 3-arm signature (contour/edge), ERF regulariser, R50, CUB full matrix.
DEFERRED = {
    "fate_lcmv_contour": dict(method="fate", sig_mode="contour"),
    "fate_lcmv_edge": dict(method="fate", sig_mode="edge"),
    "fate_lcmv_erf": dict(method="fate", lambda_erf=0.1),
}
MATRIX = STAGE1


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
