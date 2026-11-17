"""Copy small result files (manifest/config/metrics, no checkpoints) from $FATE_RUNS into tracked results/.
Usage: python scripts/collect.py [experiment]   then git add results && git commit && git push"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), ".."))
import os, shutil, sys
from fatecnn.env import paths, REPO

src, dst = paths()["FATE_RUNS"], os.path.join(REPO, "results")
only = sys.argv[1] if len(sys.argv) > 1 else None
for root, _, files in os.walk(src):
    rel = os.path.relpath(root, src)
    if only and not rel.startswith(only):
        continue
    for f in files:
        if f in ("manifest.json", "config.yaml") or f.startswith("metrics_"):
            os.makedirs(os.path.join(dst, rel), exist_ok=True); shutil.copy(os.path.join(root, f), os.path.join(dst, rel, f))
            print("collected", os.path.join(rel, f))
