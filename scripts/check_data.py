"""Verify the dataset layout under $FATE_DATA is identical on every machine.  python scripts/check_data.py"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), ".."))
import os, sys
from fatecnn.env import paths

REQ = {"CUB_200_2011": ["images.txt", "image_class_labels.txt", "train_test_split.txt", "parts/part_locs.txt", "images"]}
d = paths()["FATE_DATA"]; bad = 0
for ds, files in REQ.items():
    base = os.path.join(d, ds)
    for f in files:
        ok = os.path.exists(os.path.join(base, f)); bad += not ok
        print(("OK  " if ok else "MISS"), os.path.join(base, f))
sys.exit(1 if bad else 0)
