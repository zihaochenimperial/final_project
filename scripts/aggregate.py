"""python scripts/aggregate.py [results_dir] -> results/summary.md (mean + 95% bootstrap CI over seeds)."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), ".."))
import sys, os
from fatecnn.env import REPO
from fatecnn.stats import collect, summarize, to_markdown

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "results")
rows = summarize(collect(d))
md = to_markdown(rows)
open(os.path.join(d, "summary.md"), "w").write(md); print(md)
