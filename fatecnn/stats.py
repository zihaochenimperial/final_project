"""Seed aggregation with percentile-bootstrap CIs (docs/02_protocol.md). With only 3 seeds CIs are wide; report honestly."""
import json, os
import numpy as np


def bootstrap_ci(x, n=10000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    if len(x) == 1:
        return float(x[0]), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_bootstrap(a, b, n=10000, seed=0):
    """Mean difference a-b over paired seeds and CI."""
    return bootstrap_ci(np.asarray(a, float) - np.asarray(b, float), n, seed=seed)


def scalar_metrics(res, areas=("0.2",)):
    """Flatten one metrics_*.json into scalars."""
    out = {"clean_acc": res["clean_acc"]}
    for f, d in res["part_occ_acc"].items():
        out[f"part_occ_acc/{f}"] = float(np.mean(list(d.values())))
    for f, d in res["silence"].items():
        out[f"silence/{f}"] = float(np.mean(list(d.values())))
    for f, d in res["stability"].items():
        out[f"stability/{f}"] = float(np.mean(list(d.values())))
    for f, d in res["block_acc"].items():
        for a in areas:
            out[f"block{a}/{f}"] = d[a]
    return out


def collect(results_dir, split="test"):
    """-> {exp: {seed: scalars}} from results/<exp>/<seed>/metrics_<split>.json (exp may contain '/')."""
    runs = {}
    for root, _, files in os.walk(results_dir):
        fn = f"metrics_{split}.json"
        if fn in files:
            rel = os.path.relpath(root, results_dir); exp, seed = os.path.split(rel)
            runs.setdefault(exp, {})[seed] = scalar_metrics(json.load(open(os.path.join(root, fn))))
    return runs


def summarize(runs):
    rows = []
    for exp, seeds in sorted(runs.items()):
        keys = sorted(next(iter(seeds.values())))
        row = {"exp": exp, "n_seeds": len(seeds)}
        for k in keys:
            row[k] = bootstrap_ci([s[k] for s in seeds.values()])
        rows.append(row)
    return rows


def to_markdown(rows, keys=("clean_acc", "part_occ_acc/gray", "part_occ_acc/texture", "silence/gray", "silence/texture", "stability/gray", "block0.2/gray")):
    h = "| exp | seeds | " + " | ".join(keys) + " |\n|" + "---|" * (len(keys) + 2) + "\n"
    f = lambda t: f"{t[0]:.3f} [{t[1]:.3f},{t[2]:.3f}]" if t[1] == t[1] else f"{t[0]:.3f}"
    return h + "\n".join(f"| {r['exp']} | {r['n_seeds']} | " + " | ".join(f(r[k]) if k in r else "-" for k in keys) + " |" for r in rows) + "\n"
