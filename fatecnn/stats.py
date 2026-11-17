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


def scalar_metrics(res):
    """Flatten one metrics_*.json into scalars (macro-mean over parts; NaN-aware)."""
    pp = res["per_part"]
    def mac(prefix):
        v = [x for k, x in pp.items() if k.startswith(prefix + "/") and x == x]
        return float(np.mean(v)) if v else float("nan")
    out = {"clean_acc": res["clean_acc"]}
    for f in ("gray", "noise", "texture"):
        for name in ("acc_target", "acc_control", "drop_target", "drop_control", "selectivity", "silence",
                     "other_unconnected", "other_cooccluded"):
            out[f"{name}/{f}"] = mac(f"{name}/{f}")
    for f in ("gray", "noise", "texture"):
        out[f"silence_control/{f}"] = mac(f"silence_control/{f}")
        out[f"silence_selectivity/{f}"] = out[f"silence/{f}"] - out[f"silence_control/{f}"]
    for name in ("coverage", "false_alarm_absent", "suppress_matched_drop", "suppress_nonmatched_drop", "overlap_rate"):
        out[name] = mac(name)
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
            row[k] = bootstrap_ci([s[k] for s in seeds.values() if s[k] == s[k]] or [float('nan')])
        rows.append(row)
    return rows


def to_markdown(rows, keys=("clean_acc", "acc_target/gray", "acc_target/texture", "acc_control/texture", "selectivity/gray", "selectivity/texture", "silence/texture", "other_unconnected/texture", "suppress_matched_drop", "suppress_nonmatched_drop")):
    h = "| exp | seeds | " + " | ".join(keys) + " |\n|" + "---|" * (len(keys) + 2) + "\n"
    f = lambda t: f"{t[0]:.3f} [{t[1]:.3f},{t[2]:.3f}]" if t[1] == t[1] else f"{t[0]:.3f}"
    return h + "\n".join(f"| {r['exp']} | {r['n_seeds']} | " + " | ".join(f(r[k]) if k in r else "-" for k in keys) + " |" for r in rows) + "\n"
