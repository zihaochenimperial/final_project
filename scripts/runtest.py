"""~10 min end-to-end run test on THIS machine (synthetic data, CPU). Exercises the real CLI path:
train -> thresholds(val) -> evaluate(test) -> collect -> aggregate, for every arm, and ASSERTS the outputs.
  python scripts/runtest.py [--budget-min 10]
It checks plumbing/correctness only; the numbers are NOT results."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), ".."))
import argparse, json, math, os, shutil, subprocess, sys, tempfile, time

ap = argparse.ArgumentParser(); ap.add_argument("--budget-min", type=float, default=10.0); ap.add_argument("--n", type=int, default=512)
ap.add_argument("--epochs", type=int, default=16); a = ap.parse_args()
t0 = time.time(); fails = []
def check(name, cond, info=""):
    print(f"[{'OK' if cond else 'FAIL'}] {name} {info}", flush=True)
    if not cond: fails.append(name)
def sh(cmd, env=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env={**os.environ, **(env or {})})
    return r.returncode, r.stdout + r.stderr

tmp = tempfile.mkdtemp(prefix="fate_runtest_"); env = {"FATE_RUNS": os.path.join(tmp, "runs"), "FATE_DATA": os.path.join(tmp, "data"), "FATE_PROFILE": "local"}
print("workdir", tmp)
rc, out = sh(f"{sys.executable} -m pytest -q -x", env); check("unit tests", rc == 0, out.strip().splitlines()[-1])
ARMS = ["gap_resnet", "bottleneck", "cutout", "multitask", "fate_learned", "fate"]
common = f"n={a.n} n_eval=128 epochs={a.epochs} batch_size=32 size=64 lr=0.0003 synthetic_mode=missing seed=0"   # 'missing': LCMV-valid construction
per_arm = max(30.0, (a.budget_min * 60 - (time.time() - t0)) / len(ARMS) - 5)
for arm in ARMS:
    if time.time() - t0 > a.budget_min * 60:
        check(f"{arm} (skipped: budget)", False); continue
    out_name = f"runtest/{arm}/0"; ta = time.time()
    rc, o = sh(f"{sys.executable} -m fatecnn.train --config configs/synthetic.yaml --set method={arm} {common} out={out_name}", env)
    check(f"{arm}: train exit", rc == 0, o.strip().splitlines()[-1][:150] if o.strip() else "")
    if rc: continue
    rc, o = sh(f"{sys.executable} -m fatecnn.evaluate --run {out_name} --split test", env); check(f"{arm}: evaluate exit", rc == 0, o.strip()[-80:])
    d = os.path.join(env["FATE_RUNS"], out_name)
    need = ["config.yaml", "manifest.json", "ckpt.pt", "metrics_test.json"] + ([] if arm == "gap_resnet" else ["thresholds.json"])
    check(f"{arm}: artifacts", all(os.path.exists(os.path.join(d, f)) for f in need), f"{time.time()-ta:.0f}s")
    if rc: continue
    m = json.load(open(os.path.join(d, "metrics_test.json")))
    vals = [v for v in m["per_part"].values() if isinstance(v, float)]
    check(f"{arm}: metrics finite & in range", 0 <= m["clean_acc"] <= 1 and all(not math.isinf(v) for v in vals))
    man = json.load(open(os.path.join(d, "manifest.json"))); check(f"{arm}: manifest", man["profile"] == "local" and man["config_sha"])
    if arm in ("fate",):
        rows = [json.loads(l) for l in open(os.path.join(d, "lcmv_diag.jsonl"))]
        check("fate: lcmv_diag logged every epoch", len(rows) == a.epochs, f"last rank={rows[-1]['rank_C']}/{rows[-1]['need_rank']} resid={rows[-1]['constraint_residual']:.2g} cond_G={rows[-1]['cond_G']:.1e}")
        check("fate: LCMV valid every epoch (G1)", all(r['rank_C'] == r['need_rank'] and r['constraint_residual'] < 1e-2 for r in rows))
        check("fate: no solver failures", rows[-1]["refresh_failures"] == 0)
    if arm == "fate_learned":
        check("fate_learned: eval used thresholds from val", json.load(open(os.path.join(d, "thresholds.json")))["split"] == "val")
# determinism: same seed -> same metrics (CPU)
if time.time() - t0 < a.budget_min * 60 - 60:
    rc, _ = sh(f"{sys.executable} -m fatecnn.train --config configs/synthetic.yaml --set method=bottleneck {common} out=runtest/bottleneck_rep/0", env)
    sh(f"{sys.executable} -m fatecnn.evaluate --run runtest/bottleneck_rep/0 --split test", env)
    r = lambda n: json.load(open(os.path.join(env["FATE_RUNS"], n, "metrics_test.json")))["clean_acc"]
    check("determinism (same seed, bottleneck)", abs(r("runtest/bottleneck/0") - r("runtest/bottleneck_rep/0")) < 1e-9, f"{r('runtest/bottleneck/0'):.4f} vs {r('runtest/bottleneck_rep/0'):.4f}")
# collect + aggregate into a scratch results dir (do not pollute the repo)
res = os.path.join(tmp, "results"); os.makedirs(res)
rc, _ = sh(f"{sys.executable} -c \"import shutil,os,sys; sys.path.insert(0,'.'); from fatecnn.env import paths; [shutil.copytree(os.path.join(paths()['FATE_RUNS'],'runtest'),os.path.join('{res}','runtest'),dirs_exist_ok=True)]\"", env)
rc, o = sh(f"{sys.executable} scripts/aggregate.py {res}", env); check("aggregate -> summary.md", rc == 0 and os.path.exists(os.path.join(res, "summary.md")))
print(open(os.path.join(res, "summary.md")).read() if os.path.exists(os.path.join(res, "summary.md")) else o)
print(f"elapsed {(time.time()-t0)/60:.1f} min; failures: {fails or 'none'}")
shutil.rmtree(tmp, ignore_errors=True); sys.exit(1 if fails else 0)
