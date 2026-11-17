"""Pre-training readiness check. Run on EVERY machine before launching:  python scripts/preflight.py [--fast]
Checks env, profile, device, data layout, then one tiny train+eval step for every method/signature mode on synthetic data."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), ".."))
import subprocess, sys, tempfile, time
import torch
from fatecnn.env import detect_profile, paths, REPO, _git
from fatecnn.train import run
from fatecnn.evaluate import evaluate
from fatecnn.train import build_dataset
from fatecnn.model import build_model
from experiments import MATRIX
from torch.utils.data import DataLoader

ok = True
def rep(name, good, info=""):
    global ok; ok &= bool(good); print(f"[{'OK' if good else 'FAIL'}] {name} {info}")

rep("python/torch", True, f"{sys.version.split()[0]} / torch {torch.__version__}")
rep("profile", True, detect_profile())
rep("device", True, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu (no CUDA)")
dirty = bool(_git("status", "--porcelain"))
if detect_profile() == "hpc":
    rep("git clean", not dirty, "(dirty runs are not reportable)")
else:
    print("[WARN] uncommitted changes (not reportable)" if dirty else "[OK] git clean", (_git("rev-parse", "HEAD") or "?")[:8])
d = paths(); print("paths:", d)
r = subprocess.run([sys.executable, _o.path.join(REPO, "scripts", "check_data.py")], capture_output=True, text=True)
print("[WARN] CUB data not found (needed for real runs)" if r.returncode else "[OK] CUB data layout")
if "--fast" not in sys.argv:
    base = dict(dataset="synthetic", arch="resnet18", n=16, size=64, num_parts=4, num_classes=4, batch_size=8,
                lr=1e-3, epochs=1, lambda_sil=1.0, lambda_stab=1.0, lambda_erf=0.1, erf_every=1, device="cpu", workers=0)
    for name, ov in MATRIX.items():
        t = time.time()
        try:
            cfg = {**base, **ov}
            with tempfile.TemporaryDirectory() as td:
                import io, contextlib
                with contextlib.redirect_stdout(io.StringIO()):
                    model = run({**cfg, "out": td})
                ds, K, C = build_dataset(cfg, "test")
                res = evaluate(model, DataLoader(ds, batch_size=8), K, "cpu", areas=(0.2,))
            rep(f"smoke {name}", 0 <= res["clean_acc"] <= 1, f"{time.time()-t:.1f}s")
        except Exception as e:
            rep(f"smoke {name}", False, repr(e)[:120])
print("READY" if ok else "NOT READY"); sys.exit(0 if ok else 1)
