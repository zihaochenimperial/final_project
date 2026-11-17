"""Unified environment protocol (docs/05_unified_protocol.md).
Same code everywhere; differences come ONLY from  (1) FATE_* env vars  (2) configs/profiles/<profile>.yaml.
Precedence: experiment config < profile < --set overrides."""
import hashlib, json, os, platform, socket, subprocess, time
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def paths():
    return {
        "FATE_DATA": os.path.abspath(os.environ.get("FATE_DATA", os.path.join(REPO, "data"))),
        "FATE_RUNS": os.path.abspath(os.environ.get("FATE_RUNS", os.path.join(REPO, "runs"))),
    }


def detect_profile():
    if os.environ.get("FATE_PROFILE"):
        return os.environ["FATE_PROFILE"]
    if os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID"):
        return "hpc"
    try:
        import torch
        return "gpu" if torch.cuda.is_available() else "local"
    except Exception:
        return "local"


def _expand(v):
    if isinstance(v, str):
        for k, p in paths().items():
            v = v.replace("${%s}" % k, p)
        return os.path.expandvars(v)
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    return v


def load_config(path, overrides=(), profile=None):
    cfg = yaml.safe_load(open(path))
    profile = profile or detect_profile()
    pp = os.path.join(REPO, "configs", "profiles", f"{profile}.yaml")
    if os.path.exists(pp):
        cfg.update(yaml.safe_load(open(pp)) or {})
    for kv in overrides:
        k, v = kv.split("=", 1); cfg[k] = yaml.safe_load(v)
    cfg = _expand(cfg)
    cfg["profile"] = profile
    return cfg


def _git(*a):
    try:
        return subprocess.check_output(["git", "-C", REPO, *a], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def write_manifest(out, cfg):
    import torch
    m = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": socket.gethostname(), "profile": cfg.get("profile"),
        "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(), "torch": torch.__version__,
        "device": "cuda:" + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "config_sha": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12],
        "job_id": os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID"),
    }
    os.makedirs(out, exist_ok=True)
    json.dump(m, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    return m
