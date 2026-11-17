"""Part-level intervention evaluation. Masks are used ONLY here to build interventions; the model never receives them
at test time (forward(x) only), so they cannot enter the gate or the signature/covariance updates.
Thresholds (per-part 'active' score) are defined on the VAL split and stored in thresholds.json; test never sets them.

Per part p, per fill f (train fills gray/noise, unseen texture), over samples where p is visible in the mask AND active
when clean (clean-response coverage), all on the SAME image (paired):
  paired_drop_target   : s_p(clean) - s_p(p occluded)           [pre-gate, pre-ReLU score = max over space of raw map]
  paired_drop_control  : s_p(clean) - s_p(same-area NON-target region occluded)
  selectivity          : paired_drop_target - paired_drop_control
  silence              : fraction with s_p(occluded p) below the val active threshold
  other_rel_change     : |s_q(occ p) - s_q(clean)| / |s_q(clean)|, q != p visible+active, split into
                         unconnected (mask q does not overlap mask p) and co-occluded (overlap > 0)
  acc_*                : Top-1 clean / target-occluded / control-occluded on the same visible subset
Natural invisibility/unknown (p absent in the mask) is reported SEPARATELY (false_alarm_absent), never mixed with
artificial occlusion. Suppression probe: zero evidence of channel p vs the mean over non-matched channels q != p
(equal count: one channel), accuracy drop on samples where p is visible."""
import argparse, json, os
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .env import load_config, paths
from .model import build_model
from .occlusion import TEST_FILLS, occlude_mask, occlude_part, shifted_mask
from .train import build_dataset


def _score(model, x, suppress=None):
    lg, _ = model(x, suppress=suppress)
    return lg, model.raw.amax((2, 3))


@torch.no_grad()
def calibrate(model, dl, K, device):
    """Per-part active threshold = 0.5 x median clean score over visible val samples. Defined on VAL only."""
    model.eval(); sc = [[] for _ in range(K)]
    for x, y, m in dl:
        _, s = _score(model, x.to(device)); vis = m.flatten(2).sum(2) > 0
        for p in range(K):
            sc[p].append(s[:, p].cpu()[vis[:, p]])
    thr = [0.5 * float(torch.cat(v).median()) if len(torch.cat(v)) else 0.0 for v in sc]
    return {"active_thr": thr, "split": "val"}


class _Acc(dict):
    def add(self, k, v, n=None):
        t = self.setdefault(k, [0.0, 0])
        v = torch.as_tensor(v, dtype=torch.float64).flatten()
        t[0] += float(v.sum()); t[1] += int(v.numel() if n is None else n)

    def mean(self, k):
        t = self.get(k); return t[0] / t[1] if t and t[1] else float("nan")


@torch.no_grad()
def evaluate(model, dl, K, device, thr, fills=TEST_FILLS, seed=0):
    model.eval(); gen = torch.Generator().manual_seed(seed); A = _Acc()
    parts = getattr(model, "has_parts", True)
    thr_t = torch.tensor(thr, device=device)
    for x, y, m in dl:
        x, y, m = x.to(device), y.to(device), m.to(device)
        lg0, s0 = (None, None)
        if parts:
            lg0, s0 = _score(model, x)
        else:
            lg0 = model(x)[0]
        A.add("clean_acc", lg0.argmax(1) == y)
        vis = m.flatten(2).sum(2) > 0
        act = vis & (s0 > thr_t[None]) if parts else vis
        # overlap[b,p,q] = |mask_p ∩ mask_q| / |mask_q|
        mf = m.flatten(2); inter = torch.einsum("bpn,bqn->bpq", mf, mf); ov = inter / mf.sum(2)[:, None, :].clamp_min(1)
        for p in range(K):
            sel = act[:, p]
            A.add(f"coverage/{p}", sel[vis[:, p]], n=int(vis[:, p].sum()))
            if parts and (~vis[:, p]).any():
                A.add(f"false_alarm_absent/{p}", (s0[:, p] > thr_t[p])[~vis[:, p]])
            if int(vis[:, p].sum()) == 0:
                continue
            A.add(f"acc_clean_vis/{p}", (lg0.argmax(1) == y)[vis[:, p]])
            mc = shifted_mask(m[:, p].cpu(), gen).to(device)
            for f in fills:
                xo = occlude_part(x, m, p, f, gen); xc = occlude_mask(x, mc, f, gen)
                lgo = model(xo)[0] if not parts else None
                if parts:
                    lgo, so = _score(model, xo); lgc, sc_ = _score(model, xc)
                else:
                    lgc = model(xc)[0]
                A.add(f"acc_target/{f}/{p}", (lgo.argmax(1) == y)[vis[:, p]])
                A.add(f"acc_control/{f}/{p}", (lgc.argmax(1) == y)[vis[:, p]])
                if not parts:
                    continue
                if sel.any():
                    dt = (s0[:, p] - so[:, p])[sel]; dc = (s0[:, p] - sc_[:, p])[sel]
                    A.add(f"drop_target/{f}/{p}", dt); A.add(f"drop_control/{f}/{p}", dc)
                    A.add(f"selectivity/{f}/{p}", dt - dc)
                    A.add(f"silence/{f}/{p}", so[:, p][sel] < thr_t[p])
                    A.add(f"silence_control/{f}/{p}", sc_[:, p][sel] < thr_t[p])
                for q in range(K):
                    if q == p:
                        continue
                    rel_ok = act[:, q]
                    if not rel_ok.any():
                        continue
                    rel = ((so[:, q] - s0[:, q]).abs() / s0[:, q].abs().clamp_min(1e-3))
                    uncon = rel_ok & (ov[:, p, q] == 0) & (ov[:, q, p] == 0); co = rel_ok & ~uncon & vis[:, p]
                    if uncon.any(): A.add(f"other_unconnected/{f}/{p}", rel[uncon])
                    if co.any(): A.add(f"other_cooccluded/{f}/{p}", rel[co])
                    A.add(f"overlap_rate/{p}", (ov[:, p, q] > 0)[vis[:, p] & vis[:, q]]) if (vis[:, p] & vis[:, q]).any() else None
            if parts:
                v = vis[:, p]
                a_m = (model(x, suppress=[p])[0].argmax(1) == y)[v]
                A.add(f"suppress_matched_drop/{p}", (lg0.argmax(1) == y)[v].float() - a_m.float())
                for q in range(K):
                    if q != p:
                        a_q = (model(x, suppress=[q])[0].argmax(1) == y)[v]
                        A.add(f"suppress_nonmatched_drop/{p}", (lg0.argmax(1) == y)[v].float() - a_q.float())
                model(x)  # refresh state for contributions
                contrib = model.contributions()[torch.arange(len(y)), y, p]
                A.add(f"contribution/{p}", contrib[v])
    out = {"clean_acc": A.mean("clean_acc"), "has_parts": parts, "n_parts": K, "per_part": {}, "thresholds": list(thr)}
    for k in sorted(A):
        out["per_part"][k] = A.mean(k)
    return out


def macro(res, prefix):
    v = [x for k, x in res["per_part"].items() if k.startswith(prefix + "/") and x == x]
    return float(np.mean(v)) if v else float("nan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test"); ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    if not os.path.isabs(a.run) and not os.path.exists(a.run):
        a.run = os.path.join(paths()["FATE_RUNS"], a.run)
    cp = os.path.join(a.run, "config.yaml")
    cfg = yaml.safe_load(open(cp))   # saved config is already the final merged one: do NOT re-apply a profile
    dev = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    ds_te, K, C = build_dataset(cfg, a.split)
    model = build_model(cfg, C, K).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.run, "ckpt.pt"), map_location=dev))
    tp = os.path.join(a.run, "thresholds.json")
    if getattr(model, "has_parts", True) and (a.calibrate or not os.path.exists(tp)):
        ds_va, _, _ = build_dataset(cfg, "val")                       # thresholds from VAL, never from the evaluated split
        json.dump(calibrate(model, DataLoader(ds_va, batch_size=cfg["batch_size"]), K, dev), open(tp, "w"))
    thr = json.load(open(tp))["active_thr"] if getattr(model, "has_parts", True) else [0.0] * K
    res = evaluate(model, DataLoader(ds_te, batch_size=cfg["batch_size"]), K, dev, thr)
    if getattr(model, "has_parts", True) and hasattr(model.readout, "diagnostics"):
        res["lcmv_diagnostics"] = model.readout.diagnostics()
    json.dump(res, open(os.path.join(a.run, f"metrics_{a.split}.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("clean_acc",)}))
