"""Evaluation protocol: silence rate, other-filter stability, matched-channel suppression specificity."""
import torch


@torch.no_grad()
def silence_rate(a_clean, a_erased, p, ratio=0.1, eps=1e-3):
    """Among samples where part p is active when clean (energy > eps), fraction whose energy after erasure
    drops to <= `ratio` x clean energy. Parts that are inactive when clean are excluded (nothing to silence)."""
    e0 = a_clean[:, p].abs().mean((1, 2)); e1 = a_erased[:, p].abs().mean((1, 2))
    act = e0 > eps
    if not act.any():
        return 0.0
    return (e1[act] <= ratio * e0[act]).float().mean().item()


@torch.no_grad()
def other_stability(a_clean, a_erased, p):
    K = a_clean.shape[1]
    keep = [q for q in range(K) if q != p]
    d = (a_erased[:, keep] - a_clean[:, keep]).flatten(1).norm(dim=1)
    n = a_clean[:, keep].flatten(1).norm(dim=1).clamp_min(1e-8)
    return (1 - d / n).clamp(min=0).mean().item()  # 1 = perfectly stable


@torch.no_grad()
def suppression_specificity(acc_drop_attr_p, acc_drop_other_attrs, eps=1e-6):
    """Drop of attribute p when channel p suppressed / drop of non-p attributes."""
    return acc_drop_attr_p / (acc_drop_other_attrs + eps)
