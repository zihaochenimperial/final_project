"""Counterfactual erasure losses: silence the erased part's filters, keep the others unchanged."""
import torch
import torch.nn.functional as F


def _pool(a):
    return a.abs().mean((2, 3))  # (B, K)


def silence_loss(a_erased, p):
    """L1 energy of part-p activations after erasing part p. a_erased: (B,K,h,w)."""
    return _pool(a_erased)[:, p].mean()


def stability_loss(a_erased, a_clean, p):
    """Other parts q != p must not change when p is erased."""
    K = a_clean.shape[1]
    keep = [q for q in range(K) if q != p]
    if not keep:
        return a_clean.new_zeros(())
    return F.mse_loss(a_erased[:, keep], a_clean[:, keep].detach())


def erf_leak_loss(a, x, part_masks, p, dilate=3):
    """Fraction of |d a_p / d x| mass outside the (dilated) mask of part p. x must require grad."""
    g, = torch.autograd.grad(a[:, p].sum(), x, create_graph=True)
    g = g.abs().sum(1)  # (B,H,W)
    m = part_masks[:, p].float()[:, None]
    m = F.max_pool2d(m, dilate * 2 + 1, 1, dilate)[:, 0]
    return ((g * (1 - m)).sum((1, 2)) / (g.sum((1, 2)) + 1e-8)).mean()


def fate_loss(cls_loss, sil, stab, erf=None, l1=1.0, l2=1.0, l3=0.1):
    total = cls_loss + l1 * sil + l2 * stab
    if erf is not None:
        total = total + l3 * erf
    return total
