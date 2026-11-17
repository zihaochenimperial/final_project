"""LCMV null-constrained part read-out.

Classical derivation (Frost / MVDR-LCMV). Model z(u) = sum_q alpha_q(u) s_q + n(u), n zero-mean, covariance R.
With C = [s_1..s_K, 1] and constraint C^T w = e_p, the minimum-variance solution is
    w_p = R^-1 C (C^T R^-1 C)^-1 e_p .
ASSUMPTIONS: R is positive definite (here: diagonal loading), C has full column rank (K+1 <= channels and signatures
linearly independent), G = C^T R^-1 C well conditioned. diagnostics() reports rank(C), cond(R), cond(G).
WHAT IT GUARANTEES: zero response to the ESTIMATED signatures and the common-mode direction (linear, in the estimate).
It does NOT guarantee silence under real occlusion, nor absence of ERF leakage: those are measured (evaluate.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def edge_weights(x, h, w):
    """Generic (part-agnostic) Sobel edge magnitude in [0,1] at feature resolution."""
    g = x.mean(1, keepdim=True)
    kx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device)[None, None]
    gx, gy = F.conv2d(g, kx, padding=1), F.conv2d(g, kx.transpose(2, 3), padding=1)
    e = F.interpolate((gx ** 2 + gy ** 2).sqrt(), size=(h, w), mode="area")
    return e / (e.amax((2, 3), keepdim=True) + 1e-8)


def region_weights(masks, h, w):
    return F.interpolate(masks.float(), size=(h, w), mode="area")


def contour_weights(masks, h, w):
    """Boundary band computed at FULL resolution (morphological gradient), then area-downsampled: small parts survive."""
    r = max(1, round(masks.shape[-1] / 64)); k = 2 * r + 1
    mb = (masks > 0.5).float()
    band = (F.max_pool2d(mb, k, 1, r) + F.max_pool2d(-mb, k, 1, r)).clamp(0, 1)  # dilation - erosion
    return F.interpolate(band, size=(h, w), mode="area")


def legacy_contour_weights(masks, h, w):
    """Earlier scheme (kept only to quantify small-part disappearance): binarise coarse grid >0.5, then 3x3 gradient."""
    m = (F.interpolate(masks.float(), size=(h, w), mode="area") > 0.5).float()
    return (F.max_pool2d(m, 3, 1, 1) + F.max_pool2d(-m, 3, 1, 1)).clamp(0, 1)


def vanish_rate(masks, h, w, weight_fn):
    """Fraction of (sample, part) pairs that are visible at full resolution but get zero weight mass at (h, w)."""
    vis = masks.flatten(2).sum(2) > 0
    mass = weight_fn(masks, h, w).flatten(2).sum(2)
    return ((mass <= 1e-6) & vis).sum().item() / max(int(vis.sum()), 1)


def estimate_signatures_ex(feat, masks, mode="region", edge=None, gamma=1.0, eps=1e-6):
    """Signatures (K,C) and coverage masses. ALL arms share the region base term:
        s_k = [mu_region(k) - mu_out(k)]  +  gamma * [mu_extra(k) - mu_out(k)]   (extra term only for contour/edge)
      region : base only
      contour: extra = part's own boundary band
      edge   : extra = in-part area weighted by GENERIC Sobel edges
    Returns sig (K,C), cov_base (K,), cov_extra (K,) (weight masses over the batch)."""
    B, C, h, w = feat.shape
    f = feat.flatten(2)
    m = region_weights(masks, h, w)
    out = (1.0 - m).flatten(2)
    outside = torch.einsum("bkn,bcn->kc", out, f) / (out.sum((0, 2))[:, None] + eps)
    wb = m.flatten(2)
    base = torch.einsum("bkn,bcn->kc", wb, f) / (wb.sum((0, 2))[:, None] + eps)
    sig, cov_extra = base - outside, wb.sum((0, 2))
    if mode == "region":
        return sig, wb.sum((0, 2)), cov_extra
    if mode == "contour":
        we = contour_weights(masks, h, w).flatten(2)
    elif mode == "edge":
        assert edge is not None, "mode='edge' needs an edge map"
        we = (m * edge).flatten(2)
    else:
        raise ValueError(mode)
    extra = torch.einsum("bkn,bcn->kc", we, f) / (we.sum((0, 2))[:, None] + eps)
    return sig + gamma * (extra - outside), wb.sum((0, 2)), we.sum((0, 2))


def estimate_signatures(feat, masks, mode="region", edge=None, gamma=1.0):
    return estimate_signatures_ex(feat, masks, mode, edge, gamma)[0]


def covariance(feat):
    z = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
    z = z - z.mean(0, keepdim=True)
    return z.T @ z / max(z.shape[0] - 1, 1)


class LCMVReadout(nn.Module):
    """1x1 read-out whose weights are the closed-form LCMV solution (not trained by SGD)."""

    def __init__(self, channels, num_parts, momentum=0.1, loading=1e-2, sig_mode="region", gamma=1.0, min_cov=1.0):
        super().__init__()
        self.C, self.K, self.momentum, self.loading = channels, num_parts, momentum, loading
        self.sig_mode, self.gamma, self.min_cov = sig_mode, gamma, min_cov
        self.register_buffer("sig", torch.randn(num_parts, channels) / channels ** 0.5)
        self.register_buffer("cov", torch.eye(channels))
        self.register_buffer("weight", torch.zeros(num_parts, channels))
        self.register_buffer("ema_cov", torch.zeros(num_parts))      # effective coverage (weight mass / batch)
        self.register_buffer("n_upd", torch.zeros(num_parts))        # signature updates actually applied per part
        self.register_buffer("n_refresh_fail", torch.zeros(()))      # non-finite closed-form solves (weights kept)
        self.refresh()

    @torch.no_grad()
    def update_stats(self, feat, masks=None, x=None):
        self.cov.lerp_(covariance(feat.detach().float()), self.momentum)
        if masks is None:
            return
        edge = edge_weights(x.detach(), *feat.shape[2:]) if self.sig_mode == "edge" else None
        sig, cb, ce = estimate_signatures_ex(feat.detach().float(), masks, self.sig_mode, edge, self.gamma)
        ok = (cb >= self.min_cov) & (ce >= self.min_cov if self.sig_mode != "region" else True)
        # parts with (almost) no mass in this batch must NOT pull the EMA toward 0 / -outside-mean
        self.sig += (ok.float() * self.momentum)[:, None] * (sig - self.sig)
        self.ema_cov.lerp_(cb, self.momentum); self.n_upd += ok.float()

    def _solve(self):
        eye = torch.eye(self.C, device=self.cov.device)
        R = self.cov + self.loading * self.cov.diagonal().mean() * eye
        Cm = torch.cat([self.sig, torch.ones(1, self.C, device=self.sig.device)], 0).T   # (C, K+1)
        RiC = torch.linalg.solve(R, Cm)
        G = Cm.T @ RiC
        # tiny ridge on G: if C is rank-deficient the constraints are infeasible; we return the least-squares-like
        # solution instead of crashing, and diagnostics() reports the (large) constraint residual / rank shortfall.
        Gr = G + 1e-8 * G.diagonal().mean().abs().clamp_min(1e-12) * torch.eye(G.shape[0], device=G.device)
        return R, Cm, G, RiC @ torch.linalg.pinv(Gr, hermitian=True)

    @torch.no_grad()
    def refresh(self):
        W = self._solve()[3][:, : self.K].T
        if torch.isfinite(W).all():
            self.weight.copy_(W)
        else:                                   # keep previous weights; count the failure
            self.n_refresh_fail += 1

    @torch.no_grad()
    def diagnostics(self):
        """Assumption checks + constraint residual ||C^T W - I||_F (should be ~0), effective coverage, update counts."""
        R, Cm, G, W = self._solve()
        I = torch.eye(self.K + 1, device=W.device)
        return dict(rank_C=int(torch.linalg.matrix_rank(Cm)), need_rank=self.K + 1,
                    cond_R=float(torch.linalg.cond(R)), cond_G=float(torch.linalg.cond(G)),
                    constraint_residual=float((Cm.T @ W - I).norm()), w_norm_max=float(W.norm(dim=0).max()),
                    min_ema_coverage=float(self.ema_cov.min()), min_sig_updates=float(self.n_upd.min()), refresh_failures=float(self.n_refresh_fail))

    def leakage(self):
        """|w_p^T s_q| (estimated-signature linear response); off-diagonal should be ~0."""
        return (self.weight @ self.sig.T).abs()

    def forward(self, feat):
        return F.conv2d(feat, self.weight[:, :, None, None])
