"""LCMV null-constrained part read-out (Proposition 1 / 2 of the plan).

Feature at location u:  z(u) = sum_q alpha_q(u) s_q + n(u),  n ~ (0, R).
With C = [s_1..s_K, 1] and the constraint C^T w = e_p, the minimum-variance
solution is  w_p = R^-1 C (C^T R^-1 C)^-1 e_p, giving zero *linear* cross-talk
from every other part signature and from the common-mode direction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def estimate_signatures(feat, masks, eps=1e-6):
    """Part signatures as in-part minus out-of-part mean feature (contour-difference estimator).

    feat:  (B, C, h, w); masks: (B, K, H, W) in {0,1} (resized to h, w).
    Returns (K, C) signatures, plus (C, C) covariance of residual features.
    """
    B, C, h, w = feat.shape
    m = F.interpolate(masks.float(), size=(h, w), mode="area")  # (B,K,h,w)
    f = feat.flatten(2)                                         # (B,C,hw)
    mk = m.flatten(2)                                           # (B,K,hw)
    inside = torch.einsum("bkn,bcn->kc", mk, f) / (mk.sum((0, 2))[:, None] + eps)
    out_w = 1.0 - mk
    outside = torch.einsum("bkn,bcn->kc", out_w, f) / (out_w.sum((0, 2))[:, None] + eps)
    return inside - outside


def covariance(feat):
    z = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
    z = z - z.mean(0, keepdim=True)
    return z.T @ z / max(z.shape[0] - 1, 1)


class LCMVReadout(nn.Module):
    """1x1 read-out whose weights are the closed-form LCMV solution (not trained by SGD)."""

    def __init__(self, channels, num_parts, momentum=0.1, loading=1e-2):
        super().__init__()
        self.C, self.K, self.momentum, self.loading = channels, num_parts, momentum, loading
        self.register_buffer("sig", torch.randn(num_parts, channels) / channels ** 0.5)
        self.register_buffer("cov", torch.eye(channels))
        self.register_buffer("weight", torch.zeros(num_parts, channels))
        self.refresh()

    @torch.no_grad()
    def update_stats(self, feat, masks=None):
        self.cov.lerp_(covariance(feat.detach().float()), self.momentum)
        if masks is not None:
            self.sig.lerp_(estimate_signatures(feat.detach().float(), masks), self.momentum)

    @torch.no_grad()
    def refresh(self):
        """Recompute W (K x C) with diagonal loading (controls ||w|| -> Prop. 2)."""
        R = self.cov + self.loading * self.cov.diagonal().mean() * torch.eye(self.C, device=self.cov.device)
        Cm = torch.cat([self.sig, torch.ones(1, self.C, device=self.sig.device)], 0).T  # (C, K+1)
        RiC = torch.linalg.solve(R, Cm)                                                  # (C, K+1)
        G = Cm.T @ RiC                                                                   # (K+1,K+1)
        W = RiC @ torch.linalg.inv(G)                                                    # (C, K+1)
        self.weight.copy_(W[:, : self.K].T)

    def leakage(self):
        """|w_p^T s_q| matrix; off-diagonal should be ~0 (linear cross-talk)."""
        return (self.weight @ self.sig.T).abs()

    def forward(self, feat):
        return F.conv2d(feat, self.weight[:, :, None, None])
