"""LCMV null-constrained part read-out (Proposition 1 / 2 of the plan).

Feature at location u:  z(u) = sum_q alpha_q(u) s_q + n(u),  n ~ (0, R).
With C = [s_1..s_K, 1] and the constraint C^T w = e_p, the minimum-variance
solution is  w_p = R^-1 C (C^T R^-1 C)^-1 e_p, giving zero *linear* cross-talk
from every other part signature and from the common-mode direction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def edge_weights(x, h, w):
    """Generic (part-agnostic) Sobel edge magnitude, normalised to [0,1], at feature resolution."""
    g = x.mean(1, keepdim=True)
    kx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device)[None, None]
    gx, gy = F.conv2d(g, kx, padding=1), F.conv2d(g, kx.transpose(2, 3), padding=1)
    e = (gx ** 2 + gy ** 2).sqrt()
    e = F.interpolate(e, size=(h, w), mode="area")
    return e / (e.amax((2, 3), keepdim=True) + 1e-8)


def estimate_signatures(feat, masks, mode="region", edge=None, eps=1e-6):
    """Part signatures (K, C) = weighted in-part mean feature minus outside mean. `mode` = the three arms:
      region  : in-part mask area vs outside                           (region only)
      contour : thin band on the part's own boundary vs outside        (region + part contour)
      edge    : in-part area weighted by GENERIC Sobel edges vs outside (region + generic edge)
    feat (B,C,h,w); masks (B,K,H,W); edge (B,1,h,w) required for mode='edge'."""
    B, C, h, w = feat.shape
    m = F.interpolate(masks.float(), size=(h, w), mode="area")
    if mode == "contour":
        mb = (m > 0.5).float()
        dil = F.max_pool2d(mb, 3, 1, 1); ero = -F.max_pool2d(-mb, 3, 1, 1)
        wt = (dil - ero).clamp(0, 1)
    elif mode == "edge":
        assert edge is not None, "mode='edge' needs an edge map"
        wt = m * edge
    elif mode == "region":
        wt = m
    else:
        raise ValueError(mode)
    f = feat.flatten(2)
    wk, out_w = wt.flatten(2), (1.0 - m).flatten(2)
    inside = torch.einsum("bkn,bcn->kc", wk, f) / (wk.sum((0, 2))[:, None] + eps)
    outside = torch.einsum("bkn,bcn->kc", out_w, f) / (out_w.sum((0, 2))[:, None] + eps)
    return inside - outside


def covariance(feat):
    z = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
    z = z - z.mean(0, keepdim=True)
    return z.T @ z / max(z.shape[0] - 1, 1)


class LCMVReadout(nn.Module):
    """1x1 read-out whose weights are the closed-form LCMV solution (not trained by SGD)."""

    def __init__(self, channels, num_parts, momentum=0.1, loading=1e-2, sig_mode="region"):
        super().__init__()
        self.C, self.K, self.momentum, self.loading, self.sig_mode = channels, num_parts, momentum, loading, sig_mode
        self.register_buffer("sig", torch.randn(num_parts, channels) / channels ** 0.5)
        self.register_buffer("cov", torch.eye(channels))
        self.register_buffer("weight", torch.zeros(num_parts, channels))
        self.refresh()

    @torch.no_grad()
    def update_stats(self, feat, masks=None, x=None):
        self.cov.lerp_(covariance(feat.detach().float()), self.momentum)
        if masks is not None:
            edge = edge_weights(x.detach(), *feat.shape[2:]) if self.sig_mode == "edge" else None
            self.sig.lerp_(estimate_signatures(feat.detach().float(), masks, self.sig_mode, edge), self.momentum)

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
