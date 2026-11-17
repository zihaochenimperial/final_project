"""Occlusion protocol (docs/02_protocol.md). Train fills: gray/noise. Test adds: texture (unseen)."""
import math
import torch

TRAIN_FILLS = ("gray", "noise")
TEST_FILLS = ("gray", "noise", "texture")


def fill_like(x, kind, gen=None):
    if kind == "gray":
        return torch.full_like(x, 0.5)
    if kind == "noise":
        return (0.5 + 0.25 * torch.randn(x.shape, generator=gen)).clamp(0, 1).to(x)
    if kind == "texture":  # high-frequency sinusoidal stripes, random phase/orientation, unseen in training
        B, _, H, W = x.shape
        yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
        th = torch.rand(B, 1, 1, generator=gen) * math.pi
        ph = torch.rand(B, 1, 1, generator=gen) * 2 * math.pi
        t = 0.5 + 0.5 * torch.sin(2 * math.pi * 24 * (xx * torch.cos(th) + yy * torch.sin(th)) + ph)
        return t[:, None].expand_as(x).to(x)
    raise ValueError(kind)


def occlude_mask(x, mask, kind="gray", gen=None):
    """mask: (B,1,H,W) or (B,H,W) in {0,1}; occluded where 1."""
    m = mask.float().reshape(x.shape[0], 1, *x.shape[2:])
    return x * (1 - m) + fill_like(x, kind, gen) * m


def occlude_part(x, masks, p, kind="gray", gen=None):
    return occlude_mask(x, masks[:, p], kind, gen)


def random_block_mask(B, H, W, area_frac, gen=None):
    s = int(round(math.sqrt(area_frac) * min(H, W)))
    m = torch.zeros(B, 1, H, W)
    for b in range(B):
        y = int(torch.randint(0, H - s + 1, (1,), generator=gen)); x = int(torch.randint(0, W - s + 1, (1,), generator=gen))
        m[b, :, y:y + s, x:x + s] = 1
    return m


def shifted_mask(mask, gen=None):
    """Same-area, same-shape NON-TARGET control region: circularly shift `mask` (B,H,W) to a random zero-overlap place
    (overlap for every shift via FFT cross-correlation); if none exists take the least-overlapping shift. -> (B,1,H,W)."""
    B, H, W = mask.shape
    out = torch.zeros(B, 1, H, W)
    for b in range(B):
        m = mask[b].float()
        F = torch.fft.rfft2(m)
        ov = torch.fft.irfft2(F * F.conj(), s=(H, W)).round()          # ov[dy,dx] = |m ∩ roll(m,(dy,dx))|
        ok = (ov <= ov.min() + 0.5).flatten().nonzero().flatten()       # all minimal-overlap shifts (zero if one exists)
        k = int(ok[int(torch.randint(0, len(ok), (1,), generator=gen))])
        out[b, 0] = torch.roll(m, (k // W, k % W), (0, 1))
    return out
