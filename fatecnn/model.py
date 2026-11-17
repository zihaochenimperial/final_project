import torch
import torch.nn as nn
import torchvision

from .lcmv import LCMVReadout

# gap_resnet : standard ResNet (stem..layer4) + GAP + FC, no part bottleneck
# bottleneck : capacity-matched K-channel part bottleneck (stem..layer3 -> 1x1 K -> evidence -> FC), CE only
# cutout     : bottleneck + same-amount part occlusion, classification-only
# multitask  : bottleneck + part-mask supervision (no null/silence constraint)
# fate_learned : learned read-out (LCMV-initialised) + erasure losses      } same read-out family,
# fate         : closed-form LCMV read-out         + erasure losses      } differ only in how W is obtained
METHODS = ("gap_resnet", "bottleneck", "cutout", "multitask", "fate_learned", "fate")


class GAPResNet(nn.Module):
    has_parts = False

    def __init__(self, num_classes, arch="resnet18", pretrained=False):
        super().__init__()
        r = getattr(torchvision.models, arch)(weights="DEFAULT" if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.body = nn.Sequential(r.layer1, r.layer2, r.layer3)
        self.layer4 = r.layer4
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(r.fc.in_features, num_classes))
        self.raw = None

    def forward(self, x, masks=None, update=True, suppress=None):
        return self.head(self.layer4(self.body(self.stem(x)))), None


class FATECNN(nn.Module):
    """ResNet stem..layer3 -> K-channel part read-out (+ per-part calibration) -> visibility-gated evidence -> FC.
    Attribution caveat: what is traced here is the NEW 1x1 read-out kernels and their ERF on the input image;
    individual backbone kernels are not tracked, so they are called 'backbone features'."""
    has_parts = True

    def __init__(self, num_classes, num_parts, arch="resnet18", pretrained=False, vis_thresh=0.05,
                 readout="lcmv", sig_mode="region", loading=1e-2, gamma=1.0, min_cov=1.0, readout_init="default"):
        super().__init__()
        r = getattr(torchvision.models, arch)(weights="DEFAULT" if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.body = nn.Sequential(r.layer1, r.layer2, r.layer3)
        last = r.layer3[-1]
        c = last.conv3.out_channels if hasattr(last, "conv3") else last.conv2.out_channels
        self.kind, self.init_mode, self.sig_mode, self.loading, self.gamma, self.min_cov = readout, readout_init, sig_mode, loading, gamma, min_cov
        if readout == "lcmv":
            self.readout = LCMVReadout(c, num_parts, sig_mode=sig_mode, loading=loading, gamma=gamma, min_cov=min_cov)
        else:
            self.readout = nn.Conv2d(c, num_parts, 1, bias=False)   # bias lives in the shared calibration below
        # identical per-part calibration (gain, threshold) in BOTH read-out families
        self.cal_s = nn.Parameter(torch.ones(num_parts))
        self.cal_b = nn.Parameter(torch.full((num_parts,), -0.5))
        self.register_buffer("inited", torch.tensor(0))
        self.vis_thresh, self.num_parts, self.raw = vis_thresh, num_parts, None
        self.head = nn.Linear(num_parts, num_classes)

    def features(self, x):
        return self.body(self.stem(x))

    @torch.no_grad()
    def _init_learned_from_lcmv(self, f, masks, x):
        t = LCMVReadout(f.shape[1], self.num_parts, momentum=1.0, loading=self.loading, sig_mode=self.sig_mode,
                        gamma=self.gamma, min_cov=0.0).to(f.device)
        t.update_stats(f.detach(), masks, x); t.refresh()
        self.readout.weight.copy_(t.weight[:, :, None, None]); self.inited.fill_(1)

    def forward(self, x, masks=None, update=True, suppress=None):
        f = self.features(x)
        if self.training and update and masks is not None:
            if self.kind == "lcmv":
                self.readout.update_stats(f, masks, x); self.readout.refresh()
            elif self.init_mode == "lcmv" and int(self.inited) == 0:
                self._init_learned_from_lcmv(f, masks, x)
        raw = self.readout(f) * self.cal_s[None, :, None, None] + self.cal_b[None, :, None, None]
        self.raw = raw                                       # pre-gate, pre-ReLU score map (B,K,h,w)
        a = torch.relu(raw)
        if suppress:                                         # channel suppression probe (evaluation only)
            a = a.clone(); a[:, list(suppress)] = 0
        e = a.amax((2, 3))
        v = (e > self.vis_thresh).float()                    # visibility gate (silent => 0)
        self.evidence = e * v
        return self.head(self.evidence), a

    def contributions(self):
        """(B,C,K) per-part contribution to each class logit from the last forward: W[c,p] * evidence[p]."""
        return self.head.weight[None] * self.evidence[:, None, :]


def build_model(cfg, num_classes, num_parts):
    m = cfg.get("method", "fate")
    assert m in METHODS, m
    if m == "gap_resnet":
        return GAPResNet(num_classes, cfg["arch"], cfg.get("pretrained", False))
    return FATECNN(num_classes, num_parts, cfg["arch"], cfg.get("pretrained", False),
                   readout="lcmv" if m == "fate" else "learned",
                   sig_mode=cfg.get("sig_mode", "region"), loading=cfg.get("loading", 1e-2),
                   gamma=cfg.get("gamma", 1.0), min_cov=cfg.get("min_cov", 1.0),
                   readout_init=cfg.get("readout_init", "lcmv" if m == "fate_learned" else "default"))
