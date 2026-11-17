import torch
import torch.nn as nn
import torchvision

from .lcmv import LCMVReadout

METHODS = ("fate", "fate_learned", "plain", "cutout", "multitask")


class FATECNN(nn.Module):
    """ResNet stem..layer3 -> part read-out -> visibility-gated part evidence -> class logits.
    readout='lcmv' (closed-form null-constrained, FATE) or 'learned' (plain 1x1 conv; baselines/ablation)."""

    def __init__(self, num_classes, num_parts, arch="resnet18", pretrained=False, vis_thresh=0.05,
                 readout="lcmv", sig_mode="region", loading=1e-2):
        super().__init__()
        r = getattr(torchvision.models, arch)(weights="DEFAULT" if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.body = nn.Sequential(r.layer1, r.layer2, r.layer3)
        last = r.layer3[-1]
        c = last.conv3.out_channels if hasattr(last, "conv3") else last.conv2.out_channels
        self.kind = readout
        self.readout = LCMVReadout(c, num_parts, sig_mode=sig_mode, loading=loading) if readout == "lcmv" \
            else nn.Conv2d(c, num_parts, 1)
        self.vis_thresh = vis_thresh
        self.head = nn.Linear(num_parts, num_classes)
        self.num_parts = num_parts
        self.raw = None

    def features(self, x):
        return self.body(self.stem(x))

    def forward(self, x, masks=None, update=True):
        f = self.features(x)
        if self.kind == "lcmv" and self.training and update:
            self.readout.update_stats(f, masks, x)
            self.readout.refresh()
        self.raw = self.readout(f)                           # (B,K,h,w) pre-activation (used by multitask)
        a = torch.relu(self.raw)
        e = a.amax((2, 3))                                   # part evidence
        v = (e > self.vis_thresh).float()                    # visibility gate (silent => 0)
        return self.head(e * v), a


def build_model(cfg, num_classes, num_parts):
    m = cfg.get("method", "fate")
    assert m in METHODS, m
    return FATECNN(num_classes, num_parts, cfg["arch"], cfg.get("pretrained", False),
                   readout="lcmv" if m == "fate" else "learned",
                   sig_mode=cfg.get("sig_mode", "region"), loading=cfg.get("loading", 1e-2))
