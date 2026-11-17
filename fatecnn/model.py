import torch
import torch.nn as nn
import torchvision

from .lcmv import LCMVReadout


class FATECNN(nn.Module):
    """ResNet stem..layer3 -> LCMV part read-out -> visibility-gated part evidence -> class logits."""

    def __init__(self, num_classes, num_parts, arch="resnet18", pretrained=False, vis_thresh=0.05):
        super().__init__()
        r = getattr(torchvision.models, arch)(weights="DEFAULT" if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.body = nn.Sequential(r.layer1, r.layer2, r.layer3)
        c = r.layer3[-1].conv2.out_channels if hasattr(r.layer3[-1], "conv2") and arch in ("resnet18", "resnet34") \
            else r.layer3[-1].conv3.out_channels
        self.readout = LCMVReadout(c, num_parts)
        self.vis_thresh = vis_thresh
        self.head = nn.Linear(num_parts, num_classes)
        self.num_parts = num_parts

    def features(self, x):
        return self.body(self.stem(x))

    def forward(self, x, masks=None, update=True):
        f = self.features(x)
        if self.training and update:
            self.readout.update_stats(f, masks)
            self.readout.refresh()
        a = torch.relu(self.readout(f))                      # (B,K,h,w) part activation maps
        e = a.amax((2, 3))                                   # part evidence
        v = (e > self.vis_thresh).float()                    # visibility gate (silent => 0)
        return self.head(e * v), a
