"""Datasets. SyntheticParts is a smoke-test dataset; RIVAL10 / CUB loaders follow the same
contract: __getitem__ -> (image[3,H,W], label, part_masks[K,H,W] in {0,1})."""
import torch
from torch.utils.data import Dataset


class SyntheticParts(Dataset):
    """Class c = which subset of K colour-coded rectangles ("parts") is present with fixed colours."""

    def __init__(self, n=256, size=64, num_parts=4, num_classes=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.items = []
        palette = torch.rand(num_parts, 3, generator=g) * 0.8 + 0.2
        for i in range(n):
            y = int(torch.randint(num_classes, (1,), generator=g))
            img = torch.rand(3, size, size, generator=g) * 0.1
            masks = torch.zeros(num_parts, size, size)
            for p in range(num_parts):
                x0 = (p % 2) * size // 2 + 4; y0 = (p // 2) * size // 2 + 4
                masks[p, y0:y0 + size // 2 - 8, x0:x0 + size // 2 - 8] = 1
                if (y + p) % num_classes < num_classes - 1:  # part present
                    img += masks[p][None] * palette[p][:, None, None]
                else:
                    masks[p] = 0
            self.items.append((img.clamp(0, 1), y, masks))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def erase(x, masks, p, fill=0.5):
    """Counterfactual erasure of part p (grey fill; extend with noise/texture fills)."""
    m = masks[:, p:p + 1].float()
    return x * (1 - m) + fill * m
