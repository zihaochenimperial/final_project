"""Synthetic functional-test data. Contract: __getitem__ -> (image[3,S,S], label, part_masks[K,S,S]).

Palette is FIXED (palette_seed) across train/val/test: splits differ only in sample seed. `shift=True` permutes colour
channels = a separate colour-shift test. Modes:
  redundant : every part carries the class colour (part-specific brightness) -> class decodable from ANY remaining part.
              Use this for occlusion-robust classification diagnostics.
  missing   : class = which part is absent (legacy). NOT suitable for occlusion accuracy: erasing a present part makes
              the image look like another class."""
import math
import torch
from torch.utils.data import Dataset

SPLIT_SEED = {"train": 0, "test": 1, "val": 2, "test_shift": 1}


class SyntheticParts(Dataset):
    def __init__(self, n=256, size=64, num_parts=4, num_classes=4, seed=0, mode="redundant", palette_seed=0, shift=False):
        gp = torch.Generator().manual_seed(palette_seed)
        palette = torch.rand(max(num_classes, num_parts), 3, generator=gp) * 0.8 + 0.2
        if shift:
            palette = palette[:, [1, 2, 0]]
        g = torch.Generator().manual_seed(1000 + seed)
        gdim = math.ceil(math.sqrt(num_parts)); cell = size // gdim
        self.items = []
        for _ in range(n):
            y = int(torch.randint(num_classes, (1,), generator=g))
            img = torch.rand(3, size, size, generator=g) * 0.1
            masks = torch.zeros(num_parts, size, size)
            for p in range(num_parts):
                x0, y0 = (p % gdim) * cell + 3, (p // gdim) * cell + 3
                sl = (slice(y0, y0 + cell - 6), slice(x0, x0 + cell - 6))
                if mode == "redundant":
                    col = palette[y] * (0.75 + 0.08 * p)
                else:
                    if (y + p) % num_classes == num_classes - 1:
                        continue
                    col = palette[p]
                masks[p][sl] = 1; img[:, sl[0], sl[1]] += col[:, None, None]
            self.items.append((img.clamp(0, 1), y, masks))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]
