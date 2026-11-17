"""Real-data loaders. Contract: __getitem__ -> (image[3,S,S] float in [0,1], label:int, masks[K,S,S] float {0,1}).

CubParts   : CUB-200-2011 (needs the standard extracted folder), 15 keypoints -> K=8 part discs.
FolderParts: generic format  <root>/<split>/index.json = [{"image": rel.png, "label": int, "masks": rel.npy (K,H,W)}]
             Use it for RIVAL10 once a converter writes this format (not yet implemented; needs the data).
"""
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# CUB part ids (1-based) -> grouped parts
CUB_GROUPS = {
    "beak": [2], "head": [5, 6, 7, 10, 11, 15], "back": [1], "belly": [3],
    "breast": [4], "wing": [9, 13], "tail": [14], "legs": [8, 12],
}
CUB_PART_NAMES = list(CUB_GROUPS)


def _to_tensor(img):
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


class CubParts(Dataset):
    def __init__(self, root, split="train", size=224, flip=False, radius_frac=0.1, val_frac=0.1, seed=0):
        base = os.path.join(root, "CUB_200_2011") if os.path.isdir(os.path.join(root, "CUB_200_2011")) else root
        rd = lambda f: [l.split() for l in open(os.path.join(base, f))]
        paths = {int(i): p for i, p in rd("images.txt")}
        labels = {int(i): int(c) - 1 for i, c in rd("image_class_labels.txt")}
        is_train = {int(i): t == "1" for i, t in rd("train_test_split.txt")}
        kp = {}
        for i, pid, x, y, v in rd(os.path.join("parts", "part_locs.txt")):
            kp.setdefault(int(i), {})[int(pid)] = (float(x), float(y), v == "1")
        ids = sorted(paths)
        if split == "test":
            ids = [i for i in ids if not is_train[i]]
        else:  # train / val: stratified-by-class 10% val split from official train
            tr = [i for i in ids if is_train[i]]
            rng = np.random.RandomState(seed); val = set()
            for c in set(labels[i] for i in tr):
                ci = [i for i in tr if labels[i] == c]; rng.shuffle(ci)
                val.update(ci[: max(1, int(len(ci) * val_frac))])
            ids = [i for i in tr if (i in val) == (split == "val")]
        self.base, self.ids, self.paths, self.labels, self.kp = base, ids, paths, labels, kp
        self.size, self.flip, self.r = size, flip, radius_frac * size
        self.K = len(CUB_GROUPS)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, n):
        i = self.ids[n]
        img = Image.open(os.path.join(self.base, "images", self.paths[i])).convert("RGB")
        W, H = img.size
        img = img.resize((self.size, self.size))
        yy, xx = torch.meshgrid(torch.arange(self.size), torch.arange(self.size), indexing="ij")
        masks = torch.zeros(self.K, self.size, self.size)
        for k, pids in enumerate(CUB_GROUPS.values()):
            for pid in pids:
                x, y, vis = self.kp.get(i, {}).get(pid, (0, 0, False))
                if vis:
                    cx, cy = x / W * self.size, y / H * self.size
                    masks[k] = torch.maximum(masks[k], (((xx - cx) ** 2 + (yy - cy) ** 2) <= self.r ** 2).float())
        x = _to_tensor(img)
        if self.flip and torch.rand(()) < 0.5:
            x, masks = x.flip(-1), masks.flip(-1)  # grouped parts are L/R symmetric
        return x, self.labels[i], masks


class FolderParts(Dataset):
    def __init__(self, root, split="train", size=224):
        self.dir = os.path.join(root, split)
        self.items = json.load(open(os.path.join(self.dir, "index.json")))
        self.size = size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):
        it = self.items[n]
        img = Image.open(os.path.join(self.dir, it["image"])).convert("RGB").resize((self.size,) * 2)
        m = torch.from_numpy(np.load(os.path.join(self.dir, it["masks"]))).float()
        m = torch.nn.functional.interpolate(m[None], size=(self.size,) * 2, mode="nearest")[0]
        return _to_tensor(img), int(it["label"]), m
