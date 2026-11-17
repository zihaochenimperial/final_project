import argparse
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .data import SyntheticParts, erase
from .losses import fate_loss, silence_loss, stability_loss, erf_leak_loss
from .model import FATECNN
from . import metrics


def run(cfg):
    torch.manual_seed(cfg.get("seed", 0))
    K = cfg["num_parts"]
    ds = SyntheticParts(cfg["n"], cfg["size"], K, cfg["num_classes"])
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)
    model = FATECNN(cfg["num_classes"], K, cfg["arch"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    for ep in range(cfg["epochs"]):
        model.train()
        for x, y, m in dl:
            p = int(torch.randint(K, (1,)))
            x.requires_grad_(cfg["lambda_erf"] > 0)
            logits, a = model(x, m)
            xe = erase(x.detach(), m, p)
            _, ae = model(xe, m, update=False)
            erf = erf_leak_loss(a, x, m, p) if cfg["lambda_erf"] > 0 else None
            loss = fate_loss(F.cross_entropy(logits, y), silence_loss(ae, p),
                             stability_loss(ae, a, p), erf,
                             cfg["lambda_sil"], cfg["lambda_stab"], cfg["lambda_erf"])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            x, y, m = next(iter(dl))
            lg, a = model(x); _, ae = model(erase(x, m, 0))
            print(f"ep{ep} loss={loss.item():.3f} acc={(lg.argmax(1)==y).float().mean():.2f} "
                  f"silence={metrics.silence_rate(a, ae, 0):.2f} stab={metrics.other_stability(a, ae, 0):.2f}")
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/synthetic.yaml")
    run(yaml.safe_load(open(ap.parse_args().config)))
