import torch
from fatecnn import LCMVReadout, FATECNN, estimate_signatures
from fatecnn.losses import erf_leak_loss, silence_loss, stability_loss
from fatecnn.train import run


def test_lcmv_null_constraint():
    torch.manual_seed(0)
    C, K = 32, 4
    r = LCMVReadout(C, K, loading=1e-3)
    r.sig.copy_(torch.randn(K, C)); r.cov.copy_(torch.eye(C) * 0.5); r.refresh()
    L = r.weight @ r.sig.T
    assert torch.allclose(L, torch.eye(K), atol=1e-3)          # unit gain on own part, zero on others
    assert torch.allclose(r.weight.sum(1), torch.zeros(K), atol=1e-3)  # common-mode null


def test_signature_estimator():
    f = torch.zeros(2, 3, 8, 8); m = torch.zeros(2, 1, 8, 8); m[..., :4, :] = 1; f[:, 0, :4] = 1
    s = estimate_signatures(f, m)
    assert s[0, 0] > 0.9 and s[0, 1].abs() < 1e-3


def test_forward_and_losses():
    m = FATECNN(4, 4); x = torch.rand(2, 3, 64, 64, requires_grad=True)
    masks = torch.zeros(2, 4, 64, 64); masks[:, 0, :32, :32] = 1
    lg, a = m(x, masks)
    assert lg.shape == (2, 4) and a.shape[1] == 4
    assert silence_loss(a, 0) >= 0 and stability_loss(a, a, 0) == 0
    assert 0 <= erf_leak_loss(a, x, masks, 0) <= 1


def test_train_smoke():
    run(dict(seed=0, arch="resnet18", n=32, size=64, num_parts=4, num_classes=4, batch_size=16,
             lr=1e-3, epochs=1, lambda_sil=1.0, lambda_stab=1.0, lambda_erf=0.0))
