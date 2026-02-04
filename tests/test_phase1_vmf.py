import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1.vmf import VmfLogNormal, vmf_nll


def test_vmf_direction_prefers_alignment() -> None:
    torch.manual_seed(0)
    b, d = 4, 32
    mu = torch.randn(b, d)
    mu = mu / torch.linalg.vector_norm(mu, dim=-1, keepdim=True).clamp_min(1e-8)

    d_aligned = mu.clone()
    d_opposed = -mu.clone()
    kappa = torch.full((b,), 50.0)

    nll_aligned = vmf_nll(d_true=d_aligned, mu_dir=mu, kappa=kappa)
    nll_opposed = vmf_nll(d_true=d_opposed, mu_dir=mu, kappa=kappa)

    assert (nll_aligned < nll_opposed).all()


def test_vmf_lognormal_shapes_finite_and_grad() -> None:
    torch.manual_seed(0)
    b, din, dout = 8, 64, 16
    model = VmfLogNormal(
        input_dim=din,
        output_dim=dout,
        hidden_dim=64,
        n_hidden_layers=2,
        dropout=0.0,
        min_log_kappa=-2.0,
        max_log_kappa=8.0,
        min_log_sigma_logm=-5.0,
        max_log_sigma_logm=2.0,
    )

    ctx = torch.randn(b, din, requires_grad=True)
    dx = torch.randn(b, dout)

    nll = model.nll(ctx, dx)
    assert nll.shape == (b,)
    assert torch.isfinite(nll).all()

    mean = model.expected_mean(ctx)
    assert mean.shape == (b, dout)
    assert torch.isfinite(mean).all()

    loss = nll.mean()
    loss.backward()
    assert ctx.grad is not None
    assert any(p.grad is not None for p in model.parameters())

