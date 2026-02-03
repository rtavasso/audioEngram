import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase4.memory import KMeansDeltaMemory
from phase4.models import ParamDyn, HybridDyn, diag_gaussian_nll


def test_phase4_memory_and_models_shapes() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    k, d = 8, 6
    centroids = torch.randn(k, d)
    dz_mean = torch.randn(k, d)
    dz_var = torch.ones(k, d)
    counts = torch.arange(k)
    mem = KMeansDeltaMemory(
        centroids=centroids.to(device),
        dz_mean=dz_mean.to(device),
        dz_var=dz_var.to(device),
        counts=counts.to(device),
    )

    b = 5
    z_prev = torch.randn(b, d)
    dz = torch.randn(b, d)

    mu_mem = mem.predict_mean(z_prev)
    assert mu_mem.shape == (b, d)

    param = ParamDyn(z_dim=d, hidden_dim=16, n_layers=2, dropout=0.0, min_log_sigma=-6.0, max_log_sigma=2.0)
    p = param(z_prev)
    assert p.mu.shape == (b, d)
    assert p.log_sigma.shape == (d,)
    nll = diag_gaussian_nll(dz, p)
    assert nll.shape == (b,)
    assert torch.isfinite(nll).all()

    hybrid = HybridDyn(
        z_dim=d,
        hidden_dim=16,
        n_layers=2,
        dropout=0.0,
        gate_hidden_dim=8,
        min_log_sigma=-6.0,
        max_log_sigma=2.0,
        memory=mem,
    )
    h = hybrid(z_prev)
    assert h.mu.shape == (b, d)
    assert h.log_sigma.shape == (d,)
    nll_h = diag_gaussian_nll(dz, h)
    assert nll_h.shape == (b,)
    assert torch.isfinite(nll_h).all()

