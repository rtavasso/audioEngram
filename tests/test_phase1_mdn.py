import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1.mdn import MDN


def test_mdn_shapes_and_finite_nll() -> None:
    torch.manual_seed(0)
    b, din, dout, k = 4, 16, 8, 3
    model = MDN(input_dim=din, output_dim=dout, n_components=k, hidden_dim=32, n_hidden_layers=2)
    ctx = torch.randn(b, din)
    dx = torch.randn(b, dout)

    nll = model.nll(ctx, dx)
    assert nll.shape == (b,)
    assert torch.isfinite(nll).all()

    loss = nll.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
