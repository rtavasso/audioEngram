import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calm.model import CALM


def test_calm_shapes_and_finite_loss() -> None:
    torch.manual_seed(0)
    model = CALM(
        latent_dim=32,
        d_model=64,
        n_backbone_layers=2,
        n_short_ctx_layers=1,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
        short_ctx_k=5,
        head_hidden_dim=64,
        head_n_layers=2,
        ema_decay=0.9,
    )

    x = torch.randn(2, 16, 32)
    Z, x_targets = model.compute_conditioning(x)
    assert Z.shape == (2, 15, 64)
    assert x_targets.shape == (2, 15, 32)

    loss, mse = model.consistency_loss(x_targets, Z, head_batch_mult=2)
    assert torch.isfinite(loss)
    assert mse == pytest.approx(float(mse))
    loss.backward()


@torch.no_grad()
def test_calm_generate_zero_steps_and_shape() -> None:
    torch.manual_seed(0)
    model = CALM(
        latent_dim=32,
        d_model=64,
        n_backbone_layers=1,
        n_short_ctx_layers=1,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
        short_ctx_k=5,
        head_hidden_dim=64,
        head_n_layers=2,
        ema_decay=0.9,
    )

    prompt = torch.randn(1, 4, 32)
    out0 = model.generate(prompt, 0)
    assert out0.shape == (0, 32)

    out = model.generate(prompt, 3)
    assert out.shape == (3, 32)
