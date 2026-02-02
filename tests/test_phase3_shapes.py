import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase3.models import Factorizer
from phase3.losses import compute_losses


def test_phase3_forward_and_losses_finite() -> None:
    torch.manual_seed(0)
    b, t, x_dim = 2, 17, 32
    z_dyn_dim, z_rec_dim = 8, 4
    x = torch.randn(b, t, x_dim)
    mask = torch.ones(b, t, dtype=torch.bool)

    model = Factorizer(
        x_dim=x_dim,
        z_dyn_dim=z_dyn_dim,
        z_rec_dim=z_rec_dim,
        dyn_encoder_hidden=16,
        dyn_encoder_layers=1,
        dyn_encoder_dropout=0.0,
        dyn_model_hidden=16,
        dyn_model_layers=1,
        dyn_model_dropout=0.0,
        dyn_model_min_log_sigma=-6.0,
        dyn_model_max_log_sigma=1.0,
        posterior_hidden=16,
        posterior_layers=2,
        posterior_dropout=0.0,
        posterior_min_log_sigma=-6.0,
        posterior_max_log_sigma=1.0,
        prior_hidden=16,
        prior_layers=2,
        prior_dropout=0.0,
        prior_min_log_sigma=-6.0,
        prior_max_log_sigma=1.0,
        recon_hidden=16,
        recon_layers=2,
        recon_dropout=0.0,
    )
    out = model(x, prior_sample_prob=0.5)
    assert out.z_dyn.shape == (b, t, z_dyn_dim)
    assert out.x_hat_mixed.shape == (b, t, x_dim)
    assert out.x_hat_post.shape == (b, t, x_dim)
    assert out.x_hat_prior.shape == (b, t, x_dim)
    assert out.prior_mask.shape == (b, t, 1)

    dyn_params = model.dyn(out.z_dyn[:, :-1])
    losses = compute_losses(
        x=x,
        mask=mask,
        x_hat=out.x_hat_mixed,
        q_rec=out.q_rec,
        p_rec=out.p_rec,
        dyn_params=dyn_params,
        z_dyn_target=out.z_dyn[:, 1:],
        recon_weight=1.0,
        beta=1.0,
        free_bits_per_dim=0.0,
        z_rec_dim=z_rec_dim,
        dyn_weight=1.0,
    )
    assert torch.isfinite(losses.total)
    losses.total.backward()
