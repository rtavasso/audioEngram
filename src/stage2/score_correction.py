"""
Langevin correction for manifold projection of rollout latents.

After each dynamics model step, apply a few Langevin steps using the
trained score model to push z back toward the data manifold.
"""

from __future__ import annotations

import torch

from .score_model import ScoreNetwork


@torch.no_grad()
def langevin_correction(
    z: torch.Tensor,
    score_model: ScoreNetwork,
    n_steps: int = 3,
    step_size: float = 0.01,
    sigma: float = 0.1,
) -> torch.Tensor:
    """
    Apply Langevin-based correction to push z toward the data manifold.

    Args:
        z: [B, D] latent to correct
        score_model: trained ScoreNetwork
        n_steps: number of Langevin steps
        step_size: step size for gradient ascent
        sigma: noise level for score evaluation

    Returns:
        z_corrected: [B, D]
    """
    sigma_t = z.new_full((z.shape[0], 1), sigma)

    sigma_f = float(sigma)
    for _ in range(n_steps):
        score = score_model(z, sigma_t)  # [B, D]
        # Scale by sigma^2 so step_size is roughly comparable across sigma.
        z = z + float(step_size) * (sigma_f ** 2) * score

    return z


@torch.no_grad()
def tweedie_correction(
    z: torch.Tensor,
    score_model: ScoreNetwork,
    sigma: float = 0.1,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    One-step Tweedie denoiser correction:

        z_corrected = z + scale * sigma^2 * score(z, sigma)

    Often easier to tune than multi-step Langevin correction.
    """
    sigma_f = float(sigma)
    sigma_t = z.new_full((z.shape[0], 1), sigma_f)
    score = score_model(z, sigma_t)
    return z + (float(scale) * (sigma_f ** 2)) * score
