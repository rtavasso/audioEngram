"""
Supervised rollout projector model.

Learns a correction that maps (context, predicted_state) -> corrected_state.
This is intended to be trained on dynamics-model rollout errors, not on
Gaussian-noise denoising.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContextStateProjector(nn.Module):
    """
    MLP projector:
        delta = f([context_flat, z_hat])   -> [B, D]
        z_corr = z_hat + scale * delta
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        context_dim: int,
        hidden_dim: int = 2048,
        n_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)

        in_dim = int(context_dim) + int(latent_dim)
        h = int(hidden_dim)
        layers: list[nn.Module] = [nn.Linear(in_dim, h), nn.GELU()]
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
        for _ in range(int(n_layers) - 1):
            layers.extend([nn.Linear(h, h), nn.GELU()])
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(h, int(latent_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, ctx_flat: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ctx_flat: [B, context_dim]
            z_hat: [B, D]
        Returns:
            delta: [B, D]
        """
        x = torch.cat([ctx_flat, z_hat], dim=-1)
        return self.net(x)


def apply_projector(
    *,
    projector: ContextStateProjector,
    ctx_flat: torch.Tensor,
    z_hat: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """z_corr = z_hat + scale * projector(ctx_flat, z_hat)."""
    delta = projector(ctx_flat, z_hat)
    return z_hat + float(scale) * delta

