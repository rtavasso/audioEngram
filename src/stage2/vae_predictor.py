"""
Lightweight MLP predictor co-trained with the VAE for L_pred gradient signal.

Takes a flattened context window of VAE latents and predicts the next delta.
This is a training surrogate, not the final dynamics model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentPredictor(nn.Module):
    """2-layer MLP: Linear(W*D, hidden) -> GELU -> Linear(hidden, D)."""

    def __init__(self, latent_dim: int, window_size: int = 8, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window_size * latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, W*D] flattened context window

        Returns:
            [B, D] predicted delta
        """
        return self.net(x)


class CausalConvDeltaPredictor(nn.Module):
    """
    Lightweight causal Conv1d over a latent sequence that predicts next-step deltas.

    Input:
        z: [B, D, T]
    Output:
        delta_hat: [B, D, T]

    Interpretation:
        delta_hat[:, :, t] is trained to approximate z[:, :, t+1] - z[:, :, t].
        (The last timestep is typically ignored since z_{T} doesn't exist.)
    """

    def __init__(self, latent_dim: int, kernel_size: int = 6):
        super().__init__()
        if kernel_size < 2:
            raise ValueError(f"kernel_size must be >= 2, got {kernel_size}")
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(latent_dim, latent_dim, kernel_size=self.kernel_size, padding=0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError(f"Expected z with shape [B, D, T], got {tuple(z.shape)}")
        x = F.pad(z, (self.kernel_size - 1, 0))  # left-pad for causality
        return self.conv(x)
