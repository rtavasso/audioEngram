"""
Score network for denoising score matching on latent manifold.

Sigma-conditioned MLP that estimates nabla_z log p(z).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for sigma conditioning."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """sigma: [B, 1] -> [B, dim]."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=sigma.device, dtype=sigma.dtype) / half
        )
        args = sigma * freqs.unsqueeze(0)  # [B, half]
        return torch.cat([args.sin(), args.cos()], dim=-1)  # [B, dim]


class ScoreNetwork(nn.Module):
    """
    Sigma-conditioned MLP for score estimation.

    Architecture:
        Input: (z_noisy [B,D], sigma [B,1])
        Sigma -> sinusoidal embedding -> linear -> add to hidden
        Backbone: 4-layer MLP with skip connection (input z added to output)
        Output: score [B,D] (estimated nabla_z log p(z))
    """

    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 1024,
        n_layers: int = 4,
        sigma_embed_dim: int = 128,
        output_skip: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_skip = bool(output_skip)

        # Sigma conditioning
        self.sigma_embed = SinusoidalEmbedding(sigma_embed_dim)
        self.sigma_proj = nn.Linear(sigma_embed_dim, hidden_dim)

        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # Hidden layers
        layers = []
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            ])
        self.backbone = nn.Sequential(*layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_noisy: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Estimate score at z_noisy given noise level sigma.

        Args:
            z_noisy: [B, D] noisy latent
            sigma: [B, 1] noise level

        Returns:
            score: [B, D] estimated nabla_z log p(z)
        """
        # Sigma conditioning
        sigma_emb = self.sigma_embed(sigma)  # [B, sigma_embed_dim]
        sigma_h = self.sigma_proj(sigma_emb)  # [B, hidden_dim]

        # Input
        h = self.input_proj(z_noisy)  # [B, hidden_dim]
        h = h + sigma_h  # condition on sigma

        # Backbone with skip
        h = self.backbone(h)

        score = self.output_proj(h)
        if self.output_skip:
            score = score + z_noisy

        return score


def denoising_score_matching_loss(
    score_model: ScoreNetwork,
    z_clean: torch.Tensor,
    sigma_min: float = 0.01,
    sigma_max: float = 1.0,
    loss_weighting: str = "sigma2",
) -> torch.Tensor:
    """
    Denoising score matching loss.

    Sample sigma from geometric schedule, add noise, train to predict score.

    Args:
        score_model: ScoreNetwork
        z_clean: [B, D] clean latent frames
        sigma_min: minimum noise level
        sigma_max: maximum noise level

    Returns:
        Scalar loss.
    """
    B = z_clean.shape[0]
    device = z_clean.device

    # Sample sigma from log-uniform (geometric) distribution
    log_sigma = torch.rand(B, 1, device=device) * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min)
    sigma = log_sigma.exp()  # [B, 1]

    # Add noise
    eps = torch.randn_like(z_clean)  # [B, D]
    z_noisy = z_clean + sigma * eps  # [B, D]

    score = score_model(z_noisy, sigma)  # [B, D]

    lw = str(loss_weighting).lower().strip()
    if lw in ("sigma2", "sigma^2", "s2"):
        # Weighted DSM: sigma^2 * ||score - (-eps/sigma)||^2  ==  ||sigma*score + eps||^2
        return ((sigma * score + eps) ** 2).sum(dim=-1).mean()
    if lw in ("none", "unweighted"):
        target = -eps / sigma
        return ((score - target) ** 2).sum(dim=-1).mean()
    raise ValueError(f"Unknown loss_weighting: {loss_weighting} (expected sigma2|none)")


class ConditionalScoreNetwork(nn.Module):
    """
    Sigma-conditioned MLP for score estimation with a conditioning vector.

    This estimates the score of a conditional distribution p(z | cond), using
    denoising score matching:
        score(z_noisy, sigma, cond) ≈ ∇_z log p_sigma(z | cond)
    """

    def __init__(
        self,
        latent_dim: int = 512,
        cond_dim: int = 4096,
        hidden_dim: int = 1024,
        n_layers: int = 4,
        sigma_embed_dim: int = 128,
        output_skip: bool = False,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.cond_dim = int(cond_dim)
        self.output_skip = bool(output_skip)

        # Sigma conditioning
        self.sigma_embed = SinusoidalEmbedding(sigma_embed_dim)
        self.sigma_proj = nn.Linear(sigma_embed_dim, hidden_dim)

        # Conditioning projection
        self.cond_proj = nn.Linear(self.cond_dim, hidden_dim)

        # Input projection
        self.input_proj = nn.Linear(self.latent_dim, hidden_dim)

        # Hidden layers
        layers = []
        for _ in range(int(n_layers)):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            ])
        self.backbone = nn.Sequential(*layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, self.latent_dim)

    def forward(self, z_noisy: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_noisy: [B, D]
            sigma: [B, 1]
            cond: [B, C]
        Returns:
            score: [B, D]
        """
        sigma_emb = self.sigma_embed(sigma)
        sigma_h = self.sigma_proj(sigma_emb)
        cond_h = self.cond_proj(cond)

        h = self.input_proj(z_noisy)
        h = h + sigma_h + cond_h

        h = self.backbone(h)

        score = self.output_proj(h)
        if self.output_skip:
            score = score + z_noisy
        return score


def denoising_score_matching_loss_conditional(
    score_model: ConditionalScoreNetwork,
    z_clean: torch.Tensor,
    cond: torch.Tensor,
    sigma_min: float = 0.01,
    sigma_max: float = 1.0,
    loss_weighting: str = "sigma2",
) -> torch.Tensor:
    """
    Conditional denoising score matching loss.

    Args:
        score_model: ConditionalScoreNetwork
        z_clean: [B, D]
        cond: [B, C]
    """
    B = z_clean.shape[0]
    device = z_clean.device

    log_sigma = (
        torch.rand(B, 1, device=device)
        * (math.log(sigma_max) - math.log(sigma_min))
        + math.log(sigma_min)
    )
    sigma = log_sigma.exp()

    eps = torch.randn_like(z_clean)
    z_noisy = z_clean + sigma * eps

    score = score_model(z_noisy, sigma, cond)

    lw = str(loss_weighting).lower().strip()
    if lw in ("sigma2", "sigma^2", "s2"):
        return ((sigma * score + eps) ** 2).sum(dim=-1).mean()
    if lw in ("none", "unweighted"):
        target = -eps / sigma
        return ((score - target) ** 2).sum(dim=-1).mean()
    raise ValueError(f"Unknown loss_weighting: {loss_weighting} (expected sigma2|none)")
