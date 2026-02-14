"""
Factored Direction+Magnitude AR model for discrete direction prediction.

Models p(Δz | context) as:
  dir_index ~ Categorical(softmax(logits(ctx)))     # discrete direction from codebook
  magnitude ~ LogNormal(mu_logm(ctx), sigma_logm(ctx))  # continuous magnitude

  Δz = magnitude * codebook[dir_index]

The codebook is a fixed [K, D] matrix of unit vectors (from exp 8 spherical k-means).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DirectionAROutput:
    dir_logits: torch.Tensor    # [B, K]
    mu_logm: torch.Tensor       # [B]
    log_sigma_logm: torch.Tensor  # [B]


class FactoredDirectionMagnitudeAR(nn.Module):
    """
    AR model predicting discrete direction index + continuous magnitude.

    Direction: cross-entropy over K codebook entries
    Magnitude: LogNormal NLL
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        K: int,
        codebook: np.ndarray,
        hidden_dim: int = 1024,
        n_hidden_layers: int = 2,
        dropout: float = 0.0,
        min_log_sigma_logm: float = -5.0,
        max_log_sigma_logm: float = 0.7,
        min_mu_logm: float = -5.0,
        max_mu_logm: float = 12.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.K = int(K)
        self.hidden_dim = int(hidden_dim)
        self.n_hidden_layers = int(n_hidden_layers)
        self.dropout_rate = float(dropout)
        self._min_log_sigma_logm = float(min_log_sigma_logm)
        self._max_log_sigma_logm = float(max_log_sigma_logm)
        self._min_mu_logm = float(min_mu_logm)
        self._max_mu_logm = float(max_mu_logm)

        # Codebook: fixed unit vectors [K, D]
        cb = np.array(codebook, dtype=np.float32)
        if cb.shape != (self.K, self.output_dim):
            raise ValueError(
                f"Codebook shape {cb.shape} does not match (K={self.K}, D={self.output_dim})"
            )
        self.register_buffer("codebook", torch.from_numpy(cb))

        # Shared backbone (matches MDN capacity)
        layers: list[nn.Module] = []
        d = self.input_dim
        for _ in range(int(n_hidden_layers)):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.GELU())
            if dropout and float(dropout) > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            d = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()

        # Direction head: logits over K classes
        self.dir_head = nn.Linear(d, self.K)

        # Magnitude head: LogNormal parameters
        self.mu_logm_head = nn.Linear(d, 1)
        self.log_sigma_logm_head = nn.Linear(d, 1)

        # Init: conservative sigma
        nn.init.constant_(self.log_sigma_logm_head.bias, -0.5)

    def forward(self, ctx: torch.Tensor) -> DirectionAROutput:
        """
        Args:
            ctx: [B, W*D] flattened context window

        Returns:
            DirectionAROutput with dir_logits [B,K], mu_logm [B], log_sigma_logm [B]
        """
        h = self.backbone(ctx)
        dir_logits = self.dir_head(h)  # [B, K]

        mu_logm = self.mu_logm_head(h).squeeze(-1)  # [B]
        mu_logm = torch.clamp(mu_logm, self._min_mu_logm, self._max_mu_logm)

        log_sigma_logm = self.log_sigma_logm_head(h).squeeze(-1)  # [B]
        log_sigma_logm = torch.clamp(log_sigma_logm, self._min_log_sigma_logm, self._max_log_sigma_logm)

        return DirectionAROutput(
            dir_logits=dir_logits,
            mu_logm=mu_logm,
            log_sigma_logm=log_sigma_logm,
        )

    def nll(self, ctx: torch.Tensor, delta: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
        """
        Combined NLL: CE(direction) + LogNormal(magnitude).

        Args:
            ctx: [B, W*D]
            delta: [B, D] ground-truth Δz

        Returns:
            [B] per-sample NLL
        """
        magnitude = torch.linalg.vector_norm(delta, dim=-1).clamp_min(eps)  # [B]
        direction = delta / magnitude.unsqueeze(-1)  # [B, D]

        # Nearest codebook entry
        cos_sim = direction @ self.codebook.T  # [B, K]
        dir_index = cos_sim.argmax(dim=-1)  # [B]

        return self._nll_from_targets(ctx, dir_index, magnitude)

    def _nll_from_targets(
        self, ctx: torch.Tensor, dir_index: torch.Tensor, magnitude: torch.Tensor, *, eps: float = 1e-8,
    ) -> torch.Tensor:
        """Combined NLL given pre-computed targets."""
        out = self.forward(ctx)

        # Direction: cross-entropy
        ce = F.cross_entropy(out.dir_logits, dir_index, reduction="none")  # [B]

        # Magnitude: LogNormal NLL
        sigma = torch.exp(out.log_sigma_logm).clamp_min(eps)
        m = magnitude.clamp_min(eps)
        z = (torch.log(m) - out.mu_logm) / sigma
        log2pi = 1.8378770664093453
        mag_nll = torch.log(m) + torch.log(sigma) + 0.5 * (z * z + log2pi)

        return ce + mag_nll

    def nll_direction(self, ctx: torch.Tensor, dir_index: torch.Tensor) -> torch.Tensor:
        """Direction cross-entropy only. Returns [B]."""
        out = self.forward(ctx)
        return F.cross_entropy(out.dir_logits, dir_index, reduction="none")

    def nll_magnitude(self, ctx: torch.Tensor, magnitude: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
        """Magnitude LogNormal NLL only. Returns [B]."""
        out = self.forward(ctx)
        sigma = torch.exp(out.log_sigma_logm).clamp_min(eps)
        m = magnitude.clamp_min(eps)
        z = (torch.log(m) - out.mu_logm) / sigma
        log2pi = 1.8378770664093453
        return torch.log(m) + torch.log(sigma) + 0.5 * (z * z + log2pi)

    @torch.no_grad()
    def expected_mean(self, ctx: torch.Tensor) -> torch.Tensor:
        """
        Deterministic Δz prediction: argmax direction × median magnitude.

        Returns: [B, D]
        """
        out = self.forward(ctx)
        dir_idx = out.dir_logits.argmax(dim=-1)  # [B]
        direction = self.codebook[dir_idx]  # [B, D]
        m_median = torch.exp(out.mu_logm)  # [B] — median of LogNormal
        return direction * m_median.unsqueeze(-1)

    @torch.no_grad()
    def sample(
        self,
        ctx: torch.Tensor,
        strategy: str = "categorical",
        top_p: float = 0.9,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample direction index + magnitude.

        Args:
            ctx: [B, W*D]
            strategy: "argmax", "categorical", or "top_p"
            top_p: nucleus sampling threshold (only used if strategy="top_p")

        Returns:
            (dir_idx [B], magnitude [B])
        """
        out = self.forward(ctx)

        # Direction sampling
        if strategy == "argmax":
            dir_idx = out.dir_logits.argmax(dim=-1)  # [B]
        elif strategy == "categorical":
            probs = F.softmax(out.dir_logits, dim=-1)
            dir_idx = torch.multinomial(probs, 1).squeeze(-1)  # [B]
        elif strategy == "top_p":
            dir_idx = _top_p_sample(out.dir_logits, top_p)  # [B]
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        # Magnitude: sample from LogNormal
        sigma = torch.exp(out.log_sigma_logm)
        eps = torch.randn_like(out.mu_logm)
        magnitude = torch.exp(out.mu_logm + sigma * eps)  # [B]

        return dir_idx, magnitude

    @torch.no_grad()
    def sample_delta(self, ctx: torch.Tensor) -> torch.Tensor:
        """
        Sample Δz for rollout compatibility with Phase 1 eval code.
        Uses categorical sampling.
        """
        dir_idx, magnitude = self.sample(ctx, strategy="categorical")
        return self.reconstruct_delta(dir_idx, magnitude)

    def reconstruct_delta(self, dir_idx: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct Δz from direction index and magnitude.

        Args:
            dir_idx: [B] integer indices into codebook
            magnitude: [B] scalar magnitudes

        Returns:
            [B, D]
        """
        direction = self.codebook[dir_idx]  # [B, D]
        return direction * magnitude.unsqueeze(-1)

    def get_model_kwargs(self) -> dict:
        """Return kwargs dict for checkpoint reconstruction."""
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "K": self.K,
            "hidden_dim": self.hidden_dim,
            "n_hidden_layers": self.n_hidden_layers,
            "dropout": self.dropout_rate,
            "min_log_sigma_logm": self._min_log_sigma_logm,
            "max_log_sigma_logm": self._max_log_sigma_logm,
            "min_mu_logm": self._min_mu_logm,
            "max_mu_logm": self._max_mu_logm,
        }


def _top_p_sample(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus (top-p) sampling from logits [B, K]."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)

    # Zero out tokens beyond the nucleus
    mask = cumprobs - probs > top_p
    sorted_logits[mask] = float("-inf")

    # Sample from filtered distribution
    filtered_probs = F.softmax(sorted_logits, dim=-1)
    sampled_sorted_idx = torch.multinomial(filtered_probs, 1).squeeze(-1)  # [B]

    # Map back to original indices
    dir_idx = sorted_indices.gather(1, sampled_sorted_idx.unsqueeze(-1)).squeeze(-1)  # [B]
    return dir_idx


def load_direction_ar_checkpoint(
    path: str,
    codebook: np.ndarray,
    device: torch.device,
) -> tuple[FactoredDirectionMagnitudeAR, dict]:
    """Load a FactoredDirectionMagnitudeAR from checkpoint."""
    ckpt = torch.load(str(path), map_location=device)
    model_kwargs = ckpt["model_kwargs"]
    model = FactoredDirectionMagnitudeAR(codebook=codebook, **model_kwargs)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt
