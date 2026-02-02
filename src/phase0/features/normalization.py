"""
Delta normalization for target variable.

Computes velocity (Δx = x[t] - x[t-1]) and normalizes globally.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class NormalizationStats:
    """Global normalization statistics for delta computation."""

    mu: np.ndarray  # [D] mean
    sigma: np.ndarray  # [D] std
    n_samples: int  # number of samples used

    def save(self, path: str | Path) -> None:
        """Save stats to JSON file."""
        data = {
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
            "n_samples": self.n_samples,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationStats":
        """Load stats from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            mu=np.array(data["mu"], dtype=np.float32),
            sigma=np.array(data["sigma"], dtype=np.float32),
            n_samples=data["n_samples"],
        )


def compute_delta(x: np.ndarray, t: int) -> np.ndarray:
    """
    Compute velocity for a target frame.

    Args:
        x: Latent sequence [T, D]
        t: Target frame index (must be >= 1)

    Returns:
        Delta [D] = x[t] - x[t-1]
    """
    if t < 1:
        raise ValueError(f"Cannot compute delta for t={t}, need t >= 1")
    return x[t] - x[t - 1]


def compute_normalization_stats(
    deltas: np.ndarray,
    min_sigma: float = 1e-6,
) -> NormalizationStats:
    """
    Compute global normalization statistics from delta samples.

    Args:
        deltas: Delta array [N, D]
        min_sigma: Minimum sigma to prevent division by zero

    Returns:
        NormalizationStats with mean and std per dimension
    """
    mu = deltas.mean(axis=0)
    sigma = deltas.std(axis=0)

    # Clamp sigma to prevent division by zero
    sigma = np.maximum(sigma, min_sigma)

    return NormalizationStats(
        mu=mu.astype(np.float32),
        sigma=sigma.astype(np.float32),
        n_samples=len(deltas),
    )


def normalize_delta(
    delta: np.ndarray,
    stats: NormalizationStats,
) -> np.ndarray:
    """
    Normalize a delta vector using global stats.

    Args:
        delta: Delta vector [D] or batch [N, D]
        stats: Normalization statistics

    Returns:
        Normalized delta with same shape
    """
    return (delta - stats.mu) / stats.sigma


def collect_deltas_for_utterance(
    x: np.ndarray,
    valid_frame_indices: list[int],
) -> np.ndarray:
    """
    Collect all deltas for an utterance's valid frames.

    Args:
        x: Latent sequence [T, D]
        valid_frame_indices: List of valid target frame indices

    Returns:
        Deltas array [N, D]
    """
    deltas = np.array([compute_delta(x, t) for t in valid_frame_indices])
    return deltas
