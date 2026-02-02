"""
Quartile binning for axis-aligned discretization.

Bins each PCA dimension into quartiles and hashes to bin IDs.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class QuartileBinModel:
    """Quartile binning model with edge values."""

    edges: np.ndarray  # [n_dims, 3] quartile edges (25%, 50%, 75%) per dimension
    n_dims: int
    n_train_samples: int

    def save(self, path: str | Path) -> None:
        """Save model to pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "QuartileBinModel":
        """Load model from pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)


def fit_quartile_bins(
    features: np.ndarray,
) -> QuartileBinModel:
    """
    Fit quartile bins on features.

    Computes 25th, 50th, 75th percentile edges for each dimension.

    Args:
        features: Feature array [N, D]

    Returns:
        QuartileBinModel with fitted edges
    """
    n_dims = features.shape[1]
    edges = np.zeros((n_dims, 3), dtype=np.float32)

    for d in range(n_dims):
        edges[d, 0] = np.percentile(features[:, d], 25)
        edges[d, 1] = np.percentile(features[:, d], 50)
        edges[d, 2] = np.percentile(features[:, d], 75)

    return QuartileBinModel(
        edges=edges,
        n_dims=n_dims,
        n_train_samples=len(features),
    )


def assign_quartile_bins(
    features: np.ndarray,
    model: QuartileBinModel,
) -> np.ndarray:
    """
    Assign quartile bin IDs to features.

    Each dimension is binned to {0, 1, 2, 3} based on quartile edges.
    The final bin ID is a hash: sum(bin[d] * 4^d) for d in range(n_dims).

    For n_dims=8, this gives up to 4^8 = 65536 possible bins.

    Args:
        features: Feature array [N, D]
        model: Fitted QuartileBinModel

    Returns:
        Bin IDs [N]
    """
    n_samples = features.shape[0]
    n_dims = model.n_dims

    # Compute bin index per dimension
    bins_per_dim = np.zeros((n_samples, n_dims), dtype=np.int32)

    for d in range(n_dims):
        x = features[:, d]
        # Bin 0: x <= Q1
        # Bin 1: Q1 < x <= Q2
        # Bin 2: Q2 < x <= Q3
        # Bin 3: x > Q3
        bins_per_dim[:, d] = np.digitize(x, model.edges[d])

    # Hash to single bin ID: sum(bin[d] * 4^d)
    powers = 4 ** np.arange(n_dims)
    bin_ids = np.sum(bins_per_dim * powers, axis=1)

    return bin_ids.astype(np.int32)


def get_n_possible_bins(n_dims: int) -> int:
    """Get maximum number of possible bins for given dimensions."""
    return 4**n_dims


def compute_quartile_stats(
    bin_ids: np.ndarray,
    n_dims: int,
    min_size: int = 100,
) -> dict:
    """
    Compute statistics about quartile binning.

    Args:
        bin_ids: Bin assignments [N]
        n_dims: Number of dimensions
        min_size: Minimum bin size for effective bins

    Returns:
        Dict with binning statistics
    """
    n_possible = get_n_possible_bins(n_dims)

    # Count non-empty bins
    unique_bins, counts = np.unique(bin_ids, return_counts=True)
    n_nonempty = len(unique_bins)

    # Effective bins (>= min_size)
    effective_mask = counts >= min_size
    n_effective = np.sum(effective_mask)
    effective_bins = unique_bins[effective_mask]

    n_total = len(bin_ids)
    n_in_effective = sum(counts[effective_mask])
    excluded_mass = 1.0 - (n_in_effective / n_total) if n_total > 0 else 0.0

    return {
        "n_total_samples": n_total,
        "n_possible_bins": n_possible,
        "n_nonempty_bins": n_nonempty,
        "n_effective_bins": int(n_effective),
        "excluded_mass": excluded_mass,
        "effective_bin_ids": effective_bins.tolist(),
        "min_bin_size": int(counts.min()) if len(counts) > 0 else 0,
        "max_bin_size": int(counts.max()) if len(counts) > 0 else 0,
        "mean_bin_size": float(counts.mean()) if len(counts) > 0 else 0,
    }
