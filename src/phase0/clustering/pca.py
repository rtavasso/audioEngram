"""
PCA projection for dimensionality reduction.

Fits PCA on training data and projects to lower dimensions.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class PCAModel:
    """PCA model with metadata."""

    components: np.ndarray  # [n_components, D]
    mean: np.ndarray  # [D]
    explained_variance_ratio: np.ndarray  # [n_components]
    n_components: int
    n_train_samples: int

    def save(self, path: str | Path) -> None:
        """Save model to pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "PCAModel":
        """Load model from pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)


def fit_pca(
    features: np.ndarray,
    n_components: int = 8,
    seed: int = 42,
) -> PCAModel:
    """
    Fit PCA on features.

    Args:
        features: Feature array [N, D]
        n_components: Number of components to keep
        seed: Random seed for reproducibility

    Returns:
        PCAModel with fitted projection
    """
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(features)

    return PCAModel(
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        n_components=n_components,
        n_train_samples=len(features),
    )


def project_pca(
    features: np.ndarray,
    model: PCAModel,
) -> np.ndarray:
    """
    Project features using fitted PCA.

    Args:
        features: Feature array [N, D]
        model: Fitted PCAModel

    Returns:
        Projected features [N, n_components]
    """
    # Center and project
    centered = features - model.mean
    projected = np.dot(centered, model.components.T)
    return projected.astype(np.float32)


def compute_pca_stats(model: PCAModel) -> dict:
    """
    Compute statistics about PCA projection.

    Args:
        model: Fitted PCAModel

    Returns:
        Dict with PCA statistics
    """
    return {
        "n_components": model.n_components,
        "n_train_samples": model.n_train_samples,
        "explained_variance_ratio": model.explained_variance_ratio.tolist(),
        "cumulative_variance_ratio": np.cumsum(model.explained_variance_ratio).tolist(),
        "total_variance_explained": float(np.sum(model.explained_variance_ratio)),
    }
