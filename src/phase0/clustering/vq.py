"""
K-means vector quantization for context clustering.

Fits k-means on training data and assigns cluster IDs.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class ClusterModel:
    """K-means clustering model with metadata."""

    centroids: np.ndarray  # [K, D]
    k: int
    n_train_samples: int
    inertia: float  # sum of squared distances

    def save(self, path: str | Path) -> None:
        """Save model to pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "ClusterModel":
        """Load model from pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)


def fit_kmeans(
    features: np.ndarray,
    k: int,
    seed: int = 42,
    n_init: int = 10,
    max_iter: int = 100,
) -> ClusterModel:
    """
    Fit k-means clustering on features.

    Args:
        features: Feature array [N, D]
        k: Number of clusters
        seed: Random seed for reproducibility
        n_init: Number of initializations
        max_iter: Maximum iterations

    Returns:
        ClusterModel with fitted centroids
    """
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
    )
    kmeans.fit(features)

    return ClusterModel(
        centroids=kmeans.cluster_centers_.astype(np.float32),
        k=k,
        n_train_samples=len(features),
        inertia=float(kmeans.inertia_),
    )


def assign_clusters(
    features: np.ndarray,
    model: ClusterModel,
    batch_size: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign cluster IDs to features.

    Args:
        features: Feature array [N, D]
        model: Fitted ClusterModel
        batch_size: Process in batches to limit memory usage

    Returns:
        Tuple of (cluster_ids [N], distances [N])
    """
    n_samples = len(features)
    cluster_ids = np.empty(n_samples, dtype=np.int32)
    distances = np.empty(n_samples, dtype=np.float32)

    # Process in batches to avoid memory explosion
    # Full broadcast would create [N, K, D] array which is huge
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = features[start:end]

        # Compute distances for this batch only
        diff = batch[:, None, :] - model.centroids[None, :, :]  # [batch, K, D]
        sq_dists = np.sum(diff**2, axis=2)  # [batch, K]

        batch_ids = np.argmin(sq_dists, axis=1)
        batch_dists = np.sqrt(sq_dists[np.arange(len(batch)), batch_ids])

        cluster_ids[start:end] = batch_ids
        distances[start:end] = batch_dists

    return cluster_ids, distances


def get_cluster_sizes(cluster_ids: np.ndarray, k: int) -> np.ndarray:
    """
    Count samples per cluster.

    Args:
        cluster_ids: Cluster assignments [N]
        k: Number of clusters

    Returns:
        Counts array [K]
    """
    counts = np.bincount(cluster_ids, minlength=k)
    return counts


def get_effective_clusters(
    cluster_ids: np.ndarray,
    k: int,
    min_size: int = 100,
) -> np.ndarray:
    """
    Get cluster IDs with at least min_size samples.

    Args:
        cluster_ids: Cluster assignments [N]
        k: Number of clusters
        min_size: Minimum cluster size

    Returns:
        Array of effective cluster IDs
    """
    counts = get_cluster_sizes(cluster_ids, k)
    effective = np.where(counts >= min_size)[0]
    return effective


def compute_cluster_stats(
    cluster_ids: np.ndarray,
    k: int,
    min_size: int = 100,
) -> dict:
    """
    Compute statistics about cluster assignments.

    Args:
        cluster_ids: Cluster assignments [N]
        k: Number of clusters
        min_size: Minimum cluster size for effective clusters

    Returns:
        Dict with cluster statistics
    """
    counts = get_cluster_sizes(cluster_ids, k)
    effective = get_effective_clusters(cluster_ids, k, min_size)

    n_total = len(cluster_ids)
    n_in_effective = sum(counts[c] for c in effective)
    excluded_mass = 1.0 - (n_in_effective / n_total) if n_total > 0 else 0.0

    return {
        "n_total_samples": n_total,
        "n_clusters": k,
        "n_effective_clusters": len(effective),
        "n_in_effective_clusters": n_in_effective,
        "excluded_mass": excluded_mass,
        "cluster_sizes": counts.tolist(),
        "effective_cluster_ids": effective.tolist(),
        "min_cluster_size": int(counts.min()),
        "max_cluster_size": int(counts.max()),
        "mean_cluster_size": float(counts.mean()),
    }
