"""
Baseline clustering methods for control experiments.

Provides random cluster assignments that preserve cluster size distribution.
"""

import numpy as np


def create_random_clusters(
    n_samples: int,
    k: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Create random cluster assignments.

    Args:
        n_samples: Number of samples
        k: Number of clusters
        seed: Random seed

    Returns:
        Random cluster IDs [N]
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, k, size=n_samples).astype(np.int32)


def permute_cluster_ids(
    cluster_ids: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """
    Permute cluster IDs across samples.

    This preserves the exact cluster size distribution while breaking
    any relationship between cluster ID and sample features.

    This is the best control baseline - it has exactly the same
    cluster size histogram as the original assignments.

    Args:
        cluster_ids: Original cluster assignments [N]
        seed: Random seed

    Returns:
        Permuted cluster IDs [N]
    """
    rng = np.random.default_rng(seed)
    permuted = cluster_ids.copy()
    rng.shuffle(permuted)
    return permuted


def create_stratified_random_clusters(
    n_samples: int,
    target_counts: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """
    Create random clusters with specified size distribution.

    Args:
        n_samples: Number of samples
        target_counts: Target count per cluster [K]
        seed: Random seed

    Returns:
        Cluster IDs [N] with approximately target distribution
    """
    rng = np.random.default_rng(seed)

    k = len(target_counts)

    # Normalize to get probabilities
    probs = target_counts / target_counts.sum()

    # Sample cluster IDs
    cluster_ids = rng.choice(k, size=n_samples, p=probs)

    return cluster_ids.astype(np.int32)


def verify_baseline_preserves_histogram(
    original_ids: np.ndarray,
    permuted_ids: np.ndarray,
) -> bool:
    """
    Verify that permutation preserved cluster size histogram.

    Args:
        original_ids: Original cluster assignments
        permuted_ids: Permuted cluster assignments

    Returns:
        True if histograms match exactly
    """
    original_counts = np.bincount(original_ids)
    permuted_counts = np.bincount(permuted_ids)

    # Pad to same length
    max_len = max(len(original_counts), len(permuted_counts))
    original_counts = np.pad(original_counts, (0, max_len - len(original_counts)))
    permuted_counts = np.pad(permuted_counts, (0, max_len - len(permuted_counts)))

    return np.array_equal(np.sort(original_counts), np.sort(permuted_counts))
