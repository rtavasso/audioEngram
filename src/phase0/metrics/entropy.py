"""
Diagonal Gaussian entropy for evaluating cluster quality.

Computes entropy reduction under diagonal Gaussian assumption.
H ∝ (1/2) Σ_d log(σ_d^2)
"""

import numpy as np
from typing import Optional


def compute_diagonal_gaussian_entropy(
    data: np.ndarray,
    min_var: float = 1e-10,
) -> float:
    """
    Compute diagonal Gaussian entropy (up to constant).

    H ∝ (1/2) Σ_d log(σ_d^2)

    Args:
        data: Data array [N, D]
        min_var: Minimum variance to prevent log(0)

    Returns:
        Entropy value (relative, not absolute)
    """
    variances = data.var(axis=0)
    variances = np.maximum(variances, min_var)
    entropy = 0.5 * np.sum(np.log(variances))
    return float(entropy)


def compute_per_cluster_entropy(
    deltas: np.ndarray,
    cluster_ids: np.ndarray,
    effective_clusters: Optional[np.ndarray] = None,
    min_var: float = 1e-10,
) -> dict:
    """
    Compute entropy per cluster.

    Args:
        deltas: Normalized delta array [N, D]
        cluster_ids: Cluster assignments [N]
        effective_clusters: Optional array of cluster IDs to include
        min_var: Minimum variance

    Returns:
        Dict with per-cluster entropies and weighted average
    """
    if effective_clusters is not None:
        mask = np.isin(cluster_ids, effective_clusters)
        deltas = deltas[mask]
        cluster_ids = cluster_ids[mask]
        clusters_to_use = effective_clusters
    else:
        clusters_to_use = np.unique(cluster_ids)

    per_cluster = {}
    weighted_sum = 0.0
    total_count = 0

    for c in clusters_to_use:
        cluster_mask = cluster_ids == c
        cluster_deltas = deltas[cluster_mask]
        count = len(cluster_deltas)

        if count < 2:  # Need at least 2 samples for variance
            continue

        entropy = compute_diagonal_gaussian_entropy(cluster_deltas, min_var)
        per_cluster[int(c)] = {
            "entropy": entropy,
            "count": count,
        }

        weighted_sum += entropy * count
        total_count += count

    weighted_avg = weighted_sum / total_count if total_count > 0 else float("nan")

    return {
        "per_cluster": per_cluster,
        "weighted_average": weighted_avg,
        "n_clusters": len(per_cluster),
        "total_samples": total_count,
    }


def compute_entropy_reduction(
    deltas: np.ndarray,
    cluster_ids: np.ndarray,
    effective_clusters: Optional[np.ndarray] = None,
    min_var: float = 1e-10,
) -> dict:
    """
    Compute relative entropy reduction from clustering.

    reduction = (H_unconditional - H_conditional) / H_unconditional

    Args:
        deltas: Normalized delta array [N, D]
        cluster_ids: Cluster assignments [N]
        effective_clusters: Optional array of cluster IDs to include
        min_var: Minimum variance

    Returns:
        Dict with:
            - entropy_unconditional: H(Δx)
            - entropy_conditional: H(Δx | cluster)
            - entropy_reduction: Relative reduction
    """
    # Filter to effective clusters
    if effective_clusters is not None:
        mask = np.isin(cluster_ids, effective_clusters)
        deltas_filtered = deltas[mask]
        cluster_ids_filtered = cluster_ids[mask]
    else:
        deltas_filtered = deltas
        cluster_ids_filtered = cluster_ids

    # Unconditional entropy (on filtered samples)
    h_uncond = compute_diagonal_gaussian_entropy(deltas_filtered, min_var)

    # Conditional entropy (weighted average over clusters)
    cluster_result = compute_per_cluster_entropy(
        deltas_filtered, cluster_ids_filtered, min_var=min_var
    )
    h_cond = cluster_result["weighted_average"]

    # Compute reduction per SPEC: (H - H_cond) / H
    # Note: H can be negative after normalization, so we use H directly (not abs)
    if np.isnan(h_uncond) or np.isnan(h_cond):
        reduction = float("nan")
    elif h_uncond == 0:
        reduction = 0.0 if h_cond == 0 else float("-inf")
    else:
        reduction = (h_uncond - h_cond) / h_uncond

    return {
        "entropy_unconditional": h_uncond,
        "entropy_conditional": h_cond,
        "entropy_reduction": reduction,
        "n_samples": len(deltas_filtered),
        "n_clusters": cluster_result["n_clusters"],
    }
