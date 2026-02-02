"""
Variance ratio metric for evaluating cluster quality.

The variance ratio measures how much within-cluster variance is reduced
relative to total variance. Lower is better (clusters explain more variance).

variance_ratio = SSE_within / SSE_total

Target: < 0.6 (clusters explain >40% of variance)
"""

import numpy as np
from typing import Optional


def compute_total_sse(
    deltas: np.ndarray,
) -> float:
    """
    Compute total sum of squared errors from global mean.

    Args:
        deltas: Normalized delta array [N, D]

    Returns:
        Total SSE (scalar)
    """
    global_mean = deltas.mean(axis=0)
    diff = deltas - global_mean
    sse = np.sum(diff**2)
    return float(sse)


def compute_within_cluster_sse(
    deltas: np.ndarray,
    cluster_ids: np.ndarray,
    effective_clusters: Optional[np.ndarray] = None,
) -> float:
    """
    Compute within-cluster sum of squared errors.

    Only includes samples from effective clusters if specified.

    Args:
        deltas: Normalized delta array [N, D]
        cluster_ids: Cluster assignments [N]
        effective_clusters: Optional array of cluster IDs to include

    Returns:
        Within-cluster SSE (scalar)
    """
    if effective_clusters is not None:
        # Filter to effective clusters
        mask = np.isin(cluster_ids, effective_clusters)
        deltas = deltas[mask]
        cluster_ids = cluster_ids[mask]

    unique_clusters = np.unique(cluster_ids)
    sse = 0.0

    for c in unique_clusters:
        cluster_mask = cluster_ids == c
        cluster_deltas = deltas[cluster_mask]
        cluster_mean = cluster_deltas.mean(axis=0)
        diff = cluster_deltas - cluster_mean
        sse += np.sum(diff**2)

    return float(sse)


def compute_variance_ratio(
    deltas: np.ndarray,
    cluster_ids: np.ndarray,
    effective_clusters: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute variance ratio metric.

    IMPORTANT: Both SSE_within and SSE_total are computed on the same
    filtered sample set (samples in effective clusters only).

    Args:
        deltas: Normalized delta array [N, D]
        cluster_ids: Cluster assignments [N]
        effective_clusters: Optional array of cluster IDs to include

    Returns:
        Dict with:
            - variance_ratio: SSE_within / SSE_total
            - sse_within: Within-cluster SSE
            - sse_total: Total SSE
            - n_samples: Number of samples used
            - n_clusters: Number of clusters used
    """
    # Filter to effective clusters if specified
    if effective_clusters is not None:
        mask = np.isin(cluster_ids, effective_clusters)
        deltas = deltas[mask]
        cluster_ids = cluster_ids[mask]
        n_clusters = len(effective_clusters)
    else:
        n_clusters = len(np.unique(cluster_ids))

    n_samples = len(deltas)

    if n_samples == 0:
        return {
            "variance_ratio": float("nan"),
            "sse_within": 0.0,
            "sse_total": 0.0,
            "n_samples": 0,
            "n_clusters": 0,
        }

    # Compute SSE on the same (filtered) sample set
    sse_total = compute_total_sse(deltas)
    sse_within = compute_within_cluster_sse(deltas, cluster_ids)

    # Avoid division by zero
    if sse_total == 0:
        variance_ratio = 0.0 if sse_within == 0 else float("inf")
    else:
        variance_ratio = sse_within / sse_total

    return {
        "variance_ratio": variance_ratio,
        "sse_within": sse_within,
        "sse_total": sse_total,
        "n_samples": n_samples,
        "n_clusters": n_clusters,
    }


def compute_variance_ratio_per_speaker(
    deltas: np.ndarray,
    cluster_ids: np.ndarray,
    speaker_ids: np.ndarray,
    effective_clusters: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute variance ratio per speaker.

    Args:
        deltas: Normalized delta array [N, D]
        cluster_ids: Cluster assignments [N]
        speaker_ids: Speaker IDs [N]
        effective_clusters: Optional array of cluster IDs to include

    Returns:
        Dict with:
            - per_speaker: Dict[speaker_id -> variance_ratio]
            - mean: Mean variance ratio across speakers
            - std: Std of variance ratio across speakers
            - n_speakers: Number of speakers
    """
    # Filter to effective clusters if specified
    if effective_clusters is not None:
        mask = np.isin(cluster_ids, effective_clusters)
        deltas = deltas[mask]
        cluster_ids = cluster_ids[mask]
        speaker_ids = speaker_ids[mask]

    unique_speakers = np.unique(speaker_ids)
    per_speaker = {}

    for spk in unique_speakers:
        spk_mask = speaker_ids == spk
        spk_deltas = deltas[spk_mask]
        spk_clusters = cluster_ids[spk_mask]

        if len(spk_deltas) < 10:  # Skip speakers with too few samples
            continue

        result = compute_variance_ratio(spk_deltas, spk_clusters)
        per_speaker[int(spk)] = result["variance_ratio"]

    ratios = list(per_speaker.values())

    return {
        "per_speaker": per_speaker,
        "mean": float(np.mean(ratios)) if ratios else float("nan"),
        "std": float(np.std(ratios)) if ratios else float("nan"),
        "n_speakers": len(per_speaker),
    }


def compute_variance_explained(variance_ratio: float) -> float:
    """
    Convert variance ratio to variance explained.

    variance_explained = 1 - variance_ratio
    """
    return 1.0 - variance_ratio
