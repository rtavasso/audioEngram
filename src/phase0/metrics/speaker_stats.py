"""
Per-speaker statistics for cross-speaker analysis.

Computes metrics per speaker and aggregates to detect speaker-specific effects.
"""

import numpy as np
from typing import Optional
from scipy import stats

from .variance_ratio import compute_variance_ratio
from .entropy import compute_entropy_reduction


def compute_speaker_level_metrics(
    deltas: np.ndarray,
    cluster_ids: np.ndarray,
    speaker_ids: np.ndarray,
    effective_clusters: Optional[np.ndarray] = None,
    min_samples_per_speaker: int = 50,
) -> dict:
    """
    Compute all metrics per speaker.

    Args:
        deltas: Normalized delta array [N, D]
        cluster_ids: Cluster assignments [N]
        speaker_ids: Speaker IDs [N]
        effective_clusters: Cluster IDs to include
        min_samples_per_speaker: Minimum samples to include a speaker

    Returns:
        Dict with per-speaker metrics
    """
    # Filter to effective clusters
    if effective_clusters is not None:
        mask = np.isin(cluster_ids, effective_clusters)
        deltas = deltas[mask]
        cluster_ids = cluster_ids[mask]
        speaker_ids = speaker_ids[mask]

    unique_speakers = np.unique(speaker_ids)
    results = {}

    for spk in unique_speakers:
        spk_mask = speaker_ids == spk
        spk_deltas = deltas[spk_mask]
        spk_clusters = cluster_ids[spk_mask]

        n_samples = len(spk_deltas)
        if n_samples < min_samples_per_speaker:
            continue

        # Compute variance ratio
        vr_result = compute_variance_ratio(spk_deltas, spk_clusters)

        # Compute entropy reduction
        er_result = compute_entropy_reduction(spk_deltas, spk_clusters)

        results[int(spk)] = {
            "n_samples": n_samples,
            "variance_ratio": vr_result["variance_ratio"],
            "entropy_reduction": er_result["entropy_reduction"],
            "n_clusters_used": len(np.unique(spk_clusters)),
        }

    return results


def aggregate_speaker_metrics(
    speaker_metrics: dict,
) -> dict:
    """
    Aggregate per-speaker metrics into summary statistics.

    Args:
        speaker_metrics: Output from compute_speaker_level_metrics

    Returns:
        Dict with aggregated statistics
    """
    if not speaker_metrics:
        return {
            "n_speakers": 0,
            "variance_ratio_mean": float("nan"),
            "variance_ratio_std": float("nan"),
            "variance_ratio_ci_lower": float("nan"),
            "variance_ratio_ci_upper": float("nan"),
            "entropy_reduction_mean": float("nan"),
            "entropy_reduction_std": float("nan"),
        }

    vr_values = [m["variance_ratio"] for m in speaker_metrics.values()]
    er_values = [m["entropy_reduction"] for m in speaker_metrics.values()]
    n_samples = [m["n_samples"] for m in speaker_metrics.values()]

    # Filter out NaN values
    vr_values = [v for v in vr_values if not np.isnan(v)]
    er_values = [v for v in er_values if not np.isnan(v)]

    # Compute confidence interval for variance ratio
    if len(vr_values) >= 2:
        vr_ci = stats.t.interval(
            0.95,
            len(vr_values) - 1,
            loc=np.mean(vr_values),
            scale=stats.sem(vr_values),
        )
    else:
        vr_ci = (float("nan"), float("nan"))

    return {
        "n_speakers": len(speaker_metrics),
        "total_samples": sum(n_samples),
        "variance_ratio_mean": float(np.mean(vr_values)) if vr_values else float("nan"),
        "variance_ratio_std": float(np.std(vr_values)) if vr_values else float("nan"),
        "variance_ratio_ci_lower": float(vr_ci[0]),
        "variance_ratio_ci_upper": float(vr_ci[1]),
        "variance_ratio_min": float(np.min(vr_values)) if vr_values else float("nan"),
        "variance_ratio_max": float(np.max(vr_values)) if vr_values else float("nan"),
        "entropy_reduction_mean": float(np.mean(er_values)) if er_values else float("nan"),
        "entropy_reduction_std": float(np.std(er_values)) if er_values else float("nan"),
    }


def compute_cross_speaker_degradation(
    train_metrics: dict,
    eval_metrics: dict,
) -> dict:
    """
    Compute degradation of metrics from train to eval speakers.

    Args:
        train_metrics: Aggregated metrics for train speakers
        eval_metrics: Aggregated metrics for eval speakers

    Returns:
        Dict with degradation statistics
    """
    train_vr = train_metrics["variance_ratio_mean"]
    eval_vr = eval_metrics["variance_ratio_mean"]

    # Degradation: how much worse (higher) is eval variance ratio
    if np.isnan(train_vr) or np.isnan(eval_vr):
        vr_degradation = float("nan")
    else:
        vr_degradation = eval_vr - train_vr

    # Relative degradation
    if np.isnan(vr_degradation) or train_vr == 0:
        vr_rel_degradation = float("nan")
    else:
        vr_rel_degradation = vr_degradation / train_vr

    return {
        "train_variance_ratio": train_vr,
        "eval_variance_ratio": eval_vr,
        "variance_ratio_degradation": vr_degradation,
        "variance_ratio_relative_degradation": vr_rel_degradation,
        "train_n_speakers": train_metrics["n_speakers"],
        "eval_n_speakers": eval_metrics["n_speakers"],
    }
