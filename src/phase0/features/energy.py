"""
Frame energy utilities for confound analysis.

Computes energy thresholds and flags for high-energy frame filtering.
"""

import numpy as np


def compute_median_energy(
    all_energies: list[np.ndarray],
) -> float:
    """
    Compute global median energy across all frames.

    Args:
        all_energies: List of energy arrays, one per utterance

    Returns:
        Median energy value
    """
    # Concatenate all energies
    concatenated = np.concatenate(all_energies)
    return float(np.median(concatenated))


def is_high_energy(
    energy: float | np.ndarray,
    threshold: float,
) -> bool | np.ndarray:
    """
    Check if energy is above threshold.

    Args:
        energy: Single value or array of energy values
        threshold: Energy threshold (typically global median)

    Returns:
        Boolean or boolean array
    """
    return energy > threshold


def get_energy_stats(energies: np.ndarray) -> dict:
    """
    Compute energy statistics for an utterance.

    Args:
        energies: Energy array [T]

    Returns:
        Dict with mean, std, min, max, median
    """
    return {
        "mean": float(np.mean(energies)),
        "std": float(np.std(energies)),
        "min": float(np.min(energies)),
        "max": float(np.max(energies)),
        "median": float(np.median(energies)),
    }
