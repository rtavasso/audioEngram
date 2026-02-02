"""
Visualization utilities for Phase 0 analysis.

Creates diagnostic plots for cluster quality and structure analysis.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_cluster_sizes(
    cluster_sizes: np.ndarray,
    output_path: str | Path,
    title: str = "Cluster Size Distribution",
    min_size_line: Optional[int] = 100,
) -> None:
    """
    Plot histogram of cluster sizes.

    Args:
        cluster_sizes: Array of cluster sizes [K]
        output_path: Path to save figure
        title: Plot title
        min_size_line: Optional vertical line for minimum size
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Remove empty clusters for histogram
    non_empty = cluster_sizes[cluster_sizes > 0]

    ax.hist(non_empty, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Cluster Size", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14)

    if min_size_line is not None:
        ax.axvline(min_size_line, color="red", linestyle="--", label=f"Min size = {min_size_line}")
        ax.legend()

    # Add statistics
    stats_text = (
        f"Total clusters: {len(cluster_sizes)}\n"
        f"Non-empty: {len(non_empty)}\n"
        f"Mean size: {non_empty.mean():.1f}\n"
        f"Median: {np.median(non_empty):.1f}"
    )
    ax.text(
        0.95, 0.95, stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_variance_vs_size(
    cluster_sizes: np.ndarray,
    cluster_variances: np.ndarray,
    output_path: str | Path,
    title: str = "Per-Cluster Variance vs Size",
) -> None:
    """
    Plot per-cluster variance against cluster size.

    Args:
        cluster_sizes: Array of cluster sizes [K]
        cluster_variances: Array of within-cluster variances [K]
        output_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter to non-empty clusters
    mask = cluster_sizes > 0
    sizes = cluster_sizes[mask]
    variances = cluster_variances[mask]

    ax.scatter(sizes, variances, alpha=0.5, s=20)
    ax.set_xlabel("Cluster Size", fontsize=12)
    ax.set_ylabel("Within-Cluster Variance", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xscale("log")

    # Add trend line
    if len(sizes) > 10:
        z = np.polyfit(np.log10(sizes), variances, 1)
        p = np.poly1d(z)
        x_line = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 100)
        ax.plot(x_line, p(np.log10(x_line)), "r--", alpha=0.7, label="Trend")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_pca_scatter(
    cluster_means: np.ndarray,
    cluster_sizes: np.ndarray,
    output_path: str | Path,
    title: str = "PCA of Cluster Mean Dynamics",
) -> None:
    """
    Plot 2D PCA scatter of cluster mean delta values.

    Point size proportional to cluster count.

    Args:
        cluster_means: Mean delta per cluster [K, D]
        cluster_sizes: Cluster sizes [K]
        output_path: Path to save figure
        title: Plot title
    """
    from sklearn.decomposition import PCA

    # Filter to non-empty clusters
    mask = cluster_sizes > 0
    means = cluster_means[mask]
    sizes = cluster_sizes[mask]

    if len(means) < 2:
        return

    # Fit PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(means)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scale point sizes
    size_scaled = 20 + 200 * (sizes / sizes.max())

    scatter = ax.scatter(
        projected[:, 0],
        projected[:, 1],
        s=size_scaled,
        c=sizes,
        cmap="viridis",
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
    )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)", fontsize=12)
    ax.set_title(title, fontsize=14)

    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster Size", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confidence_curve(
    variance_ratios: np.ndarray,
    distances: np.ndarray,
    output_path: str | Path,
    title: str = "Variance Ratio vs Assignment Confidence",
    n_bins: int = 10,
) -> None:
    """
    Plot variance ratio as a function of assignment confidence.

    Confidence = inverse of distance to centroid.

    Args:
        variance_ratios: Per-sample variance contribution
        distances: Distance to assigned centroid
        output_path: Path to save figure
        title: Plot title
        n_bins: Number of confidence bins
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert distance to confidence (lower distance = higher confidence)
    # Normalize distances
    dist_normalized = distances / distances.max() if distances.max() > 0 else distances
    confidence = 1.0 - dist_normalized

    # Bin by confidence
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_means = []
    bin_stds = []

    for i in range(n_bins):
        mask = (confidence >= bin_edges[i]) & (confidence < bin_edges[i + 1])
        if mask.sum() > 10:
            bin_means.append(variance_ratios[mask].mean())
            bin_stds.append(variance_ratios[mask].std())
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)

    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)

    # Plot
    ax.errorbar(
        bin_centers, bin_means, yerr=bin_stds,
        fmt="o-", capsize=5, markersize=8
    )
    ax.set_xlabel("Assignment Confidence (1 - normalized distance)", fontsize=12)
    ax.set_ylabel("Variance Ratio", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 1)

    # Add horizontal line at 0.6 threshold
    ax.axhline(0.6, color="red", linestyle="--", alpha=0.7, label="Threshold (0.6)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_speaker_distribution(
    speaker_variance_ratios: dict,
    output_path: str | Path,
    title: str = "Variance Ratio by Speaker",
) -> None:
    """
    Plot distribution of variance ratios across speakers.

    Args:
        speaker_variance_ratios: Dict mapping speaker_id -> variance_ratio
        output_path: Path to save figure
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ratios = list(speaker_variance_ratios.values())

    # Histogram
    ax1.hist(ratios, bins=30, edgecolor="black", alpha=0.7)
    ax1.axvline(np.mean(ratios), color="red", linestyle="-", label=f"Mean: {np.mean(ratios):.3f}")
    ax1.axvline(0.6, color="green", linestyle="--", label="Threshold: 0.6")
    ax1.set_xlabel("Variance Ratio", fontsize=12)
    ax1.set_ylabel("Number of Speakers", fontsize=12)
    ax1.set_title("Distribution", fontsize=12)
    ax1.legend()

    # Sorted bar plot
    sorted_items = sorted(speaker_variance_ratios.items(), key=lambda x: x[1])
    ax2.barh(range(len(sorted_items)), [v for _, v in sorted_items], alpha=0.7)
    ax2.axvline(0.6, color="red", linestyle="--", label="Threshold: 0.6")
    ax2.set_xlabel("Variance Ratio", fontsize=12)
    ax2.set_ylabel("Speaker (sorted)", fontsize=12)
    ax2.set_title("Per-Speaker Values", fontsize=12)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_all_plots(
    metrics: list,
    cluster_data: dict,
    output_dir: str | Path,
) -> None:
    """
    Create all diagnostic plots.

    Args:
        metrics: List of metric results
        cluster_data: Dict with cluster_sizes, cluster_means, etc.
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cluster size histogram
    if "cluster_sizes" in cluster_data:
        plot_cluster_sizes(
            cluster_data["cluster_sizes"],
            output_dir / "cluster_sizes.png",
        )

    # Variance vs size
    if "cluster_sizes" in cluster_data and "cluster_variances" in cluster_data:
        plot_variance_vs_size(
            cluster_data["cluster_sizes"],
            cluster_data["cluster_variances"],
            output_dir / "variance_vs_size.png",
        )

    # PCA scatter
    if "cluster_means" in cluster_data and "cluster_sizes" in cluster_data:
        plot_pca_scatter(
            cluster_data["cluster_means"],
            cluster_data["cluster_sizes"],
            output_dir / "pca_scatter.png",
        )

    # Speaker distribution
    if "speaker_variance_ratios" in cluster_data:
        plot_speaker_distribution(
            cluster_data["speaker_variance_ratios"],
            output_dir / "speaker_distribution.png",
        )
