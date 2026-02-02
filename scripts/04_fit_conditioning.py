#!/usr/bin/env python3
"""
Fit conditioning models (PCA, k-means, quartile bins) on training data.

Usage:
    uv run python scripts/04_fit_conditioning.py [--config configs/phase0.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import yaml
from tqdm import tqdm

from phase0.data.io import LatentStore, load_frames_index
from phase0.features.context import get_context_mean, get_context_flat
from phase0.features.normalization import compute_delta, compute_normalization_stats
from phase0.clustering.vq import fit_kmeans, assign_clusters, compute_cluster_stats
from phase0.clustering.pca import fit_pca, project_pca, compute_pca_stats
from phase0.clustering.quantile import fit_quartile_bins, assign_quartile_bins, compute_quartile_stats
from phase0.utils.logging import setup_logging
from phase0.utils.seed import set_seed


def collect_train_features(
    frames,
    latent_store,
    window_size,
    lag,
    mode="mean",
    max_samples=None,
):
    """Collect features from training frames."""
    features_list = []
    deltas_list = []
    speaker_ids_list = []

    # Group by utterance
    grouped = frames.groupby("utterance_id")

    for utt_id, utt_frames in tqdm(grouped, desc=f"Collecting {mode} features"):
        if utt_id not in latent_store:
            continue

        x = latent_store.get_latents(utt_id)

        for _, row in utt_frames.iterrows():
            t = row["t"]
            speaker_id = row["speaker_id"]

            if t < window_size + lag or t < 1:
                continue

            try:
                if mode == "mean":
                    feat = get_context_mean(x, t, window_size, lag)
                else:
                    feat = get_context_flat(x, t, window_size, lag)

                delta = compute_delta(x, t)

                features_list.append(feat)
                deltas_list.append(delta)
                speaker_ids_list.append(speaker_id)

            except Exception:
                continue

            if max_samples and len(features_list) >= max_samples:
                break

        if max_samples and len(features_list) >= max_samples:
            break

    return (
        np.array(features_list, dtype=np.float32),
        np.array(deltas_list, dtype=np.float32),
        np.array(speaker_ids_list, dtype=np.int32),
    )


def main():
    parser = argparse.ArgumentParser(description="Fit conditioning models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase0.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples for fitting (for debugging)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup
    logger = setup_logging()
    set_seed(config["seed"])

    # Create output directory
    output_dir = Path(config["output"]["conditioning_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    latent_store = LatentStore(config["output"]["latents_dir"])
    frames = load_frames_index(config["output"]["frames_index"])

    # Filter to train only
    train_frames = frames[frames["split"] == "train"]
    logger.info(f"Train frames: {len(train_frames)}")

    # Get parameters
    window_size = config["context"]["window_size"]
    lags = config["context"]["lags"]
    conditions = config["clustering"]["conditions"]
    min_cluster_size = config["clustering"]["min_cluster_size"]
    seed = config["seed"]

    # Process each lag
    for lag in lags:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing lag={lag}")
        logger.info("=" * 50)

        # Collect mean-pooled features
        logger.info("Collecting mean-pooled features...")
        feat_mean, deltas, speaker_ids = collect_train_features(
            train_frames, latent_store, window_size, lag,
            mode="mean", max_samples=args.max_samples
        )
        logger.info(f"Collected {len(feat_mean)} samples")

        # Collect flat features
        logger.info("Collecting flat features...")
        feat_flat, _, _ = collect_train_features(
            train_frames, latent_store, window_size, lag,
            mode="flat", max_samples=args.max_samples
        )

        # Compute normalization stats
        logger.info("Computing normalization stats...")
        norm_stats = compute_normalization_stats(deltas)
        norm_stats.save(output_dir / f"norm_stats_lag{lag}.json")
        logger.info(f"Delta mean range: [{norm_stats.mu.min():.4f}, {norm_stats.mu.max():.4f}]")
        logger.info(f"Delta std range: [{norm_stats.sigma.min():.4f}, {norm_stats.sigma.max():.4f}]")

        # Process each condition
        for cond_cfg in conditions:
            cond_name = cond_cfg["name"]
            cond_type = cond_cfg["type"]

            logger.info(f"\n--- {cond_name} ({cond_type}) ---")

            if cond_type == "mean_pool_vq":
                k = cond_cfg["k"]
                logger.info(f"Fitting k-means with K={k}...")

                cluster_model = fit_kmeans(feat_mean, k, seed=seed)
                cluster_model.save(output_dir / f"{cond_name}_lag{lag}_kmeans.pkl")

                cluster_ids, distances = assign_clusters(feat_mean, cluster_model)
                stats = compute_cluster_stats(cluster_ids, k, min_cluster_size)

                logger.info(f"Inertia: {cluster_model.inertia:.2f}")
                logger.info(f"Effective clusters: {stats['n_effective_clusters']}/{k}")
                logger.info(f"Excluded mass: {stats['excluded_mass']:.1%}")

            elif cond_type == "pca_vq":
                pca_dim = cond_cfg["pca_dim"]
                k = cond_cfg["k"]

                logger.info(f"Fitting PCA with n_components={pca_dim}...")
                pca_model = fit_pca(feat_flat, n_components=pca_dim, seed=seed)
                pca_model.save(output_dir / f"{cond_name}_lag{lag}_pca.pkl")

                pca_stats = compute_pca_stats(pca_model)
                logger.info(f"Variance explained: {pca_stats['total_variance_explained']:.1%}")

                feat_pca = project_pca(feat_flat, pca_model)

                logger.info(f"Fitting k-means with K={k}...")
                cluster_model = fit_kmeans(feat_pca, k, seed=seed)
                cluster_model.save(output_dir / f"{cond_name}_lag{lag}_kmeans.pkl")

                cluster_ids, distances = assign_clusters(feat_pca, cluster_model)
                stats = compute_cluster_stats(cluster_ids, k, min_cluster_size)

                logger.info(f"Effective clusters: {stats['n_effective_clusters']}/{k}")
                logger.info(f"Excluded mass: {stats['excluded_mass']:.1%}")

            elif cond_type == "pca_quartile":
                pca_dim = cond_cfg["pca_dim"]

                logger.info(f"Fitting PCA with n_components={pca_dim}...")
                pca_model = fit_pca(feat_flat, n_components=pca_dim, seed=seed)
                pca_model.save(output_dir / f"{cond_name}_lag{lag}_pca.pkl")

                feat_pca = project_pca(feat_flat, pca_model)

                logger.info("Fitting quartile bins...")
                bin_model = fit_quartile_bins(feat_pca)
                bin_model.save(output_dir / f"{cond_name}_lag{lag}_bins.pkl")

                bin_ids = assign_quartile_bins(feat_pca, bin_model)
                stats = compute_quartile_stats(bin_ids, pca_dim, min_cluster_size)

                logger.info(f"Non-empty bins: {stats['n_nonempty_bins']}/{stats['n_possible_bins']}")
                logger.info(f"Effective bins: {stats['n_effective_bins']}")
                logger.info(f"Excluded mass: {stats['excluded_mass']:.1%}")

            else:
                logger.warning(f"Unknown condition type: {cond_type}")

            # Save stats
            with open(output_dir / f"{cond_name}_lag{lag}_stats.json", "w") as f:
                json.dump(stats, f, indent=2)

    logger.info("\nDone! Models saved to: " + str(output_dir))


if __name__ == "__main__":
    main()
