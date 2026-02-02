"""
Main orchestrator for Phase 0 analysis.

Runs all conditions × lags × slices and computes all metrics.

CRITICAL: Conditioning models (PCA, k-means, quartile bins) are fit ONCE per lag
on ALL train frames. Confound slices then evaluate using the same models but
filtering the sample set. This ensures confound checks test whether structure
*persists* under subset selection, not whether structure can be *rediscovered*.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from ..data.io import LatentStore, load_frames_index
from ..features.context import get_context_mean, get_context_flat, get_valid_frame_range
from ..features.normalization import (
    compute_delta,
    compute_normalization_stats,
    normalize_delta,
    NormalizationStats,
)
from ..clustering.vq import (
    fit_kmeans,
    assign_clusters,
    get_effective_clusters,
    get_cluster_sizes,
    compute_cluster_stats,
    ClusterModel,
)
from ..clustering.pca import fit_pca, project_pca, PCAModel
from ..clustering.quantile import fit_quartile_bins, assign_quartile_bins, QuartileBinModel
from ..clustering.baselines import permute_cluster_ids
from ..metrics.variance_ratio import compute_variance_ratio, compute_variance_ratio_per_speaker
from ..metrics.entropy import compute_entropy_reduction
from ..metrics.speaker_stats import (
    compute_speaker_level_metrics,
    aggregate_speaker_metrics,
    compute_cross_speaker_degradation,
)
from ..utils.seed import set_seed
from ..utils.logging import get_logger


def _optimize_frames_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory footprint of the frames index DataFrame.

    Notes:
    - utterance_id repeats heavily; categorical encoding can save GBs on large datasets.
    - Downcast numeric columns where safe.
    """
    if "utterance_id" in df.columns and df["utterance_id"].dtype != "category":
        df["utterance_id"] = df["utterance_id"].astype("category")
    if "speaker_id" in df.columns:
        df["speaker_id"] = df["speaker_id"].astype(np.int32, copy=False)
    if "t" in df.columns:
        df["t"] = df["t"].astype(np.int32, copy=False)
    if "pos_frac" in df.columns:
        df["pos_frac"] = df["pos_frac"].astype(np.float32, copy=False)
    if "is_high_energy" in df.columns:
        df["is_high_energy"] = df["is_high_energy"].astype(bool, copy=False)
    return df


def load_config(config_path: str | Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_slice_mask(
    slice_name: str,
    is_high_energy: np.ndarray,
    pos_frac: np.ndarray,
) -> np.ndarray:
    """
    Get boolean mask for a slice using per-sample metadata.

    Args:
        slice_name: Name of slice (all, high_energy, utterance_medial)
        is_high_energy: Boolean array [N] marking high-energy frames
        pos_frac: Float array [N] position fraction within utterance

    Returns:
        Boolean mask array
    """
    n = len(pos_frac)
    if slice_name == "all":
        return np.ones(n, dtype=bool)

    if slice_name == "high_energy":
        return is_high_energy.astype(bool, copy=False)

    if slice_name == "utterance_medial":
        return (pos_frac >= 0.17) & (pos_frac <= 0.83)

    raise ValueError(f"Unknown slice: {slice_name}")


def collect_all_features_and_deltas(
    frames: pd.DataFrame,
    latent_store: LatentStore,
    window_size: int,
    lag: int,
    max_samples: Optional[int] = None,
) -> dict:
    """
    Collect BOTH mean and flat features in a single pass to ensure alignment.

    This is critical: features and deltas must correspond to the same frames.

    Returns:
        Dict with:
            - features_mean: [N, D] mean-pooled context
            - features_flat: [N, W*D] flattened context
            - deltas: [N, D] velocity targets
            - speaker_ids: [N] speaker IDs
            - frame_keys: List of (utterance_id, t) tuples
    """
    logger = get_logger()

    features_mean_list = []
    features_flat_list = []
    deltas_list = []
    speaker_ids_list = []
    is_high_energy_list = []
    pos_frac_list = []
    frame_keys = []

    current_utt_id: Optional[str] = None
    current_latents: Optional[np.ndarray] = None

    # Iterate in DataFrame order (avoids expensive groupby on multi-million-row frames tables).
    for row in frames.itertuples(index=False):
        utt_id = str(row.utterance_id)
        t = int(row.t)
        speaker_id = int(row.speaker_id)

        if t < window_size + lag or t < 1:
            continue

        if current_utt_id != utt_id:
            if utt_id not in latent_store:
                current_utt_id = None
                current_latents = None
                continue

            try:
                current_latents = latent_store.get_latents(utt_id)
                current_utt_id = utt_id
            except Exception as e:
                logger.warning(f"Could not load latents for {utt_id}: {e}")
                current_utt_id = None
                current_latents = None
                continue

        try:
            x = current_latents
            if x is None:
                continue

            feat_mean = get_context_mean(x, t, window_size, lag)
            feat_flat = get_context_flat(x, t, window_size, lag)
            delta = compute_delta(x, t)

            features_mean_list.append(feat_mean)
            features_flat_list.append(feat_flat)
            deltas_list.append(delta)
            speaker_ids_list.append(speaker_id)
            is_high_energy_list.append(bool(row.is_high_energy))
            pos_frac_list.append(float(row.pos_frac))
            frame_keys.append((utt_id, t))

        except Exception as e:
            logger.warning(f"Error processing {utt_id}:{t}: {e}")
            continue

        if max_samples and len(features_mean_list) >= max_samples:
            break

    return {
        "features_mean": np.array(features_mean_list, dtype=np.float32),
        "features_flat": np.array(features_flat_list, dtype=np.float32),
        "deltas": np.array(deltas_list, dtype=np.float32),
        "speaker_ids": np.array(speaker_ids_list, dtype=np.int32),
        "is_high_energy": np.array(is_high_energy_list, dtype=bool),
        "pos_frac": np.array(pos_frac_list, dtype=np.float32),
        "frame_keys": frame_keys,
    }


def fit_condition_models(
    condition_config: dict,
    train_features_mean: np.ndarray,
    train_features_flat: np.ndarray,
    seed: int,
    output_dir: Path,
    lag: int,
) -> dict:
    """
    Fit conditioning models (PCA, k-means, bins) on ALL train data.

    Returns dict with fitted models and cluster assignments.
    """
    cond_name = condition_config["name"]
    cond_type = condition_config["type"]

    result = {"name": cond_name, "type": cond_type}

    if cond_type == "mean_pool_vq":
        k = condition_config["k"]
        cluster_model = fit_kmeans(train_features_mean, k, seed=seed)
        cluster_model.save(output_dir / f"{cond_name}_lag{lag}_kmeans.pkl")

        result["cluster_model"] = cluster_model
        result["n_clusters"] = k
        result["feature_mode"] = "mean"

    elif cond_type == "pca_vq":
        pca_dim = condition_config["pca_dim"]
        k = condition_config["k"]

        pca_model = fit_pca(train_features_flat, n_components=pca_dim, seed=seed)
        pca_model.save(output_dir / f"{cond_name}_lag{lag}_pca.pkl")

        train_pca = project_pca(train_features_flat, pca_model)
        cluster_model = fit_kmeans(train_pca, k, seed=seed)
        cluster_model.save(output_dir / f"{cond_name}_lag{lag}_kmeans.pkl")

        result["pca_model"] = pca_model
        result["cluster_model"] = cluster_model
        result["n_clusters"] = k
        result["feature_mode"] = "flat"

    elif cond_type == "pca_quartile":
        pca_dim = condition_config["pca_dim"]

        pca_model = fit_pca(train_features_flat, n_components=pca_dim, seed=seed)
        pca_model.save(output_dir / f"{cond_name}_lag{lag}_pca.pkl")

        train_pca = project_pca(train_features_flat, pca_model)
        bin_model = fit_quartile_bins(train_pca)
        bin_model.save(output_dir / f"{cond_name}_lag{lag}_bins.pkl")

        result["pca_model"] = pca_model
        result["bin_model"] = bin_model
        result["n_clusters"] = 4 ** pca_dim
        result["feature_mode"] = "flat"

    else:
        raise ValueError(f"Unknown condition type: {cond_type}")

    return result


def assign_condition_clusters(
    condition_models: dict,
    features_mean: np.ndarray,
    features_flat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign cluster IDs using fitted models.

    Returns (cluster_ids, distances).
    """
    cond_type = condition_models["type"]

    if cond_type == "mean_pool_vq":
        return assign_clusters(features_mean, condition_models["cluster_model"])

    elif cond_type == "pca_vq":
        pca_features = project_pca(features_flat, condition_models["pca_model"])
        return assign_clusters(pca_features, condition_models["cluster_model"])

    elif cond_type == "pca_quartile":
        pca_features = project_pca(features_flat, condition_models["pca_model"])
        cluster_ids = assign_quartile_bins(pca_features, condition_models["bin_model"])
        distances = np.zeros(len(cluster_ids), dtype=np.float32)  # No distance for bins
        return cluster_ids, distances

    else:
        raise ValueError(f"Unknown condition type: {cond_type}")


def compute_slice_metrics(
    deltas_norm: np.ndarray,
    cluster_ids: np.ndarray,
    speaker_ids: np.ndarray,
    effective_clusters: np.ndarray,
    slice_mask: np.ndarray,
) -> dict:
    """
    Compute metrics on a slice of the data.

    Args:
        deltas_norm: Normalized deltas [N, D]
        cluster_ids: Cluster assignments [N]
        speaker_ids: Speaker IDs [N]
        effective_clusters: Array of effective cluster IDs
        slice_mask: Boolean mask for slice [N]

    Returns:
        Dict with metrics
    """
    # Apply slice mask
    deltas_slice = deltas_norm[slice_mask]
    clusters_slice = cluster_ids[slice_mask]
    speakers_slice = speaker_ids[slice_mask]

    # Compute variance ratio
    vr = compute_variance_ratio(deltas_slice, clusters_slice, effective_clusters)

    # Compute entropy reduction
    er = compute_entropy_reduction(deltas_slice, clusters_slice, effective_clusters)

    # Compute per-speaker metrics
    speaker_metrics = compute_speaker_level_metrics(
        deltas_slice, clusters_slice, speakers_slice, effective_clusters
    )
    speaker_agg = aggregate_speaker_metrics(speaker_metrics)

    # Compute excluded mass for this slice
    in_effective = np.isin(clusters_slice, effective_clusters)
    excluded_mass = 1.0 - (in_effective.sum() / len(clusters_slice)) if len(clusters_slice) > 0 else 0.0

    return {
        "variance_ratio": vr["variance_ratio"],
        "entropy_reduction": er["entropy_reduction"],
        "n_samples": vr["n_samples"],
        "n_samples_total": len(deltas_slice),
        "excluded_mass": excluded_mass,
        "speaker_mean": speaker_agg["variance_ratio_mean"],
        "speaker_std": speaker_agg["variance_ratio_std"],
        "speaker_ci_lower": speaker_agg["variance_ratio_ci_lower"],
        "speaker_ci_upper": speaker_agg["variance_ratio_ci_upper"],
        "n_speakers": speaker_agg["n_speakers"],
    }


def run_full_analysis(
    config_path: str | Path,
    output_dir: Optional[str | Path] = None,
    max_samples: Optional[int] = None,
) -> dict:
    """
    Run the complete Phase 0 analysis.

    CRITICAL FLOW:
    1. For each lag, fit conditioning models ONCE on ALL train frames
    2. Assign cluster IDs to ALL train and eval frames
    3. Define effective clusters from train
    4. For each slice, compute metrics by FILTERING (not refitting)

    Args:
        config_path: Path to phase0.yaml config
        output_dir: Optional output directory override

    Returns:
        Dict with all metrics
    """
    logger = get_logger()
    config = load_config(config_path)

    if output_dir is None:
        output_dir = Path(config["output"]["conditioning_dir"])
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seed
    set_seed(config["seed"])

    # Load data
    logger.info("Loading data...")
    latent_store = LatentStore(config["output"]["latents_dir"])
    frames_index_path = config["output"]["frames_index"]
    frame_columns = ["utterance_id", "speaker_id", "t", "pos_frac", "is_high_energy", "split"]

    # Prefer parquet predicate pushdown to avoid loading both splits into memory at once.
    try:
        train_frames = load_frames_index(
            frames_index_path,
            columns=frame_columns,
            filters=[[("split", "==", "train")]],
        )
        eval_frames = load_frames_index(
            frames_index_path,
            columns=frame_columns,
            filters=[[("split", "==", "eval")]],
        )
    except Exception:
        frames = load_frames_index(frames_index_path, columns=frame_columns)
        train_frames = frames[frames["split"] == "train"]
        eval_frames = frames[frames["split"] == "eval"]
        del frames

    # Split is constant per DataFrame from here on; drop it to save memory.
    if "split" in train_frames.columns:
        train_frames = train_frames.drop(columns=["split"])
    if "split" in eval_frames.columns:
        eval_frames = eval_frames.drop(columns=["split"])

    train_frames = _optimize_frames_df(train_frames)
    eval_frames = _optimize_frames_df(eval_frames)

    logger.info(f"Train frames: {len(train_frames)}, Eval frames: {len(eval_frames)}")

    # Get config parameters
    window_size = config["context"]["window_size"]
    lags = config["context"]["lags"]
    conditions = config["clustering"]["conditions"]
    min_cluster_size = config["clustering"]["min_cluster_size"]
    slices = [s["name"] for s in config["slices"]]

    all_results = []

    for lag in lags:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing lag={lag}")
        logger.info("=" * 50)

        # Step 1: Collect ALL features and deltas for this lag (single pass)
        logger.info("Collecting train features...")
        train_data = collect_all_features_and_deltas(
            train_frames, latent_store, window_size, lag, max_samples=max_samples
        )
        logger.info(f"Train samples: {len(train_data['deltas'])}")

        logger.info("Collecting eval features...")
        eval_data = collect_all_features_and_deltas(
            eval_frames, latent_store, window_size, lag, max_samples=max_samples
        )
        logger.info(f"Eval samples: {len(eval_data['deltas'])}")

        # Step 2: Compute normalization stats on ALL train data
        norm_stats = compute_normalization_stats(train_data["deltas"])
        norm_stats.save(output_dir / f"norm_stats_lag{lag}.json")

        # Normalize deltas
        train_deltas_norm = normalize_delta(train_data["deltas"], norm_stats)
        eval_deltas_norm = normalize_delta(eval_data["deltas"], norm_stats)

        # Step 3: For each condition, fit models ONCE on ALL train data
        for cond_cfg in conditions:
            cond_name = cond_cfg["name"]
            logger.info(f"\n--- Fitting {cond_name} ---")

            # Fit models on ALL train data
            condition_models = fit_condition_models(
                cond_cfg,
                train_data["features_mean"],
                train_data["features_flat"],
                config["seed"],
                output_dir,
                lag,
            )

            # Assign clusters to ALL train and eval data
            train_cluster_ids, train_distances = assign_condition_clusters(
                condition_models,
                train_data["features_mean"],
                train_data["features_flat"],
            )
            eval_cluster_ids, eval_distances = assign_condition_clusters(
                condition_models,
                eval_data["features_mean"],
                eval_data["features_flat"],
            )

            # Define effective clusters from ALL train data
            n_clusters = condition_models["n_clusters"]
            effective_clusters = get_effective_clusters(train_cluster_ids, n_clusters, min_cluster_size)

            # Compute cluster stats for reporting
            cluster_stats = compute_cluster_stats(train_cluster_ids, n_clusters, min_cluster_size)
            logger.info(f"Effective clusters: {len(effective_clusters)}/{n_clusters}")
            logger.info(f"Excluded mass (train): {cluster_stats['excluded_mass']:.1%}")

            # Save cluster stats (convert numpy types to native Python for JSON)
            cluster_stats_json = {
                k: (v.item() if hasattr(v, 'item') else v) for k, v in cluster_stats.items()
            }
            with open(output_dir / f"{cond_name}_lag{lag}_cluster_stats.json", "w") as f:
                json.dump(cluster_stats_json, f, indent=2)

            # Random baseline (permute on ALL train data)
            random_train_ids = permute_cluster_ids(train_cluster_ids, seed=config["seed"] + 1000)
            random_vr = compute_variance_ratio(train_deltas_norm, random_train_ids, effective_clusters)
            random_eval_ids = permute_cluster_ids(eval_cluster_ids, seed=config["seed"] + 2000)
            random_eval_vr = compute_variance_ratio(eval_deltas_norm, random_eval_ids, effective_clusters)

            # Step 4: For each slice, compute metrics by FILTERING (not refitting)
            for slice_name in slices:
                logger.info(f"  Slice: {slice_name}")

                # Get slice masks
                train_slice_mask = get_slice_mask(
                    slice_name, train_data["is_high_energy"], train_data["pos_frac"]
                )
                eval_slice_mask = get_slice_mask(
                    slice_name, eval_data["is_high_energy"], eval_data["pos_frac"]
                )

                if train_slice_mask.sum() == 0 or eval_slice_mask.sum() == 0:
                    logger.warning(f"    Empty slice, skipping")
                    continue

                # Compute metrics on filtered data
                train_metrics = compute_slice_metrics(
                    train_deltas_norm,
                    train_cluster_ids,
                    train_data["speaker_ids"],
                    effective_clusters,
                    train_slice_mask,
                )

                eval_metrics = compute_slice_metrics(
                    eval_deltas_norm,
                    eval_cluster_ids,
                    eval_data["speaker_ids"],
                    effective_clusters,
                    eval_slice_mask,
                )

                # Cross-speaker degradation
                degradation = eval_metrics["variance_ratio"] - train_metrics["variance_ratio"]
                rel_degradation = degradation / train_metrics["variance_ratio"] if train_metrics["variance_ratio"] > 0 else 0.0

                result = {
                    "condition": cond_name,
                    "type": cond_cfg["type"],
                    "lag": lag,
                    "slice": slice_name,
                    "n_effective_clusters": len(effective_clusters),
                    "train": train_metrics,
                    "eval": eval_metrics,
                    "cross_speaker_degradation": rel_degradation,
                    "cross_speaker_degradation_absolute": degradation,
                    "random_baseline_variance_ratio": random_vr["variance_ratio"],
                    "random_baseline_eval_variance_ratio": random_eval_vr["variance_ratio"],
                    "cluster_stats": {
                        "excluded_mass_train": cluster_stats["excluded_mass"],
                        "min_cluster_size": cluster_stats["min_cluster_size"],
                        "max_cluster_size": cluster_stats["max_cluster_size"],
                    },
                }

                all_results.append(result)

                logger.info(f"    Train VR: {train_metrics['variance_ratio']:.3f}, "
                           f"Eval VR: {eval_metrics['variance_ratio']:.3f}, "
                           f"Random: {random_vr['variance_ratio']:.3f}")

    # Save results
    metrics_path = Path(config["output"]["metrics_file"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save as table
    rows = []
    for r in all_results:
        rows.append({
            "condition": r["condition"],
            "lag": r["lag"],
            "slice": r["slice"],
            "train_variance_ratio": r["train"]["variance_ratio"],
            "eval_variance_ratio": r["eval"]["variance_ratio"],
            "train_entropy_reduction": r["train"]["entropy_reduction"],
            "eval_entropy_reduction": r["eval"]["entropy_reduction"],
            "cross_speaker_degradation": r["cross_speaker_degradation"],
            "random_baseline": r["random_baseline_variance_ratio"],
            "random_baseline_eval": r["random_baseline_eval_variance_ratio"],
            "n_effective_clusters": r["n_effective_clusters"],
            "train_n_samples": r["train"]["n_samples"],
            "eval_n_samples": r["eval"]["n_samples"],
            "train_excluded_mass": r["train"]["excluded_mass"],
            "eval_excluded_mass": r["eval"]["excluded_mass"],
            "train_speaker_ci_lower": r["train"]["speaker_ci_lower"],
            "train_speaker_ci_upper": r["train"]["speaker_ci_upper"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(config["output"]["tables_file"], index=False)

    logger.info(f"\nResults saved to {metrics_path}")
    return {"results": all_results, "table": df}
