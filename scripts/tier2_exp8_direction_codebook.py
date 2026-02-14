#!/usr/bin/env python3
"""
Stage 3 - Experiment 8: Direction Codebook Feasibility (Mimi 512D latents).

Tests whether frame-to-frame direction vectors from Mimi 512D latents
cluster into a practical codebook. Three-way comparison:
  A) Normalized (spherical k-means on directions Δz/||Δz||)
  B) Unnormalized (standard k-means on raw deltas Δz)
  C) State VQ (standard k-means on raw latent vectors z_t)

Protocol:
  1. Extract directions (dz/||dz||), magnitudes, and raw vectors from Mimi latents
  2. Exp 9: Characterize near-zero magnitude distribution
  3. Three k-means variants: normalized, unnormalized, state VQ
  4. Reconstruct trajectories with all three methods
  5. Decode through Mimi decoder -> 48kHz WAV
  6. Compute mel distance, L1, trajectory divergence — compare A/B/C

Usage:
  uv run python scripts/tier2_exp8_direction_codebook.py \
      --config configs/tier2_exp8_direction_codebook.yaml

  # Single K for quick test:
  uv run python scripts/tier2_exp8_direction_codebook.py \
      --config configs/tier2_exp8_direction_codebook.yaml --k 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import yaml
from scipy import stats as scipy_stats
from sklearn.cluster import MiniBatchKMeans

from experiment import register_run, finalize_run
from phase0.data.io import LatentStore, load_latents_index
from phase0.data.splits import load_splits
from phase0.utils.logging import setup_logging
from phase0.utils.seed import set_seed
from phase1.data import sample_eval_utterances
from phase1.train_eval import _device_from_config


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Step 1: Extract directions and magnitudes
# ---------------------------------------------------------------------------


def extract_directions_and_magnitudes(
    store: LatentStore,
    utterance_ids: list[str],
    logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Extract direction unit vectors, magnitudes, raw deltas, and raw vectors from frame-to-frame changes.

    Returns:
        all_directions: [N, D] unit vectors (from deltas, N = total frames - n_utterances)
        all_magnitudes: [N] scalar magnitudes
        all_deltas: [N, D] raw unnormalized deltas
        all_vectors: [M, D] raw latent vectors (all frames, M > N)
        per_utt_magnitudes: list of per-utterance magnitude arrays
    """
    all_dirs = []
    all_mags = []
    all_dx = []
    all_vecs = []
    per_utt_mags = []

    for i, utt_id in enumerate(utterance_ids):
        if utt_id not in store:
            continue
        x = store.get_latents(utt_id).astype(np.float32, copy=False)  # [T, D]
        if x.shape[0] < 2:
            continue

        all_vecs.append(x)

        dx = x[1:] - x[:-1]  # [T-1, D]
        mags = np.linalg.norm(dx, axis=1)  # [T-1]
        # Directions: normalize, using eps to avoid div-by-zero
        dirs = dx / np.maximum(mags[:, None], 1e-8)  # [T-1, D]

        all_dirs.append(dirs)
        all_mags.append(mags)
        all_dx.append(dx)
        per_utt_mags.append(mags)

        if (i + 1) % 200 == 0:
            logger.info(f"[exp8] Extracted {i+1}/{len(utterance_ids)} utterances")

    all_directions = np.concatenate(all_dirs, axis=0)
    all_magnitudes = np.concatenate(all_mags, axis=0)
    all_deltas = np.concatenate(all_dx, axis=0)
    all_vectors = np.concatenate(all_vecs, axis=0)
    logger.info(
        f"[exp8] Extracted {len(all_magnitudes)} direction vectors, "
        f"{len(all_vectors)} raw vectors "
        f"from {len(per_utt_mags)} utterances, D={all_directions.shape[1]}"
    )
    return all_directions, all_magnitudes, all_deltas, all_vectors, per_utt_mags


# ---------------------------------------------------------------------------
# Step 2: Exp 9 — Near-zero magnitude analysis
# ---------------------------------------------------------------------------


def analyze_near_zero_magnitudes(
    all_magnitudes: np.ndarray,
    per_utt_magnitudes: list[np.ndarray],
    epsilon_fractions: list[float],
) -> dict:
    """Characterize near-zero magnitude distribution and temporal clustering."""
    median_mag = float(np.median(all_magnitudes))
    mean_mag = float(np.mean(all_magnitudes))
    std_mag = float(np.std(all_magnitudes))

    # Magnitude distribution stats
    log_mags = np.log(np.maximum(all_magnitudes, 1e-12))
    mag_dist = {
        "n_total": int(len(all_magnitudes)),
        "mean": mean_mag,
        "median": median_mag,
        "std": std_mag,
        "skew": float(scipy_stats.skew(all_magnitudes)),
        "log_mean": float(np.mean(log_mags)),
        "log_std": float(np.std(log_mags)),
        "percentiles": {
            "p5": float(np.percentile(all_magnitudes, 5)),
            "p25": float(np.percentile(all_magnitudes, 25)),
            "p50": float(np.percentile(all_magnitudes, 50)),
            "p75": float(np.percentile(all_magnitudes, 75)),
            "p95": float(np.percentile(all_magnitudes, 95)),
        },
    }

    # Per-epsilon analysis
    epsilon_results = {}
    for eps_frac in epsilon_fractions:
        threshold = eps_frac * median_mag
        is_near_zero = all_magnitudes < threshold
        frac = float(np.mean(is_near_zero))

        # Temporal clustering: compute run-lengths of consecutive near-zero frames
        run_lengths = []
        for utt_mags in per_utt_magnitudes:
            utt_nz = utt_mags < threshold
            # Find runs of consecutive True values
            if not np.any(utt_nz):
                continue
            changes = np.diff(utt_nz.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1
            # Handle edge cases
            if utt_nz[0]:
                starts = np.concatenate([[0], starts])
            if utt_nz[-1]:
                ends = np.concatenate([ends, [len(utt_nz)]])
            for s, e in zip(starts, ends):
                run_lengths.append(int(e - s))

        temporal = {}
        if run_lengths:
            temporal = {
                "n_runs": len(run_lengths),
                "mean_run_length": float(np.mean(run_lengths)),
                "median_run_length": float(np.median(run_lengths)),
                "max_run_length": int(np.max(run_lengths)),
                "p90_run_length": float(np.percentile(run_lengths, 90)),
            }

        epsilon_results[str(eps_frac)] = {
            "threshold": float(threshold),
            "fraction": frac,
            "n_near_zero": int(np.sum(is_near_zero)),
            "temporal_clustering": temporal,
        }

    # Decision recommendation
    eps_001_frac = epsilon_results.get("0.01", {}).get("fraction", 0.0)
    if eps_001_frac > 0.10:
        recommendation = "no_change_token_needed"
    elif eps_001_frac < 0.02:
        recommendation = "floor_clamp_sufficient"
    else:
        recommendation = "borderline_consider_no_change_token"

    return {
        "magnitude_distribution": mag_dist,
        "epsilon_analysis": epsilon_results,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Step 3: K-means (spherical and standard)
# ---------------------------------------------------------------------------


def run_spherical_kmeans(
    directions: np.ndarray,
    k: int,
    n_init: int = 1,
    max_iter: int = 300,
    batch_size: int = 4096,
    seed: int = 42,
    logger=None,
) -> dict:
    """Fit spherical k-means (MiniBatchKMeans) and compute quality metrics."""
    if logger:
        logger.info(f"[exp8] Running spherical MiniBatchKMeans K={k} (n_init={n_init}, batch_size={batch_size})...")

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        batch_size=batch_size,
        random_state=seed,
    )
    kmeans.fit(directions)
    labels = kmeans.labels_

    # Renormalize centroids to unit sphere
    centroids = kmeans.cluster_centers_
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(norms, 1e-8)

    # Angular quantization error
    cos_sim = np.sum(directions * centroids[labels], axis=1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    angular_errors = np.degrees(np.arccos(cos_sim))

    # Codebook utilization
    counts = np.bincount(labels, minlength=k)
    n_total = len(labels)
    threshold_count = 0.001 * n_total  # 0.1% of total
    utilized = int(np.sum(counts >= threshold_count))

    # Entropy
    probs = counts / n_total
    probs_nonzero = probs[probs > 0]
    entropy = float(-np.sum(probs_nonzero * np.log(probs_nonzero)))
    max_entropy = float(np.log(k))

    if logger:
        logger.info(
            f"[exp8] K={k}: angular_err={np.mean(angular_errors):.1f}° (median {np.median(angular_errors):.1f}°), "
            f"utilization={utilized}/{k} ({100*utilized/k:.0f}%), "
            f"entropy_ratio={entropy/max_entropy:.3f}"
        )

    return {
        "centroids": centroids,
        "labels": labels,
        "k": k,
        "angular_error_mean_deg": float(np.mean(angular_errors)),
        "angular_error_median_deg": float(np.median(angular_errors)),
        "angular_error_p95_deg": float(np.percentile(angular_errors, 95)),
        "codebook_utilization": utilized,
        "codebook_utilization_frac": float(utilized / k),
        "entropy": entropy,
        "max_entropy": max_entropy,
        "entropy_ratio": float(entropy / max_entropy) if max_entropy > 0 else 0.0,
        "cluster_sizes_min": int(np.min(counts)),
        "cluster_sizes_max": int(np.max(counts)),
        "cluster_sizes_mean": float(np.mean(counts)),
    }


def run_unnormalized_kmeans(
    deltas: np.ndarray,
    directions: np.ndarray,
    k: int,
    n_init: int = 1,
    max_iter: int = 300,
    batch_size: int = 4096,
    seed: int = 42,
    logger=None,
) -> dict:
    """Fit standard MiniBatchKMeans on raw (unnormalized) deltas.

    Centroids implicitly encode both direction and magnitude.
    We normalize them post-hoc to measure angular quantization error.
    """
    if logger:
        logger.info(f"[exp8] Running unnormalized MiniBatchKMeans K={k} (n_init={n_init}, batch_size={batch_size})...")

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        batch_size=batch_size,
        random_state=seed,
    )
    kmeans.fit(deltas)
    labels = kmeans.labels_

    # Normalize centroids to get implied directions
    centroids_raw = kmeans.cluster_centers_
    centroid_norms = np.linalg.norm(centroids_raw, axis=1, keepdims=True)
    centroids_normalized = centroids_raw / np.maximum(centroid_norms, 1e-8)

    # Angular error: compare true direction of each delta to the direction
    # implied by its assigned (unnormalized) centroid
    cos_sim = np.sum(directions * centroids_normalized[labels], axis=1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    angular_errors = np.degrees(np.arccos(cos_sim))

    # Codebook utilization
    counts = np.bincount(labels, minlength=k)
    n_total = len(labels)
    threshold_count = 0.001 * n_total
    utilized = int(np.sum(counts >= threshold_count))

    # Entropy
    probs = counts / n_total
    probs_nonzero = probs[probs > 0]
    entropy = float(-np.sum(probs_nonzero * np.log(probs_nonzero)))
    max_entropy = float(np.log(k))

    # Centroid magnitude spread
    centroid_mags = centroid_norms.squeeze()

    if logger:
        logger.info(
            f"[exp8] Unnorm K={k}: angular_err={np.mean(angular_errors):.1f}° "
            f"(median {np.median(angular_errors):.1f}°), "
            f"utilization={utilized}/{k} ({100*utilized/k:.0f}%), "
            f"centroid_mag range=[{centroid_mags.min():.3f}, {centroid_mags.max():.3f}]"
        )

    return {
        "centroids_raw": centroids_raw,
        "centroids_normalized": centroids_normalized,
        "labels": labels,
        "k": k,
        "angular_error_mean_deg": float(np.mean(angular_errors)),
        "angular_error_median_deg": float(np.median(angular_errors)),
        "angular_error_p95_deg": float(np.percentile(angular_errors, 95)),
        "codebook_utilization": utilized,
        "codebook_utilization_frac": float(utilized / k),
        "entropy": entropy,
        "max_entropy": max_entropy,
        "entropy_ratio": float(entropy / max_entropy) if max_entropy > 0 else 0.0,
        "centroid_magnitude_mean": float(np.mean(centroid_mags)),
        "centroid_magnitude_std": float(np.std(centroid_mags)),
        "centroid_magnitude_min": float(np.min(centroid_mags)),
        "centroid_magnitude_max": float(np.max(centroid_mags)),
    }


def run_state_kmeans(
    vectors: np.ndarray,
    k: int,
    n_init: int = 1,
    max_iter: int = 300,
    batch_size: int = 4096,
    seed: int = 42,
    logger=None,
) -> dict:
    """Fit standard MiniBatchKMeans on raw latent vectors z_t (state-level VQ).

    Each centroid represents a prototypical latent state. This serves as a
    baseline: VQ on the vectors themselves rather than on their deltas.
    """
    if logger:
        logger.info(f"[exp8] Running state VQ MiniBatchKMeans K={k} (n_init={n_init}, batch_size={batch_size})...")

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        batch_size=batch_size,
        random_state=seed,
    )
    kmeans.fit(vectors)
    labels = kmeans.labels_

    centroids = kmeans.cluster_centers_  # [K, D]

    # Reconstruction error: Euclidean distance between each vector and its centroid
    recon_errors = np.linalg.norm(vectors - centroids[labels], axis=1)

    # Codebook utilization
    counts = np.bincount(labels, minlength=k)
    n_total = len(labels)
    threshold_count = 0.001 * n_total
    utilized = int(np.sum(counts >= threshold_count))

    # Entropy
    probs = counts / n_total
    probs_nonzero = probs[probs > 0]
    entropy = float(-np.sum(probs_nonzero * np.log(probs_nonzero)))
    max_entropy = float(np.log(k))

    if logger:
        logger.info(
            f"[exp8] State VQ K={k}: recon_err={np.mean(recon_errors):.4f} "
            f"(median {np.median(recon_errors):.4f}), "
            f"utilization={utilized}/{k} ({100*utilized/k:.0f}%), "
            f"entropy_ratio={entropy/max_entropy:.3f}"
        )

    return {
        "centroids": centroids,
        "labels": labels,
        "k": k,
        "recon_error_mean": float(np.mean(recon_errors)),
        "recon_error_median": float(np.median(recon_errors)),
        "recon_error_p95": float(np.percentile(recon_errors, 95)),
        "codebook_utilization": utilized,
        "codebook_utilization_frac": float(utilized / k),
        "entropy": entropy,
        "max_entropy": max_entropy,
        "entropy_ratio": float(entropy / max_entropy) if max_entropy > 0 else 0.0,
        "cluster_sizes_min": int(np.min(counts)),
        "cluster_sizes_max": int(np.max(counts)),
        "cluster_sizes_mean": float(np.mean(counts)),
    }


# ---------------------------------------------------------------------------
# Step 5: Trajectory reconstruction
# ---------------------------------------------------------------------------


def reconstruct_trajectory_normalized(
    x_true: np.ndarray,
    centroids: np.ndarray,
    near_zero_threshold: float,
) -> tuple[np.ndarray, dict]:
    """Reconstruct trajectory with quantized directions + GT magnitudes.

    Spherical k-means centroids are unit vectors. Reconstruction uses
    GT magnitude with nearest codebook direction.
    """
    T, D = x_true.shape
    x_q = np.zeros_like(x_true)
    x_q[0] = x_true[0]

    angular_errors = []
    n_near_zero = 0
    codebook_indices = []

    for t in range(1, T):
        dx_true = x_true[t] - x_true[t - 1]
        mag = np.linalg.norm(dx_true)

        if mag < near_zero_threshold:
            x_q[t] = x_q[t - 1]
            n_near_zero += 1
            codebook_indices.append(-1)
            continue

        direction = dx_true / mag
        cos_sims = centroids @ direction  # [K]
        k_star = int(np.argmax(cos_sims))
        codebook_indices.append(k_star)

        cos_val = float(np.clip(cos_sims[k_star], -1.0, 1.0))
        angular_errors.append(float(np.degrees(np.arccos(cos_val))))

        # Reconstruct with GT magnitude + quantized direction
        x_q[t] = x_q[t - 1] + mag * centroids[k_star]

    # Trajectory divergence
    divergence = np.linalg.norm(x_q - x_true, axis=1)  # [T]
    if T > 2:
        t_axis = np.arange(T, dtype=np.float64)
        coeffs = np.polyfit(t_axis, divergence, 1)
        growth_rate = float(coeffs[0])
    else:
        growth_rate = 0.0

    recon_stats = {
        "n_frames": T,
        "n_near_zero": n_near_zero,
        "angular_error_mean_deg": float(np.mean(angular_errors)) if angular_errors else 0.0,
        "angular_error_median_deg": float(np.median(angular_errors)) if angular_errors else 0.0,
        "trajectory_divergence_mean": float(np.mean(divergence)),
        "trajectory_divergence_final": float(divergence[-1]) if T > 0 else 0.0,
        "trajectory_divergence_growth_rate": growth_rate,
    }
    return x_q, recon_stats


def reconstruct_trajectory_unnormalized(
    x_true: np.ndarray,
    centroids_raw: np.ndarray,
    near_zero_threshold: float,
) -> tuple[np.ndarray, dict]:
    """Reconstruct trajectory using unnormalized centroids as full delta replacements.

    Each raw centroid encodes both direction and magnitude jointly.
    The nearest centroid (by Euclidean distance) replaces the true delta entirely.
    """
    T, D = x_true.shape
    x_q = np.zeros_like(x_true)
    x_q[0] = x_true[0]

    angular_errors = []
    magnitude_errors = []
    n_near_zero = 0
    codebook_indices = []

    for t in range(1, T):
        dx_true = x_true[t] - x_true[t - 1]
        mag = np.linalg.norm(dx_true)

        if mag < near_zero_threshold:
            x_q[t] = x_q[t - 1]
            n_near_zero += 1
            codebook_indices.append(-1)
            continue

        # Find nearest centroid by Euclidean distance
        dists = np.linalg.norm(centroids_raw - dx_true[None, :], axis=1)  # [K]
        k_star = int(np.argmin(dists))
        codebook_indices.append(k_star)

        # Angular error between true direction and centroid direction
        direction = dx_true / mag
        centroid_norm = np.linalg.norm(centroids_raw[k_star])
        if centroid_norm > 1e-8:
            centroid_dir = centroids_raw[k_star] / centroid_norm
            cos_val = float(np.clip(np.dot(direction, centroid_dir), -1.0, 1.0))
            angular_errors.append(float(np.degrees(np.arccos(cos_val))))
            magnitude_errors.append(abs(mag - centroid_norm))

        # Reconstruct: centroid replaces entire delta
        x_q[t] = x_q[t - 1] + centroids_raw[k_star]

    # Trajectory divergence
    divergence = np.linalg.norm(x_q - x_true, axis=1)  # [T]
    if T > 2:
        t_axis = np.arange(T, dtype=np.float64)
        coeffs = np.polyfit(t_axis, divergence, 1)
        growth_rate = float(coeffs[0])
    else:
        growth_rate = 0.0

    recon_stats = {
        "n_frames": T,
        "n_near_zero": n_near_zero,
        "angular_error_mean_deg": float(np.mean(angular_errors)) if angular_errors else 0.0,
        "angular_error_median_deg": float(np.median(angular_errors)) if angular_errors else 0.0,
        "magnitude_error_mean": float(np.mean(magnitude_errors)) if magnitude_errors else 0.0,
        "trajectory_divergence_mean": float(np.mean(divergence)),
        "trajectory_divergence_final": float(divergence[-1]) if T > 0 else 0.0,
        "trajectory_divergence_growth_rate": growth_rate,
    }
    return x_q, recon_stats


def reconstruct_trajectory_state_vq(
    x_true: np.ndarray,
    centroids: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Reconstruct trajectory by snapping each frame to its nearest state centroid.

    No accumulation: each frame is independently quantized to the nearest
    codebook entry. This is the simplest baseline — pure VQ on states.
    """
    T, D = x_true.shape

    # Find nearest centroid for each frame
    # x_true: [T, D], centroids: [K, D]
    # Use batched distance computation
    dists = np.linalg.norm(x_true[:, None, :] - centroids[None, :, :], axis=2)  # [T, K]
    labels = np.argmin(dists, axis=1)  # [T]
    x_q = centroids[labels]  # [T, D]

    # Per-frame reconstruction error
    recon_errors = np.linalg.norm(x_q - x_true, axis=1)  # [T]

    # Trajectory divergence (cumulative deviation)
    divergence = np.linalg.norm(x_q - x_true, axis=1)  # same as recon_errors here
    if T > 2:
        t_axis = np.arange(T, dtype=np.float64)
        coeffs = np.polyfit(t_axis, divergence, 1)
        growth_rate = float(coeffs[0])
    else:
        growth_rate = 0.0

    recon_stats = {
        "n_frames": T,
        "recon_error_mean": float(np.mean(recon_errors)),
        "recon_error_median": float(np.median(recon_errors)),
        "trajectory_divergence_mean": float(np.mean(divergence)),
        "trajectory_divergence_final": float(divergence[-1]) if T > 0 else 0.0,
        "trajectory_divergence_growth_rate": growth_rate,
    }
    return x_q, recon_stats


# ---------------------------------------------------------------------------
# Step 6: Audio decode and metrics
# ---------------------------------------------------------------------------


def decode_mimi_latents_to_audio(
    z: np.ndarray,
    mimi_decoder,
    device: torch.device,
) -> np.ndarray:
    """Decode Mimi latents [T, 512] to audio via Mimi decoder."""
    z_torch = torch.from_numpy(z.T.copy()).unsqueeze(0).float().to(device)  # [1, 512, T]
    with torch.inference_mode():
        audio = mimi_decoder(z_torch)  # [1, 1, T_audio]
    return audio.squeeze().cpu().numpy()


def _mel_distance(audio: torch.Tensor, audio_hat: torch.Tensor, sr: int, n_mels: int = 80) -> float:
    """Compute mel spectrogram L1 distance."""
    mel_transform = T.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=256, n_mels=n_mels,
    ).to(audio.device)

    mel = mel_transform(audio.squeeze())
    mel_hat = mel_transform(audio_hat.squeeze())
    log_mel = torch.log(mel.clamp_min(1e-5))
    log_mel_hat = torch.log(mel_hat.clamp_min(1e-5))
    return float(torch.nn.functional.l1_loss(log_mel_hat, log_mel).item())


def _save_wav(audio: np.ndarray, path: Path, input_sr: int, output_sr: int) -> None:
    """Save audio as WAV, resampling if needed."""
    audio_t = torch.from_numpy(audio).unsqueeze(0).float()  # [1, T]
    if input_sr != output_sr:
        audio_t = torchaudio.functional.resample(audio_t, input_sr, output_sr)
    torchaudio.save(str(path), audio_t, output_sr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 3 Exp8: direction codebook feasibility")
    p.add_argument("--config", type=str, default="configs/tier2_exp8_direction_codebook.yaml")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--k", type=int, default=None, help="Run only this K value (for quick testing)")
    p.add_argument(
        "--vae-dir", type=str, default=None,
        help="Path to β-VAE variant dir (e.g. outputs/tier2/exp5_betavae/RUN/baseline_betavae). "
             "Uses VAE latents + VAE decoder instead of raw Mimi latents.",
    )
    args = p.parse_args()

    if os.environ.get("NO_TORCH_COMPILE"):
        os.environ["TORCH_COMPILE_DISABLE"] = "1"

    logger = setup_logging(name="phase0")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_id = args.run_id or _default_run_id()
    out_root = Path(cfg["output"]["out_dir"])
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    run = register_run(
        experiment="exp8_direction_codebook",
        run_id=run_id,
        config_path=args.config,
        config=cfg,
        cli_args=sys.argv[1:],
        out_dir=out_dir,
        log_name="phase0",
    )

    set_seed(int(cfg.get("seed", 42)))
    device = _device_from_config(cfg["train"]["device"])

    data_cfg = cfg["data"]
    codebook_cfg = cfg["codebook"]
    eval_cfg = cfg["eval"]
    use_vae = args.vae_dir is not None or eval_cfg.get("vae_checkpoint") is not None

    k_values = [args.k] if args.k else [int(k) for k in codebook_cfg["k_values"]]

    # -----------------------------------------------------------------------
    # Step 1: Load latents and extract directions/magnitudes (train split)
    # -----------------------------------------------------------------------
    latents_dir = data_cfg["latents_dir"]
    latents_index_path = data_cfg["latents_index"]
    if args.vae_dir is not None:
        vae_dir = Path(args.vae_dir)
        latents_dir = str(vae_dir / "latents.zarr")
        latents_index_path = str(vae_dir / "latents_index.parquet")
        logger.info(f"[exp8] Using VAE latents from {vae_dir}")
    elif eval_cfg.get("vae_checkpoint") is not None:
        logger.info(f"[exp8] Using VAE latents from config: {latents_dir}")
    else:
        logger.info("[exp8] Using raw Mimi latents")

    logger.info("[exp8] Loading latents index and splits...")
    latents_index = load_latents_index(latents_index_path)
    splits = load_splits(data_cfg["splits_dir"])
    train_speaker_set = set(splits.train_speakers)

    train_utts = latents_index[latents_index["speaker_id"].isin(train_speaker_set)]["utterance_id"].astype(str).tolist()
    logger.info(f"[exp8] {len(train_utts)} train utterances for direction extraction")

    store = LatentStore(latents_dir)
    all_directions, all_magnitudes, all_deltas, all_vectors, per_utt_magnitudes = extract_directions_and_magnitudes(
        store, train_utts, logger,
    )
    logger.info(f"[exp8] Latent dimensionality: D={all_directions.shape[1]}")

    # -----------------------------------------------------------------------
    # Step 2: Exp 9 — Near-zero magnitude analysis
    # -----------------------------------------------------------------------
    logger.info("[exp8] Running Exp 9: near-zero magnitude analysis...")
    epsilon_fractions = [float(e) for e in codebook_cfg["near_zero_epsilons"]]
    exp9_result = analyze_near_zero_magnitudes(
        all_magnitudes, per_utt_magnitudes, epsilon_fractions,
    )
    with open(out_dir / "exp9_near_zero_analysis.json", "w") as f:
        json.dump(exp9_result, f, indent=2)
    logger.info(
        f"[exp8] Exp 9 recommendation: {exp9_result['recommendation']}. "
        f"Magnitude median={exp9_result['magnitude_distribution']['median']:.4f}"
    )
    for eps_key, eps_data in exp9_result["epsilon_analysis"].items():
        logger.info(
            f"[exp8]   eps={eps_key}*median -> threshold={eps_data['threshold']:.6f}, "
            f"fraction={eps_data['fraction']:.4f} ({eps_data['n_near_zero']} frames)"
        )

    with open(out_dir / "magnitude_distribution.json", "w") as f:
        json.dump(exp9_result["magnitude_distribution"], f, indent=2)

    # -----------------------------------------------------------------------
    # Step 3: Filter near-zero directions and run BOTH k-means variants
    # -----------------------------------------------------------------------
    median_mag = float(np.median(all_magnitudes))
    nz_threshold = 0.01 * median_mag
    valid_mask = all_magnitudes >= nz_threshold
    train_directions = all_directions[valid_mask]
    train_deltas = all_deltas[valid_mask]
    logger.info(
        f"[exp8] Filtering near-zero: {np.sum(~valid_mask)} removed "
        f"({100*np.mean(~valid_mask):.1f}%), {len(train_directions)} directions for k-means"
    )

    # --- Normalized (spherical k-means) ---
    norm_codebook_results = {}
    norm_centroids = {}

    for k in k_values:
        result = run_spherical_kmeans(
            train_directions,
            k=k,
            n_init=int(codebook_cfg.get("kmeans_n_init", 1)),
            max_iter=int(codebook_cfg["kmeans_max_iter"]),
            batch_size=int(codebook_cfg.get("kmeans_batch_size", 4096)),
            seed=int(cfg.get("seed", 42)),
            logger=logger,
        )
        centroids = result.pop("centroids")
        result.pop("labels")
        norm_codebook_results[str(k)] = result
        norm_centroids[k] = centroids
        np.save(str(out_dir / f"codebook_normalized_K{k:04d}.npy"), centroids)

    with open(out_dir / "codebook_results_normalized.json", "w") as f:
        json.dump(norm_codebook_results, f, indent=2)

    # --- Unnormalized (standard k-means on raw deltas) ---
    unnorm_codebook_results = {}
    unnorm_centroids_raw = {}

    for k in k_values:
        result = run_unnormalized_kmeans(
            train_deltas,
            train_directions,
            k=k,
            n_init=int(codebook_cfg.get("kmeans_n_init", 1)),
            max_iter=int(codebook_cfg["kmeans_max_iter"]),
            batch_size=int(codebook_cfg.get("kmeans_batch_size", 4096)),
            seed=int(cfg.get("seed", 42)),
            logger=logger,
        )
        raw = result.pop("centroids_raw")
        result.pop("centroids_normalized")
        result.pop("labels")
        unnorm_codebook_results[str(k)] = result
        unnorm_centroids_raw[k] = raw
        np.save(str(out_dir / f"codebook_unnormalized_K{k:04d}.npy"), raw)

    with open(out_dir / "codebook_results_unnormalized.json", "w") as f:
        json.dump(unnorm_codebook_results, f, indent=2)

    # --- State VQ (k-means on raw latent vectors z_t) ---
    state_codebook_results = {}
    state_centroids = {}

    for k in k_values:
        result = run_state_kmeans(
            all_vectors,
            k=k,
            n_init=int(codebook_cfg.get("kmeans_n_init", 1)),
            max_iter=int(codebook_cfg["kmeans_max_iter"]),
            batch_size=int(codebook_cfg.get("kmeans_batch_size", 4096)),
            seed=int(cfg.get("seed", 42)),
            logger=logger,
        )
        centroids = result.pop("centroids")
        result.pop("labels")
        state_codebook_results[str(k)] = result
        state_centroids[k] = centroids
        np.save(str(out_dir / f"codebook_state_vq_K{k:04d}.npy"), centroids)

    with open(out_dir / "codebook_results_state_vq.json", "w") as f:
        json.dump(state_codebook_results, f, indent=2)

    # Log comparison
    logger.info("[exp8] === Normalized vs Unnormalized vs State VQ comparison ===")
    for k in k_values:
        norm = norm_codebook_results[str(k)]
        unnorm = unnorm_codebook_results[str(k)]
        state = state_codebook_results[str(k)]
        logger.info(
            f"[exp8]   K={k}: norm_ang={norm['angular_error_mean_deg']:.1f}° "
            f"unnorm_ang={unnorm['angular_error_mean_deg']:.1f}° "
            f"state_recon_err={state['recon_error_mean']:.4f} "
            f"(util: {norm['codebook_utilization_frac']:.0%} / "
            f"{unnorm['codebook_utilization_frac']:.0%} / "
            f"{state['codebook_utilization_frac']:.0%})"
        )

    # -----------------------------------------------------------------------
    # Step 4: Load decoder for audio decode
    # -----------------------------------------------------------------------
    from mimi_autoencoder import load_mimi_autoencoder
    autoencoder = None

    if use_vae:
        # Two supported VAE checkpoint formats:
        # - Exp5 ARFriendlyVAE checkpoints (vae_final.pt)
        # - Exp9 PocketMimiVAE-GAN checkpoints (vae_gan_final.pt)
        if args.vae_dir is not None:
            vae_ckpt_path = Path(args.vae_dir) / "checkpoints" / "vae_final.pt"
            if not vae_ckpt_path.exists():
                ckpts = sorted((Path(args.vae_dir) / "checkpoints").glob("vae_step*.pt"))
                vae_ckpt_path = ckpts[-1] if ckpts else vae_ckpt_path
        else:
            vae_ckpt_path = Path(eval_cfg["vae_checkpoint"])

        ckpt_data = torch.load(str(vae_ckpt_path), map_location=device)

        if "mimi_full" in ckpt_data:
            logger.info("[exp8] Loading Exp9 PocketMimiVAE decoder (dec_proj + Mimi decoder) ...")
            from stage2.pocket_mimi_vae import build_pocket_mimi_vae

            latent_dim = int(ckpt_data.get("latent_dim", 32))
            dec_hidden_dim = int(ckpt_data.get("dec_hidden_dim", 256))

            vae = build_pocket_mimi_vae(
                latent_dim=latent_dim,
                dec_hidden_dim=dec_hidden_dim,
                freeze_encoder=False,
                freeze_decoder=False,
                device=str(device),
                load_pretrained_mimi=False,
            )

            vae.mimi.load_state_dict(ckpt_data["mimi_full"], strict=True)
            vae.mu_proj.load_state_dict(ckpt_data["vae_bottleneck"]["mu_proj"], strict=True)
            vae.logvar_proj.load_state_dict(ckpt_data["vae_bottleneck"]["logvar_proj"], strict=True)
            vae.dec_proj.load_state_dict(ckpt_data["vae_bottleneck"]["dec_proj"], strict=True)
            vae.eval()
            logger.info(f"[exp8] Loaded PocketMimiVAE checkpoint from {vae_ckpt_path} (latent_dim={vae.latent_dim})")

            def _decode_to_audio(z: np.ndarray) -> np.ndarray:
                z_torch = torch.from_numpy(z.T.copy()).unsqueeze(0).float().to(device)
                with torch.inference_mode():
                    audio = vae.decode(z_torch)  # [1, 1, T_audio]
                return audio.squeeze().cpu().numpy()
        else:
            logger.info("[exp8] Loading Exp5 ARFriendlyVAE decoder (VAE dec_proj + Mimi decoder)...")
            from stage2.vae import ARFriendlyVAE
            from stage2.vae_train import load_vae_checkpoint

            autoencoder = load_mimi_autoencoder(
                checkpoint_path=eval_cfg.get("mimi_checkpoint"),
                device=str(device),
            )
            latent_dim = int(ckpt_data.get("latent_dim", 32))
            encoder_dim = int(ckpt_data.get("encoder_dim", 512))
            dec_hidden_dim = ckpt_data.get("dec_hidden_dim")

            if dec_hidden_dim is None:
                vae = ARFriendlyVAE(
                    mimi_encoder=autoencoder.encoder,
                    mimi_decoder=autoencoder.decoder,
                    latent_dim=latent_dim,
                    encoder_dim=encoder_dim,
                    dec_hidden_dim=256,
                ).to(device)
                vae.dec_proj = torch.nn.Conv1d(latent_dim, encoder_dim, 1).to(device)
                vae.dec_hidden_dim = None
            else:
                vae = ARFriendlyVAE(
                    mimi_encoder=autoencoder.encoder,
                    mimi_decoder=autoencoder.decoder,
                    latent_dim=latent_dim,
                    encoder_dim=encoder_dim,
                    dec_hidden_dim=int(dec_hidden_dim),
                ).to(device)
            load_vae_checkpoint(vae_ckpt_path, vae, device=device)
            vae.eval()
            logger.info(f"[exp8] Loaded VAE checkpoint from {vae_ckpt_path} (latent_dim={vae.latent_dim})")

            def _decode_to_audio(z: np.ndarray) -> np.ndarray:
                z_torch = torch.from_numpy(z.T.copy()).unsqueeze(0).float().to(device)
                with torch.inference_mode():
                    audio = vae.decode(z_torch)
                return audio.squeeze().cpu().numpy()
    else:
        logger.info("[exp8] Loading Mimi decoder...")
        autoencoder = load_mimi_autoencoder(
            checkpoint_path=eval_cfg.get("mimi_checkpoint"),
            device=str(device),
        )
        mimi_decoder = autoencoder.decoder

        def _decode_to_audio(z: np.ndarray) -> np.ndarray:
            return decode_mimi_latents_to_audio(z, mimi_decoder, device)

    mimi_sr = 24000
    output_sr = int(eval_cfg["output_sample_rate"])

    # -----------------------------------------------------------------------
    # Step 5: Eval — reconstruct + decode for BOTH methods
    # -----------------------------------------------------------------------
    logger.info("[exp8] Sampling eval utterances for perceptual evaluation...")
    eval_utt_ids = sample_eval_utterances(
        splits_dir=data_cfg["splits_dir"],
        latents_index_path=latents_index_path,
        n_utterances=int(eval_cfg["n_utterances"]),
        seed=42,
    )
    logger.info(f"[exp8] {len(eval_utt_ids)} eval utterances for reconstruction + audio decode")

    eval_results = []

    for i, utt_id in enumerate(eval_utt_ids):
        if utt_id not in store:
            logger.warning(f"[exp8] Utterance {utt_id} not in store, skipping")
            continue

        x_true = store.get_latents(utt_id).astype(np.float32, copy=False)  # [T, 512]
        if x_true.shape[0] < 10:
            continue

        utt_dir = out_dir / f"utt_{i:03d}_{utt_id}"
        utt_dir.mkdir(parents=True, exist_ok=True)

        # Decode GT
        try:
            audio_gt = _decode_to_audio(x_true)
            _save_wav(audio_gt, utt_dir / "GT.wav", mimi_sr, output_sr)
            audio_gt_torch = torch.from_numpy(audio_gt).float().to(device)
        except Exception as e:
            logger.warning(f"[exp8] GT decode failed for {utt_id}: {e}")
            continue

        for k in k_values:
            # --- Normalized reconstruction ---
            x_q_norm, recon_stats_norm = reconstruct_trajectory_normalized(
                x_true, norm_centroids[k], near_zero_threshold=nz_threshold,
            )
            try:
                audio_norm = _decode_to_audio(x_q_norm)
                audio_norm_torch = torch.from_numpy(audio_norm).float().to(device)
                min_len = min(len(audio_gt), len(audio_norm))
                mel_dist_norm = _mel_distance(audio_gt_torch[:min_len], audio_norm_torch[:min_len], mimi_sr)
                l1_dist_norm = float(torch.nn.functional.l1_loss(audio_norm_torch[:min_len], audio_gt_torch[:min_len]).item())
                _save_wav(audio_norm, utt_dir / f"normalized_K{k:04d}.wav", mimi_sr, output_sr)

                eval_results.append({
                    "utterance_id": utt_id,
                    "K": k,
                    "method": "normalized",
                    "mel_distance": mel_dist_norm,
                    "l1_distance": l1_dist_norm,
                    **recon_stats_norm,
                })
            except Exception as e:
                logger.warning(f"[exp8] Normalized decode failed for {utt_id} K={k}: {e}")

            # --- Unnormalized reconstruction ---
            x_q_unnorm, recon_stats_unnorm = reconstruct_trajectory_unnormalized(
                x_true, unnorm_centroids_raw[k], near_zero_threshold=nz_threshold,
            )
            try:
                audio_unnorm = _decode_to_audio(x_q_unnorm)
                audio_unnorm_torch = torch.from_numpy(audio_unnorm).float().to(device)
                min_len = min(len(audio_gt), len(audio_unnorm))
                mel_dist_unnorm = _mel_distance(audio_gt_torch[:min_len], audio_unnorm_torch[:min_len], mimi_sr)
                l1_dist_unnorm = float(torch.nn.functional.l1_loss(audio_unnorm_torch[:min_len], audio_gt_torch[:min_len]).item())
                _save_wav(audio_unnorm, utt_dir / f"unnormalized_K{k:04d}.wav", mimi_sr, output_sr)

                eval_results.append({
                    "utterance_id": utt_id,
                    "K": k,
                    "method": "unnormalized",
                    "mel_distance": mel_dist_unnorm,
                    "l1_distance": l1_dist_unnorm,
                    **recon_stats_unnorm,
                })
            except Exception as e:
                logger.warning(f"[exp8] Unnormalized decode failed for {utt_id} K={k}: {e}")

            # --- State VQ reconstruction ---
            x_q_state, recon_stats_state = reconstruct_trajectory_state_vq(
                x_true, state_centroids[k],
            )
            try:
                audio_state = _decode_to_audio(x_q_state)
                audio_state_torch = torch.from_numpy(audio_state).float().to(device)
                min_len = min(len(audio_gt), len(audio_state))
                mel_dist_state = _mel_distance(audio_gt_torch[:min_len], audio_state_torch[:min_len], mimi_sr)
                l1_dist_state = float(torch.nn.functional.l1_loss(audio_state_torch[:min_len], audio_gt_torch[:min_len]).item())
                _save_wav(audio_state, utt_dir / f"state_vq_K{k:04d}.wav", mimi_sr, output_sr)

                eval_results.append({
                    "utterance_id": utt_id,
                    "K": k,
                    "method": "state_vq",
                    "mel_distance": mel_dist_state,
                    "l1_distance": l1_dist_state,
                    **recon_stats_state,
                })
            except Exception as e:
                logger.warning(f"[exp8] State VQ decode failed for {utt_id} K={k}: {e}")

        logger.info(f"[exp8] [{i+1}/{len(eval_utt_ids)}] {utt_id}: T={x_true.shape[0]} frames")

    # Free GPU memory
    if autoencoder is not None:
        del autoencoder
    if use_vae:
        del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Step 6: Save results and summary
    # -----------------------------------------------------------------------
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    if eval_results:
        df = pd.DataFrame(eval_results)
        summary_rows = []
        for k in k_values:
            for method in ["normalized", "unnormalized", "state_vq"]:
                m_df = df[(df["K"] == k) & (df["method"] == method)]
                if m_df.empty:
                    continue
                if method == "normalized":
                    cb = norm_codebook_results[str(k)]
                elif method == "unnormalized":
                    cb = unnorm_codebook_results[str(k)]
                else:
                    cb = state_codebook_results[str(k)]
                row = {
                    "K": k,
                    "method": method,
                    "codebook_utilization": cb["codebook_utilization"],
                    "codebook_utilization_frac": cb["codebook_utilization_frac"],
                    "entropy_ratio": cb["entropy_ratio"],
                    "mel_distance_mean": float(m_df["mel_distance"].mean()),
                    "mel_distance_std": float(m_df["mel_distance"].std()),
                    "l1_distance_mean": float(m_df["l1_distance"].mean()),
                    "trajectory_divergence_mean": float(m_df["trajectory_divergence_mean"].mean()),
                    "trajectory_divergence_growth_rate": float(m_df["trajectory_divergence_growth_rate"].mean()),
                }
                # Add angular error for direction-based methods
                if method in ("normalized", "unnormalized"):
                    row["angular_error_mean_deg"] = cb["angular_error_mean_deg"]
                    row["angular_error_median_deg"] = cb["angular_error_median_deg"]
                    row["angular_error_p95_deg"] = cb["angular_error_p95_deg"]
                else:
                    row["recon_error_mean"] = cb["recon_error_mean"]
                    row["recon_error_median"] = cb["recon_error_median"]
                    row["recon_error_p95"] = cb["recon_error_p95"]
                summary_rows.append(row)
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(str(out_dir / "summary.csv"), index=False)
        logger.info(f"[exp8] Summary:\n{summary.to_string(index=False)}")

        # Best method per K
        for k in k_values:
            k_summary = summary[summary["K"] == k]
            if len(k_summary) >= 2:
                best_idx = k_summary["mel_distance_mean"].idxmin()
                winner = k_summary.loc[best_idx, "method"]
                mel_strs = ", ".join(
                    f"{row['method']}={row['mel_distance_mean']:.4f}"
                    for _, row in k_summary.iterrows()
                )
                logger.info(f"[exp8] K={k} winner: {winner} ({mel_strs})")
    else:
        summary = pd.DataFrame()

    # Key metrics for finalize
    best_mel = 999.0
    best_method = "unknown"
    best_angular = 0.0
    if not summary.empty:
        best_idx = summary["mel_distance_mean"].idxmin()
        best_mel = float(summary.loc[best_idx, "mel_distance_mean"])
        best_method = summary.loc[best_idx, "method"]
        if "angular_error_mean_deg" in summary.columns and pd.notna(summary.loc[best_idx].get("angular_error_mean_deg")):
            best_angular = float(summary.loc[best_idx, "angular_error_mean_deg"])

    eps_001_frac = exp9_result["epsilon_analysis"].get("0.01", {}).get("fraction", 0.0)

    logger.info(
        f"[exp8] Done. Best method={best_method}, mel_dist={best_mel:.4f}"
    )
    finalize_run(run, key_metrics={
        "best_method": best_method,
        "best_angular_error_deg": best_angular,
        "best_mel_distance": best_mel,
        "near_zero_frac_001": eps_001_frac,
        "exp9_recommendation": exp9_result["recommendation"],
        "latent_dim": int(all_directions.shape[1]),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
