#!/usr/bin/env python3
"""
Phase 0 sanity checks: delta computation and context/target alignment.

Usage:
  uv run python scripts/07_sanity_check_phase0.py --config configs/phase0.yaml
  uv run python scripts/07_sanity_check_phase0.py --utterance-id <utt> --t 200 --lag 1
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import yaml

from phase0.data.io import load_latents_index, load_latents_zarr
from phase0.features.context import get_valid_frame_range
from phase0.features.normalization import compute_delta
from phase0.utils.logging import setup_logging


def _l2_norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt((x * x).sum(axis=axis))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 sanity checks")
    parser.add_argument("--config", type=str, default="configs/phase0.yaml")
    parser.add_argument("--utterance-id", type=str, default=None)
    parser.add_argument("--t", type=int, default=None, help="Target frame index to inspect")
    parser.add_argument("--lag", type=int, default=None, help="Lag to inspect (defaults to first config lag)")
    parser.add_argument("--window-size", type=int, default=None, help="Window size (defaults to config)")
    parser.add_argument("--sample-frames", type=int, default=2000, help="How many frames to sample for stats")
    args = parser.parse_args()

    logger = setup_logging()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    zarr_path = config["output"]["latents_dir"]
    latents_index_path = config["output"]["latents_index"]

    window_size = int(args.window_size or config["context"]["window_size"])
    lag = int(args.lag or config["context"]["lags"][0])

    latents_index = load_latents_index(latents_index_path)
    if len(latents_index) == 0:
        raise RuntimeError(f"Empty latents index: {latents_index_path}")

    if args.utterance_id is None:
        utt_id = str(latents_index.iloc[0]["utterance_id"])
        logger.info(f"No --utterance-id provided; using first: {utt_id}")
    else:
        utt_id = args.utterance_id

    x, energy, timestamps, speaker_id = load_latents_zarr(utt_id, zarr_path)
    n_frames, d = x.shape

    first_valid, last_valid = get_valid_frame_range(n_frames, window_size, max_lag=max(config["context"]["lags"]))
    if args.t is None:
        t = int((first_valid + last_valid - 1) // 2)
    else:
        t = int(args.t)

    if not (0 <= t < n_frames):
        raise ValueError(f"t={t} out of range for n_frames={n_frames}")

    end = t - lag  # inclusive end of context
    start = end - window_size + 1

    logger.info("=" * 70)
    logger.info("PHASE 0 SANITY CHECK")
    logger.info("=" * 70)
    logger.info(f"Utterance: {utt_id} | speaker_id={speaker_id} | T={n_frames} | D={d}")
    logger.info(f"window_size(W)={window_size} | lag(L)={lag}")
    logger.info(f"Valid t range for max_lag={max(config['context']['lags'])}: [{first_valid}, {last_valid})")
    logger.info(f"Chosen t={t}")

    logger.info("-" * 70)
    logger.info("Index alignment")
    logger.info(f"Context end frame index (t-L): end={end}")
    logger.info(f"Context start index: start={start}")
    logger.info(f"Context indices: [{start}..{end}] (len={end - start + 1})")
    if lag == 1:
        logger.info(f"Expected context to include x[t-1]=x[{t-1}] as last frame: {end == t - 1}")

    if t >= 1:
        delta = compute_delta(x, t)
        logger.info("-" * 70)
        logger.info("Magnitude checks (single frame)")
        logger.info(f"||x[t]||_2 = {float(_l2_norm(x[t])):.6f}")
        logger.info(f"||x[t-1]||_2 = {float(_l2_norm(x[t-1])):.6f}")
        logger.info(f"||Δx[t]||_2 = {float(_l2_norm(delta)):.6f}")
        logger.info(f"||Δx[t]|| / ||x[t]|| = {float(_l2_norm(delta) / (_l2_norm(x[t]) + 1e-12)):.6f}")

    logger.info("-" * 70)
    logger.info("Distribution checks (sampled frames)")

    # Sample a set of valid t values for stats; avoid huge allocations.
    valid_ts = np.arange(first_valid, last_valid, dtype=np.int32)
    if len(valid_ts) == 0:
        raise RuntimeError("No valid frames found for this utterance with current W/L.")

    n_sample = min(int(args.sample_frames), len(valid_ts))
    rng = np.random.default_rng(0)
    sample_ts = rng.choice(valid_ts, size=n_sample, replace=False)

    sampled_x = x[sample_ts]
    sampled_prev = x[sample_ts - 1]
    sampled_delta = sampled_x - sampled_prev

    x_norm = _l2_norm(sampled_x)
    dx_norm = _l2_norm(sampled_delta)

    logger.info(f"Sampled frames: {n_sample}")
    logger.info(f"x:   mean||x||={float(x_norm.mean()):.6f}, std||x||={float(x_norm.std()):.6f}")
    logger.info(f"Δx:  mean||Δx||={float(dx_norm.mean()):.6f}, std||Δx||={float(dx_norm.std()):.6f}")
    logger.info(f"ratio mean(||Δx||/||x||)={float((dx_norm / (x_norm + 1e-12)).mean()):.6f}")

    per_dim_var = sampled_delta.var(axis=0)
    vmin = float(per_dim_var.min())
    vmax = float(per_dim_var.max())
    vrange_ratio = float(vmax / (vmin + 1e-12))

    logger.info(f"Per-dim var(Δx): min={vmin:.6e}, max={vmax:.6e}, max/min={vrange_ratio:.3e}")
    topk = 8
    top_idx = np.argsort(per_dim_var)[-topk:][::-1]
    top_share = float(per_dim_var[top_idx].sum() / (per_dim_var.sum() + 1e-12))
    logger.info(f"Top-{topk} dims explain {top_share:.1%} of per-dim variance mass")
    logger.info(f"Top-{topk} dims: {top_idx.tolist()}")


if __name__ == "__main__":
    main()

