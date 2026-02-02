#!/usr/bin/env python3
"""
Build the Phase 0 frame index from extracted latents.

Usage:
    uv run python scripts/03_build_phase0_dataset.py [--config configs/phase0.yaml]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import yaml

from phase0.data.io import load_latents_index, load_latents_zarr, save_frames_index, get_zarr_utterance_ids
from phase0.data.splits import load_splits
from phase0.features.context import get_valid_frame_range
from phase0.features.energy import compute_median_energy
from phase0.utils.logging import setup_logging
from phase0.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Build phase0 frame index")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase0.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup
    logger = setup_logging()
    set_seed(config["seed"])

    # Load splits
    splits = load_splits(config["output"]["splits_dir"])
    train_speaker_set = set(splits.train_speakers)
    eval_speaker_set = set(splits.eval_speakers)

    # Load latents index
    latents_index = load_latents_index(config["output"]["latents_index"])
    logger.info(f"Loaded {len(latents_index)} utterances from index")

    # Get valid frame range parameters
    window_size = config["context"]["window_size"]
    max_lag = max(config["context"]["lags"])
    min_duration = config["data"]["min_duration_sec"]

    logger.info(f"Window size: {window_size}, Max lag: {max_lag}")

    # Build frame index
    frames_list = []
    train_energies = []

    zarr_path = config["output"]["latents_dir"]

    for _, row in latents_index.iterrows():
        utt_id = row["utterance_id"]
        speaker_id = row["speaker_id"]
        n_frames = row["n_frames"]
        duration = row["duration_sec"]

        # Skip short utterances
        if duration < min_duration:
            continue

        # Determine split
        if speaker_id in train_speaker_set:
            split = "train"
        elif speaker_id in eval_speaker_set:
            split = "eval"
        else:
            continue  # Skip speakers not in our splits

        # Load energy for this utterance
        try:
            _, energy, _, _ = load_latents_zarr(utt_id, zarr_path)
        except Exception as e:
            logger.warning(f"Could not load {utt_id}: {e}")
            continue

        # Collect train energies for threshold computation
        if split == "train":
            train_energies.append(energy)

        # Get valid frame range
        first_valid, last_valid = get_valid_frame_range(n_frames, window_size, max_lag)

        for t in range(first_valid, last_valid):
            frames_list.append({
                "utterance_id": utt_id,
                "speaker_id": speaker_id,
                "t": t,
                "pos_frac": t / n_frames,
                "energy": energy[t],
                "split": split,
            })

    logger.info(f"Total frames: {len(frames_list)}")

    # Create DataFrame
    frames = pd.DataFrame(frames_list)

    # Compute global median energy (train only)
    logger.info("Computing energy threshold...")
    median_energy = compute_median_energy(train_energies)
    logger.info(f"Median energy (train): {median_energy:.6f}")

    # Add is_high_energy flag
    frames["is_high_energy"] = frames["energy"] > median_energy

    # Save
    output_path = config["output"]["frames_index"]
    save_frames_index(frames, output_path)
    logger.info(f"Saved frame index to {output_path}")

    # Summary statistics
    train_frames = frames[frames["split"] == "train"]
    eval_frames = frames[frames["split"] == "eval"]

    logger.info("=" * 50)
    logger.info("FRAME INDEX SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total frames: {len(frames)}")
    logger.info(f"Train frames: {len(train_frames)}")
    logger.info(f"Eval frames: {len(eval_frames)}")
    logger.info(f"Train speakers: {train_frames['speaker_id'].nunique()}")
    logger.info(f"Eval speakers: {eval_frames['speaker_id'].nunique()}")
    logger.info(f"Train high-energy: {train_frames['is_high_energy'].sum()} ({train_frames['is_high_energy'].mean():.1%})")
    logger.info(f"Eval high-energy: {eval_frames['is_high_energy'].sum()} ({eval_frames['is_high_energy'].mean():.1%})")

    # Position distribution
    medial_mask = (frames["pos_frac"] >= 0.17) & (frames["pos_frac"] <= 0.83)
    logger.info(f"Utterance-medial frames: {medial_mask.sum()} ({medial_mask.mean():.1%})")


if __name__ == "__main__":
    main()
