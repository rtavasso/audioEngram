#!/usr/bin/env python3
"""
Extract continuous latents from audio using Mimi autoencoder.

Usage:
    uv run python scripts/02_infer_latents.py [--config configs/phase0.yaml] [--device cuda]
"""

import argparse
import sys
from pathlib import Path

# Add src and project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import yaml
import torch

from phase0.data.librispeech import get_utterances, UtteranceInfo
from phase0.data.splits import load_splits
from phase0.data.io import save_latents_index
from phase0.vae.infer_latents import batch_infer_latents, verify_latent_store
from phase0.utils.logging import setup_logging
from phase0.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Extract latents from audio")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase0.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to Mimi checkpoint (optional, downloads from HF if not provided)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup
    logger = setup_logging()
    set_seed(config["seed"])

    logger.info(f"Device: {args.device}")

    # Load Mimi autoencoder
    logger.info("Loading Mimi autoencoder...")
    from mimi_autoencoder import load_mimi_autoencoder

    autoencoder = load_mimi_autoencoder(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    autoencoder.eval()

    logger.info(f"Sample rate: {autoencoder.sample_rate} Hz")
    logger.info(f"Frame rate: {autoencoder.frame_rate} Hz")
    logger.info(f"Latent dim: {autoencoder.latent_dim}")
    logger.info(f"Frame size: {autoencoder.frame_size} samples")

    # Verify config matches
    assert autoencoder.sample_rate == config["vae"]["sample_rate"]
    assert autoencoder.frame_rate == config["vae"]["frame_rate"]
    assert autoencoder.latent_dim == config["vae"]["latent_dim"]

    # Load splits
    splits = load_splits(config["output"]["splits_dir"])
    logger.info(f"Train speakers: {len(splits.train_speakers)}")
    logger.info(f"Eval speakers: {len(splits.eval_speakers)}")

    # Get all utterances
    librispeech_path = config["data"]["librispeech_path"]
    subset = config["data"]["subset"]

    all_speakers = splits.train_speakers + splits.eval_speakers
    utterances = get_utterances(librispeech_path, all_speakers, subset)
    logger.info(f"Total utterances: {len(utterances)}")

    # Filter by minimum duration
    min_duration = config["data"]["min_duration_sec"]
    utterances = [u for u in utterances if u.duration_sec >= min_duration]
    logger.info(f"Utterances >= {min_duration}s: {len(utterances)}")

    # Run inference
    zarr_path = config["output"]["latents_dir"]
    logger.info(f"Saving latents to {zarr_path}")

    index_entries = batch_infer_latents(
        utterances=utterances,
        autoencoder=autoencoder,
        zarr_path=zarr_path,
        device=args.device,
        show_progress=True,
    )

    # Save index
    index_path = config["output"]["latents_index"]
    save_latents_index(index_entries, index_path)
    logger.info(f"Index saved to {index_path}")

    # Verify
    logger.info("Verifying latent store...")
    stats = verify_latent_store(
        zarr_path,
        expected_frame_rate=config["vae"]["frame_rate"],
        expected_dim=config["vae"]["latent_dim"],
    )

    logger.info("=" * 50)
    logger.info("VERIFICATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Utterances: {stats['n_utterances']}")
    logger.info(f"Total frames: {stats['n_frames_total']}")
    logger.info(f"NaN utterances: {stats['n_nan_utterances']}")
    logger.info(f"Dimensions: {stats['dims']}")
    if stats['mean_frame_rate']:
        logger.info(f"Mean frame rate: {stats['mean_frame_rate']:.2f} Hz")
    logger.info(f"Valid: {stats['valid']}")

    if not stats['valid']:
        logger.error("Verification FAILED!")
        sys.exit(1)

    logger.info("Done!")


if __name__ == "__main__":
    main()
