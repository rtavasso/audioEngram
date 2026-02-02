#!/usr/bin/env python3
"""
Create speaker and utterance splits for Phase 0 analysis.

Usage:
    uv run python scripts/01_make_speaker_splits.py [--config configs/phase0.yaml]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

from phase0.data.librispeech import get_speaker_ids, get_utterances
from phase0.data.splits import (
    create_speaker_splits,
    create_utterance_splits,
    save_splits,
)
from phase0.utils.logging import setup_logging
from phase0.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Create speaker splits")
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

    logger.info("Creating speaker splits...")

    # Get all speakers
    librispeech_path = config["data"]["librispeech_path"]
    subset = config["data"]["subset"]

    logger.info(f"Loading speakers from {librispeech_path}/{subset}")
    speaker_ids = get_speaker_ids(librispeech_path, subset)
    logger.info(f"Found {len(speaker_ids)} speakers")

    # Create speaker splits
    n_train = config["splits"]["n_train_speakers"]
    n_eval = config["splits"]["n_eval_speakers"]
    seed = config["splits"]["seed"]

    train_speakers, eval_speakers = create_speaker_splits(
        speaker_ids, n_train=n_train, n_eval=n_eval, seed=seed
    )
    logger.info(f"Train speakers: {len(train_speakers)}, Eval speakers: {len(eval_speakers)}")

    # Get all utterances
    logger.info("Loading utterances...")
    all_speakers = train_speakers + eval_speakers
    utterances = get_utterances(librispeech_path, all_speakers, subset)
    logger.info(f"Found {len(utterances)} utterances")

    # Create utterance splits
    holdout_frac = config["splits"]["utt_holdout_frac"]
    splits = create_utterance_splits(
        utterances, train_speakers, eval_speakers, holdout_frac=holdout_frac, seed=seed
    )

    # Save splits
    output_dir = config["output"]["splits_dir"]
    save_splits(splits, output_dir)
    logger.info(f"Splits saved to {output_dir}")

    # Summary
    logger.info("=" * 50)
    logger.info("SPLIT SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Train speakers: {len(splits.train_speakers)}")
    logger.info(f"Eval speakers: {len(splits.eval_speakers)}")
    logger.info(f"Train utterances: {len(splits.train_utterances)}")
    logger.info(f"Eval utterances: {len(splits.eval_utterances)}")
    logger.info(f"Train utterances (k-means train): {len(splits.train_utt_train)}")
    logger.info(f"Train utterances (k-means val): {len(splits.train_utt_val)}")


if __name__ == "__main__":
    main()
