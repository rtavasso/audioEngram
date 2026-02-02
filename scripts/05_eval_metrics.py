#!/usr/bin/env python3
"""
Evaluate all metrics on train and eval splits.

Usage:
    uv run python scripts/05_eval_metrics.py [--config configs/phase0.yaml]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

from phase0.analysis.run_phase0 import run_full_analysis
from phase0.utils.logging import setup_logging
from phase0.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 0 metrics")
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

    logger.info("Running Phase 0 analysis...")

    # Run full analysis
    results = run_full_analysis(args.config)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    df = results["table"]
    for _, row in df[df["slice"] == "all"].iterrows():
        logger.info(
            f"{row['condition']:20s} lag={row['lag']} | "
            f"Train VR: {row['train_variance_ratio']:.3f} | "
            f"Eval VR: {row['eval_variance_ratio']:.3f} | "
            f"Degradation: {row['cross_speaker_degradation']:.3f} | "
            f"Random: {row['random_baseline']:.3f}"
        )

    logger.info("\nResults saved to:")
    logger.info(f"  Metrics: {config['output']['metrics_file']}")
    logger.info(f"  Table: {config['output']['tables_file']}")


if __name__ == "__main__":
    main()
