#!/usr/bin/env python3
"""
Generate Phase 0 analysis report with decision and visualizations.

Usage:
    uv run python scripts/06_make_report.py [--config configs/phase0.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import yaml

from phase0.analysis.report import generate_report, make_decision
from phase0.analysis.plots import (
    plot_cluster_sizes,
    plot_speaker_distribution,
)
from phase0.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 0 report")
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

    # Load metrics
    metrics_path = config["output"]["metrics_file"]
    with open(metrics_path) as f:
        metrics = json.load(f)

    logger.info(f"Loaded {len(metrics)} metric results")

    # Generate text report
    report_path = Path(config["output"]["plots_dir"]).parent / "report.txt"
    report = generate_report(metrics_path, report_path, config)

    print("\n" + "=" * 70)
    print(report)
    print("=" * 70 + "\n")

    # Make decision
    decision = make_decision(
        metrics,
        variance_ratio_threshold=config["metrics"]["variance_ratio_threshold"],
        cross_speaker_degradation_max=config["metrics"]["cross_speaker_degradation_max"],
    )

    # Create plots directory
    plots_dir = Path(config["output"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load cluster stats for plots
    conditioning_dir = Path(config["output"]["conditioning_dir"])

    for cond_cfg in config["clustering"]["conditions"]:
        cond_name = cond_cfg["name"]

        # Try to load cluster stats
        stats_path = conditioning_dir / f"{cond_name}_lag1_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)

            # Plot cluster sizes
            if "cluster_sizes" in stats:
                sizes = np.array(stats["cluster_sizes"])
                plot_cluster_sizes(
                    sizes,
                    plots_dir / f"{cond_name}_cluster_sizes.png",
                    title=f"Cluster Sizes: {cond_name}",
                    min_size_line=config["clustering"]["min_cluster_size"],
                )
                logger.info(f"Created cluster size plot for {cond_name}")

    # Extract per-speaker variance ratios from metrics
    for m in metrics:
        if m["slice"] == "all" and m["lag"] == 1:
            cond_name = m["condition"]
            if "train" in m and "speaker_mean" in m["train"]:
                # We don't have per-speaker data here, but we have aggregates
                # For a full implementation, we'd save per-speaker data in run_phase0
                pass

    # Save decision as JSON
    decision_path = plots_dir.parent / "decision.json"
    with open(decision_path, "w") as f:
        json.dump(decision, f, indent=2)

    logger.info(f"\nReport saved to: {report_path}")
    logger.info(f"Decision saved to: {decision_path}")
    logger.info(f"Plots saved to: {plots_dir}")

    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 0 DECISION: " + ("PASS" if decision["pass"] else "FAIL"))
    print("=" * 70)

    if decision["pass"]:
        print(f"\nBest condition: {decision['best_condition']}")
        print("Proceed to Phase 1: Minimal Engram Modeling")
    else:
        print("\nAudio latents do not exhibit sufficient reusable structure.")
        print("Consider this a negative structural result about audio latent spaces.")

    return 0 if decision["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
