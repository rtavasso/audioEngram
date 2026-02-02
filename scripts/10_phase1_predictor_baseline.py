#!/usr/bin/env python3
"""
Phase 1: Predictor baseline on Mimi latents (lagged-context primary).

Runs an MDN for p(Δx_t | context ending at t-k) for k in horizons_k.
Reports ΔNLL(k) vs unconditional p(Δx), plus direction/magnitude metrics.

Secondary diagnostic: rollout-context corruption gap.

Usage:
  uv run python scripts/10_phase1_predictor_baseline.py --config configs/phase1.yaml
  uv run python scripts/10_phase1_predictor_baseline.py --config configs/phase1.yaml --slice all
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import yaml

from phase0.utils.logging import setup_logging
from phase1.train_eval import _device_from_config, train_and_eval_for_k, write_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 predictor baseline")
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    parser.add_argument(
        "--slice",
        type=str,
        default="all",
        help="Slice name: all|high_energy|utterance_medial",
    )
    args = parser.parse_args()

    logger = setup_logging(name="phase1")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = _device_from_config(cfg["train"]["device"])
    window_size = int(cfg["context"]["window_size"])
    horizons_k = [int(k) for k in cfg["context"]["horizons_k"]]

    frames_index = cfg["data"]["frames_index"]
    latents_dir = cfg["data"]["latents_dir"]
    splits_dir = cfg["data"]["splits_dir"]

    # Phase 0 latents index path is derived from phase0.yaml convention.
    # Default: outputs/phase0/latents_index.parquet next to latents_dir.
    latents_index_path = Path(latents_dir).parent / "latents_index.parquet"

    out_dir = cfg["output"]["out_dir"]
    metrics_path = cfg["output"]["metrics_file"]
    tables_path = cfg["output"]["tables_file"]

    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {device}")
    logger.info(f"Slice: {args.slice}")
    logger.info(f"Horizons k: {horizons_k}")
    logger.info(f"Frames: {frames_index}")
    logger.info(f"Latents: {latents_dir}")
    logger.info(f"Outputs: {out_dir}")

    results = []

    for k in horizons_k:
        r = train_and_eval_for_k(
            frames_index_path=frames_index,
            latents_dir=latents_dir,
            splits_dir=splits_dir,
            latents_index_path=latents_index_path,
            out_dir=out_dir,
            horizon_k=k,
            window_size=window_size,
            slice_name=args.slice,
            seed=int(cfg["train"]["seed"]),
            device=device,
            n_components=int(cfg["model"]["n_components"]),
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            n_hidden_layers=int(cfg["model"]["n_hidden_layers"]),
            dropout=float(cfg["model"]["dropout"]),
            min_log_sigma=float(cfg["model"]["min_log_sigma"]),
            max_log_sigma=float(cfg["model"]["max_log_sigma"]),
            batch_size=int(cfg["train"]["batch_size"]),
            num_workers=int(cfg["train"]["num_workers"]),
            max_steps=int(cfg["train"]["max_steps"]),
            lr=float(cfg["train"]["lr"]),
            weight_decay=float(cfg["train"]["weight_decay"]),
            grad_clip_norm=float(cfg["train"]["grad_clip_norm"]),
            log_every=int(cfg["train"]["log_every"]),
            eval_every=int(cfg["train"]["eval_every"]),
            save_every=int(cfg["train"]["save_every"]),
            shuffle_buffer=int(cfg["train"]["shuffle_buffer"]),
            max_train_samples=cfg["train"].get("max_train_samples"),
            max_eval_samples=cfg["train"].get("max_eval_samples"),
            rollout_enabled=bool(cfg["rollout"]["enabled"]),
            rollout_n_eval_utterances=int(cfg["rollout"]["n_eval_utterances"]),
            rollout_max_frames_per_utt=int(cfg["rollout"]["max_frames_per_utterance"]),
            rollout_sample_from_mixture=bool(cfg["rollout"]["sample_from_mixture"]),
        )
        results.append(r)

        if cfg.get("metrics", {}).get("enable_early_stop", False):
            eps = float(cfg.get("metrics", {}).get("eps_dnll_stop", 0.0))
            dnll = r.eval.get("dnll")
            if dnll is not None and abs(float(dnll)) < eps:
                logger.info(f"[phase1] Early stop: |eval ΔNLL|={abs(float(dnll)):.4f} < {eps}")
                break

    write_results(results, metrics_path=metrics_path, tables_path=tables_path)
    logger.info(f"Wrote metrics to: {metrics_path}")
    logger.info(f"Wrote table to: {tables_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

