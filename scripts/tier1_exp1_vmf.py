#!/usr/bin/env python3
"""
Tier 1 - Experiment 1: vMF direction + LogNormal magnitude factorization.

Single-command runner:
  uv run python scripts/tier1_exp1_vmf.py --config configs/tier1_exp1_vmf.yaml
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import yaml

from phase0.utils.logging import setup_logging
from phase1.train_eval import _device_from_config, train_and_eval_for_k, write_results


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> int:
    p = argparse.ArgumentParser(description="Tier1 Exp1: vMF factorization")
    p.add_argument("--config", type=str, default="configs/tier1_exp1_vmf.yaml")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--slice", type=str, default=None, help="Override slice (all|high_energy|utterance_medial)")
    args = p.parse_args()

    logger = setup_logging(name="tier1-exp1-vmf")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_id = args.run_id or _default_run_id()

    device = _device_from_config(cfg["train"]["device"])
    window_size = int(cfg["context"]["window_size"])
    horizons_k = [int(k) for k in cfg["context"]["horizons_k"]]
    slice_name = str(args.slice or cfg.get("slice", "all"))

    frames_index = cfg["data"]["frames_index"]
    latents_dir = cfg["data"]["latents_dir"]
    splits_dir = cfg["data"]["splits_dir"]

    latents_index_path = cfg["data"].get("latents_index")
    if latents_index_path is None:
        latents_parent = Path(latents_dir).parent
        latents_index_path = str((latents_parent / "latents_index.parquet"))

    out_root = Path(cfg["output"]["out_dir"])
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.json"
    tables_path = out_dir / "tables.csv"

    logger.info(f"Config: {args.config}")
    logger.info(f"Run id: {run_id}")
    logger.info(f"Device: {device}")
    logger.info(f"Slice: {slice_name}")
    logger.info(f"Horizons k: {horizons_k}")
    logger.info(f"Frames: {frames_index}")
    logger.info(f"Latents: {latents_dir}")
    logger.info(f"Latents index: {latents_index_path}")
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
            slice_name=slice_name,
            seed=int(cfg["train"]["seed"]),
            device=device,
            # Reuse generic fields for the vMF model
            n_components=int(cfg["model"].get("n_components", 1)),
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            n_hidden_layers=int(cfg["model"]["n_hidden_layers"]),
            dropout=float(cfg["model"]["dropout"]),
            # Not used by vMF (but required by signature)
            min_log_sigma=float(cfg["model"].get("min_log_sigma", -7.0)),
            max_log_sigma=float(cfg["model"].get("max_log_sigma", 2.0)),
            model_type="vmf",
            vmf_min_log_kappa=float(cfg["model"]["min_log_kappa"]),
            vmf_max_log_kappa=float(cfg["model"]["max_log_kappa"]),
            vmf_min_log_sigma_logm=float(cfg["model"]["min_log_sigma_logm"]),
            vmf_max_log_sigma_logm=float(cfg["model"]["max_log_sigma_logm"]),
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
            rollout_sample_from_mixture=bool(cfg["rollout"]["sample_from_model"]),
            compile_model=bool(cfg["train"].get("compile", False)),
            compile_mode=str(cfg["train"].get("compile_mode", "default")),
            amp=bool(cfg["train"].get("amp", False)),
            amp_dtype=str(cfg["train"].get("amp_dtype", "bf16")),
        )
        results.append(r)

    write_results(results, metrics_path=str(metrics_path), tables_path=str(tables_path))
    logger.info(f"Wrote metrics to: {metrics_path}")
    logger.info(f"Wrote table to: {tables_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

