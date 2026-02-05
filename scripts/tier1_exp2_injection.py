#!/usr/bin/env python3
"""
Tier 1 - Experiment 2: Teacher-forcing injection diagnostic.

Runs modes:
  A: teacher forcing (always true state)
  B: periodic correction (default inject after steps 4,8,12 for K=16)
  C: one-shot correction (inject after step 1)
  D: pure rollout (no injection)

Usage:
  uv run python scripts/tier1_exp2_injection.py --config configs/tier1_exp2_injection.yaml
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

import pandas as pd
import yaml

from phase0.utils.logging import setup_logging
from phase1.checkpoints import load_phase1_checkpoint
from phase1.injection_diag import run_injection_diagnostic
from phase1.train_eval import _device_from_config, fit_factorized_baseline, fit_unconditional_baseline


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_plots(out_dir: Path, per_step: pd.DataFrame) -> None:
    # This script only saves figures to disk; always use a non-interactive backend.
    # In notebook environments (e.g., Colab) MPLBACKEND may be set to an inline backend
    # string that isn't valid in a plain Python process, which would otherwise crash
    # at import time.
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _plot(y_col: str, title: str, fname: str) -> None:
        plt.figure(figsize=(8, 4))
        for mode, dfm in per_step.groupby("mode"):
            plt.plot(dfm["step"], dfm[y_col], label=mode)
        plt.xlabel("rollout step")
        plt.ylabel(y_col)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(plots_dir / fname), dpi=160)
        plt.close()

    _plot("dnll", "ΔNLL vs baseline per step", "dnll_per_step.png")
    _plot("cos", "Direction cosine per step", "cos_per_step.png")
    _plot("mag_ratio", "||Δx̂|| / ||Δx|| per step", "mag_ratio_per_step.png")
    _plot("state_err", "||x_hat - x_true|| (pre-step) per step", "state_err_per_step.png")


def main() -> int:
    p = argparse.ArgumentParser(description="Tier1 Exp2: injection diagnostic")
    p.add_argument("--config", type=str, default="configs/tier1_exp2_injection.yaml")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path from config")
    args = p.parse_args()

    logger = setup_logging(name="tier1-exp2-injection")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_id = args.run_id or _default_run_id()
    out_root = Path(cfg["output"]["out_dir"])
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _device_from_config(cfg["device"])

    ckpt_path = Path(args.checkpoint or cfg["checkpoint"]).expanduser()
    model, ckpt = load_phase1_checkpoint(ckpt_path, device=device)
    model_type = str(ckpt.get("model_type", "mdn")).lower().strip()

    data_cfg = cfg["data"]
    frames_index = data_cfg["frames_index"]
    latents_dir = data_cfg["latents_dir"]
    splits_dir = data_cfg["splits_dir"]
    latents_index = data_cfg.get("latents_index")
    if latents_index is None:
        latents_parent = Path(latents_dir).parent
        latents_index = str((latents_parent / "latents_index.parquet"))

    diag_cfg = cfg["diag"]
    horizon_k = int(diag_cfg["horizon_k"])
    window_size = int(diag_cfg["window_size"])
    k_steps = int(diag_cfg["k_steps"])

    slice_name = str(diag_cfg.get("slice", "all"))
    seed = int(diag_cfg.get("seed", 42))

    # Fit baseline matched to the model objective.
    if model_type == "mdn":
        baseline = fit_unconditional_baseline(
            frames_index_path=frames_index,
            latents_dir=latents_dir,
            window_size=window_size,
            horizon_k=horizon_k,
            slice_name=slice_name,
            max_samples=diag_cfg.get("max_train_samples"),
        )
    else:
        baseline = fit_factorized_baseline(
            frames_index_path=frames_index,
            latents_dir=latents_dir,
            window_size=window_size,
            horizon_k=horizon_k,
            slice_name=slice_name,
            max_samples=diag_cfg.get("max_train_samples"),
        )

    mode_inject = {
        "A_teacher": None,
        "B_periodic": [int(x) for x in diag_cfg.get("inject_after_steps_periodic", [4, 8, 12])],
        "C_one_shot": [int(x) for x in diag_cfg.get("inject_after_steps_one_shot", [1])],
        "D_rollout": [],
    }

    logger.info(f"Run id: {run_id}")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Device: {device}")
    logger.info(f"Latents: {latents_dir}")
    logger.info(f"Frames: {frames_index}")
    logger.info(f"Horizon k: {horizon_k}  window: {window_size}  rollout K: {k_steps}")

    res = run_injection_diagnostic(
        model=model,
        baseline=baseline,
        latents_dir=latents_dir,
        latents_index_path=latents_index,
        splits_dir=splits_dir,
        horizon_k=horizon_k,
        window_size=window_size,
        k_steps=k_steps,
        n_eval_utterances=int(diag_cfg["n_eval_utterances"]),
        segments_per_utt=int(diag_cfg["segments_per_utt"]),
        max_frames_per_utt=int(diag_cfg["max_frames_per_utterance"]),
        seed=seed,
        device=device,
        mode_inject_after_steps=mode_inject,
        sample_from_model=bool(diag_cfg.get("sample_from_model", False)),
    )

    # Flatten per-step to CSV
    rows = []
    for mode, md in res["modes"].items():
        for s in md["per_step"]:
            rows.append({"mode": mode, **s})
    per_step = pd.DataFrame(rows)
    per_step_path = out_dir / "per_step.csv"
    per_step.to_csv(str(per_step_path), index=False)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(res, f, indent=2)

    _write_plots(out_dir, per_step)

    logger.info(f"Wrote: {metrics_path}")
    logger.info(f"Wrote: {per_step_path}")
    logger.info(f"Wrote: {out_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

