#!/usr/bin/env python3
"""
Tier 4 - Experiment 12: State-Conditioned Directional Structure in VAE Latent Dynamics.

Implements Experiments 1–6 from `TIER4_EXPERIMENTs.md` on continuous VAE latents.

Usage:
  uv run python scripts/tier4_exp12_state_conditioned_directions.py \
      --config configs/tier4_exp12_state_conditioned_directions.yaml

  # Quick sanity run (few utterances/deltas, skip perceptual):
  uv run python scripts/tier4_exp12_state_conditioned_directions.py \
      --config configs/tier4_exp12_state_conditioned_directions.yaml \
      --stages exp1,exp2,exp3 --max-utterances 200 --max-deltas 200000 --no-perceptual
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import yaml

from experiment import finalize_run, register_run
from phase0.utils.logging import setup_logging
from phase0.utils.seed import set_seed
from tier4.state_conditioned_directions import (
    Tier4Exp12Runner,
    parse_stages,
)


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> int:
    p = argparse.ArgumentParser(description="Tier4 Exp12: state-conditioned directional structure")
    p.add_argument("--config", type=str, default="configs/tier4_exp12_state_conditioned_directions.yaml")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument(
        "--stages",
        type=str,
        default="exp1,exp2,exp3,exp4,exp5,exp6",
        help="Comma-separated subset: exp1,exp2,exp3,exp4,exp5,exp6",
    )
    p.add_argument("--max-utterances", type=int, default=None, help="Override preprocess.max_utterances")
    p.add_argument("--max-deltas", type=int, default=None, help="Override preprocess.max_deltas")
    p.add_argument("--overwrite-cache", action="store_true", help="Rebuild cached arrays even if present")
    p.add_argument("--no-perceptual", action="store_true", help="Disable Exp5 perceptual validation")
    p.add_argument("--exp5-k1", type=int, default=None, help="Override Exp5 K1 (state codebook size)")
    p.add_argument(
        "--exp5-k2",
        type=str,
        default=None,
        help="Override Exp5 K2 values, comma-separated (e.g. 16,32,64)",
    )
    args = p.parse_args()

    logger = setup_logging(name="tier4-exp12")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_id = args.run_id or _default_run_id()
    out_root = Path(cfg["output"]["out_dir"])
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional compute controls.
    omp_threads = int(cfg.get("compute", {}).get("omp_num_threads", 1))
    os.environ.setdefault("OMP_NUM_THREADS", str(omp_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(omp_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(omp_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(omp_threads))

    # Register run
    run = register_run(
        experiment="exp12_state_cond_dirs",
        run_id=run_id,
        config_path=args.config,
        config=cfg,
        cli_args=sys.argv[1:],
        out_dir=out_dir,
        log_name="tier4-exp12",
    )

    # Seed: controls any local sampling and default RNG.
    set_seed(int(cfg.get("seed", 42)))

    # Apply CLI overrides
    if args.max_utterances is not None:
        cfg.setdefault("preprocess", {})["max_utterances"] = int(args.max_utterances)
    if args.max_deltas is not None:
        cfg.setdefault("preprocess", {})["max_deltas"] = int(args.max_deltas)
    if args.no_perceptual:
        cfg.setdefault("exp5_perceptual", {})["enabled"] = False
    if args.exp5_k1 is not None:
        cfg.setdefault("exp5_perceptual", {})["k1"] = int(args.exp5_k1)
    if args.exp5_k2 is not None:
        k2_vals = [int(x.strip()) for x in args.exp5_k2.split(",") if x.strip()]
        if not k2_vals:
            raise ValueError("--exp5-k2 provided but no values parsed.")
        cfg.setdefault("exp5_perceptual", {})["k2_values"] = k2_vals

    stages = parse_stages(args.stages)
    if args.no_perceptual and "exp5" in stages:
        raise ValueError("--no-perceptual conflicts with --stages including exp5.")

    try:
        runner = Tier4Exp12Runner(cfg=cfg, out_dir=out_dir, logger=logger)
        results = runner.run(
            stages=stages,
            overwrite_cache=bool(args.overwrite_cache),
        )
        finalize_run(run, key_metrics=results.get("key_metrics", {}))
        return 0
    except Exception as e:
        logger.exception(f"[tier4-exp12] Failed: {e}")
        # Still mark run complete with failure marker to aid debugging.
        finalize_run(run, key_metrics={"status": "failed", "error": str(e)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
