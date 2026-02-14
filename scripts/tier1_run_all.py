#!/usr/bin/env python3
"""
Convenience wrapper to run all Tier 1 experiments sequentially.

Usage:
  uv run python scripts/tier1_run_all.py \
    --exp1-config configs/tier1_exp1_vmf.yaml \
    --exp2-config configs/tier1_exp2_injection.yaml \
    --exp3-config configs/tier1_exp3_rep_compare.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from experiment import register_run, finalize_run


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> int:
    p = argparse.ArgumentParser(description="Run all Tier 1 experiments")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--exp1-config", type=str, default="configs/tier1_exp1_vmf.yaml")
    p.add_argument("--exp2-config", type=str, default="configs/tier1_exp2_injection.yaml")
    p.add_argument("--exp3-config", type=str, default="configs/tier1_exp3_rep_compare.yaml")
    p.add_argument("--exp2-horizon-k", type=int, default=1, help="Which horizon_k checkpoint to use for Exp2")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiments if their output for --run-id already exists",
    )
    args = p.parse_args()

    run_id = args.run_id or _default_run_id()

    import yaml

    # Tracking: create a combined output dir for the orchestrator
    run_all_dir = Path("outputs/tier1/run_all") / run_id
    run_all_dir.mkdir(parents=True, exist_ok=True)
    run = register_run(
        experiment="tier1_run_all", run_id=run_id, config_path="(orchestrator)",
        config={"exp1_config": args.exp1_config, "exp2_config": args.exp2_config, "exp3_config": args.exp3_config},
        cli_args=sys.argv[1:], out_dir=run_all_dir,
    )

    # Exp1

    n_experiments_run = 0

    with open(args.exp1_config) as f:
        exp1_cfg = yaml.safe_load(f)
    exp1_out_root = Path(exp1_cfg["output"]["out_dir"])
    exp1_out = exp1_out_root / run_id
    exp1_done = (exp1_out / "metrics.json").exists() and (exp1_out / "tables.csv").exists()

    if args.resume and exp1_done:
        pass
    else:
        subprocess.check_call(
            [sys.executable, "scripts/tier1_exp1_vmf.py", "--config", args.exp1_config, "--run-id", run_id],
        )
        n_experiments_run += 1

    # Locate the k=... final checkpoint for Exp2
    ckpt = exp1_out / "checkpoints" / f"vmf_k{int(args.exp2_horizon_k)}_final.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Expected Exp1 checkpoint not found: {ckpt}")

    # Exp2
    with open(args.exp2_config) as f:
        exp2_cfg = yaml.safe_load(f)
    exp2_out_root = Path(exp2_cfg["output"]["out_dir"])
    exp2_out = exp2_out_root / run_id
    exp2_done = (exp2_out / "metrics.json").exists()

    if args.resume and exp2_done:
        pass
    else:
        subprocess.check_call(
            [
                sys.executable,
                "scripts/tier1_exp2_injection.py",
                "--config",
                args.exp2_config,
                "--run-id",
                run_id,
                "--checkpoint",
                str(ckpt),
            ],
        )
        n_experiments_run += 1

    # Exp3
    with open(args.exp3_config) as f:
        exp3_cfg = yaml.safe_load(f)
    exp3_out_root = Path(exp3_cfg["output"]["out_dir"])
    exp3_out = exp3_out_root / run_id
    exp3_done = (exp3_out / "summary.csv").exists()

    if args.resume and exp3_done:
        pass
    else:
        subprocess.check_call(
            [sys.executable, "scripts/tier1_exp3_rep_compare.py", "--config", args.exp3_config, "--run-id", run_id],
        )
        n_experiments_run += 1

    finalize_run(run, key_metrics={"n_experiments_run": n_experiments_run})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

