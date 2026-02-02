#!/usr/bin/env python3
"""
Phase 3: Train RSSM-style factorizer on frozen Mimi latents (single-rate).

Usage:
  uv run python scripts/20_phase3_train_factorizer.py --config configs/phase3.yaml
  uv run python scripts/20_phase3_train_factorizer.py --config configs/phase3.yaml --resume outputs/phase3/checkpoints/phase3_step5000.pt
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import yaml

from phase0.utils.logging import setup_logging
from phase3.train_eval import train


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Phase 3 factorizer")
    parser.add_argument("--config", type=str, default="configs/phase3.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    setup_logging(name="phase3")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume_checkpoint=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

