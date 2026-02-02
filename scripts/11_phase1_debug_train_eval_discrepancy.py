#!/usr/bin/env python3
"""
Debug Phase 1 train/eval discrepancy by re-evaluating streams and surfacing outliers.

Usage:
  uv run python scripts/11_phase1_debug_train_eval_discrepancy.py --config configs/phase1.yaml --k 1
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import yaml
import torch

from phase0.utils.logging import setup_logging
from phase1.train_eval import _device_from_config, fit_unconditional_baseline
from phase1.data import iter_phase1_samples
from phase1.mdn import MDN
from phase1.debug import eval_stream_with_debug


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 debug for train/eval discrepancy")
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    parser.add_argument("--k", type=int, required=True, help="Horizon k to debug")
    parser.add_argument("--slice", type=str, default="all")
    parser.add_argument("--max-samples", type=int, default=200000)
    parser.add_argument("--skip-samples", type=int, default=0, help="Skip this many samples in each stream before scoring")
    parser.add_argument("--shuffle-buffer", type=int, default=0, help="If >0, apply buffered shuffle of this size before scoring (approximate)")
    parser.add_argument("--reservoir", type=int, default=50000)
    parser.add_argument("--top-worst", type=int, default=50)
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path; defaults to outputs/phase1/checkpoints/mdn_k{K}_final.pt")
    args = parser.parse_args()

    logger = setup_logging(name="phase1-debug")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = _device_from_config(cfg["train"]["device"])
    window_size = int(cfg["context"]["window_size"])
    k = int(args.k)

    frames_index = cfg["data"]["frames_index"]
    latents_dir = cfg["data"]["latents_dir"]

    # Fit baseline on train
    baseline = fit_unconditional_baseline(
        frames_index_path=frames_index,
        latents_dir=latents_dir,
        window_size=window_size,
        horizon_k=k,
        slice_name=args.slice,
        max_samples=cfg["train"].get("max_train_samples"),
    )

    # Load checkpoint
    if args.ckpt is None:
        ckpt_path = Path(cfg["output"]["checkpoints_dir"]) / f"mdn_k{k}_final.pt"
    else:
        ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    input_dim = window_size * 512
    output_dim = 512
    model = MDN(
        input_dim=input_dim,
        output_dim=output_dim,
        n_components=int(cfg["model"]["n_components"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        n_hidden_layers=int(cfg["model"]["n_hidden_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        min_log_sigma=float(cfg["model"]["min_log_sigma"]),
        max_log_sigma=float(cfg["model"]["max_log_sigma"]),
    ).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model"])

    logger.info(f"Loaded checkpoint: {ckpt_path}")
    logger.info(f"Evaluating k={k} slice={args.slice} max_samples={args.max_samples}")

    train_stream_base = iter_phase1_samples(
        frames_index_path=frames_index,
        latents_dir=latents_dir,
        split="train",
        window_size=window_size,
        horizon_k=k,
        slice_name=args.slice,
        max_samples=None,
    )
    eval_stream_base = iter_phase1_samples(
        frames_index_path=frames_index,
        latents_dir=latents_dir,
        split="eval",
        window_size=window_size,
        horizon_k=k,
        slice_name=args.slice,
        max_samples=None,
    )

    def apply_skip_and_shuffle(stream, seed_offset: int):
        s = stream
        if args.skip_samples and args.skip_samples > 0:
            n_skip = int(args.skip_samples)
            for _ in range(n_skip):
                try:
                    next(s)
                except StopIteration:
                    break
        if args.shuffle_buffer and args.shuffle_buffer > 0:
            from phase1.data import BufferedShuffle

            shuf = BufferedShuffle(buffer_size=int(args.shuffle_buffer), seed=int(cfg["train"]["seed"]) + seed_offset + k)
            s = shuf(s)
        return s

    train_stream = apply_skip_and_shuffle(iter(train_stream_base), seed_offset=10_000)
    eval_stream = apply_skip_and_shuffle(iter(eval_stream_base), seed_offset=20_000)

    train_dbg = eval_stream_with_debug(
        model=model,
        baseline=baseline,
        samples=train_stream,
        device=device,
        max_samples=args.max_samples,
        reservoir_size=args.reservoir,
        top_worst=args.top_worst,
    )
    eval_dbg = eval_stream_with_debug(
        model=model,
        baseline=baseline,
        samples=eval_stream,
        device=device,
        max_samples=args.max_samples,
        reservoir_size=args.reservoir,
        top_worst=args.top_worst,
    )

    out = {
        "k": k,
        "slice": args.slice,
        "train": train_dbg,
        "eval": eval_dbg,
    }
    out_path = Path(cfg["output"]["out_dir"]) / f"debug_k{k}_{args.slice}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    logger.info(f"Train dnll_mean={train_dbg.get('dnll_mean')} (n={train_dbg.get('n')})")
    logger.info(f"Train dnll_quantiles={train_dbg.get('dnll_quantiles')}")
    logger.info(f"Train nll_quantiles={train_dbg.get('nll_quantiles')}")
    if train_dbg.get("worst_by_dnll"):
        logger.info("Train worst_by_dnll (top 5):")
        for w in train_dbg["worst_by_dnll"][:5]:
            logger.info(f"  {w}")

    logger.info(f"Eval  dnll_mean={eval_dbg.get('dnll_mean')} (n={eval_dbg.get('n')})")
    logger.info(f"Eval  dnll_quantiles={eval_dbg.get('dnll_quantiles')}")
    logger.info(f"Eval  nll_quantiles={eval_dbg.get('nll_quantiles')}")
    if eval_dbg.get("worst_by_dnll"):
        logger.info("Eval worst_by_dnll (top 5):")
        for w in eval_dbg["worst_by_dnll"][:5]:
            logger.info(f"  {w}")

    logger.info(f"Wrote debug JSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
