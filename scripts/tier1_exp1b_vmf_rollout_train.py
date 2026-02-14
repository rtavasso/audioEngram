#!/usr/bin/env python3
"""
Tier 1 - Experiment 1B: vMF rollout fine-tuning.

Loads a pretrained VmfLogNormal checkpoint and fine-tunes with K-step
rollout loss to improve stability under free-running rollout.

Key ideas:
  - State advance uses rollout_mean() (median magnitude, no sigma blow-up)
  - Loss = vMF NLL (direction + magnitude) accumulated over K rollout steps
  - Scheduled sampling: mix teacher-forced and model-predicted states
  - Delta clipping for stability

Usage:
  uv run python scripts/tier1_exp1b_vmf_rollout_train.py \
    --config configs/tier1_exp1b_vmf_rollout.yaml \
    --checkpoint outputs/tier1/exp1_vmf/<RUN_ID>/checkpoints/vmf_k1_final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import torch
import yaml
from torch.utils.data import IterableDataset, DataLoader

from experiment import register_run, finalize_run
from phase0.utils.logging import setup_logging, get_logger
from phase0.utils.seed import set_seed

from phase1.checkpoints import load_phase1_checkpoint
from phase1.data import iter_rollout_segments, BufferedShuffle, sample_eval_utterances
from phase1.injection_diag import run_injection_diagnostic
from phase1.train_eval import _device_from_config, fit_factorized_baseline


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _clip_l2(x: torch.Tensor, max_l2: float) -> torch.Tensor:
    if not max_l2 or max_l2 <= 0:
        return x
    l2 = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(max_l2 / l2, max=1.0)
    return x * scale


def _linear_schedule(step: int, start: float, end: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return float(end)
    a = min(max(float(step) / float(warmup_steps), 0.0), 1.0)
    return float(start + a * (end - start))


class _RolloutSegmentDataset(IterableDataset):
    def __init__(self, iterator_fn):
        super().__init__()
        self._iterator_fn = iterator_fn

    def __iter__(self):
        return self._iterator_fn()


def _collate_rollout(batch: list[dict]) -> dict:
    return {
        "z_window": torch.from_numpy(np.stack([b["z_window"] for b in batch], axis=0)),
        "dz_seq": torch.from_numpy(np.stack([b["dz_seq"] for b in batch], axis=0)),
        "z_seq": torch.from_numpy(np.stack([b["z_seq"] for b in batch], axis=0)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Tier1 Exp1B: vMF rollout fine-tuning")
    p.add_argument("--config", type=str, default="configs/tier1_exp1b_vmf_rollout.yaml")
    p.add_argument("--checkpoint", type=str, default=None, help="Pretrained VmfLogNormal checkpoint")
    p.add_argument("--run-id", type=str, default=None)
    # Quick overrides
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    args = p.parse_args()

    logger = setup_logging(name="tier1-exp1b-vmf-rollout")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_id = args.run_id or _default_run_id()
    out_root = Path(cfg["output"]["out_dir"])
    out_dir = out_root / run_id
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run = register_run(
        experiment="exp1b_vmf_rollout", run_id=run_id, config_path=args.config,
        config=cfg, cli_args=sys.argv[1:], out_dir=out_dir, log_name="tier1-exp1b-vmf-rollout",
    )

    device = _device_from_config("auto")
    seed = int(cfg["rollout_train"].get("seed", 42))
    set_seed(seed)

    # Load pretrained checkpoint
    ckpt_path = args.checkpoint or cfg.get("checkpoint")
    if ckpt_path is None:
        logger.error("No checkpoint provided. Use --checkpoint or set checkpoint in config.")
        return 1
    ckpt_path = Path(ckpt_path).expanduser()
    model, ckpt = load_phase1_checkpoint(ckpt_path, device=device)

    # Tighten sigma clamp for rollout stability: exp(2.0)=7.39 → exp(0.7)=2.01
    finetune_max_log_sigma = float(cfg.get("rollout_train", {}).get("max_log_sigma_logm", 0.7))
    if hasattr(model, "_max_log_sigma_logm"):
        old_val = model._max_log_sigma_logm
        model._max_log_sigma_logm = finetune_max_log_sigma
        logger.info(f"[exp1b] Tightened max_log_sigma_logm: {old_val:.2f} -> {finetune_max_log_sigma:.2f}")

    model.train()

    window_size = int(cfg["context"]["window_size"])
    horizon_k = int(cfg["context"]["horizon_k"])

    data_cfg = cfg["data"]
    latents_dir = data_cfg["latents_dir"]
    latents_index = data_cfg["latents_index"]
    if latents_index is None:
        latents_index = str(Path(latents_dir).parent / "latents_index.parquet")
    splits_dir = data_cfg["splits_dir"]
    frames_index = data_cfg["frames_index"]

    rtcfg = cfg["rollout_train"]
    k = args.k or int(rtcfg.get("k", 16))
    batch_size = int(rtcfg.get("batch_size", 256))
    max_steps = args.max_steps or int(rtcfg.get("max_steps", 5000))
    lr = args.lr or float(rtcfg.get("lr", 2e-4))
    grad_clip = float(rtcfg.get("grad_clip_norm", 1.0))
    clip_dz_l2 = float(rtcfg.get("clip_dz_l2", 5.0))
    rollout_weight = float(rtcfg.get("rollout_weight", 1.0))
    teacher_weight = float(rtcfg.get("teacher_weight", 0.1))
    log_every = int(rtcfg.get("log_every", 100))
    save_every = int(rtcfg.get("save_every", 1000))
    segments_per_utt = int(rtcfg.get("segments_per_utt", 8))

    sched_cfg = rtcfg.get("sched_teacher_prob", {})
    sched_p_start = float(sched_cfg.get("start", 0.2))
    sched_p_end = float(sched_cfg.get("end", 0.0))
    sched_warmup = int(sched_cfg.get("warmup", 2000))

    logger.info(f"Run id: {run_id}")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Device: {device}")
    logger.info(f"K={k} batch={batch_size} steps={max_steps} lr={lr:g}")
    logger.info(f"w_rollout={rollout_weight:g} w_teacher={teacher_weight:g}")
    logger.info(f"sched_p={sched_p_start:g}->{sched_p_end:g}@{sched_warmup}")

    # Data loader
    shuffler = BufferedShuffle(buffer_size=4096, seed=seed + 500)
    seg_ds = _RolloutSegmentDataset(
        lambda: shuffler(
            iter_rollout_segments(
                latents_dir=latents_dir,
                latents_index_path=latents_index,
                splits_dir=splits_dir,
                split="train",
                window_size=window_size,
                horizon_k=horizon_k,
                k_steps=k,
                segments_per_utt=segments_per_utt,
                seed=seed + 1001,
                min_duration_sec=float(data_cfg.get("min_duration_sec", 3.0)),
            )
        )
    )
    seg_loader = DataLoader(
        seg_ds,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=_collate_rollout,
        pin_memory=(device.type == "cuda"),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(rtcfg.get("weight_decay", 1e-4)))

    # Training loop
    seg_iter = iter(seg_loader)
    for step in range(1, max_steps + 1):
        try:
            batch = next(seg_iter)
        except StopIteration:
            seg_iter = iter(seg_loader)
            batch = next(seg_iter)

        z_window = batch["z_window"].to(device)  # [B, W, D]
        dz_seq = batch["dz_seq"].to(device)  # [B, K, D]
        z_seq = batch["z_seq"].to(device)  # [B, K+1, D]

        bsz = z_window.shape[0]
        d = z_window.shape[2]

        # z_hat starts at last frame of initial window
        z_hat = z_window[:, -1].clone()  # [B, D]
        rolling_window = z_window.clone()  # [B, W, D]

        opt.zero_grad(set_to_none=True)
        loss_roll = torch.tensor(0.0, device=device)
        loss_tf = torch.tensor(0.0, device=device)

        p_teacher = _linear_schedule(step, sched_p_start, sched_p_end, sched_warmup)

        for i in range(k):
            dz_true = dz_seq[:, i]  # [B, D]

            # Context from rolling window
            ctx_flat = rolling_window.reshape(bsz, -1)  # [B, W*D]

            # Rollout NLL
            loss_roll = loss_roll + model.nll(ctx_flat, dz_true).mean()

            # Teacher-forced NLL (optional auxiliary loss)
            if teacher_weight > 0:
                # Teacher context uses ground-truth states
                if i == 0:
                    tf_window = z_window
                else:
                    # Build teacher window: last W frames ending at z_seq[i]
                    # z_seq[0] = frame after window, z_seq[i] = current frame
                    # We need W frames ending at z_seq[i-1] (the state before dz_true[i])
                    tf_end = z_seq[:, :i]  # [B, i, D]
                    if i < window_size:
                        tf_window = torch.cat([z_window[:, i:], tf_end], dim=1)  # [B, W, D]
                    else:
                        tf_window = tf_end[:, i - window_size : i]  # [B, W, D]
                tf_ctx = tf_window.reshape(bsz, -1)
                loss_tf = loss_tf + model.nll(tf_ctx, dz_true).mean()

            # State advance using rollout_mean (median magnitude)
            with torch.no_grad():
                dz_hat = model.rollout_mean(ctx_flat)
                dz_hat = _clip_l2(dz_hat, clip_dz_l2)

            # Scheduled sampling
            if p_teacher > 0:
                use_true = (torch.rand((bsz, 1), device=device) < p_teacher).to(dtype=z_hat.dtype)
                z_hat = use_true * z_seq[:, i + 1] + (1.0 - use_true) * (z_hat + dz_hat)
            else:
                z_hat = z_hat + dz_hat

            # Update rolling window
            rolling_window = torch.cat([rolling_window[:, 1:], z_hat.unsqueeze(1)], dim=1)

        loss_roll = loss_roll / float(k)
        loss_tf = loss_tf / float(k) if teacher_weight > 0 else loss_tf

        loss = rollout_weight * loss_roll + teacher_weight * loss_tf

        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            logger.warning(f"[exp1b] step={step} NaN/Inf loss detected, skipping update")
            continue

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        opt.step()

        if log_every and step % log_every == 0:
            logger.info(
                f"[exp1b] step={step}/{max_steps} loss={float(loss.item()):.4f} "
                f"L_roll={float(loss_roll.item()):.4f} L_tf={float(loss_tf.item()) if teacher_weight > 0 else 0:.4f} "
                f"p_teacher={p_teacher:.3f}"
            )

        if save_every and step % save_every == 0:
            ckpt_out = ckpt_dir / f"vmf_rollout_step{step}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_type": "vmf",
                    "step": step,
                    "horizon_k": horizon_k,
                    "window_size": window_size,
                    "input_dim": ckpt["input_dim"],
                    "output_dim": ckpt["output_dim"],
                    "model_kwargs": ckpt["model_kwargs"],
                },
                ckpt_out,
            )

    # Save final checkpoint
    final_ckpt = ckpt_dir / "vmf_rollout_final.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_type": "vmf",
            "step": max_steps,
            "horizon_k": horizon_k,
            "window_size": window_size,
            "input_dim": ckpt["input_dim"],
            "output_dim": ckpt["output_dim"],
            "model_kwargs": ckpt["model_kwargs"],
        },
        final_ckpt,
    )
    logger.info(f"[exp1b] Saved final checkpoint: {final_ckpt}")

    # Post-training evaluation: injection diagnostic
    model.eval()
    eval_cfg = cfg.get("eval", {})

    baseline = fit_factorized_baseline(
        frames_index_path=frames_index,
        latents_dir=latents_dir,
        window_size=window_size,
        horizon_k=horizon_k,
        slice_name="all",
        max_samples=None,
    )

    inj_result = run_injection_diagnostic(
        model=model,
        baseline=baseline,
        latents_dir=latents_dir,
        latents_index_path=latents_index,
        splits_dir=splits_dir,
        horizon_k=horizon_k,
        window_size=window_size,
        k_steps=int(eval_cfg.get("k_steps", 16)),
        n_eval_utterances=int(eval_cfg.get("n_eval_utterances", 16)),
        segments_per_utt=int(eval_cfg.get("segments_per_utt", 8)),
        max_frames_per_utt=int(eval_cfg.get("max_frames_per_utterance", 2000)),
        seed=seed + 7777,
        device=device,
        mode_inject_after_steps={"A_teacher": None, "D_rollout": []},
        sample_from_model=False,
    )

    inj_path = out_dir / "injection_diag.json"
    with open(inj_path, "w") as f:
        json.dump(inj_result, f, indent=2)
    logger.info(f"[exp1b] Wrote injection diagnostic: {inj_path}")

    # Summary
    summary = {
        "run_id": run_id,
        "checkpoint_in": str(ckpt_path),
        "checkpoint_out": str(final_ckpt),
        "rollout_train": {
            "k": k,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "lr": lr,
            "clip_dz_l2": clip_dz_l2,
            "rollout_weight": rollout_weight,
            "teacher_weight": teacher_weight,
        },
        "injection_diag": inj_result,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"[exp1b] Wrote summary: {summary_path}")

    km = {"final_loss": float(loss.item()) if 'loss' in dir() else None}
    d_mode = inj_result.get("modes", {}).get("D_rollout", {})
    d_steps = d_mode.get("per_step", [])
    if d_steps:
        km["D_rollout_cos_final"] = d_steps[-1].get("cos")
    finalize_run(run, key_metrics=km)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
