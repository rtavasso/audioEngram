"""
CALM training loop with consistency loss, EMA updates, checkpointing,
and audio sample generation.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from calm.model import CALM

logger = logging.getLogger("phase0")


def train_calm(
    *,
    model: CALM,
    dataloader: torch.utils.data.DataLoader,
    out_dir: Path,
    device: torch.device,
    max_steps: int = 50000,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip_norm: float = 1.0,
    head_batch_mult: int = 4,
    amp: bool = True,
    log_every: int = 100,
    save_every: int = 5000,
    sample_every: int = 10000,
    # For sample generation
    eval_dataloader: torch.utils.data.DataLoader | None = None,
    prompt_frames: int = 38,
    generate_frames: int = 125,
    temperature: float = 0.8,
    n_samples: int = 5,
) -> dict:
    """
    Train CALM model with consistency loss.

    Returns dict with final metrics and checkpoint path.
    """
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    # fp16 AMP
    use_amp = amp and device.type == "cuda"
    scaler = torch.GradScaler("cuda") if use_amp else None

    def make_autocast():
        return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp)

    param_counts = model.param_count()
    logger.info(
        f"[calm] Training: max_steps={max_steps} lr={lr} "
        f"head_batch_mult={head_batch_mult} amp={use_amp} "
        f"params={param_counts}"
    )

    # Accumulators
    loss_accum = 0.0
    res_mse_accum = 0.0
    log_count = 0
    step = 0
    data_iter = iter(dataloader)

    while step < max_steps:
        try:
            x_seq = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x_seq = next(data_iter)

        x_seq = x_seq.to(device, non_blocking=True)  # [B, S, D]

        optimizer.zero_grad(set_to_none=True)

        with make_autocast():
            Z, x_targets = model.compute_conditioning(x_seq)
            loss, raw_mse = model.consistency_loss(x_targets, Z, head_batch_mult=head_batch_mult)

        if not torch.isfinite(loss):
            logger.warning(f"[calm] Non-finite loss at step {step}, skipping")
            optimizer.zero_grad(set_to_none=True)
            step += 1
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=grad_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=grad_clip_norm,
            )
            optimizer.step()

        # EMA update
        model.update_ema()

        step += 1
        loss_accum += float(loss.item())
        res_mse_accum += raw_mse
        log_count += 1

        if log_every and step % log_every == 0:
            avg_loss = loss_accum / max(log_count, 1)
            avg_res_mse = res_mse_accum / max(log_count, 1)
            logger.info(
                f"[calm] step={step}/{max_steps} loss={avg_loss:.6f} "
                f"consistency_res_mse={avg_res_mse:.6f}"
            )
            loss_accum = 0.0
            res_mse_accum = 0.0
            log_count = 0

        if save_every and step % save_every == 0:
            save_checkpoint(model, optimizer, scaler, step, ckpt_dir / f"calm_step{step}.pt")

        if sample_every and step % sample_every == 0 and eval_dataloader is not None:
            _generate_samples(
                model=model,
                eval_dataloader=eval_dataloader,
                out_dir=out_dir,
                step=step,
                prompt_frames=prompt_frames,
                generate_frames=generate_frames,
                temperature=temperature,
                n_samples=n_samples,
                device=device,
            )

    # Final checkpoint and samples
    final_path = ckpt_dir / "calm_final.pt"
    save_checkpoint(model, optimizer, scaler, step, final_path)

    if eval_dataloader is not None:
        _generate_samples(
            model=model,
            eval_dataloader=eval_dataloader,
            out_dir=out_dir,
            step=step,
            prompt_frames=prompt_frames,
            generate_frames=generate_frames,
            temperature=temperature,
            n_samples=n_samples,
            device=device,
        )

    logger.info(f"[calm] Training done at step {step}. Final checkpoint: {final_path}")
    return {
        "final_checkpoint": str(final_path),
        "steps": step,
    }


def save_checkpoint(
    model: CALM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.GradScaler | None,
    step: int,
    path: Path,
) -> None:
    """Save full training checkpoint."""
    state = {
        "model": model.state_dict(),
        "head_ema": model.head_ema.state_dict(),
        "w_psi": model.w_psi.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "model_kwargs": {
            "latent_dim": model.latent_dim,
            "d_model": model.d_model,
            "ema_decay": model.ema_decay,
        },
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)
    logger.info(f"[calm] Saved checkpoint at step {step}: {path}")


def load_checkpoint(
    path: Path,
    model: CALM,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.GradScaler | None = None,
    device: torch.device = torch.device("cpu"),
) -> int:
    """Load checkpoint into model. Returns the step number."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.head_ema.load_state_dict(ckpt["head_ema"])
    model.w_psi.load_state_dict(ckpt["w_psi"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    step = ckpt.get("step", 0)
    logger.info(f"[calm] Loaded checkpoint from step {step}: {path}")
    return step


# ---------------------------------------------------------------------------
# Rollout evaluation
# ---------------------------------------------------------------------------

def evaluate_rollout(
    model: CALM,
    eval_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    prompt_frames: int = 38,
    rollout_lengths: list[int] | None = None,
    n_utterances: int = 20,
    temperature: float = 0.8,
) -> dict:
    """
    Evaluate rollout quality at various generation lengths.

    Returns dict with per-length MSE and cosine similarity vs ground truth.
    """
    if rollout_lengths is None:
        rollout_lengths = [1, 2, 4, 8, 16]

    model.eval()
    results = {}

    eval_iter = iter(eval_dataloader)
    sequences = []
    for _ in range(n_utterances):
        try:
            x = next(eval_iter)
        except StopIteration:
            break
        sequences.append(x)

    for k in rollout_lengths:
        mse_list = []
        cos_list = []

        for x_batch in sequences:
            x_batch = x_batch.to(device)
            B, S, D = x_batch.shape

            if S < prompt_frames + k:
                continue

            for b in range(B):
                x_full = x_batch[b:b+1]  # [1, S, D]
                prompt = x_full[:, :prompt_frames]  # [1, P, D]
                gt = x_full[0, prompt_frames:prompt_frames + k]  # [k, D]

                with torch.no_grad():
                    generated = model.generate(prompt, k, temperature=temperature)  # [k, D]

                mse = float(((generated - gt) ** 2).mean().item())
                mse_list.append(mse)

                # Cosine similarity of deltas at the LAST generated frame
                # This shows degradation: k=1 compares frame 1, k=16 compares frame 16
                last = k - 1
                prev_frame = gt[last - 1] if last > 0 else x_full[0, prompt_frames - 1]
                gen_delta = generated[last] - prev_frame
                gt_delta = gt[last] - prev_frame
                norm_prod = gen_delta.norm() * gt_delta.norm()
                if norm_prod > 1e-8:
                    cos_sim = float(torch.nn.functional.cosine_similarity(
                        gen_delta.unsqueeze(0), gt_delta.unsqueeze(0)
                    ).item())
                else:
                    cos_sim = 0.0
                cos_list.append(cos_sim)

        results[f"k={k}"] = {
            "mse_mean": float(np.mean(mse_list)) if mse_list else float("nan"),
            "mse_std": float(np.std(mse_list)) if mse_list else float("nan"),
            "cos_mean": float(np.mean(cos_list)) if cos_list else float("nan"),
            "cos_std": float(np.std(cos_list)) if cos_list else float("nan"),
            "n_samples": len(mse_list),
        }

    model.train()
    return results


# ---------------------------------------------------------------------------
# Audio sample generation
# ---------------------------------------------------------------------------

def _generate_samples(
    model: CALM,
    eval_dataloader: torch.utils.data.DataLoader,
    out_dir: Path,
    step: int,
    prompt_frames: int,
    generate_frames: int,
    temperature: float,
    n_samples: int,
    device: torch.device,
) -> None:
    """Generate latent continuations and save as .pt files (audio decode done in script)."""
    samples_dir = out_dir / "samples" / f"step_{step:06d}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    eval_iter = iter(eval_dataloader)
    count = 0

    try:
        x_batch = next(eval_iter).to(device)  # [B, S, D]
    except StopIteration:
        model.train()
        return

    B, S, D = x_batch.shape
    for b in range(min(B, n_samples)):
        if S < prompt_frames + 1:
            continue

        prompt = x_batch[b:b+1, :prompt_frames]  # [1, P, D]
        n_gen = min(generate_frames, S - prompt_frames)

        with torch.no_grad():
            generated = model.generate(prompt, n_gen, temperature=temperature)

        # Save prompt + generated as .pt for later audio decode
        full_seq = torch.cat([prompt.squeeze(0), generated], dim=0)  # [P+G, D]
        gt_seq = x_batch[b, :prompt_frames + n_gen]  # [P+G, D]

        torch.save({
            "prompt": prompt.squeeze(0).cpu(),
            "generated": generated.cpu(),
            "full_sequence": full_seq.cpu(),
            "ground_truth": gt_seq.cpu(),
            "prompt_frames": prompt_frames,
        }, samples_dir / f"sample_{count:02d}.pt")
        count += 1

    model.train()
    logger.info(f"[calm] Saved {count} latent samples at step {step} -> {samples_dir}")
