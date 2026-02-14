#!/usr/bin/env python3
"""
Stage 3 - Experiment 10: Discrete Direction AR Model.

Tests whether an AR model predicting discrete direction indices + continuous
magnitudes survives multi-step rollout better than continuous-vector prediction.

Direction: codebook of K unit vectors from exp 8 spherical k-means.
Magnitude: LogNormal (continuous).

Protocol:
  1. Load β-VAE latents + codebook from exp 8
  2. Train FactoredDirectionMagnitudeAR model
  3. Teacher-forced evaluation (direction accuracy, magnitude NLL)
  4. Rollout evaluation at k=1,2,4,8,16 with argmax/categorical/top_p
  5. Perceptual rollout evaluation → decode through VAE+Mimi → 48kHz WAV
  6. Save results

Usage:
  uv run python scripts/tier3_exp10_direction_ar.py \
      --config configs/tier3_exp10_direction_ar.yaml

  # Quick smoke test:
  uv run python scripts/tier3_exp10_direction_ar.py \
      --config configs/tier3_exp10_direction_ar.yaml --max-steps 100 --skip-audio
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
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import yaml

from experiment import register_run, finalize_run
from phase0.data.io import LatentStore
from phase0.features.context import get_context_flat
from phase0.features.normalization import compute_delta
from phase0.utils.logging import setup_logging
from phase0.utils.seed import set_seed, get_rng
from phase1.data import (
    Phase1Sample,
    iter_phase1_samples,
    BufferedShuffle,
    sample_eval_utterances,
)
from phase1.direction_ar import FactoredDirectionMagnitudeAR
from phase1.train_eval import Phase1IterableDataset, _device_from_config


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Collate: Phase1Sample -> batch with direction indices + magnitudes
# ---------------------------------------------------------------------------


def _make_collate(codebook_np: np.ndarray, near_zero_threshold: float):
    """Create a collate function that computes direction indices on-the-fly."""

    def _collate(samples: list[Phase1Sample]) -> dict:
        ctx = np.stack([s.context_flat for s in samples], axis=0).astype(np.float32, copy=False)
        dx = np.stack([s.delta for s in samples], axis=0).astype(np.float32, copy=False)

        magnitudes = np.linalg.norm(dx, axis=1)  # [B]
        safe_mag = np.maximum(magnitudes, 1e-8)
        directions = dx / safe_mag[:, None]  # [B, D]

        # Nearest codebook entry: argmax(codebook @ direction^T)
        cos_sim = directions @ codebook_np.T  # [B, K]
        dir_indices = np.argmax(cos_sim, axis=1).astype(np.int64)  # [B]

        return {
            "context": torch.from_numpy(ctx),
            "delta": torch.from_numpy(dx),
            "magnitude": torch.from_numpy(magnitudes.astype(np.float32, copy=False)),
            "dir_index": torch.from_numpy(dir_indices),
        }

    return _collate


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_model(
    *,
    model: FactoredDirectionMagnitudeAR,
    cfg: dict,
    codebook_np: np.ndarray,
    near_zero_threshold: float,
    device: torch.device,
    out_dir: Path,
    logger,
) -> dict:
    """Train the factored direction+magnitude AR model."""
    data_cfg = cfg["data"]
    ctx_cfg = cfg["context"]
    train_cfg = cfg["train"]
    seed = int(cfg.get("seed", 42))

    window_size = int(ctx_cfg["window_size"])
    horizon_k = int(ctx_cfg["horizon_k"])
    batch_size = int(train_cfg["batch_size"])
    max_steps = int(train_cfg["max_steps"])
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])
    grad_clip_norm = float(train_cfg["grad_clip_norm"])
    log_every = int(train_cfg["log_every"])
    eval_every = int(train_cfg["eval_every"])
    save_every = int(train_cfg["save_every"])
    shuffle_buffer = int(train_cfg["shuffle_buffer"])
    num_workers = int(train_cfg["num_workers"])

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    collate_fn = _make_collate(codebook_np, near_zero_threshold)

    # Train iterator
    base_it = lambda: iter_phase1_samples(
        frames_index_path=data_cfg["frames_index"],
        latents_dir=data_cfg["latents_dir"],
        split="train",
        window_size=window_size,
        horizon_k=horizon_k,
    )
    shuffler = BufferedShuffle(buffer_size=shuffle_buffer, seed=seed + 10_000)
    train_it = lambda: shuffler(base_it())
    train_ds = Phase1IterableDataset(train_it)

    train_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": (device.type == "cuda"),
    }
    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = 2
    train_loader = torch.utils.data.DataLoader(train_ds, **train_loader_kwargs)

    # Eval loader
    eval_ds = Phase1IterableDataset(
        lambda: iter_phase1_samples(
            frames_index_path=data_cfg["frames_index"],
            latents_dir=data_cfg["latents_dir"],
            split="eval",
            window_size=window_size,
            horizon_k=horizon_k,
            max_samples=5000,
        )
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=batch_size, num_workers=0,
        collate_fn=collate_fn, pin_memory=(device.type == "cuda"),
    )

    # Optimizer
    opt_kwargs = {"lr": lr, "weight_decay": weight_decay}
    if device.type == "cuda":
        opt_kwargs["fused"] = True
    try:
        opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
    except TypeError:
        opt_kwargs.pop("fused", None)
        opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)

    logger.info(
        f"[exp10] Training: K={model.K} D={model.output_dim} "
        f"hidden={model.hidden_dim} layers={model.n_hidden_layers} "
        f"batch={batch_size} steps={max_steps} lr={lr}"
    )

    step = 0
    model.train()
    train_iter = iter(train_loader)
    train_losses = []

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ctx = batch["context"].to(device, non_blocking=True)
        dir_index = batch["dir_index"].to(device, non_blocking=True)
        magnitude = batch["magnitude"].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        loss = model._nll_from_targets(ctx, dir_index, magnitude).mean()
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        opt.step()

        step += 1
        train_losses.append(float(loss.item()))

        if log_every and step % log_every == 0:
            avg_loss = np.mean(train_losses[-log_every:])
            logger.info(f"[exp10] step={step}/{max_steps} train_nll={avg_loss:.4f}")

        if save_every and step % save_every == 0:
            _save_checkpoint(model, ckpt_dir / f"direction_ar_step{step}.pt", step)

        if eval_every and step % eval_every == 0:
            eval_metrics = _eval_teacher_forced(model, eval_loader, device)
            logger.info(
                f"[exp10] step={step} eval: dir_acc={eval_metrics['dir_top1_acc']:.4f} "
                f"dir_top5_acc={eval_metrics['dir_top5_acc']:.4f} "
                f"mag_nll={eval_metrics['mag_nll']:.4f} "
                f"total_nll={eval_metrics['total_nll']:.4f}"
            )

    # Final checkpoint
    final_ckpt_path = ckpt_dir / "direction_ar_final.pt"
    _save_checkpoint(model, final_ckpt_path, step)

    # Final eval
    eval_metrics = _eval_teacher_forced(model, eval_loader, device)
    logger.info(
        f"[exp10] Final eval: dir_acc={eval_metrics['dir_top1_acc']:.4f} "
        f"dir_top5_acc={eval_metrics['dir_top5_acc']:.4f} "
        f"mag_nll={eval_metrics['mag_nll']:.4f} total_nll={eval_metrics['total_nll']:.4f}"
    )

    return {
        "final_step": step,
        "final_train_nll": float(np.mean(train_losses[-100:])) if train_losses else float("nan"),
        "eval": eval_metrics,
        "checkpoint": str(final_ckpt_path),
    }


def _save_checkpoint(model: FactoredDirectionMagnitudeAR, path: Path, step: int) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "model_type": "direction_ar",
            "step": step,
            "model_kwargs": model.get_model_kwargs(),
            "K": model.K,
            "output_dim": model.output_dim,
            "input_dim": model.input_dim,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Teacher-forced evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def _eval_teacher_forced(
    model: FactoredDirectionMagnitudeAR,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Teacher-forced evaluation: direction accuracy, magnitude NLL."""
    model.eval()

    n = 0
    ce_sum = 0.0
    mag_nll_sum = 0.0
    total_nll_sum = 0.0
    top1_correct = 0
    top5_correct = 0

    for batch in loader:
        ctx = batch["context"].to(device)
        dir_index = batch["dir_index"].to(device)
        magnitude = batch["magnitude"].to(device)
        bsz = ctx.shape[0]

        out = model(ctx)

        # Direction accuracy
        pred_top1 = out.dir_logits.argmax(dim=-1)
        top1_correct += int((pred_top1 == dir_index).sum().item())

        _, pred_top5 = out.dir_logits.topk(5, dim=-1)
        top5_match = (pred_top5 == dir_index.unsqueeze(-1)).any(dim=-1)
        top5_correct += int(top5_match.sum().item())

        # Losses
        ce = torch.nn.functional.cross_entropy(out.dir_logits, dir_index, reduction="sum")
        ce_sum += float(ce.item())

        eps = 1e-8
        sigma = torch.exp(out.log_sigma_logm).clamp_min(eps)
        m = magnitude.clamp_min(eps)
        z = (torch.log(m) - out.mu_logm) / sigma
        log2pi = 1.8378770664093453
        mag_nll = torch.log(m) + torch.log(sigma) + 0.5 * (z * z + log2pi)
        mag_nll_sum += float(mag_nll.sum().item())

        total_nll_sum += float(ce.item()) + float(mag_nll.sum().item())
        n += bsz

    model.train()

    if n == 0:
        return {"n": 0, "dir_top1_acc": 0.0, "dir_top5_acc": 0.0, "dir_ce": 0.0, "mag_nll": 0.0, "total_nll": 0.0}

    return {
        "n": n,
        "dir_top1_acc": top1_correct / n,
        "dir_top5_acc": top5_correct / n,
        "dir_ce": ce_sum / n,
        "mag_nll": mag_nll_sum / n,
        "total_nll": total_nll_sum / n,
    }


# ---------------------------------------------------------------------------
# Rollout evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def rollout_evaluation(
    *,
    model: FactoredDirectionMagnitudeAR,
    cfg: dict,
    device: torch.device,
    logger,
) -> dict:
    """Run multi-step rollout evaluation at various k values and sampling strategies."""
    data_cfg = cfg["data"]
    ctx_cfg = cfg["context"]
    rollout_cfg = cfg["rollout"]

    window_size = int(ctx_cfg["window_size"])
    horizon_k = int(ctx_cfg["horizon_k"])
    k_values = [int(k) for k in rollout_cfg["k_values"]]
    strategies = [str(s) for s in rollout_cfg["sampling_strategies"]]
    top_p = float(rollout_cfg.get("top_p", 0.9))
    n_eval_utterances = int(rollout_cfg["n_eval_utterances"])
    segments_per_utt = int(rollout_cfg["segments_per_utt"])
    max_frames_per_utt = int(rollout_cfg["max_frames_per_utt"])
    seed = int(cfg.get("seed", 42))

    store = LatentStore(data_cfg["latents_dir"])
    codebook_np = model.codebook.cpu().numpy()

    utt_ids = sample_eval_utterances(
        splits_dir=data_cfg["splits_dir"],
        latents_index_path=data_cfg["latents_index"],
        n_utterances=n_eval_utterances,
        seed=seed + 30_000,
    )

    if not utt_ids:
        logger.warning("[exp10] No eval utterances found for rollout")
        return {}

    model.eval()
    max_k = max(k_values)
    min_t = max(1, (window_size - 1) + horizon_k)
    rng = get_rng(seed + 40_000)

    # Collect segments
    segments = []
    for utt_id in utt_ids:
        if utt_id not in store:
            continue
        x = store.get_latents(utt_id).astype(np.float32, copy=False)
        t_total = int(min(x.shape[0], max_frames_per_utt))
        min_len = min_t + max_k + 2
        if t_total < min_len:
            continue

        t0_max = t_total - max_k - 1
        if t0_max <= min_t:
            continue
        n_segs = min(segments_per_utt, t0_max - min_t)
        starts = rng.choice(np.arange(min_t, t0_max, dtype=np.int64), size=n_segs, replace=False)
        for t0 in starts:
            segments.append((x, int(t0)))

    logger.info(f"[exp10] Rollout: {len(segments)} segments from {len(utt_ids)} utterances")

    results = {}
    per_step_rows = []

    for strategy in strategies:
        for k_max in k_values:
            # Per-step accumulators
            step_n = [0] * k_max
            step_dir_correct = [0] * k_max
            step_mag_err_sum = [0.0] * k_max
            step_state_err_sum = [0.0] * k_max
            step_cos_sum = [0.0] * k_max

            for x_np, t0 in segments:
                if t0 + k_max >= x_np.shape[0]:
                    continue

                x_t = torch.from_numpy(x_np).to(device)
                D = x_t.shape[1]

                # Build initial context window ending at t0 - horizon_k
                ctx_end = t0 - horizon_k
                ctx_start = ctx_end - window_size + 1
                if ctx_start < 0:
                    continue
                ctx_window = x_t[ctx_start:ctx_end + 1].clone()  # [W, D]

                x_hat_prev = x_t[t0 - 1].clone()  # state before first step

                for s in range(k_max):
                    t = t0 + s

                    # GT delta and direction index
                    dx_true = x_t[t] - x_t[t - 1]
                    mag_true = torch.linalg.vector_norm(dx_true).clamp_min(1e-8)
                    dir_true = dx_true / mag_true
                    cos_sim = dir_true @ model.codebook.T
                    gt_dir_idx = cos_sim.argmax().item()

                    # Predict
                    ctx_flat = ctx_window.reshape(1, -1)  # [1, W*D]
                    dir_idx, magnitude = model.sample(ctx_flat, strategy=strategy, top_p=top_p)
                    dir_idx = dir_idx[0]
                    magnitude = magnitude[0]

                    # Direction accuracy
                    pred_dir = model.codebook[dir_idx]
                    step_dir_correct[s] += int(dir_idx.item() == gt_dir_idx)

                    # Magnitude error
                    step_mag_err_sum[s] += abs(float(magnitude.item()) - float(mag_true.item()))

                    # Advance state
                    dx_hat = pred_dir * magnitude
                    x_hat_curr = x_hat_prev + dx_hat

                    # State error
                    state_err = float(torch.linalg.vector_norm(x_hat_curr - x_t[t]).item())
                    step_state_err_sum[s] += state_err

                    # Direction cosine of predicted delta vs GT delta
                    cos_val = float((dx_hat @ dx_true).item() / (
                        torch.linalg.vector_norm(dx_hat).clamp_min(1e-8).item() *
                        torch.linalg.vector_norm(dx_true).clamp_min(1e-8).item()
                    ))
                    step_cos_sum[s] += cos_val

                    step_n[s] += 1
                    x_hat_prev = x_hat_curr

                    # Update context window: slide by 1, append new frame
                    if s < k_max - 1:
                        # The new frame entering the window is the predicted state
                        # (at the position that would correspond to t - horizon_k + 1 in rollout)
                        if s < horizon_k:
                            # Early steps: still using GT for context window update
                            new_frame = x_t[t0 + s - horizon_k + 1] if t0 + s - horizon_k + 1 >= 0 else x_hat_curr
                        else:
                            new_frame = x_hat_curr
                        ctx_window = torch.cat([ctx_window[1:], new_frame.unsqueeze(0)], dim=0)

            # Aggregate
            key = f"{strategy}_k{k_max}"
            per_k_results = {}
            for s in range(k_max):
                if step_n[s] == 0:
                    continue
                n = step_n[s]
                step_result = {
                    "step": s + 1,
                    "strategy": strategy,
                    "k_max": k_max,
                    "n": n,
                    "dir_top1_acc": step_dir_correct[s] / n,
                    "mag_abs_err": step_mag_err_sum[s] / n,
                    "state_err": step_state_err_sum[s] / n,
                    "direction_cosine": step_cos_sum[s] / n,
                }
                per_step_rows.append(step_result)
                per_k_results[f"step_{s+1}"] = step_result

            # Summary for this (strategy, k) combination
            total_n = sum(step_n)
            if total_n > 0:
                results[key] = {
                    "strategy": strategy,
                    "k_max": k_max,
                    "n_total": total_n,
                    "mean_dir_acc": sum(step_dir_correct) / total_n,
                    "mean_state_err": sum(step_state_err_sum) / total_n,
                    "mean_direction_cosine": sum(step_cos_sum) / total_n,
                    "final_step_state_err": step_state_err_sum[-1] / max(step_n[-1], 1),
                    "final_step_dir_acc": step_dir_correct[-1] / max(step_n[-1], 1),
                    "per_step": per_k_results,
                }

            logger.info(
                f"[exp10] Rollout {key}: n={total_n} "
                f"mean_dir_acc={sum(step_dir_correct)/max(total_n,1):.4f} "
                f"final_state_err={step_state_err_sum[-1]/max(step_n[-1],1):.4f}"
            )

    return {
        "results": results,
        "per_step": per_step_rows,
    }


# ---------------------------------------------------------------------------
# Perceptual rollout evaluation (audio decode)
# ---------------------------------------------------------------------------


def _mel_distance(audio: torch.Tensor, audio_hat: torch.Tensor, sr: int, n_mels: int = 80) -> float:
    mel_transform = T.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=256, n_mels=n_mels,
    ).to(audio.device)
    mel = mel_transform(audio.squeeze())
    mel_hat = mel_transform(audio_hat.squeeze())
    log_mel = torch.log(mel.clamp_min(1e-5))
    log_mel_hat = torch.log(mel_hat.clamp_min(1e-5))
    return float(torch.nn.functional.l1_loss(log_mel_hat, log_mel).item())


def _save_wav(audio: np.ndarray, path: Path, input_sr: int, output_sr: int) -> None:
    audio_t = torch.from_numpy(audio).unsqueeze(0).float()
    if input_sr != output_sr:
        audio_t = torchaudio.functional.resample(audio_t, input_sr, output_sr)
    torchaudio.save(str(path), audio_t, output_sr)


@torch.no_grad()
def perceptual_rollout_eval(
    *,
    model: FactoredDirectionMagnitudeAR,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
    logger,
) -> list[dict]:
    """Decode rollout trajectories through VAE+Mimi → WAV and compute mel distance."""
    from mimi_autoencoder import load_mimi_autoencoder
    from stage2.vae import ARFriendlyVAE
    from stage2.vae_train import load_vae_checkpoint

    data_cfg = cfg["data"]
    ctx_cfg = cfg["context"]
    eval_cfg = cfg["eval"]
    rollout_cfg = cfg["rollout"]

    window_size = int(ctx_cfg["window_size"])
    horizon_k = int(ctx_cfg["horizon_k"])
    audio_k_values = [int(k) for k in eval_cfg["rollout_audio_k_values"]]
    n_utterances = int(eval_cfg["n_utterances"])
    output_sr = int(eval_cfg["output_sample_rate"])
    mimi_sr = 24000
    seed = int(cfg.get("seed", 42))

    audio_dir = out_dir / "rollout_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Load VAE + Mimi decoder
    autoencoder = load_mimi_autoencoder(
        checkpoint_path=eval_cfg.get("mimi_checkpoint"),
        device=str(device),
    )
    vae_ckpt_path = Path(eval_cfg["vae_checkpoint"])
    ckpt_data = torch.load(str(vae_ckpt_path), map_location=device)
    latent_dim = int(ckpt_data.get("latent_dim", 32))
    encoder_dim = int(ckpt_data.get("encoder_dim", 512))
    dec_hidden_dim = ckpt_data.get("dec_hidden_dim")

    if dec_hidden_dim is None:
        vae = ARFriendlyVAE(
            mimi_encoder=autoencoder.encoder,
            mimi_decoder=autoencoder.decoder,
            latent_dim=latent_dim,
            encoder_dim=encoder_dim,
            dec_hidden_dim=256,
        ).to(device)
        vae.dec_proj = torch.nn.Conv1d(latent_dim, encoder_dim, 1).to(device)
        vae.dec_hidden_dim = None
    else:
        vae = ARFriendlyVAE(
            mimi_encoder=autoencoder.encoder,
            mimi_decoder=autoencoder.decoder,
            latent_dim=latent_dim,
            encoder_dim=encoder_dim,
            dec_hidden_dim=int(dec_hidden_dim),
        ).to(device)
    load_vae_checkpoint(vae_ckpt_path, vae, device=device)
    vae.eval()
    logger.info(f"[exp10] Loaded VAE decoder (latent_dim={latent_dim})")

    def _decode_to_audio(z: np.ndarray) -> np.ndarray:
        """z: [T, D_vae] -> audio [T_audio]"""
        z_torch = torch.from_numpy(z.T.copy()).unsqueeze(0).float().to(device)
        with torch.inference_mode():
            audio = vae.decode(z_torch)
        return audio.squeeze().cpu().numpy()

    # Sample eval utterances
    store = LatentStore(data_cfg["latents_dir"])
    utt_ids = sample_eval_utterances(
        splits_dir=data_cfg["splits_dir"],
        latents_index_path=data_cfg["latents_index"],
        n_utterances=n_utterances,
        seed=seed + 50_000,
    )

    model.eval()
    min_t = max(1, (window_size - 1) + horizon_k)
    max_k = max(audio_k_values)
    rng = get_rng(seed + 60_000)
    results = []

    for i, utt_id in enumerate(utt_ids):
        if utt_id not in store:
            continue
        x_true = store.get_latents(utt_id).astype(np.float32, copy=False)
        t_total = x_true.shape[0]
        if t_total < min_t + max_k + 20:
            continue

        # Pick a start point with enough margin for prefix + rollout + suffix
        prefix_len = 10
        suffix_len = 10
        t_start_min = min_t + prefix_len
        t_start_max = t_total - max_k - suffix_len
        if t_start_max <= t_start_min:
            continue
        t_start = int(rng.integers(t_start_min, t_start_max))

        # Decode GT
        try:
            audio_gt = _decode_to_audio(x_true)
            _save_wav(audio_gt, audio_dir / f"utt{i:02d}_{utt_id}_GT.wav", mimi_sr, output_sr)
            audio_gt_torch = torch.from_numpy(audio_gt).float().to(device)
        except Exception as e:
            logger.warning(f"[exp10] GT decode failed for {utt_id}: {e}")
            continue

        for k_rollout in audio_k_values:
            if t_start + k_rollout >= t_total:
                continue

            # Run rollout from t_start for k_rollout steps using argmax
            x_t = torch.from_numpy(x_true).to(device)
            ctx_end = t_start - horizon_k
            ctx_start = ctx_end - window_size + 1
            if ctx_start < 0:
                continue
            ctx_window = x_t[ctx_start:ctx_end + 1].clone()

            x_hat_prev = x_t[t_start - 1].clone()
            rollout_frames = []

            for s in range(k_rollout):
                ctx_flat = ctx_window.reshape(1, -1)
                dir_idx, magnitude = model.sample(ctx_flat, strategy="argmax")
                dx_hat = model.reconstruct_delta(dir_idx, magnitude)[0]
                x_hat_curr = x_hat_prev + dx_hat
                rollout_frames.append(x_hat_curr.cpu().numpy())
                x_hat_prev = x_hat_curr

                if s < k_rollout - 1:
                    if s < horizon_k:
                        new_idx = t_start + s - horizon_k + 1
                        new_frame = x_t[new_idx] if new_idx >= 0 else x_hat_curr
                    else:
                        new_frame = x_hat_curr
                    ctx_window = torch.cat([ctx_window[1:], new_frame.unsqueeze(0)], dim=0)

            # Splice: GT prefix → rollout → GT suffix
            rollout_arr = np.stack(rollout_frames, axis=0)  # [k, D]
            spliced = np.concatenate([
                x_true[t_start - prefix_len:t_start],     # prefix
                rollout_arr,                               # rollout
                x_true[t_start + k_rollout:t_start + k_rollout + suffix_len],  # suffix
            ], axis=0)

            try:
                audio_spliced = _decode_to_audio(spliced)
                wav_path = audio_dir / f"utt{i:02d}_{utt_id}_rollout_k{k_rollout:02d}.wav"
                _save_wav(audio_spliced, wav_path, mimi_sr, output_sr)

                # Mel distance vs GT segment
                gt_segment = x_true[t_start - prefix_len:t_start + k_rollout + suffix_len]
                audio_gt_seg = _decode_to_audio(gt_segment)
                min_len = min(len(audio_gt_seg), len(audio_spliced))
                mel_dist = _mel_distance(
                    torch.from_numpy(audio_gt_seg[:min_len]).float().to(device),
                    torch.from_numpy(audio_spliced[:min_len]).float().to(device),
                    mimi_sr,
                )

                results.append({
                    "utterance_id": utt_id,
                    "k_rollout": k_rollout,
                    "t_start": t_start,
                    "mel_distance": mel_dist,
                    "wav_path": str(wav_path),
                })
                logger.info(f"[exp10] Audio utt={utt_id} k={k_rollout}: mel_dist={mel_dist:.4f}")
            except Exception as e:
                logger.warning(f"[exp10] Decode failed for {utt_id} k={k_rollout}: {e}")

    # Clean up
    del autoencoder, vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description="Exp 10: Discrete Direction AR Model")
    p.add_argument("--config", type=str, default="configs/tier3_exp10_direction_ar.yaml")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--exp8-dir", type=str, default=None, help="Override exp 8 output dir")
    p.add_argument("--k-codebook", type=int, default=None, help="Override codebook K")
    p.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    p.add_argument("--skip-audio", action="store_true", help="Skip perceptual audio eval")
    args = p.parse_args()

    if os.environ.get("NO_TORCH_COMPILE"):
        os.environ["TORCH_COMPILE_DISABLE"] = "1"

    logger = setup_logging(name="phase0")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if args.k_codebook is not None:
        cfg["codebook"]["K"] = args.k_codebook
    if args.exp8_dir is not None:
        cfg["codebook"]["exp8_dir"] = args.exp8_dir

    run_id = args.run_id or _default_run_id()
    out_root = Path(cfg["output"]["out_dir"])
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    run = register_run(
        experiment="exp10_direction_ar",
        run_id=run_id,
        config_path=args.config,
        config=cfg,
        cli_args=sys.argv[1:],
        out_dir=out_dir,
        log_name="phase0",
    )

    set_seed(int(cfg.get("seed", 42)))
    device = _device_from_config(cfg["train"]["device"])

    # -----------------------------------------------------------------------
    # Step 1: Load codebook from exp 8
    # -----------------------------------------------------------------------
    codebook_cfg = cfg["codebook"]
    exp8_dir = Path(codebook_cfg["exp8_dir"])
    K = int(codebook_cfg["K"])

    # Try both naming conventions (older exp8 used codebook_K0256.npy, newer uses codebook_normalized_K0256.npy)
    codebook_path = exp8_dir / f"codebook_K{K:04d}.npy"
    if not codebook_path.exists():
        codebook_path = exp8_dir / f"codebook_normalized_K{K:04d}.npy"
    if not codebook_path.exists():
        logger.error(f"[exp10] Codebook not found at {exp8_dir}/codebook_K{K:04d}.npy")
        return 1

    codebook_np = np.load(str(codebook_path)).astype(np.float32)
    logger.info(f"[exp10] Loaded codebook: {codebook_path} shape={codebook_np.shape}")

    # Verify unit norms
    norms = np.linalg.norm(codebook_np, axis=1)
    logger.info(f"[exp10] Codebook norm range: [{norms.min():.6f}, {norms.max():.6f}]")

    D = codebook_np.shape[1]  # output_dim (32 for β-VAE)
    window_size = int(cfg["context"]["window_size"])
    input_dim = window_size * D

    # Compute near-zero threshold from magnitude distribution
    nz_frac = float(codebook_cfg.get("near_zero_threshold_frac", 0.01))
    mag_dist_path = exp8_dir / "magnitude_distribution.json"
    if mag_dist_path.exists():
        with open(mag_dist_path) as f:
            mag_dist = json.load(f)
        median_mag = float(mag_dist["median"])
    else:
        # Fallback: compute from data
        store = LatentStore(cfg["data"]["latents_dir"])
        sample_utts = sample_eval_utterances(
            splits_dir=cfg["data"]["splits_dir"],
            latents_index_path=cfg["data"]["latents_index"],
            n_utterances=100,
            seed=42,
        )
        all_mags = []
        for uid in sample_utts[:50]:
            if uid not in store:
                continue
            x = store.get_latents(uid).astype(np.float32, copy=False)
            if x.shape[0] < 2:
                continue
            dx = x[1:] - x[:-1]
            all_mags.append(np.linalg.norm(dx, axis=1))
        median_mag = float(np.median(np.concatenate(all_mags)))
        logger.info(f"[exp10] Computed median magnitude from data: {median_mag:.6f}")

    near_zero_threshold = nz_frac * median_mag
    logger.info(f"[exp10] Near-zero threshold: {near_zero_threshold:.6f} (frac={nz_frac}, median={median_mag:.6f})")

    # -----------------------------------------------------------------------
    # Step 2: Build model
    # -----------------------------------------------------------------------
    model_cfg = cfg["model"]
    model = FactoredDirectionMagnitudeAR(
        input_dim=input_dim,
        output_dim=D,
        K=K,
        codebook=codebook_np,
        hidden_dim=int(model_cfg["hidden_dim"]),
        n_hidden_layers=int(model_cfg["n_hidden_layers"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
        min_log_sigma_logm=float(model_cfg.get("min_log_sigma_logm", -5.0)),
        max_log_sigma_logm=float(model_cfg.get("max_log_sigma_logm", 0.7)),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[exp10] Model: {n_params:,} parameters, K={K}, D={D}, input_dim={input_dim}")

    # -----------------------------------------------------------------------
    # Step 3: Train
    # -----------------------------------------------------------------------
    train_result = train_model(
        model=model,
        cfg=cfg,
        codebook_np=codebook_np,
        near_zero_threshold=near_zero_threshold,
        device=device,
        out_dir=out_dir,
        logger=logger,
    )

    with open(out_dir / "teacher_forced_metrics.json", "w") as f:
        json.dump(train_result, f, indent=2)

    # -----------------------------------------------------------------------
    # Step 4: Rollout evaluation
    # -----------------------------------------------------------------------
    logger.info("[exp10] Running rollout evaluation...")
    rollout_result = rollout_evaluation(
        model=model,
        cfg=cfg,
        device=device,
        logger=logger,
    )

    with open(out_dir / "rollout_metrics.json", "w") as f:
        json.dump(rollout_result.get("results", {}), f, indent=2, default=str)

    # Save per-step CSV
    if rollout_result.get("per_step"):
        per_step_df = pd.DataFrame(rollout_result["per_step"])
        per_step_df.to_csv(str(out_dir / "per_step.csv"), index=False)

    # -----------------------------------------------------------------------
    # Step 5: Perceptual rollout evaluation (audio)
    # -----------------------------------------------------------------------
    audio_results = []
    if not args.skip_audio:
        logger.info("[exp10] Running perceptual rollout evaluation (audio decode)...")
        audio_results = perceptual_rollout_eval(
            model=model,
            cfg=cfg,
            device=device,
            out_dir=out_dir,
            logger=logger,
        )
        with open(out_dir / "audio_eval_results.json", "w") as f:
            json.dump(audio_results, f, indent=2)
    else:
        logger.info("[exp10] Skipping audio eval (--skip-audio)")

    # -----------------------------------------------------------------------
    # Step 6: Summary
    # -----------------------------------------------------------------------
    summary_rows = []

    # Teacher-forced row
    eval_m = train_result.get("eval", {})
    summary_rows.append({
        "condition": "teacher_forced",
        "dir_top1_acc": eval_m.get("dir_top1_acc"),
        "dir_top5_acc": eval_m.get("dir_top5_acc"),
        "dir_ce": eval_m.get("dir_ce"),
        "mag_nll": eval_m.get("mag_nll"),
        "total_nll": eval_m.get("total_nll"),
    })

    # Rollout rows
    for key, rdata in rollout_result.get("results", {}).items():
        summary_rows.append({
            "condition": key,
            "strategy": rdata.get("strategy"),
            "k_max": rdata.get("k_max"),
            "n": rdata.get("n_total"),
            "mean_dir_acc": rdata.get("mean_dir_acc"),
            "mean_state_err": rdata.get("mean_state_err"),
            "mean_direction_cosine": rdata.get("mean_direction_cosine"),
            "final_dir_acc": rdata.get("final_step_dir_acc"),
            "final_state_err": rdata.get("final_step_state_err"),
        })

    # Audio rows
    if audio_results:
        audio_df = pd.DataFrame(audio_results)
        for k_val in audio_df["k_rollout"].unique():
            k_data = audio_df[audio_df["k_rollout"] == k_val]
            summary_rows.append({
                "condition": f"audio_k{k_val}",
                "mel_distance_mean": float(k_data["mel_distance"].mean()),
                "mel_distance_std": float(k_data["mel_distance"].std()),
                "n_audio": len(k_data),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(str(out_dir / "summary.csv"), index=False)
    logger.info(f"[exp10] Summary:\n{summary_df.to_string(index=False)}")

    # Key metrics for finalize
    key_metrics = {
        "dir_top1_acc": eval_m.get("dir_top1_acc", 0.0),
        "dir_top5_acc": eval_m.get("dir_top5_acc", 0.0),
        "total_nll": eval_m.get("total_nll", 0.0),
        "n_params": n_params,
        "K": K,
        "D": D,
    }
    # Add best rollout metric
    for key, rdata in rollout_result.get("results", {}).items():
        if "argmax_k4" in key:
            key_metrics["rollout_k4_dir_acc"] = rdata.get("final_step_dir_acc", 0.0)
            key_metrics["rollout_k4_state_err"] = rdata.get("final_step_state_err", 0.0)

    if audio_results:
        audio_df = pd.DataFrame(audio_results)
        key_metrics["audio_mel_mean"] = float(audio_df["mel_distance"].mean())

    finalize_run(run, key_metrics=key_metrics)
    logger.info(f"[exp10] Done. Results at {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
