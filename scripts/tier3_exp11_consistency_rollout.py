#!/usr/bin/env python3
"""
Stage 3 - Experiment 11: Consistency Model for Rollout Stability.

Trains a one-step Tweedie denoiser on 32D beta-VAE latents to correct AR
prediction errors after each rollout step, extending horizon before divergence.

Key insight: z_corrected = z + sigma^2 * score(z, sigma)  (one-step Tweedie)
Applied after EACH rollout step to project predictions back onto the data manifold.

Protocol:
  1. Load pre-trained models (Direction AR, MDN, codebook)
  2. Train score model on 32D latents via denoising score matching
  3. Run rollout evaluation across 6 conditions x sigma sweep x k values
  4. Find best sigma per condition
  5. Optional: decode best-sigma rollouts to audio via VAE+Mimi

Conditions:
  1. dir_ar_argmax         - Direction AR, argmax, no correction
  2. dir_ar_argmax_tweedie - Direction AR, argmax, Tweedie correction
  3. dir_ar_categorical         - Direction AR, categorical, no correction
  4. dir_ar_categorical_tweedie - Direction AR, categorical, Tweedie correction
  5. mdn_baseline          - MDN expected_mean, no correction
  6. mdn_tweedie           - MDN expected_mean, Tweedie correction

Usage:
  uv run python scripts/tier3_exp11_consistency_rollout.py \
      --config configs/tier3_exp11_consistency_rollout.yaml

  # Smoke test:
  uv run python scripts/tier3_exp11_consistency_rollout.py \
      --config configs/tier3_exp11_consistency_rollout.yaml \
      --score-max-steps 100 --sigma 0.1 --skip-audio --skip-mdn

  # Joint training (AR + Tweedie end-to-end):
  uv run python scripts/tier3_exp11_consistency_rollout.py \
      --config configs/tier3_exp11_consistency_rollout.yaml \
      --joint-train --joint-train-steps 100 --joint-train-sigma 0.1 \
      --score-max-steps 100 --sigma 0.1 --skip-audio --skip-mdn
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
from phase0.data.io import LatentStore, load_latents_index
from phase0.utils.logging import setup_logging
from phase0.utils.seed import set_seed, get_rng
from phase1.checkpoints import load_phase1_checkpoint
from phase1.data import sample_eval_utterances
from phase1.direction_ar import load_direction_ar_checkpoint
from phase1.train_eval import _device_from_config


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Tweedie correction
# ---------------------------------------------------------------------------


@torch.no_grad()
def tweedie_correct(z, score_model, sigma, device):
    """One-step Tweedie denoiser: z_corrected = z + sigma^2 * score(z, sigma).

    Args:
        z: [B, D] latent to correct
        score_model: trained ScoreNetwork
        sigma: float, noise level for score evaluation
        device: torch device

    Returns:
        z_corrected: [B, D]
    """
    sigma_t = z.new_full((z.shape[0], 1), sigma)
    score = score_model(z, sigma_t)
    return z + (sigma ** 2) * score


# ---------------------------------------------------------------------------
# Joint training: AR + Tweedie end-to-end
# ---------------------------------------------------------------------------


def soft_predict(model, ctx_flat):
    """Differentiable prediction: softmax-weighted direction x median magnitude.

    Unlike model.expected_mean() which uses non-differentiable argmax,
    this uses soft attention over the codebook for gradient flow.

    Args:
        model: FactoredDirectionMagnitudeAR
        ctx_flat: [B, W*D]

    Returns:
        dx_hat: [B, D] differentiable delta prediction
    """
    out = model(ctx_flat)
    dir_probs = torch.softmax(out.dir_logits, dim=-1)  # [B, K]
    direction = dir_probs @ model.codebook  # [B, D]
    magnitude = torch.exp(out.mu_logm)  # [B] median of LogNormal
    return direction * magnitude.unsqueeze(-1)


def joint_train_ar_with_tweedie(
    *,
    model,
    score_model,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
    sigma: float,
    max_steps_override: int | None = None,
    logger,
) -> dict:
    """Fine-tune Direction AR with Tweedie correction in the unrolled loop.

    K-step BPTT: predict -> Tweedie correct -> update context -> repeat.
    Score model is frozen; only AR model parameters are updated.
    Loss: L2 between corrected predictions and GT states at each step.
    """
    joint_cfg = cfg.get("joint_train", {})
    data_cfg = cfg["data"]

    unroll_k = int(joint_cfg.get("unroll_k", 4))
    max_steps = max_steps_override or int(joint_cfg.get("max_steps", 5000))
    batch_size = int(joint_cfg.get("batch_size", 64))
    lr = float(joint_cfg.get("lr", 1e-4))
    weight_decay = float(joint_cfg.get("weight_decay", 1e-4))
    grad_clip_norm = float(joint_cfg.get("grad_clip_norm", 1.0))
    log_every = int(joint_cfg.get("log_every", 100))
    save_every = int(joint_cfg.get("save_every", 1000))
    seed = int(cfg.get("seed", 42))

    window_size = int(cfg["context"]["window_size"])
    horizon_k = int(cfg["context"]["horizon_k"])

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Freeze score model but keep it differentiable-through
    score_model.eval()
    for p in score_model.parameters():
        p.requires_grad_(False)

    # Load training data into memory (32D latents are small: ~100MB)
    store = LatentStore(data_cfg["latents_dir"])
    idx = load_latents_index(data_cfg["latents_index"])
    from phase0.data.splits import load_splits
    splits = load_splits(data_cfg["splits_dir"])
    train_speaker_set = set(splits.train_speakers)
    train_utts = idx[idx["speaker_id"].isin(train_speaker_set)]["utterance_id"].astype(str).tolist()

    min_t = max(1, (window_size - 1) + horizon_k)
    min_len = min_t + unroll_k + 1

    train_tensors = []
    for utt_id in train_utts:
        if utt_id not in store:
            continue
        x = store.get_latents(utt_id).astype(np.float32, copy=False)
        if x.shape[0] >= min_len:
            train_tensors.append(torch.from_numpy(x))

    if not train_tensors:
        logger.error("[exp11] No valid training utterances for joint training")
        return {"status": "failed"}

    logger.info(
        f"[exp11] Joint training: {len(train_tensors)} utterances, "
        f"unroll_k={unroll_k}, sigma={sigma}, batch={batch_size}, "
        f"steps={max_steps}, lr={lr}"
    )

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = get_rng(seed + 70_000)
    losses = []

    for step in range(1, max_steps + 1):
        # Sample batch of segments
        utt_indices = rng.integers(0, len(train_tensors), size=batch_size)

        ctx_windows_list = []
        gt_seqs_list = []
        x_prev_list = []

        for b in range(batch_size):
            x = train_tensors[int(utt_indices[b])]
            t0_max = x.shape[0] - unroll_k
            if t0_max <= min_t:
                t0 = min_t
            else:
                t0 = int(rng.integers(min_t, t0_max))

            ctx_end = t0 - horizon_k
            ctx_start = ctx_end - window_size + 1

            ctx_windows_list.append(x[ctx_start:ctx_end + 1])  # [W, D]
            gt_seqs_list.append(x[t0:t0 + unroll_k])           # [K, D]
            x_prev_list.append(x[t0 - 1])                      # [D]

        ctx_windows = torch.stack(ctx_windows_list).to(device)  # [B, W, D]
        gt_seqs = torch.stack(gt_seqs_list).to(device)          # [B, K, D]
        x_prev = torch.stack(x_prev_list).to(device)            # [B, D]

        B = ctx_windows.shape[0]

        total_loss = torch.tensor(0.0, device=device)

        for s in range(unroll_k):
            ctx_flat = ctx_windows.reshape(B, -1)  # [B, W*D]

            # Differentiable prediction
            dx_hat = soft_predict(model, ctx_flat)  # [B, D]
            x_hat = x_prev + dx_hat

            # Tweedie correction (differentiable through AR model)
            sigma_t = x_hat.new_full((B, 1), sigma)
            score = score_model(x_hat, sigma_t)
            x_corrected = x_hat + (sigma ** 2) * score

            # L2 loss vs GT
            total_loss = total_loss + ((x_corrected - gt_seqs[:, s]) ** 2).sum(dim=-1).mean()

            # Update state and context for next step (full BPTT)
            x_prev = x_corrected
            if s < unroll_k - 1:
                ctx_windows = torch.cat([
                    ctx_windows[:, 1:],
                    x_corrected.unsqueeze(1),
                ], dim=1)

        loss = total_loss / unroll_k

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        losses.append(float(loss.item()))

        if log_every and step % log_every == 0:
            avg_loss = np.mean(losses[-log_every:])
            logger.info(f"[exp11] joint step={step}/{max_steps} loss={avg_loss:.6f}")

        if save_every and step % save_every == 0:
            ckpt_path = ckpt_dir / f"direction_ar_joint_step{step}.pt"
            torch.save({
                "model": model.state_dict(),
                "model_type": "direction_ar",
                "step": step,
                "model_kwargs": model.get_model_kwargs(),
                "K": model.K,
                "output_dim": model.output_dim,
                "input_dim": model.input_dim,
                "joint_trained": True,
                "joint_sigma": sigma,
                "joint_unroll_k": unroll_k,
            }, ckpt_path)

    # Save final checkpoint
    final_path = ckpt_dir / "direction_ar_joint_final.pt"
    torch.save({
        "model": model.state_dict(),
        "model_type": "direction_ar",
        "step": max_steps,
        "model_kwargs": model.get_model_kwargs(),
        "K": model.K,
        "output_dim": model.output_dim,
        "input_dim": model.input_dim,
        "joint_trained": True,
        "joint_sigma": sigma,
        "joint_unroll_k": unroll_k,
    }, final_path)
    logger.info(f"[exp11] Joint training done. Checkpoint: {final_path}")

    model.eval()

    return {
        "final_step": max_steps,
        "final_loss": float(np.mean(losses[-100:])) if losses else float("nan"),
        "checkpoint": str(final_path),
        "sigma": sigma,
        "unroll_k": unroll_k,
    }


# ---------------------------------------------------------------------------
# Rollout evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def rollout_evaluation(
    *,
    dir_ar_model,
    mdn_model,
    score_model,
    cfg: dict,
    device: torch.device,
    sigma_values: list[float],
    skip_mdn: bool,
    logger,
) -> dict:
    """Run multi-step rollout evaluation across conditions and sigma sweep."""
    data_cfg = cfg["data"]
    ctx_cfg = cfg["context"]
    rollout_cfg = cfg["rollout"]

    window_size = int(ctx_cfg["window_size"])
    horizon_k = int(ctx_cfg["horizon_k"])
    k_values = [int(k) for k in rollout_cfg["k_values"]]
    n_eval_utterances = int(rollout_cfg["n_eval_utterances"])
    segments_per_utt = int(rollout_cfg["segments_per_utt"])
    max_frames_per_utt = int(rollout_cfg["max_frames_per_utt"])
    seed = int(cfg.get("seed", 42))

    store = LatentStore(data_cfg["latents_dir"])

    utt_ids = sample_eval_utterances(
        splits_dir=data_cfg["splits_dir"],
        latents_index_path=data_cfg["latents_index"],
        n_utterances=n_eval_utterances,
        seed=seed + 30_000,
    )
    if not utt_ids:
        logger.warning("[exp11] No eval utterances found for rollout")
        return {"all_results": [], "segments": []}

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

    logger.info(f"[exp11] Rollout: {len(segments)} segments from {len(utt_ids)} utterances")

    # Define conditions
    conditions = [
        ("dir_ar_argmax", "dir_ar", "argmax", False),
        ("dir_ar_argmax_tweedie", "dir_ar", "argmax", True),
        ("dir_ar_categorical", "dir_ar", "categorical", False),
        ("dir_ar_categorical_tweedie", "dir_ar", "categorical", True),
    ]
    if not skip_mdn:
        conditions.extend([
            ("mdn_baseline", "mdn", None, False),
            ("mdn_tweedie", "mdn", None, True),
        ])

    all_results = []

    for sigma in sigma_values:
        logger.info(f"[exp11] === Sigma={sigma} ===")

        for cond_name, model_type, sampling_strategy, use_tweedie in conditions:
            # Skip sigma sweep for uncorrected conditions (only run once at sigma=sigma_values[0])
            if not use_tweedie and sigma != sigma_values[0]:
                continue

            for k_max in k_values:
                # Per-step accumulators
                step_n = [0] * k_max
                step_state_err_sum = [0.0] * k_max
                step_cos_sum = [0.0] * k_max
                step_mag_ratio_sum = [0.0] * k_max
                step_nan_count = [0] * k_max

                for x_np, t0 in segments:
                    if t0 + k_max >= x_np.shape[0]:
                        continue

                    x_t = torch.from_numpy(x_np).to(device)
                    D = x_t.shape[1]

                    # Build initial context window
                    ctx_end = t0 - horizon_k
                    ctx_start = ctx_end - window_size + 1
                    if ctx_start < 0:
                        continue
                    ctx_window = x_t[ctx_start:ctx_end + 1].clone()  # [W, D]

                    x_hat_prev = x_t[t0 - 1].clone()

                    for s in range(k_max):
                        t = t0 + s
                        dx_true = x_t[t] - x_t[t - 1]

                        # Predict delta
                        ctx_flat = ctx_window.reshape(1, -1)  # [1, W*D]

                        if model_type == "dir_ar":
                            dir_idx, magnitude = dir_ar_model.sample(
                                ctx_flat, strategy=sampling_strategy,
                            )
                            dx_hat = dir_ar_model.reconstruct_delta(dir_idx, magnitude)[0]
                        else:
                            dx_hat = mdn_model.expected_mean(ctx_flat).squeeze(0)

                        # Advance state
                        x_hat_curr = x_hat_prev + dx_hat

                        # Apply Tweedie correction
                        if use_tweedie:
                            x_hat_curr_2d = x_hat_curr.unsqueeze(0)  # [1, D]
                            x_hat_curr_2d = tweedie_correct(
                                x_hat_curr_2d, score_model, sigma, device,
                            )
                            x_hat_curr = x_hat_curr_2d.squeeze(0)

                        # Check for NaN/Inf
                        if torch.isnan(x_hat_curr).any() or torch.isinf(x_hat_curr).any():
                            step_nan_count[s] += 1
                            step_n[s] += 1
                            # Don't continue rollout with NaN state
                            break

                        # State error (L2 vs GT)
                        state_err = float(torch.linalg.vector_norm(x_hat_curr - x_t[t]).item())
                        step_state_err_sum[s] += state_err

                        # Direction cosine of predicted delta vs GT delta
                        dx_hat_norm = torch.linalg.vector_norm(dx_hat).clamp_min(1e-8)
                        dx_true_norm = torch.linalg.vector_norm(dx_true).clamp_min(1e-8)
                        cos_val = float((dx_hat @ dx_true).item() / (dx_hat_norm.item() * dx_true_norm.item()))
                        step_cos_sum[s] += cos_val

                        # Magnitude ratio
                        mag_pred = float(torch.linalg.vector_norm(dx_hat).item())
                        mag_true = float(dx_true_norm.item())
                        mag_ratio = mag_pred / max(mag_true, 1e-8)
                        step_mag_ratio_sum[s] += mag_ratio

                        step_n[s] += 1
                        x_hat_prev = x_hat_curr

                        # Update context window
                        if s < k_max - 1:
                            if s < horizon_k:
                                new_idx = t0 + s - horizon_k + 1
                                new_frame = x_t[new_idx] if new_idx >= 0 else x_hat_curr
                            else:
                                new_frame = x_hat_curr
                            ctx_window = torch.cat([ctx_window[1:], new_frame.unsqueeze(0)], dim=0)

                # Aggregate per-step results
                per_step = []
                for s in range(k_max):
                    n = step_n[s]
                    if n == 0:
                        per_step.append({"step": s + 1, "n": 0})
                        continue
                    valid_n = n - step_nan_count[s]
                    per_step.append({
                        "step": s + 1,
                        "n": n,
                        "state_err": step_state_err_sum[s] / max(valid_n, 1),
                        "direction_cosine": step_cos_sum[s] / max(valid_n, 1),
                        "magnitude_ratio": step_mag_ratio_sum[s] / max(valid_n, 1),
                        "nan_count": step_nan_count[s],
                    })

                # Summary
                total_n = sum(step_n)
                total_valid = total_n - sum(step_nan_count)
                final_n = step_n[-1]
                final_valid = final_n - step_nan_count[-1]

                result_entry = {
                    "condition": cond_name,
                    "sigma": sigma if use_tweedie else None,
                    "k_max": k_max,
                    "n_total": total_n,
                    "n_valid": total_valid,
                    "nan_total": sum(step_nan_count),
                    "mean_state_err": sum(step_state_err_sum) / max(total_valid, 1),
                    "mean_direction_cosine": sum(step_cos_sum) / max(total_valid, 1),
                    "final_state_err": step_state_err_sum[-1] / max(final_valid, 1),
                    "final_direction_cosine": step_cos_sum[-1] / max(final_valid, 1),
                    "per_step": per_step,
                }
                all_results.append(result_entry)

                logger.info(
                    f"[exp11] {cond_name} sigma={sigma if use_tweedie else 'N/A':>5} k={k_max:>2}: "
                    f"state_err={result_entry['final_state_err']:.4f} "
                    f"cos={result_entry['final_direction_cosine']:.4f} "
                    f"nan={result_entry['nan_total']}"
                )

    return {"all_results": all_results}


# ---------------------------------------------------------------------------
# Find best sigma
# ---------------------------------------------------------------------------


def find_best_sigma(all_results: list[dict], target_k: int = 4) -> dict:
    """For each Tweedie condition, find sigma minimizing state_err at target k."""
    best = {}

    # Group by condition
    conditions = set(r["condition"] for r in all_results)
    for cond in conditions:
        if "tweedie" not in cond:
            continue

        cond_results = [
            r for r in all_results
            if r["condition"] == cond and r["k_max"] == target_k
        ]
        if not cond_results:
            # Fall back to max available k
            cond_k_results = [r for r in all_results if r["condition"] == cond]
            if not cond_k_results:
                continue
            max_avail_k = max(r["k_max"] for r in cond_k_results)
            cond_results = [r for r in cond_k_results if r["k_max"] == max_avail_k]

        # Find sigma with lowest final_state_err, excluding runs with many NaNs
        valid = [r for r in cond_results if r["nan_total"] < r["n_total"] * 0.5]
        if not valid:
            valid = cond_results

        best_result = min(valid, key=lambda r: r["final_state_err"])
        best[cond] = {
            "best_sigma": best_result["sigma"],
            "state_err": best_result["final_state_err"],
            "direction_cosine": best_result["final_direction_cosine"],
            "nan_total": best_result["nan_total"],
            "k_max": best_result["k_max"],
        }

    return best


# ---------------------------------------------------------------------------
# Perceptual audio evaluation
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
def perceptual_audio_eval(
    *,
    dir_ar_model,
    mdn_model,
    score_model,
    best_sigmas: dict,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
    skip_mdn: bool,
    logger,
) -> list[dict]:
    """Decode best-sigma Tweedie rollouts through VAE+Mimi to WAVs."""
    from mimi_autoencoder import load_mimi_autoencoder
    from stage2.vae import ARFriendlyVAE
    from stage2.vae_train import load_vae_checkpoint

    data_cfg = cfg["data"]
    ctx_cfg = cfg["context"]
    eval_cfg = cfg["eval"]

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
    logger.info(f"[exp11] Loaded VAE decoder (latent_dim={latent_dim})")

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

    min_t = max(1, (window_size - 1) + horizon_k)
    max_k = max(audio_k_values)
    rng = get_rng(seed + 60_000)
    results = []

    # Conditions to decode: uncorrected baseline + best Tweedie for each corrected condition
    decode_conditions = [
        ("dir_ar_argmax", "dir_ar", "argmax", False, None),
    ]
    if "dir_ar_argmax_tweedie" in best_sigmas:
        sigma = best_sigmas["dir_ar_argmax_tweedie"]["best_sigma"]
        decode_conditions.append(("dir_ar_argmax_tweedie", "dir_ar", "argmax", True, sigma))
    if "dir_ar_categorical_tweedie" in best_sigmas:
        sigma = best_sigmas["dir_ar_categorical_tweedie"]["best_sigma"]
        decode_conditions.append(("dir_ar_categorical_tweedie", "dir_ar", "categorical", True, sigma))
    if not skip_mdn:
        decode_conditions.append(("mdn_baseline", "mdn", None, False, None))
        if "mdn_tweedie" in best_sigmas:
            sigma = best_sigmas["mdn_tweedie"]["best_sigma"]
            decode_conditions.append(("mdn_tweedie", "mdn", None, True, sigma))

    for i, utt_id in enumerate(utt_ids):
        if utt_id not in store:
            continue
        x_true = store.get_latents(utt_id).astype(np.float32, copy=False)
        t_total = x_true.shape[0]
        if t_total < min_t + max_k + 20:
            continue

        prefix_len = 10
        suffix_len = 10
        t_start_min = min_t + prefix_len
        t_start_max = t_total - max_k - suffix_len
        if t_start_max <= t_start_min:
            continue
        t_start = int(rng.integers(t_start_min, t_start_max))

        # Decode GT segment
        try:
            audio_gt = _decode_to_audio(x_true)
            _save_wav(audio_gt, audio_dir / f"utt{i:02d}_{utt_id}_GT.wav", mimi_sr, output_sr)
        except Exception as e:
            logger.warning(f"[exp11] GT decode failed for {utt_id}: {e}")
            continue

        for cond_name, model_type, sampling_strategy, use_tweedie, sigma in decode_conditions:
            for k_rollout in audio_k_values:
                if t_start + k_rollout >= t_total:
                    continue

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

                    if model_type == "dir_ar":
                        dir_idx, magnitude = dir_ar_model.sample(
                            ctx_flat, strategy=sampling_strategy,
                        )
                        dx_hat = dir_ar_model.reconstruct_delta(dir_idx, magnitude)[0]
                    else:
                        dx_hat = mdn_model.expected_mean(ctx_flat).squeeze(0)

                    x_hat_curr = x_hat_prev + dx_hat

                    if use_tweedie and sigma is not None:
                        x_hat_curr_2d = x_hat_curr.unsqueeze(0)
                        x_hat_curr_2d = tweedie_correct(
                            x_hat_curr_2d, score_model, sigma, device,
                        )
                        x_hat_curr = x_hat_curr_2d.squeeze(0)

                    rollout_frames.append(x_hat_curr.cpu().numpy())
                    x_hat_prev = x_hat_curr

                    if s < k_rollout - 1:
                        if s < horizon_k:
                            new_idx = t_start + s - horizon_k + 1
                            new_frame = x_t[new_idx] if new_idx >= 0 else x_hat_curr
                        else:
                            new_frame = x_hat_curr
                        ctx_window = torch.cat([ctx_window[1:], new_frame.unsqueeze(0)], dim=0)

                # Splice: GT prefix -> rollout -> GT suffix
                rollout_arr = np.stack(rollout_frames, axis=0)
                spliced = np.concatenate([
                    x_true[t_start - prefix_len:t_start],
                    rollout_arr,
                    x_true[t_start + k_rollout:t_start + k_rollout + suffix_len],
                ], axis=0)

                try:
                    audio_spliced = _decode_to_audio(spliced)
                    wav_path = audio_dir / f"utt{i:02d}_{utt_id}_{cond_name}_k{k_rollout:02d}.wav"
                    _save_wav(audio_spliced, wav_path, mimi_sr, output_sr)

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
                        "condition": cond_name,
                        "sigma": sigma,
                        "k_rollout": k_rollout,
                        "t_start": t_start,
                        "mel_distance": mel_dist,
                        "wav_path": str(wav_path),
                    })
                    logger.info(
                        f"[exp11] Audio {cond_name} utt={utt_id} k={k_rollout}: "
                        f"mel_dist={mel_dist:.4f}"
                    )
                except Exception as e:
                    logger.warning(f"[exp11] Decode failed for {utt_id} {cond_name} k={k_rollout}: {e}")

    # Clean up
    del autoencoder, vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description="Exp 11: Consistency Model for Rollout Stability")
    p.add_argument("--config", type=str, default="configs/tier3_exp11_consistency_rollout.yaml")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--score-checkpoint", type=str, default=None,
                   help="Pre-trained score model checkpoint (skip score training)")
    p.add_argument("--score-max-steps", type=int, default=None,
                   help="Override score training steps")
    p.add_argument("--skip-audio", action="store_true", help="Skip perceptual audio eval")
    p.add_argument("--skip-mdn", action="store_true", help="Skip MDN conditions")
    p.add_argument("--sigma", type=float, default=None,
                   help="Single sigma value instead of sweep")
    p.add_argument("--joint-train", action="store_true",
                   help="Fine-tune AR model with Tweedie in the unrolled loop")
    p.add_argument("--joint-train-steps", type=int, default=None,
                   help="Override joint training max steps")
    p.add_argument("--joint-train-sigma", type=float, default=None,
                   help="Override sigma for joint training (default: from config)")
    args = p.parse_args()

    if os.environ.get("NO_TORCH_COMPILE"):
        os.environ["TORCH_COMPILE_DISABLE"] = "1"

    logger = setup_logging(name="phase0")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.score_max_steps is not None:
        cfg["score_train"]["max_steps"] = args.score_max_steps

    run_id = args.run_id or _default_run_id()
    out_root = Path(cfg["output"]["out_dir"])
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    run = register_run(
        experiment="exp11_consistency_rollout",
        run_id=run_id,
        config_path=args.config,
        config=cfg,
        cli_args=sys.argv[1:],
        out_dir=out_dir,
        log_name="phase0",
    )

    set_seed(int(cfg.get("seed", 42)))
    device = _device_from_config(cfg["train"]["device"])

    data_cfg = cfg["data"]
    ckpt_cfg = cfg["checkpoints"]

    # -------------------------------------------------------------------
    # Step 1: Load pre-trained models
    # -------------------------------------------------------------------
    logger.info("[exp11] Step 1: Loading pre-trained models...")

    # Load codebook
    codebook_path = Path(ckpt_cfg["codebook"])
    codebook_np = np.load(str(codebook_path)).astype(np.float32)
    logger.info(f"[exp11] Loaded codebook: {codebook_path} shape={codebook_np.shape}")

    # Load Direction AR
    dir_ar_ckpt_path = ckpt_cfg["direction_ar"]
    dir_ar_model, dir_ar_ckpt = load_direction_ar_checkpoint(
        path=dir_ar_ckpt_path,
        codebook=codebook_np,
        device=device,
    )
    dir_ar_model.eval()
    window_size = int(cfg["context"]["window_size"])
    D = dir_ar_model.output_dim
    logger.info(f"[exp11] Loaded Direction AR: K={dir_ar_model.K} D={D}")

    # Load MDN
    mdn_model = None
    if not args.skip_mdn:
        mdn_ckpt_path = ckpt_cfg["mdn"]
        mdn_model, mdn_ckpt = load_phase1_checkpoint(mdn_ckpt_path, device=device)
        mdn_model.eval()
        logger.info(f"[exp11] Loaded MDN from {mdn_ckpt_path}")

    # -------------------------------------------------------------------
    # Step 2: Train (or load) score model on 32D latents
    # -------------------------------------------------------------------
    logger.info("[exp11] Step 2: Score model...")

    if args.score_checkpoint:
        from stage2.score_train import load_score_checkpoint
        score_model, score_ckpt = load_score_checkpoint(args.score_checkpoint, device=device)
        logger.info(f"[exp11] Loaded pre-trained score model from {args.score_checkpoint}")
    else:
        from stage2.score_train import train_score_model
        from phase0.data.splits import load_splits

        idx = load_latents_index(data_cfg["latents_index"])
        splits = load_splits(data_cfg["splits_dir"])
        train_speaker_set = set(splits.train_speakers)
        train_utts = idx[idx["speaker_id"].isin(train_speaker_set)]["utterance_id"].astype(str).tolist()
        logger.info(f"[exp11] {len(train_utts)} train utterances for score model")

        score_cfg = cfg["score_model"]
        score_train_cfg = cfg["score_train"]

        score_model, score_ckpt_path = train_score_model(
            latents_dir=data_cfg["latents_dir"],
            utterance_ids=train_utts,
            out_dir=out_dir / "score_model",
            latent_dim=int(score_cfg["latent_dim"]),
            hidden_dim=int(score_cfg["hidden_dim"]),
            n_layers=int(score_cfg["n_layers"]),
            sigma_min=float(score_cfg["sigma_min"]),
            sigma_max=float(score_cfg["sigma_max"]),
            batch_size=int(score_train_cfg["batch_size"]),
            num_workers=int(score_train_cfg.get("num_workers", 0)),
            max_steps=int(score_train_cfg["max_steps"]),
            lr=float(score_train_cfg["lr"]),
            weight_decay=float(score_train_cfg["weight_decay"]),
            grad_clip_norm=float(score_train_cfg["grad_clip_norm"]),
            log_every=int(score_train_cfg["log_every"]),
            save_every=int(score_train_cfg["save_every"]),
            seed=int(cfg.get("seed", 42)),
            device=device,
        )
        logger.info(f"[exp11] Score model trained. Checkpoint: {score_ckpt_path}")

    score_model.eval()
    n_score_params = sum(p.numel() for p in score_model.parameters())
    logger.info(f"[exp11] Score model: {n_score_params:,} parameters")

    # -------------------------------------------------------------------
    # Step 2.5: Joint training (AR + Tweedie end-to-end)
    # -------------------------------------------------------------------
    joint_result = None
    if args.joint_train:
        joint_sigma = args.joint_train_sigma
        if joint_sigma is None:
            joint_sigma = float(cfg.get("joint_train", {}).get("sigma", 0.1))

        logger.info(f"[exp11] Step 2.5: Joint training with sigma={joint_sigma}...")
        joint_result = joint_train_ar_with_tweedie(
            model=dir_ar_model,
            score_model=score_model,
            cfg=cfg,
            device=device,
            out_dir=out_dir,
            sigma=joint_sigma,
            max_steps_override=args.joint_train_steps,
            logger=logger,
        )

        with open(out_dir / "joint_train_result.json", "w") as f:
            json.dump(joint_result, f, indent=2)

    # -------------------------------------------------------------------
    # Step 3-4: Rollout evaluation with sigma sweep
    # -------------------------------------------------------------------
    logger.info("[exp11] Step 3-4: Rollout evaluation...")

    if args.sigma is not None:
        sigma_values = [args.sigma]
    else:
        sigma_values = [float(s) for s in cfg["rollout"]["sigma_sweep"]]

    rollout_result = rollout_evaluation(
        dir_ar_model=dir_ar_model,
        mdn_model=mdn_model,
        score_model=score_model,
        cfg=cfg,
        device=device,
        sigma_values=sigma_values,
        skip_mdn=args.skip_mdn,
        logger=logger,
    )

    all_results = rollout_result["all_results"]

    # Save full results
    with open(out_dir / "rollout_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save sigma sweep data
    sigma_sweep_data = [
        {k: v for k, v in r.items() if k != "per_step"}
        for r in all_results
    ]
    with open(out_dir / "sigma_sweep.json", "w") as f:
        json.dump(sigma_sweep_data, f, indent=2, default=str)

    # -------------------------------------------------------------------
    # Step 5: Find best sigma
    # -------------------------------------------------------------------
    logger.info("[exp11] Step 5: Finding best sigma per condition...")

    best_sigmas = find_best_sigma(all_results, target_k=4)
    for cond, info in best_sigmas.items():
        logger.info(
            f"[exp11] Best sigma for {cond}: sigma={info['best_sigma']} "
            f"state_err={info['state_err']:.4f} cos={info['direction_cosine']:.4f}"
        )

    with open(out_dir / "best_sigmas.json", "w") as f:
        json.dump(best_sigmas, f, indent=2)

    # -------------------------------------------------------------------
    # Step 6: Build summary table
    # -------------------------------------------------------------------
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "condition": r["condition"],
            "sigma": r.get("sigma"),
            "k_max": r["k_max"],
            "n": r["n_total"],
            "state_err": r["final_state_err"],
            "direction_cosine": r["final_direction_cosine"],
            "nan_count": r["nan_total"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(str(out_dir / "summary.csv"), index=False)
    logger.info(f"[exp11] Summary ({len(summary_df)} rows):\n{summary_df.to_string(index=False)}")

    # -------------------------------------------------------------------
    # Step 7: Perceptual audio eval (optional)
    # -------------------------------------------------------------------
    audio_results = []
    if not args.skip_audio and best_sigmas:
        logger.info("[exp11] Step 6: Perceptual audio evaluation...")
        audio_results = perceptual_audio_eval(
            dir_ar_model=dir_ar_model,
            mdn_model=mdn_model,
            score_model=score_model,
            best_sigmas=best_sigmas,
            cfg=cfg,
            device=device,
            out_dir=out_dir,
            skip_mdn=args.skip_mdn,
            logger=logger,
        )
        with open(out_dir / "audio_eval_results.json", "w") as f:
            json.dump(audio_results, f, indent=2)

        # Add audio metrics to summary
        if audio_results:
            audio_df = pd.DataFrame(audio_results)
            for cond in audio_df["condition"].unique():
                for k_val in audio_df["k_rollout"].unique():
                    cond_k = audio_df[(audio_df["condition"] == cond) & (audio_df["k_rollout"] == k_val)]
                    if len(cond_k) > 0:
                        summary_rows.append({
                            "condition": f"audio_{cond}_k{k_val}",
                            "mel_distance_mean": float(cond_k["mel_distance"].mean()),
                            "mel_distance_std": float(cond_k["mel_distance"].std()),
                            "n": len(cond_k),
                        })
            # Re-save summary with audio rows
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(str(out_dir / "summary.csv"), index=False)
    elif args.skip_audio:
        logger.info("[exp11] Skipping audio eval (--skip-audio)")

    # -------------------------------------------------------------------
    # Finalize
    # -------------------------------------------------------------------
    key_metrics = {
        "n_score_params": n_score_params,
        "n_sigma_values": len(sigma_values),
        "n_conditions": len(set(r["condition"] for r in all_results)),
    }

    # Add best-sigma metrics
    for cond, info in best_sigmas.items():
        key_metrics[f"best_sigma_{cond}"] = info["best_sigma"]
        key_metrics[f"best_state_err_{cond}"] = info["state_err"]

    # Add uncorrected baselines for comparison
    for r in all_results:
        if r["condition"] == "dir_ar_argmax" and r["k_max"] == 4:
            key_metrics["baseline_dir_ar_k4_state_err"] = r["final_state_err"]
        if r["condition"] == "mdn_baseline" and r["k_max"] == 4:
            key_metrics["baseline_mdn_k4_state_err"] = r["final_state_err"]

    if audio_results:
        audio_df = pd.DataFrame(audio_results)
        key_metrics["audio_mel_mean"] = float(audio_df["mel_distance"].mean())

    if joint_result:
        key_metrics["joint_trained"] = True
        key_metrics["joint_final_loss"] = joint_result.get("final_loss", float("nan"))
        key_metrics["joint_sigma"] = joint_result.get("sigma")
        key_metrics["joint_unroll_k"] = joint_result.get("unroll_k")

    finalize_run(run, key_metrics=key_metrics)
    logger.info(f"[exp11] Done. Results at {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
