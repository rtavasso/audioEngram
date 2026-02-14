#!/usr/bin/env python3
"""
Stage 2 - Experiment 9: Pocket-TTS Mimi VAE-GAN.

Trains pocket-tts Mimi with 32D VAE bottleneck using full VAE-GAN loss:
  L = L_recon + beta_kl*L_KL + lambda_pred*L_pred + lambda_adv*L_adv + lambda_feat*L_feat + lambda_distill*L_distill

Pipeline:
  1. Train VAE-GAN (alternating generator/discriminator updates)
  2. Extract 32D latents -> zarr
  3. Build frames index
  4. Run Phase 1 diagnostic battery (train_and_eval_for_k for k in horizons)
  5. Run injection diagnostic (4 modes)
  6. Compute reconstruction quality (mel distance on eval utterances)
  7. Write summary CSV + experiment tracking

Usage:
  uv run python scripts/tier2_exp9_pocket_mimi_vae.py --config configs/tier2_exp9_pocket_mimi_vae.yaml
  uv run python scripts/tier2_exp9_pocket_mimi_vae.py --config configs/tier2_exp9_pocket_mimi_vae.yaml --max-steps 10
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
import torch.nn as nn
import torchaudio
import yaml

from experiment import register_run, finalize_run
from phase0.data.librispeech import get_utterances, load_audio
from phase0.data.splits import load_splits
from phase0.utils.logging import setup_logging
from phase0.utils.seed import set_seed
from phase1.train_eval import _device_from_config, train_and_eval_for_k, write_results
from stage2.vae_train import AudioSegmentDataset
from stage2.vae_losses import reconstruction_loss, kl_loss, temporal_prediction_loss_from_delta
from stage2.vae_extract import extract_vae_latents as _extract_vae_latents_orig
from stage2.vae_extract import build_frames_index


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _reconstruct_audio(vae, audio: torch.Tensor, mode: str = "mu") -> torch.Tensor:
    """
    Reconstruct audio with one of:
      - stochastic: full VAE forward (samples z ~ q)
      - mu: deterministic VAE decode using posterior mean
      - pretrained_mimi: bypass VAE bottleneck, use pretrained Mimi path
    """
    length = audio.shape[-1]
    if mode == "stochastic":
        audio_hat = vae(audio)["audio_hat"]
    elif mode == "mu":
        _, mu, _ = vae.encode(audio)
        audio_hat = vae.decode(mu, length=length)
    elif mode == "pretrained_mimi":
        audio_hat = vae.reconstruct_with_pretrained_mimi(audio, length=length)
    else:
        raise ValueError(f"Unknown reconstruction mode: {mode}")
    return audio_hat[..., :length]


# ── VAE-GAN training loop ──────────────────────────────────────────────


def train_vae_gan(
    *,
    audio_paths: list[str],
    vae,  # PocketMimiVAE
    discriminator,  # MultiScaleSTFTDiscriminator
    wavlm_distill,  # WavLMDistillation or None
    out_dir: Path,
    # Loss weights
    lambda_recon: float = 1.0,
    beta_kl: float = 0.01,
    kl_normalize_by_elements: bool = True,
    kl_start_steps: int = 0,
    kl_ramp_steps: int = 0,
    # Temporal prediction loss (latent dynamics surrogate)
    lambda_pred: float = 0.0,
    pred_alpha: float = 0.5,
    pred_kind: str = "causal_conv",  # "causal_conv" | "mlp"
    pred_kernel_size: int = 6,
    pred_window_size: int = 6,
    pred_hidden_dim: int = 256,
    pred_start_steps: int = 0,
    pred_use_mu: bool = True,
    lambda_adv: float = 1.0,
    lambda_feat: float = 2.0,
    lambda_distill: float = 1.0,
    lambda_latent_align: float = 0.0,
    distill_start_steps: int = 0,
    # Training
    lr_schedule: str = "constant",
    segment_sec: float = 4.0,
    sample_rate: int = 24000,
    batch_size: int = 8,
    num_workers: int = 2,
    max_steps: int = 50000,
    lr_generator: float = 1e-4,
    lr_discriminator: float = 1e-4,
    weight_decay: float = 1e-4,
    grad_clip_norm: float = 1.0,
    gan_warmup_steps: int = 0,
    deterministic_warmup_steps: int = 0,
    log_every: int = 100,
    save_every: int = 5000,
    sample_every: int = 5000,
    sample_n_samples: int = 3,
    sample_output_sr: int = 48000,
    sample_mode: str = "mu",
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
    amp: bool = True,
    amp_dtype: str = "fp16",
) -> dict:
    """
    Train PocketMimiVAE with GAN loss.

    Returns:
        dict with final loss values and checkpoint path.
    """
    from stage2.gan_losses import adversarial_g_loss, adversarial_d_loss, feature_matching_loss
    from stage2.vae_predictor import CausalConvDeltaPredictor, LatentPredictor

    logger = logging.getLogger("phase0")

    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    segment_samples = int(segment_sec * sample_rate)

    dataset = AudioSegmentDataset(
        audio_paths=audio_paths,
        segment_samples=segment_samples,
        sample_rate=sample_rate,
        seed=seed,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Temporal predictor (optional) — trained jointly with the generator.
    pred_kind_n = str(pred_kind).strip().lower()
    temporal_pred = None
    temporal_pred_cfg = None
    if lambda_pred > 0:
        if pred_kind_n in ("causal_conv", "causal-conv", "conv", "causalconv"):
            temporal_pred = CausalConvDeltaPredictor(
                latent_dim=int(vae.latent_dim),
                kernel_size=int(pred_kernel_size),
            ).to(device)
            temporal_pred_cfg = {
                "kind": "causal_conv",
                "kernel_size": int(pred_kernel_size),
                "alpha": float(pred_alpha),
                "use_mu": bool(pred_use_mu),
                "start_steps": int(pred_start_steps),
            }
        elif pred_kind_n in ("mlp", "short_mlp", "context_mlp", "short-context-mlp"):
            temporal_pred = LatentPredictor(
                latent_dim=int(vae.latent_dim),
                window_size=int(pred_window_size),
                hidden_dim=int(pred_hidden_dim),
            ).to(device)
            temporal_pred_cfg = {
                "kind": "mlp",
                "window_size": int(pred_window_size),
                "hidden_dim": int(pred_hidden_dim),
                "alpha": float(pred_alpha),
                "use_mu": bool(pred_use_mu),
                "start_steps": int(pred_start_steps),
            }
        else:
            raise ValueError(f"Unknown pred_kind={pred_kind!r} (expected 'causal_conv' or 'mlp')")

    # Optimizers
    gen_params = list(vae.trainable_parameters())
    if temporal_pred is not None:
        gen_params.extend([p for p in temporal_pred.parameters() if p.requires_grad])
    if wavlm_distill is not None:
        distill_params = [p for p in wavlm_distill.parameters() if p.requires_grad]
        if distill_params:
            gen_params.extend(distill_params)
    opt_g = torch.optim.AdamW(gen_params, lr=lr_generator, weight_decay=weight_decay)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=lr_discriminator, weight_decay=weight_decay)

    # AMP setup
    use_amp = amp and device.type == "cuda"
    amp_dtype_t = torch.float16 if amp_dtype.lower() in ("fp16", "float16") else torch.bfloat16
    scaler_g = torch.GradScaler("cuda") if use_amp and amp_dtype_t == torch.float16 else None
    scaler_d = torch.GradScaler("cuda") if use_amp and amp_dtype_t == torch.float16 else None

    def make_autocast():
        return torch.autocast(device_type=device.type, dtype=amp_dtype_t, enabled=use_amp)

    # LR schedulers
    scheduler_g = None
    scheduler_d = None
    if lr_schedule == "cosine":
        scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=max_steps)
        scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=max_steps)

    logger.info(
        f"[exp9] latent_dim={vae.latent_dim} lambda_recon={lambda_recon} beta_kl={beta_kl} "
        f"kl_norm={kl_normalize_by_elements} kl_start={kl_start_steps} kl_ramp={kl_ramp_steps} "
        f"distill_start={distill_start_steps} det_warmup={deterministic_warmup_steps} "
        f"lambda_pred={lambda_pred} pred_kind={pred_kind_n} pred_alpha={pred_alpha} "
        f"pred_kernel={pred_kernel_size} pred_window={pred_window_size} pred_hidden={pred_hidden_dim} "
        f"pred_start={pred_start_steps} pred_use_mu={pred_use_mu} "
        f"lambda_adv={lambda_adv} lambda_feat={lambda_feat} "
        f"lambda_distill={lambda_distill} lambda_latent_align={lambda_latent_align} "
        f"batch={batch_size} lr_g={lr_generator} lr_d={lr_discriminator} "
        f"lr_schedule={lr_schedule} steps={max_steps} gan_warmup={gan_warmup_steps} amp={use_amp}"
    )

    vae.train()
    discriminator.train()
    if temporal_pred is not None:
        temporal_pred.train()

    step = 0
    data_iter = iter(loader)
    loss_accum = {
        "g_total": 0.0, "recon": 0.0, "kl": 0.0,
        "kl_raw": 0.0,
        "latent_align": 0.0,
        "pred": 0.0,
        "pred_cos": 0.0,
        "pred_mse": 0.0,
        "mu_bt_std": 0.0,
        "mu_abs_mean": 0.0,
        "mu_abs_max": 0.0,
        "mu_std": 0.0,
        "mu_mean_bt_abs": 0.0,
        "h_enc_abs_mean": 0.0,
        "h_enc_abs_max": 0.0,
        "h_enc_std": 0.0,
        "h_enc_bt_std": 0.0,
        "logvar_mean": 0.0,
        "logvar_std": 0.0,
        "grad_mu_proj": 0.0,
        "grad_dec_proj0": 0.0,
        "grad_logvar_bias": 0.0,
        "adv_g": 0.0, "feat": 0.0, "distill": 0.0, "d_total": 0.0,
    }
    log_count = 0

    def _safe_grad_norm(param: torch.Tensor) -> float:
        g = getattr(param, "grad", None)
        if g is None:
            return 0.0
        return float(g.detach().float().norm().item())

    while step < max_steps:
        try:
            audio = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            audio = next(data_iter)

        audio = audio.to(device, non_blocking=True)  # [B, 1, T]
        use_gan = step >= gan_warmup_steps

        # ── Generator update ──────────────────────────────────────────
        opt_g.zero_grad(set_to_none=True)

        with make_autocast():
            if step < deterministic_warmup_steps:
                z, mu, logvar, h_enc = vae.encode(audio, return_h=True)
                audio_hat = vae.decode(mu, length=audio.shape[-1])
                out = {"audio_hat": audio_hat, "z": mu, "mu": mu, "logvar": logvar, "h_enc": h_enc}
            else:
                out = vae(audio)
                audio_hat = out["audio_hat"]

            # Reconstruction loss (multi-scale STFT + time-domain L1)
            l_recon = reconstruction_loss(audio, audio_hat)

            # KL loss
            l_kl_raw = kl_loss(out["mu"], out["logvar"])
            if kl_normalize_by_elements:
                denom = max(1, int(out["mu"].shape[1] * out["mu"].shape[2]))
                l_kl = l_kl_raw / denom
            else:
                l_kl = l_kl_raw

            if step < kl_start_steps:
                kl_mult = 0.0
            elif kl_ramp_steps > 0:
                kl_mult = min(1.0, float(step - kl_start_steps + 1) / float(kl_ramp_steps))
            else:
                kl_mult = 1.0
            g_loss = lambda_recon * l_recon + (beta_kl * kl_mult) * l_kl

            # Keep bottleneck informative: dec_proj(mu) should match encoder hidden.
            l_latent_align = audio.new_tensor(0.0)
            if lambda_latent_align > 0:
                if "h_enc" in out:
                    h_ref = out["h_enc"].detach()
                else:
                    _, _, _, h_ref = vae.encode(audio, return_h=True)
                    h_ref = h_ref.detach()
                h_hat = vae.dec_proj(out["mu"])
                l_latent_align = nn.functional.l1_loss(h_hat, h_ref)
                g_loss = g_loss + lambda_latent_align * l_latent_align

            # Temporal prediction loss on short context (cheap, causal).
            l_pred = audio.new_tensor(0.0)
            l_pred_cos = audio.new_tensor(0.0)
            l_pred_mse = audio.new_tensor(0.0)
            if lambda_pred > 0 and temporal_pred is not None and step >= pred_start_steps:
                z_src = out["mu"] if pred_use_mu else out["z"]  # [B, D, T']
                z_src_f = z_src.float()
                B, D, Tz = z_src_f.shape
                if pred_kind_n in ("causal_conv", "causal-conv", "conv", "causalconv"):
                    k = int(pred_kernel_size)
                    start_t = max(0, k - 1)
                    if Tz > start_t + 1:
                        delta_pred_all = temporal_pred(z_src_f)  # [B, D, T']
                        delta_pred = delta_pred_all[:, :, start_t:-1]  # t in [start_t, T'-2]
                        if delta_pred.numel() > 0:
                            l_pred, l_pred_cos, l_pred_mse = temporal_prediction_loss_from_delta(
                                z_src_f, delta_pred, alpha=float(pred_alpha), start_t=start_t
                            )
                            g_loss = g_loss + lambda_pred * l_pred
                elif pred_kind_n in ("mlp", "short_mlp", "context_mlp", "short-context-mlp"):
                    w = int(pred_window_size)
                    start_t = max(0, w - 1)
                    if Tz > w:
                        # windows: [B, D, T'-w+1, w] where window i ends at t=i+w-1
                        windows = z_src_f.unfold(dimension=2, size=w, step=1)
                        windows = windows[:, :, :-1, :]  # drop window ending at t=T'-1 (no t+1 target)
                        N = int(windows.shape[2])
                        if N > 0:
                            windows_flat = windows.permute(0, 2, 1, 3).reshape(B * N, D * w)
                            delta_pred_flat = temporal_pred(windows_flat)  # [B*N, D]
                            delta_pred = delta_pred_flat.reshape(B, N, D).permute(0, 2, 1)  # [B, D, N]
                            l_pred, l_pred_cos, l_pred_mse = temporal_prediction_loss_from_delta(
                                z_src_f, delta_pred, alpha=float(pred_alpha), start_t=start_t
                            )
                            g_loss = g_loss + lambda_pred * l_pred
                else:
                    raise ValueError(f"Unknown pred_kind={pred_kind!r}")

            # Adversarial + feature matching (after warmup)
            l_adv = audio.new_tensor(0.0)
            l_feat = audio.new_tensor(0.0)
            use_adv_loss = use_gan and lambda_adv > 0
            use_feat_loss = use_gan and lambda_feat > 0
            if use_adv_loss or use_feat_loss:
                # disc_real: no_grad since we only need features as targets
                with torch.no_grad():
                    disc_real = discriminator(audio)
                disc_fake = discriminator(audio_hat)
                if use_adv_loss:
                    l_adv = adversarial_g_loss(disc_fake)
                    g_loss = g_loss + lambda_adv * l_adv
                if use_feat_loss:
                    l_feat = feature_matching_loss(disc_real, disc_fake)
                    g_loss = g_loss + lambda_feat * l_feat

            # WavLM distillation (on same device, under autocast)
            l_distill_val = 0.0
            if wavlm_distill is not None and lambda_distill > 0 and step >= distill_start_steps:
                l_distill = wavlm_distill(audio, out["mu"])
                g_loss = g_loss + lambda_distill * l_distill
                l_distill_val = float(l_distill.item())

        if not torch.isfinite(audio_hat).all():
            raise FloatingPointError(f"[exp9] non-finite audio_hat at step={step}")
        if not torch.isfinite(g_loss):
            raise FloatingPointError(
                f"[exp9] non-finite generator loss at step={step}: "
                f"recon={float(l_recon.item()):.4f} kl={float(l_kl.item()):.4f} "
                f"lat_align={float(l_latent_align.item()):.4f} "
                f"pred={float(l_pred.item()):.4f} pred_cos={float(l_pred_cos.item()):.4f} pred_mse={float(l_pred_mse.item()):.4f} "
                f"adv={float(l_adv.item()):.4f} feat={float(l_feat.item()):.4f} "
                f"distill={l_distill_val:.4f}"
            )

        mu_f = out["mu"].detach().float()
        logvar_f = out["logvar"].detach().float()
        h_enc_f = out["h_enc"].detach().float() if "h_enc" in out else None
        mu_abs_mean = float(mu_f.abs().mean().item())
        mu_abs_max = float(mu_f.abs().max().item())
        mu_std = float(mu_f.std().item())
        mu_mean_bt_abs = float(mu_f.mean(dim=(0, 2)).abs().mean().item())
        if h_enc_f is None:
            h_enc_abs_mean = 0.0
            h_enc_abs_max = 0.0
            h_enc_std = 0.0
            h_enc_bt_std = 0.0
        else:
            h_enc_abs_mean = float(h_enc_f.abs().mean().item())
            h_enc_abs_max = float(h_enc_f.abs().max().item())
            h_enc_std = float(h_enc_f.std().item())
            h_enc_bt_std = float(h_enc_f.std(dim=(0, 2)).mean().item())
        logvar_mean = float(logvar_f.mean().item())
        logvar_std = float(logvar_f.std().item())

        if scaler_g is not None:
            scaler_g.scale(g_loss).backward()
            scaler_g.unscale_(opt_g)
            grad_mu_proj = _safe_grad_norm(vae.mu_proj.weight)
            grad_dec_proj0 = _safe_grad_norm(vae.dec_proj[0].weight)
            grad_logvar_bias = _safe_grad_norm(vae.logvar_proj.bias)
            if grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(gen_params, max_norm=grad_clip_norm)
            scaler_g.step(opt_g)
            scaler_g.update()
        else:
            g_loss.backward()
            grad_mu_proj = _safe_grad_norm(vae.mu_proj.weight)
            grad_dec_proj0 = _safe_grad_norm(vae.dec_proj[0].weight)
            grad_logvar_bias = _safe_grad_norm(vae.logvar_proj.bias)
            if grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(gen_params, max_norm=grad_clip_norm)
            opt_g.step()

        # ── Discriminator update (after warmup) ───────────────────────
        d_loss_val = 0.0
        if use_gan and (lambda_adv > 0 or lambda_feat > 0):
            opt_d.zero_grad(set_to_none=True)

            with make_autocast():
                disc_real_d = discriminator(audio.detach())
                disc_fake_d = discriminator(audio_hat.detach())
                d_loss = adversarial_d_loss(disc_real_d, disc_fake_d)
            if not torch.isfinite(d_loss):
                raise FloatingPointError(
                    f"[exp9] non-finite discriminator loss at step={step}: "
                    f"d_loss={float(d_loss.item())}"
                )

            if scaler_d is not None:
                scaler_d.scale(d_loss).backward()
                if grad_clip_norm > 0:
                    scaler_d.unscale_(opt_d)
                    nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=grad_clip_norm)
                scaler_d.step(opt_d)
                scaler_d.update()
            else:
                d_loss.backward()
                if grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=grad_clip_norm)
                opt_d.step()
            d_loss_val = float(d_loss.item())

        if scheduler_g is not None:
            scheduler_g.step()
        if scheduler_d is not None:
            scheduler_d.step()

        step += 1
        loss_accum["g_total"] += float(g_loss.item())
        loss_accum["recon"] += float(l_recon.item())
        loss_accum["kl"] += float(l_kl.item())
        loss_accum["kl_raw"] += float(l_kl_raw.item())
        loss_accum["latent_align"] += float(l_latent_align.item())
        loss_accum["pred"] += float(l_pred.item())
        loss_accum["pred_cos"] += float(l_pred_cos.item())
        loss_accum["pred_mse"] += float(l_pred_mse.item())
        # Std over batch+time averaged across channels; near zero => collapse.
        mu_bt_std = out["mu"].float().std(dim=(0, 2)).mean()
        loss_accum["mu_bt_std"] += float(mu_bt_std.item())
        loss_accum["mu_abs_mean"] += mu_abs_mean
        loss_accum["mu_abs_max"] += mu_abs_max
        loss_accum["mu_std"] += mu_std
        loss_accum["mu_mean_bt_abs"] += mu_mean_bt_abs
        loss_accum["h_enc_abs_mean"] += h_enc_abs_mean
        loss_accum["h_enc_abs_max"] += h_enc_abs_max
        loss_accum["h_enc_std"] += h_enc_std
        loss_accum["h_enc_bt_std"] += h_enc_bt_std
        loss_accum["logvar_mean"] += logvar_mean
        loss_accum["logvar_std"] += logvar_std
        loss_accum["grad_mu_proj"] += grad_mu_proj
        loss_accum["grad_dec_proj0"] += grad_dec_proj0
        loss_accum["grad_logvar_bias"] += grad_logvar_bias
        loss_accum["adv_g"] += float(l_adv.item())
        loss_accum["feat"] += float(l_feat.item())
        loss_accum["distill"] += l_distill_val
        loss_accum["d_total"] += d_loss_val
        log_count += 1

        if log_every and step % log_every == 0:
            avg = {k: v / max(log_count, 1) for k, v in loss_accum.items()}
            warmup_tag = " [warmup]" if not use_gan else ""
            logger.info(
                f"[exp9] step={step}/{max_steps}{warmup_tag} "
                f"g={avg['g_total']:.4f} recon={avg['recon']:.4f} "
                f"kl={avg['kl']:.4f} kl_raw={avg['kl_raw']:.2f} "
                f"lat_align={avg['latent_align']:.4f} "
                f"pred={avg['pred']:.4f} pred_cos={avg['pred_cos']:.4f} pred_mse={avg['pred_mse']:.4f} "
                f"mu_bt_std={avg['mu_bt_std']:.3e} "
                f"mu_abs_mean={avg['mu_abs_mean']:.3e} mu_abs_max={avg['mu_abs_max']:.3e} "
                f"mu_std={avg['mu_std']:.3e} mu_mean_bt_abs={avg['mu_mean_bt_abs']:.3e} "
                f"h_abs_mean={avg['h_enc_abs_mean']:.3e} h_abs_max={avg['h_enc_abs_max']:.3e} "
                f"h_std={avg['h_enc_std']:.3e} h_bt_std={avg['h_enc_bt_std']:.3e} "
                f"logvar_mean={avg['logvar_mean']:.3e} logvar_std={avg['logvar_std']:.3e} "
                f"gmu_proj={avg['grad_mu_proj']:.3e} gdec0={avg['grad_dec_proj0']:.3e} "
                f"glv_b={avg['grad_logvar_bias']:.3e} "
                f"kl_mult={kl_mult:.3f} "
                f"adv_g={avg['adv_g']:.4f} "
                f"feat={avg['feat']:.4f} distill={avg['distill']:.4f} "
                f"d={avg['d_total']:.4f}"
            )
            loss_accum = {k: 0.0 for k in loss_accum}
            log_count = 0

        if save_every and step % save_every == 0:
            _save_checkpoint(vae, discriminator, opt_g, opt_d, step,
                             ckpt_dir / f"vae_gan_step{step}.pt",
                             temporal_pred=temporal_pred,
                             temporal_pred_config=temporal_pred_cfg)

        if sample_every and step % sample_every == 0:
            _save_audio_samples(
                vae, audio_paths, step, out_dir,
                sample_rate=sample_rate,
                output_sr=sample_output_sr,
                device=device,
                amp_ctx=make_autocast(),
                sample_mode=sample_mode,
                n_samples=sample_n_samples,
            )

    # Final checkpoint
    final_path = ckpt_dir / "vae_gan_final.pt"
    _save_checkpoint(
        vae,
        discriminator,
        opt_g,
        opt_d,
        step,
        final_path,
        temporal_pred=temporal_pred,
        temporal_pred_config=temporal_pred_cfg,
    )
    _save_audio_samples(
        vae,
        audio_paths,
        step=step,
        out_dir=out_dir,
        sample_rate=sample_rate,
        output_sr=sample_output_sr,
        device=device,
        amp_ctx=make_autocast(),
        sample_mode=sample_mode,
        n_samples=sample_n_samples,
        subdir_name="final",
    )
    logger.info(f"[exp9] Training done. Final checkpoint: {final_path}")

    return {
        "final_checkpoint": str(final_path),
        "steps": step,
    }


def _save_checkpoint(
    vae,
    discriminator,
    opt_g,
    opt_d,
    step,
    path,
    *,
    temporal_pred=None,
    temporal_pred_config: dict | None = None,
):
    state = {
        "format_version": 2,
        "vae_bottleneck": {
            "mu_proj": vae.mu_proj.state_dict(),
            "logvar_proj": vae.logvar_proj.state_dict(),
            "dec_proj": vae.dec_proj.state_dict(),
        },
        # Self-contained full Mimi state for strict/offline reload.
        "mimi_full": vae.mimi.state_dict(),
        "mimi_decoder": vae.mimi.decoder.state_dict(),
        "mimi_decoder_transformer": vae.mimi.decoder_transformer.state_dict(),
        "discriminator": discriminator.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "step": step,
        "latent_dim": vae.latent_dim,
        "encoder_dim": vae.encoder_dim,
        "dec_hidden_dim": vae.dec_hidden_dim,
    }
    if temporal_pred is not None:
        state["temporal_pred"] = temporal_pred.state_dict()
        if temporal_pred_config is not None:
            state["temporal_pred_config"] = dict(temporal_pred_config)
    if hasattr(vae.mimi, 'upsample'):
        state["mimi_upsample"] = vae.mimi.upsample.state_dict()
    torch.save(state, path)


def _load_checkpoint_into_vae(
    vae,
    ckpt_path: Path,
    device: torch.device,
    *,
    require_full_mimi: bool = False,
) -> dict:
    """
    Load exp9 checkpoint into an already-built PocketMimiVAE.

    Supports both:
      - v2 checkpoints with full Mimi state ("mimi_full")
      - older checkpoints with decoder-only Mimi state
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    if "mimi_full" in ckpt:
        vae.mimi.load_state_dict(ckpt["mimi_full"], strict=True)
    else:
        if require_full_mimi:
            raise KeyError(
                "Checkpoint does not contain mimi_full. "
                "Use a newer exp9 checkpoint or allow pretrained Mimi download."
            )
        if "mimi_decoder" not in ckpt or "mimi_decoder_transformer" not in ckpt:
            raise KeyError("Checkpoint missing Mimi weights (mimi_full or decoder/transformer keys).")
        vae.mimi.decoder.load_state_dict(ckpt["mimi_decoder"], strict=True)
        vae.mimi.decoder_transformer.load_state_dict(ckpt["mimi_decoder_transformer"], strict=True)
        if hasattr(vae.mimi, "upsample") and "mimi_upsample" in ckpt:
            vae.mimi.upsample.load_state_dict(ckpt["mimi_upsample"], strict=True)

    vae.mu_proj.load_state_dict(ckpt["vae_bottleneck"]["mu_proj"], strict=True)
    vae.logvar_proj.load_state_dict(ckpt["vae_bottleneck"]["logvar_proj"], strict=True)
    vae.dec_proj.load_state_dict(ckpt["vae_bottleneck"]["dec_proj"], strict=True)
    return ckpt


def _save_audio_samples(
    vae,
    sample_paths,
    step,
    out_dir,
    sample_rate,
    output_sr,
    device,
    amp_ctx,
    sample_mode: str = "mu",
    n_samples: int = 3,
    subdir_name: str | None = None,
):
    """Decode a few utterances and save GT + reconstruction WAVs."""
    logger = logging.getLogger("phase0")
    if subdir_name:
        samples_dir = out_dir / "samples" / subdir_name
    else:
        samples_dir = out_dir / "samples" / f"step_{step:06d}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    vae.eval()
    for i, path in enumerate(sample_paths[:n_samples]):
        try:
            wav, sr = torchaudio.load(path)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            if wav.shape[0] > 1:
                wav = wav[:1]
            wav = wav.unsqueeze(0).to(device)  # [1, 1, T]

            with torch.inference_mode(), amp_ctx:
                audio_hat = _reconstruct_audio(vae, wav, mode=sample_mode)

            gt_out = wav.squeeze(0).cpu()
            recon_out = audio_hat.squeeze(0).float().cpu()
            min_len = min(gt_out.shape[-1], recon_out.shape[-1])
            gt_out = gt_out[..., :min_len]
            recon_out = recon_out[..., :min_len]

            if sample_rate != output_sr:
                gt_out = torchaudio.functional.resample(gt_out, sample_rate, output_sr)
                recon_out = torchaudio.functional.resample(recon_out, sample_rate, output_sr)

            torchaudio.save(str(samples_dir / f"sample_{i:02d}_gt.wav"), gt_out, output_sr)
            torchaudio.save(str(samples_dir / f"sample_{i:02d}_recon.wav"), recon_out, output_sr)
        except Exception as e:
            logger.warning(f"[exp9] Sample {i} failed: {e}")

    vae.train()
    logger.info(f"[exp9] Saved {n_samples} audio samples at step {step} -> {samples_dir}")


# ── Latent extraction wrapper (PocketMimiVAE compatible) ──────────────


@torch.no_grad()
def extract_vae_latents(
    *,
    vae,
    utterances,
    zarr_path,
    index_path,
    device,
    sample_rate=24000,
    frame_size=1920,
):
    """Extract VAE latents using PocketMimiVAE.extract_latents().

    Same interface as stage2.vae_extract.extract_vae_latents but works
    with PocketMimiVAE (which has the same extract_latents API).
    """
    from phase0.data.io import save_latents_zarr, save_latents_index

    logger = logging.getLogger("phase0")
    vae.eval()
    entries = []

    for i, utt in enumerate(utterances):
        try:
            wav, sr = load_audio(utt.audio_path, target_sr=sample_rate)
            audio = wav.unsqueeze(0).to(device)  # [1, 1, T]

            mu = vae.extract_latents(audio)  # [1, D_vae, T']
            latents = mu.squeeze(0).permute(1, 0).cpu().numpy()  # [T', D_vae]
            n_frames = latents.shape[0]

            # Per-frame energy
            audio_np = wav.squeeze().cpu().numpy()
            T = len(audio_np)
            energy = np.zeros(n_frames, dtype=np.float32)
            chunk = max(1, T // max(n_frames, 1))
            for j in range(n_frames):
                start = j * chunk
                end = min(start + chunk, T)
                if start < T:
                    energy[j] = float(np.mean(audio_np[start:end] ** 2))

            timestamps = (np.arange(n_frames, dtype=np.float32) / 12.5).astype(np.float32)

            save_latents_zarr(
                latents=latents, energy=energy, timestamps=timestamps,
                speaker_id=utt.speaker_id, utterance_id=utt.utterance_id,
                zarr_path=zarr_path,
            )
            entries.append({
                "utterance_id": utt.utterance_id,
                "speaker_id": utt.speaker_id,
                "n_frames": n_frames,
                "duration_sec": utt.duration_sec,
                "audio_path": str(utt.audio_path),
            })

            if (i + 1) % 50 == 0:
                logger.info(f"[exp9] {i+1}/{len(utterances)} utterances extracted")

        except Exception as e:
            logger.warning(f"[exp9] Failed for {utt.utterance_id}: {e}")
            continue

    save_latents_index(entries, index_path)
    logger.info(f"[exp9] Done. {len(entries)} utterances -> {zarr_path}")


# ── Reconstruction metrics ────────────────────────────────────────────


def _mel_distance(audio, audio_hat, sr, n_mels=80):
    import torchaudio.transforms as T
    mel_transform = T.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=256, n_mels=n_mels,
    ).to(audio.device)
    mel = mel_transform(audio.squeeze())
    mel_hat = mel_transform(audio_hat.squeeze())
    log_mel = torch.log(mel.clamp_min(1e-5))
    log_mel_hat = torch.log(mel_hat.clamp_min(1e-5))
    return float(nn.functional.l1_loss(log_mel_hat, log_mel).item())


def _compute_recon_metrics(
    vae,
    utterances,
    device,
    n_utterances: int = 50,
    max_duration_sec: float = 10.0,
    mode: str = "mu",
    logger=None,
):
    vae.eval()
    l1_errors = []
    mel_errors = []
    selected = [u for u in utterances if u.duration_sec <= max_duration_sec][:n_utterances]

    for utt in selected:
        try:
            wav, sr = load_audio(utt.audio_path, target_sr=24000)
            audio = wav.unsqueeze(0).to(device)
            length = audio.shape[-1]

            with torch.inference_mode():
                audio_hat = _reconstruct_audio(vae, audio, mode=mode)[..., :length]

            l1 = float(nn.functional.l1_loss(audio_hat, audio).item())
            l1_errors.append(l1)

            mel_err = _mel_distance(audio.squeeze(), audio_hat.squeeze(), 24000)
            mel_errors.append(mel_err)
        except Exception as e:
            if logger:
                logger.warning(f"[exp9] Recon eval failed for {utt.utterance_id}: {e}")

    return {
        "n_utterances": len(l1_errors),
        "l1_mean": float(np.mean(l1_errors)) if l1_errors else float("nan"),
        "l1_std": float(np.std(l1_errors)) if l1_errors else float("nan"),
        "mel_distance_mean": float(np.mean(mel_errors)) if mel_errors else float("nan"),
        "mel_distance_std": float(np.std(mel_errors)) if mel_errors else float("nan"),
    }


# ── Main ──────────────────────────────────────────────────────────────


import logging


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 2 Exp9: Pocket-TTS Mimi VAE-GAN")
    p.add_argument("--config", type=str, default="configs/tier2_exp9_pocket_mimi_vae.yaml")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="Override VAE-GAN max_steps")
    p.add_argument("--phase1-max-steps", type=int, default=None, help="Override Phase 1 max_steps")
    p.add_argument(
        "--benchmark-pretrained-only",
        action="store_true",
        help="Benchmark pretrained Mimi reconstruction and exit before training.",
    )
    p.add_argument(
        "--export-samples-only",
        action="store_true",
        help="Load --checkpoint-path, export audio samples, and exit.",
    )
    p.add_argument(
        "--extract-latents-only",
        action="store_true",
        help="Load --checkpoint-path (or the run's final checkpoint), extract latents.zarr + indices, and exit.",
    )
    p.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Checkpoint to load for --export-samples-only/--extract-latents-only.",
    )
    args = p.parse_args()

    if os.environ.get("NO_TORCH_COMPILE"):
        os.environ["TORCH_COMPILE_DISABLE"] = "1"

    logger = setup_logging(name="phase0")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_id = args.run_id or _default_run_id()
    out_root = Path(cfg["output"]["out_dir"])
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    run = register_run(
        experiment="exp9_pocket_mimi_vae", run_id=run_id, config_path=args.config,
        config=cfg, cli_args=sys.argv[1:], out_dir=out_dir, log_name="phase0",
    )

    set_seed(int(cfg.get("seed", 42)))
    device = _device_from_config(cfg["train"]["device"])

    # ── Load utterances ───────────────────────────────────────────────
    data_cfg = cfg["data"]
    splits = load_splits(data_cfg["splits_dir"])
    all_speakers = splits.train_speakers + splits.eval_speakers
    utterances = get_utterances(data_cfg["librispeech_path"], all_speakers, data_cfg["subset"])
    utterances = [u for u in utterances if u.duration_sec >= float(data_cfg["min_duration_sec"])]
    max_utts = data_cfg.get("max_utterances")
    if max_utts is not None:
        utterances = utterances[:int(max_utts)]

    audio_paths = [str(u.audio_path) for u in utterances]
    logger.info(f"[exp9] Run id: {run_id}, device: {device}, utterances: {len(utterances)}")

    eval_speaker_set = set(splits.eval_speakers)
    eval_utterances = [u for u in utterances if u.speaker_id in eval_speaker_set]

    # ── Build models ──────────────────────────────────────────────────
    logger.info("[exp9] Building PocketMimiVAE...")
    from stage2.pocket_mimi_vae import build_pocket_mimi_vae
    from stage2.discriminator import MultiScaleSTFTDiscriminator

    vae_cfg = cfg["vae"]
    load_pretrained_mimi_cfg = bool(vae_cfg.get("load_pretrained_mimi", True))
    # When operating on an existing checkpoint, avoid downloading pretrained Mimi weights.
    load_pretrained_mimi = load_pretrained_mimi_cfg and not (args.export_samples_only or args.extract_latents_only)
    vae = build_pocket_mimi_vae(
        latent_dim=int(vae_cfg["latent_dim"]),
        dec_hidden_dim=int(vae_cfg.get("dec_hidden_dim", 256)),
        freeze_encoder=bool(vae_cfg.get("freeze_encoder", True)),
        freeze_decoder=bool(vae_cfg.get("freeze_decoder", False)),
        device=str(device),
        seanet_ratios=vae_cfg.get("seanet_ratios"),
        transformer_num_layers=vae_cfg.get("transformer_num_layers"),
        transformer_context=vae_cfg.get("transformer_context"),
        allow_partial_pretrained_load=bool(vae_cfg.get("allow_partial_pretrained_load", False)),
        load_pretrained_mimi=load_pretrained_mimi,
    )
    logger.info(f"[exp9] VAE built. Trainable params: {sum(p.numel() for p in vae.trainable_parameters()):,}")

    vae_train_cfg = cfg["vae_train"]

    if args.export_samples_only:
        if not args.checkpoint_path:
            raise ValueError("--export-samples-only requires --checkpoint-path")
        ckpt_path = Path(args.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        _load_checkpoint_into_vae(
            vae,
            ckpt_path,
            device,
            require_full_mimi=not load_pretrained_mimi,
        )
        sample_mode = str(vae_train_cfg.get("sample_mode", "mu"))
        sample_n = int(vae_train_cfg.get("sample_n_samples", 8))
        sample_rate = int(vae_train_cfg.get("sample_rate", 24000))
        output_sr = int(vae_train_cfg.get("sample_output_sr", 48000))
        use_amp = bool(vae_train_cfg.get("amp", True)) and device.type == "cuda"
        amp_dtype = str(vae_train_cfg.get("amp_dtype", "fp16")).lower()
        amp_dtype_t = torch.float16 if amp_dtype in ("fp16", "float16") else torch.bfloat16
        _save_audio_samples(
            vae,
            [str(u.audio_path) for u in eval_utterances],
            step=0,
            out_dir=out_dir,
            sample_rate=sample_rate,
            output_sr=output_sr,
            device=device,
            amp_ctx=torch.autocast(device_type=device.type, dtype=amp_dtype_t, enabled=use_amp),
            sample_mode=sample_mode,
            n_samples=sample_n,
            subdir_name="checkpoint_export",
        )
        logger.info(
            f"[exp9] Exported checkpoint samples to {out_dir / 'samples' / 'checkpoint_export'} "
            f"(mode={sample_mode}, n={sample_n})"
        )
        finalize_run(run, key_metrics={})
        return 0

    if args.extract_latents_only:
        if args.checkpoint_path:
            ckpt_path = Path(args.checkpoint_path)
        else:
            ckpt_path = out_dir / "checkpoints" / "vae_gan_final.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. "
                "Pass --checkpoint-path or set --run-id to an existing exp9 run directory."
            )
        _load_checkpoint_into_vae(
            vae,
            ckpt_path,
            device,
            require_full_mimi=not load_pretrained_mimi,
        )

        logger.info("[exp9] Extracting VAE latents...")
        latents_zarr = out_dir / "latents.zarr"
        latents_index = out_dir / "latents_index.parquet"
        extract_vae_latents(
            vae=vae,
            utterances=utterances,
            zarr_path=latents_zarr,
            index_path=latents_index,
            device=device,
        )

        window_size = int(cfg["context"]["window_size"])
        horizons_k = [int(k) for k in cfg["context"]["horizons_k"]]
        max_lag = max(horizons_k)
        frames_index = out_dir / "phase0_frames.parquet"
        build_frames_index(
            splits_dir=data_cfg["splits_dir"],
            latents_index_path=latents_index,
            latents_zarr_path=latents_zarr,
            window_size=window_size,
            max_lag=max_lag,
            min_duration_sec=float(data_cfg["min_duration_sec"]),
            out_frames_index_path=frames_index,
        )

        logger.info(f"[exp9] Extracted latents to {latents_zarr} and frames index to {frames_index}")
        finalize_run(run, key_metrics={})
        return 0

    disc_n_ffts = tuple(vae_train_cfg.get("disc_n_ffts", [256, 512, 1024, 2048]))
    disc_channels = int(vae_train_cfg.get("disc_channels", 32))
    discriminator = MultiScaleSTFTDiscriminator(
        n_ffts=disc_n_ffts, channels=disc_channels,
    ).to(device)
    logger.info(f"[exp9] Discriminator built. Params: {sum(p.numel() for p in discriminator.parameters()):,}")

    # WavLM distillation
    wavlm = None
    lambda_distill = float(vae_train_cfg.get("lambda_distill", 1.0))
    if lambda_distill > 0:
        try:
            from stage2.wavlm_distill import WavLMDistillation
            wavlm_device = vae_train_cfg.get("wavlm_device", "auto")
            if wavlm_device == "auto":
                wavlm_device = str(device)
            wavlm_layer = int(vae_train_cfg.get("wavlm_layer", 7))
            wavlm = WavLMDistillation(
                latent_dim=int(vae_cfg["latent_dim"]),
                layer=wavlm_layer,
                device=wavlm_device,
            )
            logger.info(f"[exp9] WavLM distillation loaded (device={wavlm_device}, layer={wavlm_layer})")
        except Exception as e:
            logger.warning(f"[exp9] Could not load WavLM, disabling distillation: {e}")
            wavlm = None
            lambda_distill = 0.0

    # Pre-training benchmark on pretrained checkpoint quality.
    pre_cfg = cfg.get("pretrain_benchmark", {})
    if bool(pre_cfg.get("enabled", True)):
        n_bench = int(pre_cfg.get("n_utterances", 50))
        max_dur_bench = float(pre_cfg.get("max_duration_sec", 10.0))
        logger.info("[exp9] Benchmarking pretrained reconstruction (before training)...")
        pretrain_metrics = {
            "mimi_pretrained": _compute_recon_metrics(
                vae,
                eval_utterances,
                device,
                n_utterances=n_bench,
                max_duration_sec=max_dur_bench,
                mode="pretrained_mimi",
                logger=logger,
            ),
            "vae_mu_init": _compute_recon_metrics(
                vae,
                eval_utterances,
                device,
                n_utterances=n_bench,
                max_duration_sec=max_dur_bench,
                mode="mu",
                logger=logger,
            ),
        }
        with open(out_dir / "pretrain_recon_metrics.json", "w") as f:
            json.dump(pretrain_metrics, f, indent=2)
        logger.info(
            "[exp9] Pretrain benchmark: "
            f"mimi_pretrained mel={pretrain_metrics['mimi_pretrained']['mel_distance_mean']:.4f} "
            f"l1={pretrain_metrics['mimi_pretrained']['l1_mean']:.4f} | "
            f"vae_mu_init mel={pretrain_metrics['vae_mu_init']['mel_distance_mean']:.4f} "
            f"l1={pretrain_metrics['vae_mu_init']['l1_mean']:.4f}"
        )
        if bool(pre_cfg.get("save_audio", True)):
            sample_n = int(pre_cfg.get("n_samples", 8))
            use_amp = bool(vae_train_cfg.get("amp", True)) and device.type == "cuda"
            amp_dtype = str(vae_train_cfg.get("amp_dtype", "fp16")).lower()
            amp_dtype_t = torch.float16 if amp_dtype in ("fp16", "float16") else torch.bfloat16
            _save_audio_samples(
                vae,
                [str(u.audio_path) for u in eval_utterances],
                step=0,
                out_dir=out_dir,
                sample_rate=int(vae_train_cfg["sample_rate"]),
                output_sr=int(vae_train_cfg.get("sample_output_sr", 48000)),
                device=device,
                amp_ctx=torch.autocast(device_type=device.type, dtype=amp_dtype_t, enabled=use_amp),
                sample_mode="pretrained_mimi",
                n_samples=sample_n,
                subdir_name="pretrain_mimi",
            )
            _save_audio_samples(
                vae,
                [str(u.audio_path) for u in eval_utterances],
                step=0,
                out_dir=out_dir,
                sample_rate=int(vae_train_cfg["sample_rate"]),
                output_sr=int(vae_train_cfg.get("sample_output_sr", 48000)),
                device=device,
                amp_ctx=torch.autocast(device_type=device.type, dtype=amp_dtype_t, enabled=use_amp),
                sample_mode="mu",
                n_samples=sample_n,
                subdir_name="pretrain_vae_mu",
            )
            logger.info(
                f"[exp9] Wrote pretrain audio samples (n={sample_n}) to "
                f"{out_dir / 'samples' / 'pretrain_mimi'} and {out_dir / 'samples' / 'pretrain_vae_mu'}"
            )
        if args.benchmark_pretrained_only:
            km = {
                "pretrain_mimi_recon_mel": pretrain_metrics["mimi_pretrained"].get("mel_distance_mean", float("nan")),
                "pretrain_mimi_recon_l1": pretrain_metrics["mimi_pretrained"].get("l1_mean", float("nan")),
                "pretrain_vae_mu_recon_mel": pretrain_metrics["vae_mu_init"].get("mel_distance_mean", float("nan")),
                "pretrain_vae_mu_recon_l1": pretrain_metrics["vae_mu_init"].get("l1_mean", float("nan")),
            }
            finalize_run(run, key_metrics=km)
            logger.info("[exp9] benchmark-pretrained-only complete; exiting before training.")
            return 0

    # ── 1. Train VAE-GAN ──────────────────────────────────────────────
    logger.info("[exp9] Training VAE-GAN...")
    vae_max_steps = args.max_steps or int(vae_train_cfg["max_steps"])

    train_result = train_vae_gan(
        audio_paths=audio_paths,
        vae=vae,
        discriminator=discriminator,
        wavlm_distill=wavlm,
        out_dir=out_dir,
        lambda_recon=float(vae_train_cfg.get("lambda_recon", 1.0)),
        beta_kl=float(vae_train_cfg["beta_kl"]),
        kl_normalize_by_elements=bool(vae_train_cfg.get("kl_normalize_by_elements", True)),
        kl_start_steps=int(vae_train_cfg.get("kl_start_steps", 0)),
        kl_ramp_steps=int(vae_train_cfg.get("kl_ramp_steps", 0)),
        lambda_pred=float(vae_train_cfg.get("lambda_pred", 0.0)),
        pred_alpha=float(vae_train_cfg.get("pred_alpha", 0.5)),
        pred_kind=str(vae_train_cfg.get("pred_kind", "causal_conv")),
        pred_kernel_size=int(vae_train_cfg.get("pred_kernel_size", 6)),
        pred_window_size=int(vae_train_cfg.get("pred_window_size", 6)),
        pred_hidden_dim=int(vae_train_cfg.get("pred_hidden_dim", 256)),
        pred_start_steps=int(vae_train_cfg.get("pred_start_steps", 0)),
        pred_use_mu=bool(vae_train_cfg.get("pred_use_mu", True)),
        lambda_adv=float(vae_train_cfg["lambda_adv"]),
        lambda_feat=float(vae_train_cfg["lambda_feat"]),
        lambda_distill=lambda_distill,
        lambda_latent_align=float(vae_train_cfg.get("lambda_latent_align", 0.0)),
        distill_start_steps=int(vae_train_cfg.get("distill_start_steps", 0)),
        lr_schedule=str(vae_train_cfg.get("lr_schedule", "constant")),
        segment_sec=float(vae_train_cfg["segment_sec"]),
        sample_rate=int(vae_train_cfg["sample_rate"]),
        batch_size=int(vae_train_cfg["batch_size"]),
        num_workers=int(vae_train_cfg["num_workers"]),
        max_steps=vae_max_steps,
        lr_generator=float(vae_train_cfg["lr_generator"]),
        lr_discriminator=float(vae_train_cfg["lr_discriminator"]),
        weight_decay=float(vae_train_cfg["weight_decay"]),
        grad_clip_norm=float(vae_train_cfg["grad_clip_norm"]),
        gan_warmup_steps=int(vae_train_cfg.get("gan_warmup_steps", 0)),
        deterministic_warmup_steps=int(vae_train_cfg.get("deterministic_warmup_steps", 0)),
        log_every=int(vae_train_cfg["log_every"]),
        save_every=int(vae_train_cfg["save_every"]),
        sample_every=int(vae_train_cfg.get("sample_every", 5000)),
        sample_n_samples=int(vae_train_cfg.get("sample_n_samples", 3)),
        sample_output_sr=int(vae_train_cfg.get("sample_output_sr", 48000)),
        sample_mode=str(vae_train_cfg.get("sample_mode", "mu")),
        seed=int(cfg.get("seed", 42)),
        device=device,
        amp=bool(vae_train_cfg.get("amp", True)),
        amp_dtype=str(vae_train_cfg.get("amp_dtype", "fp16")),
    )
    logger.info(f"[exp9] VAE-GAN training done: {train_result['final_checkpoint']}")

    # Free discriminator and WavLM to save memory for Phase 1
    del discriminator, wavlm
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── 2. Extract latents ────────────────────────────────────────────
    logger.info("[exp9] Extracting VAE latents...")
    latents_zarr = out_dir / "latents.zarr"
    latents_index = out_dir / "latents_index.parquet"

    extract_vae_latents(
        vae=vae,
        utterances=utterances,
        zarr_path=latents_zarr,
        index_path=latents_index,
        device=device,
    )

    # ── 3. Build frames index ─────────────────────────────────────────
    window_size = int(cfg["context"]["window_size"])
    horizons_k = [int(k) for k in cfg["context"]["horizons_k"]]
    max_lag = max(horizons_k)

    frames_index = out_dir / "phase0_frames.parquet"
    build_frames_index(
        splits_dir=data_cfg["splits_dir"],
        latents_index_path=latents_index,
        latents_zarr_path=latents_zarr,
        window_size=window_size,
        max_lag=max_lag,
        min_duration_sec=float(data_cfg["min_duration_sec"]),
        out_frames_index_path=frames_index,
    )

    # ── 4. Phase 1 diagnostic battery ─────────────────────────────────
    logger.info("[exp9] Phase 1 diagnostic battery...")
    phase1_dir = out_dir / "phase1"
    phase1_dir.mkdir(parents=True, exist_ok=True)

    phase1_max_steps = args.phase1_max_steps or int(cfg["train"]["max_steps"])

    results = []
    for k in horizons_k:
        r = train_and_eval_for_k(
            frames_index_path=frames_index,
            latents_dir=latents_zarr,
            splits_dir=data_cfg["splits_dir"],
            latents_index_path=latents_index,
            out_dir=phase1_dir,
            horizon_k=k,
            window_size=window_size,
            slice_name=str(cfg.get("slice", "all")),
            seed=int(cfg["train"]["seed"]),
            device=device,
            n_components=int(cfg["model"]["n_components"]),
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            n_hidden_layers=int(cfg["model"]["n_hidden_layers"]),
            dropout=float(cfg["model"]["dropout"]),
            min_log_sigma=float(cfg["model"]["min_log_sigma"]),
            max_log_sigma=float(cfg["model"]["max_log_sigma"]),
            batch_size=int(cfg["train"]["batch_size"]),
            num_workers=int(cfg["train"]["num_workers"]),
            max_steps=phase1_max_steps,
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
            rollout_sample_from_mixture=bool(cfg["rollout"]["sample_from_mixture"]),
            model_type="mdn",
            compile_model=bool(cfg["train"].get("compile", False)),
            compile_mode=str(cfg["train"].get("compile_mode", "default")),
            amp=bool(cfg["train"].get("amp", False)),
            amp_dtype=str(cfg["train"].get("amp_dtype", "bf16")),
        )
        results.append(r)

    write_results(
        results,
        metrics_path=str(phase1_dir / "metrics.json"),
        tables_path=str(phase1_dir / "tables.csv"),
    )

    # ── 5. Injection diagnostic ───────────────────────────────────────
    ckpt_path = phase1_dir / "checkpoints" / "mdn_k1_final.pt"
    inj_result = None
    if ckpt_path.exists():
        from phase1.checkpoints import load_phase1_checkpoint
        from phase1.injection_diag import run_injection_diagnostic
        from phase1.train_eval import fit_unconditional_baseline

        inj_model, _ = load_phase1_checkpoint(ckpt_path, device=device)
        inj_baseline = fit_unconditional_baseline(
            frames_index_path=frames_index,
            latents_dir=latents_zarr,
            window_size=window_size,
            horizon_k=1,
            slice_name=str(cfg.get("slice", "all")),
            max_samples=cfg["train"].get("max_train_samples"),
        )
        inj_cfg = cfg.get("injection", {})
        inj_result = run_injection_diagnostic(
            model=inj_model,
            baseline=inj_baseline,
            latents_dir=latents_zarr,
            latents_index_path=latents_index,
            splits_dir=data_cfg["splits_dir"],
            horizon_k=1,
            window_size=window_size,
            k_steps=int(inj_cfg.get("k_steps", 16)),
            n_eval_utterances=int(inj_cfg.get("n_eval_utterances", 16)),
            segments_per_utt=int(inj_cfg.get("segments_per_utt", 8)),
            max_frames_per_utt=int(inj_cfg.get("max_frames_per_utterance", 2000)),
            seed=42,
            device=device,
            mode_inject_after_steps={
                "A_teacher": None,
                "B_periodic": [int(x) for x in inj_cfg.get("inject_after_steps_periodic", [4, 8, 12])],
                "C_one_shot": [int(x) for x in inj_cfg.get("inject_after_steps_one_shot", [1])],
                "D_rollout": [],
            },
            sample_from_model=False,
        )
        with open(phase1_dir / "injection_diag.json", "w") as f:
            json.dump(inj_result, f, indent=2)
        logger.info("[exp9] Injection diagnostic written")

    # ── 6. Reconstruction quality ─────────────────────────────────────
    recon_cfg = cfg.get("recon_eval", {})
    recon_mode = str(recon_cfg.get("mode", "mu"))
    recon_metrics = _compute_recon_metrics(
        vae, eval_utterances, device,
        n_utterances=int(recon_cfg.get("n_utterances", 50)),
        max_duration_sec=float(recon_cfg.get("max_duration_sec", 10.0)),
        mode=recon_mode,
        logger=logger,
    )
    with open(out_dir / "recon_metrics.json", "w") as f:
        json.dump(recon_metrics, f, indent=2)

    # ── 7. Summary ────────────────────────────────────────────────────
    best_dnll = min(
        (float(r.eval.get("dnll")) for r in results if r.eval.get("dnll") is not None),
        default=float("nan"),
    )
    summary_row = {
        "experiment": "exp9_pocket_mimi_vae",
        "latent_dim": int(vae_cfg["latent_dim"]),
        "best_eval_dnll": best_dnll,
        "recon_l1": recon_metrics.get("l1_mean", float("nan")),
        "recon_mel": recon_metrics.get("mel_distance_mean", float("nan")),
    }
    if inj_result and "modes" in inj_result:
        d_mode = inj_result["modes"].get("D_rollout", {})
        summary_row["rollout_state_err"] = d_mode.get("state_err", float("nan"))
        summary_row["rollout_cos"] = d_mode.get("cos", float("nan"))

    summary = pd.DataFrame([summary_row])
    summary_path = out_dir / "summary.csv"
    summary.to_csv(str(summary_path), index=False)
    logger.info(f"[exp9] Summary written: {summary_path}")
    logger.info(f"\n{summary.to_string(index=False)}")

    km = {
        "best_dnll": best_dnll,
        "recon_mel": recon_metrics.get("mel_distance_mean", float("nan")),
        "recon_l1": recon_metrics.get("l1_mean", float("nan")),
    }
    finalize_run(run, key_metrics=km)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
