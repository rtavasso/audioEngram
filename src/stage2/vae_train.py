"""
VAE training loop for AR-Friendly VAE.

Loads audio segments from LibriSpeech, trains VAE with:
  L = L_recon + beta*L_kl + lambda_smooth*L_smooth + lambda_pred*L_pred

Two optimizers: one for VAE params, one for predictor params.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .vae import ARFriendlyVAE
from .vae_losses import reconstruction_loss, kl_loss, smoothness_loss, predictability_loss
from .vae_predictor import LatentPredictor


logger = logging.getLogger("phase0")


class AudioSegmentDataset(Dataset):
    """Dataset yielding fixed-length audio segments from LibriSpeech utterances."""

    def __init__(
        self,
        audio_paths: list[str],
        segment_samples: int,
        sample_rate: int = 24000,
        seed: int = 42,
    ):
        self.audio_paths = audio_paths
        self.segment_samples = segment_samples
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        import torchaudio

        path = self.audio_paths[idx]
        wav, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # Take first channel if multi-channel
        if wav.shape[0] > 1:
            wav = wav[:1]

        T = wav.shape[-1]
        if T >= self.segment_samples:
            # Random crop
            start = int(self.rng.integers(0, T - self.segment_samples + 1))
            wav = wav[:, start:start + self.segment_samples]
        else:
            # Pad with zeros
            wav = torch.nn.functional.pad(wav, (0, self.segment_samples - T))

        return wav  # [1, segment_samples]


def _save_audio_samples(
    vae: ARFriendlyVAE,
    sample_paths: list[str],
    step: int,
    out_dir: Path,
    sample_rate: int,
    output_sr: int,
    device: torch.device,
    amp_ctx,
    n_samples: int = 3,
) -> None:
    """Decode a few utterances and save GT + reconstruction WAVs."""
    import torchaudio

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
                out = vae(wav)
                audio_hat = out["audio_hat"]

            # Save GT and reconstruction
            gt_out = wav.squeeze(0).cpu()  # [1, T]
            recon_out = audio_hat.squeeze(0).float().cpu()  # [1, T]
            min_len = min(gt_out.shape[-1], recon_out.shape[-1])
            gt_out = gt_out[..., :min_len]
            recon_out = recon_out[..., :min_len]

            if sample_rate != output_sr:
                gt_out = torchaudio.functional.resample(gt_out, sample_rate, output_sr)
                recon_out = torchaudio.functional.resample(recon_out, sample_rate, output_sr)

            torchaudio.save(str(samples_dir / f"sample_{i:02d}_gt.wav"), gt_out, output_sr)
            torchaudio.save(str(samples_dir / f"sample_{i:02d}_recon.wav"), recon_out, output_sr)
        except Exception as e:
            logger.warning(f"[vae_train] Sample {i} failed: {e}")

    vae.train()
    logger.info(f"[vae_train] Saved {n_samples} audio samples at step {step} -> {samples_dir}")


def train_vae(
    *,
    audio_paths: list[str],
    vae: ARFriendlyVAE,
    out_dir: Path,
    # Loss weights
    beta: float = 0.01,
    lambda_smooth: float = 0.0,
    lambda_pred: float = 0.0,
    pred_window_size: int = 8,
    pred_hidden_dim: int = 256,
    # Training
    segment_sec: float = 4.0,
    sample_rate: int = 24000,
    batch_size: int = 4,
    num_workers: int = 2,
    max_steps: int = 50000,
    lr: float = 1e-4,
    lr_predictor: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip_norm: float = 1.0,
    log_every: int = 100,
    save_every: int = 5000,
    sample_every: int = 5000,
    sample_output_sr: int = 48000,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
    amp: bool = True,
    amp_dtype: str = "fp16",
) -> dict:
    """
    Train the AR-Friendly VAE.

    Returns:
        dict with final loss values and checkpoint path.
    """
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
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Predictor (for L_pred if enabled)
    predictor = None
    opt_pred = None
    if lambda_pred > 0:
        predictor = LatentPredictor(
            latent_dim=vae.latent_dim,
            window_size=pred_window_size,
            hidden_dim=pred_hidden_dim,
        ).to(device)
        opt_pred = torch.optim.AdamW(predictor.parameters(), lr=lr_predictor, weight_decay=weight_decay)

    # VAE optimizer: bottleneck + any unfrozen encoder/decoder params
    vae_params = list(vae.trainable_parameters())
    opt_vae = torch.optim.AdamW(vae_params, lr=lr, weight_decay=weight_decay)

    # AMP setup
    use_amp = amp and device.type == "cuda"
    amp_dtype_t = torch.float16 if amp_dtype.lower() in ("fp16", "float16") else torch.bfloat16
    scaler = torch.GradScaler("cuda") if use_amp and amp_dtype_t == torch.float16 else None
    autocast = torch.autocast(device_type=device.type, dtype=amp_dtype_t, enabled=use_amp)

    logger.info(
        f"[vae_train] latent_dim={vae.latent_dim} beta={beta} "
        f"lambda_smooth={lambda_smooth} lambda_pred={lambda_pred} "
        f"batch={batch_size} lr={lr} steps={max_steps} amp={use_amp}"
    )

    vae.train()
    if predictor is not None:
        predictor.train()

    step = 0
    data_iter = iter(loader)
    loss_accum = {"total": 0.0, "recon": 0.0, "kl": 0.0, "smooth": 0.0, "pred": 0.0}
    log_count = 0

    while step < max_steps:
        try:
            audio = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            audio = next(data_iter)

        audio = audio.to(device, non_blocking=True)  # [B, 1, T]

        opt_vae.zero_grad(set_to_none=True)
        if opt_pred is not None:
            opt_pred.zero_grad(set_to_none=True)

        with autocast:
            out = vae(audio)
            l_recon = reconstruction_loss(audio, out["audio_hat"])
            l_kl = kl_loss(out["mu"], out["logvar"])

            loss = l_recon + beta * l_kl

            l_smooth = audio.new_tensor(0.0)
            if lambda_smooth > 0:
                l_smooth = smoothness_loss(out["z"])
                loss = loss + lambda_smooth * l_smooth

            l_pred = audio.new_tensor(0.0)
            if lambda_pred > 0 and predictor is not None:
                l_pred = predictability_loss(out["z"], predictor, window_size=pred_window_size)
                loss = loss + lambda_pred * l_pred

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(opt_vae)
                torch.nn.utils.clip_grad_norm_(vae_params, max_norm=grad_clip_norm)
                if opt_pred is not None and predictor is not None:
                    scaler.unscale_(opt_pred)
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=grad_clip_norm)
            scaler.step(opt_vae)
            if opt_pred is not None:
                scaler.step(opt_pred)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(vae_params, max_norm=grad_clip_norm)
                if predictor is not None:
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=grad_clip_norm)
            opt_vae.step()
            if opt_pred is not None:
                opt_pred.step()

        step += 1
        loss_accum["total"] += float(loss.item())
        loss_accum["recon"] += float(l_recon.item())
        loss_accum["kl"] += float(l_kl.item())
        loss_accum["smooth"] += float(l_smooth.item())
        loss_accum["pred"] += float(l_pred.item())
        log_count += 1

        if log_every and step % log_every == 0:
            avg = {k: v / max(log_count, 1) for k, v in loss_accum.items()}
            logger.info(
                f"[vae_train] step={step}/{max_steps} "
                f"loss={avg['total']:.4f} recon={avg['recon']:.4f} "
                f"kl={avg['kl']:.4f} smooth={avg['smooth']:.4f} pred={avg['pred']:.4f}"
            )
            loss_accum = {k: 0.0 for k in loss_accum}
            log_count = 0

        if save_every and step % save_every == 0:
            _save_checkpoint(vae, predictor, step, ckpt_dir / f"vae_step{step}.pt")

        if sample_every and step % sample_every == 0:
            _save_audio_samples(
                vae, audio_paths, step, out_dir,
                sample_rate=sample_rate,
                output_sr=sample_output_sr,
                device=device,
                amp_ctx=autocast,
            )

    # Final checkpoint
    final_path = ckpt_dir / "vae_final.pt"
    _save_checkpoint(vae, predictor, step, final_path)
    logger.info(f"[vae_train] Done. Final checkpoint: {final_path}")

    return {
        "final_checkpoint": str(final_path),
        "steps": step,
    }


def _save_checkpoint(
    vae: ARFriendlyVAE,
    predictor: Optional[LatentPredictor],
    step: int,
    path: Path,
) -> None:
    state = {
        "vae_bottleneck": {
            "mu_proj": vae.mu_proj.state_dict(),
            "logvar_proj": vae.logvar_proj.state_dict(),
            "dec_proj": vae.dec_proj.state_dict(),
        },
        "step": step,
        "latent_dim": vae.latent_dim,
        "encoder_dim": vae.encoder_dim,
        "dec_hidden_dim": vae.dec_hidden_dim,
    }
    # Save decoder state if it has trainable params
    decoder_params = {k: v for k, v in vae.mimi_decoder.state_dict().items()}
    if decoder_params:
        state["mimi_decoder"] = decoder_params
    # Save encoder state if it has trainable params
    encoder_trainable = any(p.requires_grad for p in vae.mimi_encoder.parameters())
    if encoder_trainable:
        state["mimi_encoder"] = vae.mimi_encoder.state_dict()
    if predictor is not None:
        state["predictor"] = predictor.state_dict()
    torch.save(state, path)


def load_vae_checkpoint(
    path: str | Path,
    vae: ARFriendlyVAE,
    predictor: Optional[LatentPredictor] = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Load VAE checkpoint into existing model."""
    ckpt = torch.load(str(path), map_location=device)
    vae.mu_proj.load_state_dict(ckpt["vae_bottleneck"]["mu_proj"])
    vae.logvar_proj.load_state_dict(ckpt["vae_bottleneck"]["logvar_proj"])
    vae.dec_proj.load_state_dict(ckpt["vae_bottleneck"]["dec_proj"])
    if "mimi_decoder" in ckpt:
        vae.mimi_decoder.load_state_dict(ckpt["mimi_decoder"])
    if "mimi_encoder" in ckpt:
        vae.mimi_encoder.load_state_dict(ckpt["mimi_encoder"])
    if predictor is not None and "predictor" in ckpt:
        predictor.load_state_dict(ckpt["predictor"])
    return ckpt
