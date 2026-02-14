#!/usr/bin/env python3
"""
CALM Speech Continuation experiment.

Trains a Continuous Audio Language Model on pre-extracted 32D VAE latents
from Exp 9 (PocketMimiVAE). Uses consistency-model-based one-step denoising
for autoregressive generation of speech continuations.

Pipeline:
  1. Load pre-extracted latents (zarr + parquet index)
  2. Build CALM model (backbone + short-context transformer + consistency head)
  3. Train with consistency distillation loss
  4. Evaluate rollout quality at k=1,2,4,8,16
  5. Generate audio continuations through VAE decoder
  6. Save metrics and audio samples

Usage:
  uv run python scripts/calm_speech_continuation.py --config configs/calm_speech_continuation.yaml
  uv run python scripts/calm_speech_continuation.py --config configs/calm_speech_continuation.yaml --max-steps 10
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torchaudio
import yaml

from experiment import register_run, finalize_run
from phase0.utils.logging import setup_logging
from phase0.utils.seed import set_seed


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _device_from_config(dev_str: str) -> torch.device:
    if dev_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev_str)


def _load_vae(vae_checkpoint: str | Path, device: torch.device):
    """Load trained PocketMimiVAE from checkpoint for audio decode."""
    from stage2.pocket_mimi_vae import build_pocket_mimi_vae

    ckpt = torch.load(vae_checkpoint, map_location=device)
    latent_dim = ckpt.get("latent_dim", 32)
    dec_hidden_dim = ckpt.get("dec_hidden_dim", 256)

    vae = build_pocket_mimi_vae(
        latent_dim=latent_dim,
        dec_hidden_dim=dec_hidden_dim,
        freeze_encoder=True,
        freeze_decoder=True,
        device=str(device),
        load_pretrained_mimi=False,
    )

    # Load weights
    if "mimi_full" in ckpt:
        vae.mimi.load_state_dict(ckpt["mimi_full"], strict=True)
    vae.mu_proj.load_state_dict(ckpt["vae_bottleneck"]["mu_proj"], strict=True)
    vae.logvar_proj.load_state_dict(ckpt["vae_bottleneck"]["logvar_proj"], strict=True)
    vae.dec_proj.load_state_dict(ckpt["vae_bottleneck"]["dec_proj"], strict=True)

    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    return vae


def decode_latents_to_audio(
    vae,
    latent_seq: torch.Tensor,
    device: torch.device,
    output_sr: int = 48000,
) -> torch.Tensor:
    """
    Decode latent sequence to audio through VAE decoder.

    Args:
        vae: PocketMimiVAE (eval mode)
        latent_seq: [T, D] latent frames
        device: torch device
        output_sr: target sample rate for output

    Returns:
        audio: [1, T_audio] at output_sr
    """
    # VAE decode expects [B, D, T] (channels-first)
    z = latent_seq.T.unsqueeze(0).to(device)  # [1, D, T]
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        z = z.float()
        audio = vae.decode(z)  # [1, 1, T_audio]

    audio = audio.squeeze(0).float().cpu()  # [1, T_audio]

    # Resample VAE sample rate -> output_sr (PocketMimiVAE is typically 24kHz)
    in_sr = int(getattr(getattr(vae, "mimi", None), "sample_rate", 24000))
    if output_sr != in_sr:
        audio = torchaudio.functional.resample(audio, in_sr, output_sr)

    return audio


def generate_audio_samples(
    calm_model,
    vae,
    eval_dataloader,
    out_dir: Path,
    device: torch.device,
    prompt_frames: int = 38,
    generate_frames: int = 125,
    temperature: float = 0.8,
    n_samples: int = 10,
    output_sr: int = 48000,
) -> list[dict]:
    """Generate audio continuations and save as WAVs."""
    logger = logging.getLogger("phase0")
    audio_dir = out_dir / "audio_samples"
    audio_dir.mkdir(parents=True, exist_ok=True)

    calm_model.eval()
    eval_iter = iter(eval_dataloader)
    manifests: list[dict] = []
    success = 0
    attempts = 0
    max_attempts = max(n_samples * 10, n_samples + 10)

    while success < n_samples and attempts < max_attempts:
        try:
            x_batch = next(eval_iter).to(device)
        except StopIteration:
            break

        B, S, D = x_batch.shape

        for b in range(B):
            if success >= n_samples or attempts >= max_attempts:
                break
            if S < prompt_frames + 1:
                continue

            attempts += 1
            prompt = x_batch[b:b+1, :prompt_frames]
            n_gen = min(generate_frames, S - prompt_frames)
            gt_seq = x_batch[b, :prompt_frames + n_gen]

            with torch.no_grad():
                generated = calm_model.generate(prompt, n_gen, temperature=temperature)

            full_gen = torch.cat([prompt.squeeze(0), generated], dim=0)

            try:
                # Decode generated continuation
                gen_audio = decode_latents_to_audio(vae, full_gen, device, output_sr)
                gen_path = audio_dir / f"sample_{success:02d}_generated.wav"
                torchaudio.save(str(gen_path), gen_audio, output_sr)

                # Decode ground truth for comparison
                gt_audio = decode_latents_to_audio(vae, gt_seq, device, output_sr)
                gt_path = audio_dir / f"sample_{success:02d}_gt.wav"
                torchaudio.save(str(gt_path), gt_audio, output_sr)

                manifests.append({
                    "sample_id": success,
                    "generated_path": str(gen_path),
                    "gt_path": str(gt_path),
                    "prompt_frames": prompt_frames,
                    "generated_frames": n_gen,
                    "temperature": temperature,
                })
                success += 1
            except Exception as e:
                logger.warning(f"[calm] Audio decode failed (attempt {attempts}, ok={success}): {e}")

    # Save manifest
    manifest_path = audio_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifests, f, indent=2)

    calm_model.train()
    if attempts >= max_attempts and success < n_samples:
        logger.warning(
            f"[calm] Stopped audio generation early: ok={success}/{n_samples}, attempts={attempts}/{max_attempts}"
        )
    logger.info(f"[calm] Generated {len(manifests)} audio samples -> {audio_dir}")
    return manifests


def main() -> int:
    p = argparse.ArgumentParser(description="CALM Speech Continuation")
    p.add_argument("--config", type=str, default="configs/calm_speech_continuation.yaml")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    p.add_argument("--skip-audio", action="store_true", help="Skip audio generation (faster)")
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
        experiment="calm_speech_continuation",
        run_id=run_id,
        config_path=args.config,
        config=cfg,
        cli_args=sys.argv[1:],
        out_dir=out_dir,
        log_name="phase0",
    )

    set_seed(int(cfg.get("seed", 42)))
    device = _device_from_config(cfg["train"]["device"])
    logger.info(f"[calm] Run id: {run_id}, device: {device}")

    # ── 1. Build data loaders ────────────────────────────────────────
    from calm.data import build_calm_dataloader

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    train_loader = build_calm_dataloader(
        latents_dir=data_cfg["latents_dir"],
        latents_index_path=data_cfg["latents_index_path"],
        splits_dir=data_cfg["splits_dir"],
        split="train",
        seq_len=int(data_cfg["seq_len"]),
        min_utterance_len=int(data_cfg.get("min_utterance_len", 32)),
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg["num_workers"]),
        seed=int(cfg.get("seed", 42)),
    )

    eval_loader = build_calm_dataloader(
        latents_dir=data_cfg["latents_dir"],
        latents_index_path=data_cfg["latents_index_path"],
        splits_dir=data_cfg["splits_dir"],
        split="eval",
        seq_len=int(data_cfg["seq_len"]),
        min_utterance_len=int(data_cfg.get("min_utterance_len", 32)),
        batch_size=int(train_cfg["batch_size"]),
        num_workers=0,
        seed=int(cfg.get("seed", 42)) + 999,
    )

    # ── 2. Build CALM model ──────────────────────────────────────────
    from calm.model import CALM

    model_cfg = cfg["model"]
    model = CALM(
        latent_dim=int(cfg["vae"]["latent_dim"]),
        d_model=int(model_cfg["d_model"]),
        n_backbone_layers=int(model_cfg["n_backbone_layers"]),
        n_short_ctx_layers=int(model_cfg["n_short_ctx_layers"]),
        n_heads=int(model_cfg["n_heads"]),
        d_ff=int(model_cfg["d_ff"]),
        dropout=float(model_cfg["dropout"]),
        short_ctx_k=int(model_cfg["short_ctx_k"]),
        head_hidden_dim=int(model_cfg["head_hidden_dim"]),
        head_n_layers=int(model_cfg["head_n_layers"]),
        ema_decay=float(model_cfg["ema_decay"]),
        normalize_latents=bool(model_cfg.get("normalize_latents", True)),
        norm_momentum=float(model_cfg.get("norm_momentum", 0.01)),
        norm_eps=float(model_cfg.get("norm_eps", 1e-5)),
        use_noise_injection=bool(model_cfg.get("use_noise_injection", True)),
        use_short_context=bool(model_cfg.get("use_short_context", True)),
    )

    param_counts = model.param_count()
    logger.info(f"[calm] Model built: {param_counts}")

    # ── 3. Train ─────────────────────────────────────────────────────
    from calm.train import train_calm, evaluate_rollout

    gen_cfg = cfg.get("generation", {})
    frame_rate = 12.5
    prompt_frames = int(float(gen_cfg.get("prompt_sec", 3.0)) * frame_rate)
    generate_frames = int(float(gen_cfg.get("generate_sec", 10.0)) * frame_rate)

    max_steps = args.max_steps or int(train_cfg["max_steps"])

    train_result = train_calm(
        model=model,
        dataloader=train_loader,
        out_dir=out_dir,
        device=device,
        max_steps=max_steps,
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        grad_clip_norm=float(train_cfg["grad_clip_norm"]),
        head_batch_mult=int(train_cfg["head_batch_mult"]),
        amp=bool(train_cfg.get("amp", True)),
        log_every=int(train_cfg["log_every"]),
        save_every=int(train_cfg["save_every"]),
        sample_every=int(train_cfg["sample_every"]),
        eval_dataloader=eval_loader,
        prompt_frames=prompt_frames,
        generate_frames=generate_frames,
        temperature=float(gen_cfg.get("temperature", 0.8)),
        n_samples=int(gen_cfg.get("n_samples", 10)),
    )

    # ── 4. Rollout evaluation ────────────────────────────────────────
    logger.info("[calm] Evaluating rollout quality...")
    eval_cfg = cfg.get("eval", {})
    rollout_lengths = [int(k) for k in eval_cfg.get("rollout_lengths", [1, 2, 4, 8, 16])]

    rollout_metrics = evaluate_rollout(
        model=model,
        eval_dataloader=eval_loader,
        device=device,
        prompt_frames=prompt_frames,
        rollout_lengths=rollout_lengths,
        n_utterances=int(eval_cfg.get("n_utterances", 20)),
        temperature=float(gen_cfg.get("temperature", 0.8)),
    )

    with open(out_dir / "rollout_metrics.json", "w") as f:
        json.dump(rollout_metrics, f, indent=2)
    logger.info(f"[calm] Rollout metrics: {json.dumps(rollout_metrics, indent=2)}")

    # ── 5. Audio generation ──────────────────────────────────────────
    audio_manifests = []
    if not args.skip_audio:
        vae_ckpt = cfg["vae"]["checkpoint"]
        if Path(vae_ckpt).exists():
            logger.info("[calm] Loading VAE for audio generation...")
            vae = _load_vae(vae_ckpt, device)

            audio_manifests = generate_audio_samples(
                calm_model=model,
                vae=vae,
                eval_dataloader=eval_loader,
                out_dir=out_dir,
                device=device,
                prompt_frames=prompt_frames,
                generate_frames=generate_frames,
                temperature=float(gen_cfg.get("temperature", 0.8)),
                n_samples=int(gen_cfg.get("n_samples", 10)),
            )
            del vae
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            logger.warning(f"[calm] VAE checkpoint not found: {vae_ckpt}, skipping audio generation")
    else:
        logger.info("[calm] Skipping audio generation (--skip-audio)")

    # ── 6. Summary ───────────────────────────────────────────────────
    summary = {
        "experiment": "calm_speech_continuation",
        "run_id": run_id,
        "steps": train_result["steps"],
        "final_checkpoint": train_result["final_checkpoint"],
        "param_counts": param_counts,
        "rollout_metrics": rollout_metrics,
        "n_audio_samples": len(audio_manifests),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Key metrics for experiment tracking
    km = {}
    for k_str, metrics in rollout_metrics.items():
        k_val = k_str.replace("k=", "")
        km[f"rollout_mse_k{k_val}"] = metrics.get("mse_mean", float("nan"))
        km[f"rollout_cos_k{k_val}"] = metrics.get("cos_mean", float("nan"))

    finalize_run(run, key_metrics=km)
    logger.info(f"[calm] Experiment complete. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
