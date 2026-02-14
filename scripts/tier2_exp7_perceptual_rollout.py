#!/usr/bin/env python3
"""
Stage 2 - Experiment 7: Perceptual rollout evaluation.

Decodes actual MDN rollout trajectories through Mimi decoder to produce
WAV files for listening tests.

For each eval utterance, at various rollout lengths:
  1. Use GT prefix up to rollout start
  2. Run dynamics model in free-running mode for K steps
  3. Resume GT suffix after rollout
  4. Decode full trajectory through Mimi decoder -> 48kHz WAV
  5. Also decode pure GT as reference

Usage:
  uv run python scripts/tier2_exp7_perceptual_rollout.py \
      --config configs/tier2_exp7_perceptual_rollout.yaml \
      --checkpoint outputs/tier1/exp1_vmf/<RUN>/checkpoints/vmf_k1_final.pt
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
import torch
import torchaudio
import yaml

from experiment import register_run, finalize_run
from phase0.data.io import LatentStore
from phase0.utils.logging import setup_logging
from phase0.utils.seed import set_seed, get_rng
from phase1.checkpoints import load_phase1_checkpoint
from phase1.data import sample_eval_utterances
from phase1.train_eval import _device_from_config


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def decode_latents_to_audio(
    x: np.ndarray,
    decoder,
    device: torch.device,
) -> np.ndarray:
    """Decode latent trajectory [T, D] to audio using Mimi decoder."""
    # Mimi decoder expects [B, D, T]
    emb = torch.from_numpy(x.T).unsqueeze(0).float().to(device)
    with torch.inference_mode():
        audio = decoder(emb)
    return audio.squeeze().cpu().numpy()


@torch.no_grad()
def generate_rollout_trajectory(
    *,
    x_true: np.ndarray,
    model,
    window_size: int,
    rollout_start: int,
    rollout_length: int,
    device: torch.device,
) -> np.ndarray:
    """
    Generate spliced trajectory: GT prefix -> rollout -> GT suffix.

    Args:
        x_true: [T, D] ground-truth latent trajectory
        model: Phase 1 dynamics model
        window_size: context window W
        rollout_start: frame index to begin rollout
        rollout_length: number of rollout steps
        device: torch device

    Returns:
        x_spliced: [T, D] trajectory with rollout segment inserted
    """
    T, D = x_true.shape
    rollout_end = min(rollout_start + rollout_length, T)
    actual_length = rollout_end - rollout_start

    x_spliced = x_true.copy()

    # Run rollout from rollout_start
    x_hat = torch.from_numpy(x_true).float().to(device)

    for s in range(actual_length):
        t = rollout_start + s
        if t >= T:
            break

        # Build context window ending at t-1
        ctx_end = t - 1  # k=1, so horizon_k=1 means context ends at t-1
        ctx_start = ctx_end - window_size + 1
        if ctx_start < 0:
            break

        ctx = x_hat[ctx_start:ctx_end + 1]  # [W, D]
        ctx_flat = ctx.reshape(1, -1)  # [1, W*D]

        if hasattr(model, "rollout_mean"):
            dx_hat = model.rollout_mean(ctx_flat)
        else:
            dx_hat = model.expected_mean(ctx_flat)

        x_hat[t] = x_hat[t - 1] + dx_hat.squeeze(0)
        x_spliced[t] = x_hat[t].cpu().numpy()

    return x_spliced


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 2 Exp7: perceptual rollout")
    p.add_argument("--config", type=str, default="configs/tier2_exp7_perceptual_rollout.yaml")
    p.add_argument("--checkpoint", type=str, default=None, help="Phase 1 checkpoint path")
    p.add_argument("--run-id", type=str, default=None)
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
        experiment="exp7_perceptual_rollout", run_id=run_id, config_path=args.config,
        config=cfg, cli_args=sys.argv[1:], out_dir=out_dir, log_name="phase0",
    )

    set_seed(int(cfg.get("seed", 42)))
    device = _device_from_config(cfg["train"]["device"])

    data_cfg = cfg["data"]
    latents_dir = data_cfg["latents_dir"]
    latents_index = data_cfg["latents_index"]
    splits_dir = data_cfg["splits_dir"]

    # Load Phase 1 checkpoint
    ckpt_path = args.checkpoint or data_cfg.get("phase1_checkpoint")
    if not ckpt_path:
        logger.error("No Phase 1 checkpoint. Use --checkpoint or set data.phase1_checkpoint in config.")
        finalize_run(run, status="failed")
        return 1

    logger.info(f"[exp7] Loading Phase 1 checkpoint: {ckpt_path}")
    model, ckpt = load_phase1_checkpoint(ckpt_path, device=device)
    window_size = int(ckpt.get("window_size", cfg["rollout"]["window_size"]))

    # Load Mimi decoder
    logger.info("[exp7] Loading Mimi autoencoder...")
    from mimi_autoencoder import load_mimi_autoencoder

    autoencoder = load_mimi_autoencoder(
        checkpoint_path=cfg.get("mimi_checkpoint"),
        device=str(device),
    )
    autoencoder.eval()
    decoder = autoencoder.decoder
    mimi_sr = int(autoencoder.sample_rate)  # 24000

    output_sr = int(cfg["output"].get("sample_rate", 48000))

    # Sample eval utterances
    rollout_cfg = cfg["rollout"]
    n_utterances = int(rollout_cfg["n_utterances"])
    max_frames = int(rollout_cfg["max_frames"])
    rollout_lengths = [int(l) for l in rollout_cfg["lengths"]]

    utt_ids = sample_eval_utterances(
        splits_dir=splits_dir,
        latents_index_path=latents_index,
        n_utterances=n_utterances,
        seed=42,
    )

    store = LatentStore(latents_dir)
    logger.info(f"[exp7] {len(utt_ids)} utterances, rollout_lengths={rollout_lengths}")

    manifest = []

    for i, utt_id in enumerate(utt_ids):
        if utt_id not in store:
            logger.warning(f"[exp7] Utterance {utt_id} not in store, skipping")
            continue

        x_true = store.get_latents(utt_id).astype(np.float32, copy=False)
        T = min(x_true.shape[0], max_frames)
        x_true = x_true[:T]

        if T < window_size + max(rollout_lengths) + 2:
            continue

        utt_dir = out_dir / f"utt_{i:03d}_{utt_id}"
        utt_dir.mkdir(parents=True, exist_ok=True)

        # Decode ground truth
        audio_gt = decode_latents_to_audio(x_true, decoder, device)
        _save_wav(audio_gt, utt_dir / "GT.wav", mimi_sr, output_sr)

        # Rollout start: middle of the utterance
        rollout_start = T // 3 + window_size

        for rl in rollout_lengths:
            if rollout_start + rl > T:
                continue

            x_spliced = generate_rollout_trajectory(
                x_true=x_true,
                model=model,
                window_size=window_size,
                rollout_start=rollout_start,
                rollout_length=rl,
                device=device,
            )

            audio_spliced = decode_latents_to_audio(x_spliced, decoder, device)
            wav_name = f"rollout_k{rl:02d}.wav"
            _save_wav(audio_spliced, utt_dir / wav_name, mimi_sr, output_sr)

            manifest.append({
                "utterance_id": utt_id,
                "rollout_length": rl,
                "rollout_start": rollout_start,
                "total_frames": T,
                "wav_path": str(utt_dir / wav_name),
            })

        logger.info(f"[exp7] [{i+1}/{len(utt_ids)}] {utt_id}: {len(rollout_lengths)} rollout WAVs -> {utt_dir}")

    # Write manifest
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"[exp7] Done. {len(manifest)} WAVs in {out_dir}")
    finalize_run(run, key_metrics={
        "n_utterances": len(utt_ids),
        "n_wavs": len(manifest),
        "rollout_lengths": rollout_lengths,
    })
    return 0


def _save_wav(
    audio: np.ndarray,
    path: Path,
    input_sr: int,
    output_sr: int,
) -> None:
    """Save audio as WAV, resampling if needed."""
    audio_t = torch.from_numpy(audio).unsqueeze(0).float()  # [1, T]
    if input_sr != output_sr:
        audio_t = torchaudio.functional.resample(audio_t, input_sr, output_sr)
    torchaudio.save(str(path), audio_t, output_sr)


if __name__ == "__main__":
    raise SystemExit(main())
