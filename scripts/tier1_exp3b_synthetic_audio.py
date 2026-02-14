#!/usr/bin/env python3
"""
Tier 1 - Experiment 3B: Synthetic trajectory perceptual validation.

Creates synthetic latent trajectories with controlled direction/magnitude and
decodes through Mimi decoder to produce audio for listening tests.

Modes:
  GT             — original trajectory (reference)
  A_stationary   — frozen at first frame (dx=0)
  B_slow_a{α}    — true direction, α × true magnitude
  C_wrong_dir    — random direction, true magnitude

Usage:
  uv run python scripts/tier1_exp3b_synthetic_audio.py --config configs/tier1_exp3b_synthetic.yaml
"""

from __future__ import annotations

import argparse
import sys
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
from phase0.utils.logging import setup_logging, get_logger
from phase0.utils.seed import set_seed, get_rng
from phase1.data import sample_eval_utterances
from phase1.train_eval import _device_from_config


def synthesize_modes(
    x_true: np.ndarray,
    alphas: list[float],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Given ground-truth trajectory x_true [T, D], produce synthetic trajectories.

    Returns dict: mode_name -> trajectory [T, D].
    """
    T, D = x_true.shape
    eps = 1e-8
    modes: dict[str, np.ndarray] = {}

    # Ground truth
    modes["GT"] = x_true.copy()

    # Mode A: stationary (frozen at frame 0)
    x_a = np.tile(x_true[0:1], (T, 1))
    modes["A_stationary"] = x_a

    # Precompute true deltas
    dx_true = x_true[1:] - x_true[:-1]  # [T-1, D]
    dx_norms = np.linalg.norm(dx_true, axis=-1, keepdims=True).clip(min=eps)  # [T-1, 1]
    dx_dirs = dx_true / dx_norms  # [T-1, D] unit directions

    # Mode B: true direction, scaled magnitude
    for alpha in alphas:
        x_b = np.zeros_like(x_true)
        x_b[0] = x_true[0]
        for t in range(1, T):
            dx_scaled = alpha * dx_norms[t - 1] * dx_dirs[t - 1]
            x_b[t] = x_b[t - 1] + dx_scaled
        modes[f"B_slow_a{alpha:.2f}"] = x_b

    # Mode C: random direction, true magnitude
    x_c = np.zeros_like(x_true)
    x_c[0] = x_true[0]
    for t in range(1, T):
        d_random = rng.standard_normal(D).astype(np.float32)
        d_random = d_random / max(np.linalg.norm(d_random), eps)
        x_c[t] = x_c[t - 1] + float(dx_norms[t - 1]) * d_random
    modes["C_wrong_dir"] = x_c

    return modes


def decode_latents_to_audio(
    x_synth: np.ndarray,
    decoder,
    device: torch.device,
) -> np.ndarray:
    """
    Decode latent trajectory [T, D] to audio using Mimi decoder.

    Returns audio as 1D numpy array.
    """
    # Mimi decoder expects [B, D, T]
    emb = torch.from_numpy(x_synth.T).unsqueeze(0).float().to(device)  # [1, D, T]
    with torch.inference_mode():
        audio = decoder(emb)  # [1, 1, samples]
    return audio.squeeze().cpu().numpy()


def main() -> int:
    p = argparse.ArgumentParser(description="Tier1 Exp3B: synthetic audio")
    p.add_argument("--config", type=str, default="configs/tier1_exp3b_synthetic.yaml")
    args = p.parse_args()

    logger = setup_logging(name="tier1-exp3b-synthetic")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(42)
    rng = get_rng(42)
    device = _device_from_config(cfg.get("device", "auto"))

    data_cfg = cfg["data"]
    latents_dir = data_cfg["latents_dir"]
    latents_index = data_cfg.get("latents_index")
    if latents_index is None:
        latents_index = str(Path(latents_dir).parent / "latents_index.parquet")
    splits_dir = data_cfg["splits_dir"]

    n_utterances = int(cfg.get("n_utterances", 10))
    max_frames = int(cfg.get("max_frames", 200))
    alphas = [float(a) for a in cfg.get("alphas", [0.25, 0.5, 0.75])]

    out_dir = Path(cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = register_run(
        experiment="exp3b_synthetic", run_id=run_id, config_path=args.config,
        config=cfg, cli_args=sys.argv[1:], out_dir=out_dir, log_name="tier1-exp3b-synthetic",
    )

    # Load Mimi decoder
    logger.info("[exp3b] Loading Mimi autoencoder...")
    from mimi_autoencoder import load_mimi_autoencoder

    autoencoder = load_mimi_autoencoder(
        checkpoint_path=cfg.get("mimi_checkpoint"),
        device=str(device),
    )
    autoencoder.eval()
    decoder = autoencoder.decoder
    sample_rate = int(autoencoder.sample_rate)  # 24000

    # Sample eval utterances
    utt_ids = sample_eval_utterances(
        splits_dir=splits_dir,
        latents_index_path=latents_index,
        n_utterances=n_utterances,
        seed=42,
    )

    store = LatentStore(latents_dir)
    logger.info(f"[exp3b] {len(utt_ids)} utterances, max_frames={max_frames}, alphas={alphas}")

    for i, utt_id in enumerate(utt_ids):
        if utt_id not in store:
            logger.warning(f"[exp3b] Utterance {utt_id} not in store, skipping")
            continue

        x_true = store.get_latents(utt_id).astype(np.float32, copy=False)
        T = min(int(x_true.shape[0]), max_frames)
        x_true = x_true[:T]

        if T < 4:
            continue

        utt_dir = out_dir / f"utt_{i:03d}_{utt_id}"
        utt_dir.mkdir(parents=True, exist_ok=True)

        modes = synthesize_modes(x_true, alphas, rng)

        for mode_name, x_synth in modes.items():
            audio = decode_latents_to_audio(x_synth, decoder, device)
            wav_path = utt_dir / f"{mode_name}.wav"
            audio_t = torch.from_numpy(audio).unsqueeze(0).float()  # [1, samples]
            # Resample to 48kHz for broad player compatibility
            if sample_rate != 48000:
                audio_t = torchaudio.functional.resample(audio_t, sample_rate, 48000)
            torchaudio.save(str(wav_path), audio_t, 48000)

        logger.info(f"[exp3b] [{i+1}/{len(utt_ids)}] {utt_id}: {len(modes)} modes -> {utt_dir}")

    logger.info(f"[exp3b] Done. Outputs in {out_dir}")
    finalize_run(run, key_metrics={"n_utterances": len(utt_ids), "n_modes": len(alphas) + 3})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
