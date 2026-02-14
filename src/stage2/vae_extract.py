"""
Extract VAE latents to zarr store + build frames index.

Reuses Phase 0 IO utilities for zarr/parquet storage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from phase0.data.io import save_latents_zarr, save_latents_index, save_frames_index
from phase0.data.io import load_latents_index
from phase0.data.librispeech import UtteranceInfo, load_audio
from phase0.data.splits import load_splits
from phase0.features.context import get_valid_frame_range
from phase0.features.energy import compute_median_energy

from .vae import ARFriendlyVAE


logger = logging.getLogger("phase0")


@torch.no_grad()
def extract_vae_latents(
    *,
    vae: ARFriendlyVAE,
    utterances: list[UtteranceInfo],
    zarr_path: str | Path,
    index_path: str | Path,
    device: torch.device,
    sample_rate: int = 24000,
    frame_size: int = 1920,
) -> None:
    """
    Extract deterministic VAE latents (mu) for all utterances into zarr.

    Args:
        vae: Trained ARFriendlyVAE in eval mode
        utterances: List of UtteranceInfo from LibriSpeech
        zarr_path: Output zarr store path
        index_path: Output latents index parquet path
        device: Torch device
        sample_rate: Audio sample rate (24kHz for Mimi)
        frame_size: Samples per latent frame (1920 for Mimi)
    """
    vae.eval()
    entries = []

    for i, utt in enumerate(utterances):
        try:
            wav, sr = load_audio(utt.audio_path, target_sr=sample_rate)
            # wav: [1, T]
            audio = wav.unsqueeze(0).to(device)  # [1, 1, T]

            mu = vae.extract_latents(audio)  # [1, D_vae, T']
            # Convert to [T', D_vae] for zarr storage
            latents = mu.squeeze(0).permute(1, 0).cpu().numpy()  # [T', D_vae]
            n_frames = latents.shape[0]

            # Compute per-frame energy from audio
            energy = _compute_frame_energy(wav, frame_size, n_frames)

            # Timestamps
            timestamps = (np.arange(n_frames, dtype=np.float32) / 12.5).astype(np.float32)

            save_latents_zarr(
                latents=latents,
                energy=energy,
                timestamps=timestamps,
                speaker_id=utt.speaker_id,
                utterance_id=utt.utterance_id,
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
                logger.info(f"[vae_extract] {i+1}/{len(utterances)} utterances extracted")

        except Exception as e:
            logger.warning(f"[vae_extract] Failed for {utt.utterance_id}: {e}")
            continue

    save_latents_index(entries, index_path)
    logger.info(f"[vae_extract] Done. {len(entries)} utterances -> {zarr_path}")


def _compute_frame_energy(wav: torch.Tensor, frame_size: int, n_frames: int) -> np.ndarray:
    """Compute per-frame energy by partitioning audio into n_frames chunks."""
    audio_np = wav.squeeze().cpu().numpy()
    T = len(audio_np)
    energy = np.zeros(n_frames, dtype=np.float32)
    chunk = max(1, T // max(n_frames, 1))
    for i in range(n_frames):
        start = i * chunk
        end = min(start + chunk, T)
        if start < T:
            energy[i] = float(np.mean(audio_np[start:end] ** 2))
    return energy


def build_frames_index(
    *,
    splits_dir: str | Path,
    latents_index_path: str | Path,
    latents_zarr_path: str | Path,
    window_size: int,
    max_lag: int,
    min_duration_sec: float,
    out_frames_index_path: str | Path,
) -> None:
    """
    Build frames index for VAE latents, matching Phase 0 format.

    Reuses pattern from tier1_exp3_rep_compare.py _build_frames_index().
    """
    from phase0.data.io import load_latents_zarr

    splits = load_splits(splits_dir)
    train_speaker_set = set(splits.train_speakers)
    eval_speaker_set = set(splits.eval_speakers)

    latents_index = load_latents_index(latents_index_path)
    frames_list = []
    train_energies = []

    for _, row in latents_index.iterrows():
        utt_id = str(row["utterance_id"])
        speaker_id = int(row["speaker_id"])
        n_frames = int(row["n_frames"])
        duration = float(row["duration_sec"])
        if duration < min_duration_sec:
            continue

        if speaker_id in train_speaker_set:
            split = "train"
        elif speaker_id in eval_speaker_set:
            split = "eval"
        else:
            continue

        try:
            _x, energy, _ts, _spk = load_latents_zarr(utt_id, latents_zarr_path)
        except Exception as e:
            logger.warning(f"[vae_extract] Could not load {utt_id}: {e}")
            continue

        if split == "train":
            train_energies.append(energy)

        first_valid, last_valid = get_valid_frame_range(n_frames, window_size, max_lag)
        for t in range(int(first_valid), int(last_valid)):
            frames_list.append({
                "utterance_id": utt_id,
                "speaker_id": speaker_id,
                "t": t,
                "pos_frac": float(t) / float(n_frames),
                "energy": float(energy[t]),
                "split": split,
            })

    frames = pd.DataFrame(frames_list)
    if frames.empty:
        raise RuntimeError("No frames found when building frames index.")

    median_energy = compute_median_energy(train_energies)
    frames["is_high_energy"] = frames["energy"] > float(median_energy)
    save_frames_index(frames, out_frames_index_path)
    logger.info(f"[vae_extract] Built frames index: {len(frames)} frames -> {out_frames_index_path}")
