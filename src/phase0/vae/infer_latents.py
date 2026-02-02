"""
Batch latent extraction using Mimi autoencoder.

Runs VAE encoder on audio files and saves continuous latents.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from phase0.data.librispeech import UtteranceInfo, load_audio
from phase0.data.io import save_latents_zarr


def compute_frame_energy(
    waveform: torch.Tensor,
    frame_size: int,
    n_frames: int,
) -> np.ndarray:
    """
    Compute RMS energy per latent frame, aligned with VAE frames.

    Args:
        waveform: Audio tensor [1, T]
        frame_size: Samples per latent frame (1920 for Mimi)
        n_frames: Number of latent frames

    Returns:
        Energy array [n_frames]
    """
    wav = waveform.squeeze().numpy()
    energy = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        start = i * frame_size
        end = min(start + frame_size, len(wav))
        if start < len(wav):
            segment = wav[start:end]
            energy[i] = np.sqrt(np.mean(segment**2))

    return energy


def infer_utterance_latents(
    utterance: UtteranceInfo,
    autoencoder,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run VAE encoder on a single utterance.

    Args:
        utterance: UtteranceInfo object
        autoencoder: MimiAutoencoder instance
        device: Device to run on

    Returns:
        Tuple of (latents [T, D], energy [T], timestamps [T])
    """
    # Load and prepare audio
    waveform, sr = load_audio(utterance.audio_path, target_sr=autoencoder.sample_rate)
    waveform = waveform.unsqueeze(0).to(device)  # [1, 1, T]

    # Run encoder
    with torch.no_grad():
        latents = autoencoder.encode(waveform)  # [1, D, T']

    # Convert to numpy [T', D]
    latents = latents.squeeze(0).permute(1, 0).cpu().numpy()
    n_frames = latents.shape[0]

    # Compute frame-aligned energy
    energy = compute_frame_energy(
        waveform.squeeze(0).cpu(),
        autoencoder.frame_size,
        n_frames,
    )

    # Compute timestamps
    timestamps = np.arange(n_frames, dtype=np.float32) / autoencoder.frame_rate

    # Validation
    assert not np.any(np.isnan(latents)), f"NaN in latents for {utterance.utterance_id}"
    assert latents.shape[1] == autoencoder.latent_dim

    return latents, energy, timestamps


def batch_infer_latents(
    utterances: list[UtteranceInfo],
    autoencoder,
    zarr_path: str | Path,
    device: str = "cpu",
    show_progress: bool = True,
) -> list[dict]:
    """
    Run VAE encoder on multiple utterances and save to zarr.

    Args:
        utterances: List of UtteranceInfo objects
        autoencoder: MimiAutoencoder instance
        zarr_path: Path to zarr store
        device: Device to run on
        show_progress: Whether to show progress bar

    Returns:
        List of index entries (dicts for parquet)
    """
    zarr_path = Path(zarr_path)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)

    index_entries = []
    iterator = tqdm(utterances, desc="Encoding") if show_progress else utterances

    for utt in iterator:
        try:
            latents, energy, timestamps = infer_utterance_latents(
                utt, autoencoder, device
            )

            # Save to zarr
            save_latents_zarr(
                latents=latents,
                energy=energy,
                timestamps=timestamps,
                speaker_id=utt.speaker_id,
                utterance_id=utt.utterance_id,
                zarr_path=zarr_path,
            )

            # Record index entry
            index_entries.append(
                {
                    "utterance_id": utt.utterance_id,
                    "speaker_id": utt.speaker_id,
                    "n_frames": latents.shape[0],
                    "duration_sec": utt.duration_sec,
                    "audio_path": str(utt.audio_path),
                }
            )

        except Exception as e:
            print(f"Error processing {utt.utterance_id}: {e}")
            continue

    return index_entries


def verify_latent_store(
    zarr_path: str | Path,
    expected_frame_rate: float = 12.5,
    expected_dim: int = 512,
) -> dict:
    """
    Verify latent store integrity.

    Args:
        zarr_path: Path to zarr store
        expected_frame_rate: Expected frame rate
        expected_dim: Expected latent dimension

    Returns:
        Dict with verification stats
    """
    import zarr

    store = zarr.open(str(zarr_path), mode="r")
    stats = {
        "n_utterances": 0,
        "n_frames_total": 0,
        "n_nan_utterances": 0,
        "frame_rates": [],
        "dims": set(),
    }

    for utt_id in store.keys():
        grp = store[utt_id]
        latents = np.array(grp["x"])
        timestamps = np.array(grp["timestamps"])

        stats["n_utterances"] += 1
        stats["n_frames_total"] += latents.shape[0]
        stats["dims"].add(latents.shape[1])

        if np.any(np.isnan(latents)):
            stats["n_nan_utterances"] += 1

        # Estimate frame rate from timestamps
        if len(timestamps) > 1:
            frame_rate = 1.0 / np.mean(np.diff(timestamps))
            stats["frame_rates"].append(frame_rate)

    # Summarize
    stats["dims"] = list(stats["dims"])
    if stats["frame_rates"]:
        stats["mean_frame_rate"] = np.mean(stats["frame_rates"])
        stats["std_frame_rate"] = np.std(stats["frame_rates"])
    else:
        stats["mean_frame_rate"] = None
        stats["std_frame_rate"] = None

    # Check expectations
    stats["valid"] = (
        stats["n_nan_utterances"] == 0
        and len(stats["dims"]) == 1
        and stats["dims"][0] == expected_dim
        and (
            stats["mean_frame_rate"] is None
            or abs(stats["mean_frame_rate"] - expected_frame_rate) < 0.5
        )
    )

    return stats
