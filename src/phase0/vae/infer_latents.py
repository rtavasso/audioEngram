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
import contextlib

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
    wav = waveform.squeeze().numpy().astype(np.float32, copy=False)
    total = int(n_frames) * int(frame_size)
    if wav.shape[0] < total:
        pad = np.zeros((total - wav.shape[0],), dtype=np.float32)
        wav = np.concatenate([wav, pad], axis=0)
    else:
        wav = wav[:total]
    frames = wav.reshape(int(n_frames), int(frame_size))
    return np.sqrt(np.mean(frames * frames, axis=1)).astype(np.float32, copy=False)


def _infer_batch_latents(
    waveforms: torch.Tensor,
    lengths: torch.Tensor,
    autoencoder,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> list[np.ndarray]:
    """
    Encode a padded batch and slice each output to its per-utterance latent length.

    waveforms: [B, 1, T_max] float32 on CPU
    lengths:   [B] original lengths in samples (after resample)
    returns: list of [T_i, D] float32 arrays
    """
    waveforms = waveforms.to(device, non_blocking=True)
    lengths = lengths.to(device, non_blocking=True)

    if str(device).startswith("cuda"):
        autocast_ctx = torch.cuda.amp.autocast(enabled=bool(use_amp), dtype=amp_dtype)
    else:
        autocast_ctx = contextlib.nullcontext()

    with torch.inference_mode(), autocast_ctx:
        lat = autoencoder.encode(waveforms)  # [B, D, T']

    lat = lat.detach().float().cpu()  # keep storage float32
    b, d, t_prime = lat.shape
    outs: list[np.ndarray] = []
    frame_size = int(autoencoder.frame_size)

    for i in range(b):
        # expected latent length is ceil(length / frame_size)
        n_frames = int((int(lengths[i].item()) + frame_size - 1) // frame_size)
        li = lat[i, :, :n_frames].permute(1, 0).contiguous().numpy()
        outs.append(li)
    return outs


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
    with torch.inference_mode():
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
    batch_size: int = 1,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    amp: bool = True,
    amp_dtype: str = "bf16",
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

    # AMP config
    amp_dtype_l = str(amp_dtype).lower()
    if amp_dtype_l not in ("bf16", "fp16"):
        raise ValueError("amp_dtype must be bf16 or fp16")
    torch_amp_dtype = torch.bfloat16 if amp_dtype_l == "bf16" else torch.float16

    index_entries = []

    if int(batch_size) <= 1:
        iterator = tqdm(utterances, desc="Encoding") if show_progress else utterances
        for utt in iterator:
            try:
                latents, energy, timestamps = infer_utterance_latents(utt, autoencoder, device)
                save_latents_zarr(
                    latents=latents,
                    energy=energy,
                    timestamps=timestamps,
                    speaker_id=utt.speaker_id,
                    utterance_id=utt.utterance_id,
                    zarr_path=zarr_path,
                )
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

    # Batched path
    from torch.utils.data import Dataset, DataLoader

    class _UttDataset(Dataset):
        def __init__(self, utts: list[UtteranceInfo]):
            self.utts = utts

        def __len__(self) -> int:
            return len(self.utts)

        def __getitem__(self, idx: int) -> dict:
            utt = self.utts[idx]
            wav, _ = load_audio(utt.audio_path, target_sr=autoencoder.sample_rate)
            # wav: [1,T] -> [1,1,T] for encoder
            wav = wav.unsqueeze(0).contiguous()
            length = int(wav.shape[-1])
            return {
                "utt": utt,
                "wav": wav,  # [1,1,T]
                "length": length,
            }

    def _collate(items: list[dict]) -> dict:
        utts = [it["utt"] for it in items]
        lengths = torch.tensor([it["length"] for it in items], dtype=torch.int64)
        max_len = int(max(lengths).item())
        w = torch.zeros((len(items), 1, max_len), dtype=torch.float32)
        for i, it in enumerate(items):
            wav = it["wav"].squeeze(0)  # [1,T]
            t = int(wav.shape[-1])
            w[i, :, :t] = wav
        return {"utts": utts, "waveforms": w, "lengths": lengths}

    ds = _UttDataset(utterances)
    loader_kwargs = dict(
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        persistent_workers=(int(num_workers) > 0),
        pin_memory=device.startswith("cuda"),
        collate_fn=_collate,
    )
    if int(num_workers) > 0:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    loader = DataLoader(ds, **loader_kwargs)

    iterator = tqdm(loader, desc="Encoding(batched)") if show_progress else loader
    for batch in iterator:
        utts = batch["utts"]
        waveforms = batch["waveforms"]
        lengths = batch["lengths"]

        latents_list = _infer_batch_latents(
            waveforms=waveforms,
            lengths=lengths,
            autoencoder=autoencoder,
            device=device,
            use_amp=bool(amp),
            amp_dtype=torch_amp_dtype,
        )

        for i, (utt, latents) in enumerate(zip(utts, latents_list)):
            try:
                length = int(lengths[i].item())
                wav = waveforms[i, :, :length].cpu().contiguous().squeeze(0)  # [1,T] -> [T]
                n_frames = int(latents.shape[0])
                energy = compute_frame_energy(wav.unsqueeze(0), int(autoencoder.frame_size), n_frames)
                timestamps = np.arange(n_frames, dtype=np.float32) / float(autoencoder.frame_rate)

                save_latents_zarr(
                    latents=latents,
                    energy=energy,
                    timestamps=timestamps,
                    speaker_id=utt.speaker_id,
                    utterance_id=utt.utterance_id,
                    zarr_path=zarr_path,
                )
                index_entries.append(
                    {
                        "utterance_id": utt.utterance_id,
                        "speaker_id": utt.speaker_id,
                        "n_frames": n_frames,
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
    max_utterances: Optional[int] = None,
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

    utt_ids = list(store.keys())
    if max_utterances is not None and max_utterances > 0 and len(utt_ids) > int(max_utterances):
        rng = np.random.default_rng(0)
        utt_ids = rng.choice(utt_ids, size=int(max_utterances), replace=False).tolist()

    for utt_id in utt_ids:
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
