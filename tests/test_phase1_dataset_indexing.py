import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("zarr")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase0.data.io import save_frames_index, save_latents_zarr
from phase1.data import iter_phase1_samples


def test_phase1_iter_samples_context_and_delta(tmp_path: Path) -> None:
    # Make a tiny utterance with deterministic latents so context is checkable.
    utt_id = "utt_a"
    speaker_id = 1
    t_total, d = 10, 3
    x = np.stack([np.full((d,), float(t), dtype=np.float32) for t in range(t_total)], axis=0)
    energy = np.ones((t_total,), dtype=np.float32)
    timestamps = np.arange(t_total, dtype=np.float32) / 12.5

    latents_dir = tmp_path / "latents.zarr"
    save_latents_zarr(
        latents=x,
        energy=energy,
        timestamps=timestamps,
        speaker_id=speaker_id,
        utterance_id=utt_id,
        zarr_path=latents_dir,
    )

    # Build frames parquet (all frames, mark as train).
    rows = []
    for t in range(t_total):
        rows.append(
            {
                "utterance_id": utt_id,
                "speaker_id": speaker_id,
                "t": t,
                "pos_frac": float(t / t_total),
                "energy": float(energy[t]),
                "split": "train",
                "is_high_energy": True,
            }
        )
    frames_path = tmp_path / "frames.parquet"
    save_frames_index(pd.DataFrame(rows), frames_path)

    # window W=2, horizon k=1: context uses frames [t-1-(W-1) .. t-1] = [t-2..t-1]
    window_size = 2
    k = 1
    samples = list(
        iter_phase1_samples(
            frames_index_path=frames_path,
            latents_dir=latents_dir,
            split="train",
            window_size=window_size,
            horizon_k=k,
            slice_name="all",
            max_samples=1,
        )
    )
    assert len(samples) == 1
    s = samples[0]

    # First valid t is max(1, (W-1)+k) = 2, so sample should be at t=2 in row order.
    assert s.t == 2

    # Context frames are x[0] and x[1], each is [0,0,0] and [1,1,1]
    expected_ctx = np.concatenate([x[0], x[1]], axis=0)
    assert np.allclose(s.context_flat, expected_ctx)

    # Delta at t=2 is x[2] - x[1] = [1,1,1]
    assert np.allclose(s.delta, np.ones((d,), dtype=np.float32))


def test_phase1_slice_filtering(tmp_path: Path) -> None:
    utt_id = "utt_b"
    speaker_id = 1
    t_total, d = 12, 2
    x = np.stack([np.full((d,), float(t), dtype=np.float32) for t in range(t_total)], axis=0)
    energy = np.ones((t_total,), dtype=np.float32)
    timestamps = np.arange(t_total, dtype=np.float32) / 12.5

    latents_dir = tmp_path / "latents.zarr"
    save_latents_zarr(
        latents=x,
        energy=energy,
        timestamps=timestamps,
        speaker_id=speaker_id,
        utterance_id=utt_id,
        zarr_path=latents_dir,
    )

    rows = []
    for t in range(t_total):
        rows.append(
            {
                "utterance_id": utt_id,
                "speaker_id": speaker_id,
                "t": t,
                "pos_frac": float(t / t_total),
                "energy": float(energy[t]),
                "split": "train",
                "is_high_energy": (t % 2 == 0),
            }
        )
    frames_path = tmp_path / "frames.parquet"
    save_frames_index(pd.DataFrame(rows), frames_path)

    window_size = 2
    k = 1

    all_n = sum(
        1
        for _ in iter_phase1_samples(
            frames_index_path=frames_path,
            latents_dir=latents_dir,
            split="train",
            window_size=window_size,
            horizon_k=k,
            slice_name="all",
            max_samples=None,
        )
    )
    high_n = sum(
        1
        for _ in iter_phase1_samples(
            frames_index_path=frames_path,
            latents_dir=latents_dir,
            split="train",
            window_size=window_size,
            horizon_k=k,
            slice_name="high_energy",
            max_samples=None,
        )
    )
    assert 0 < high_n < all_n
