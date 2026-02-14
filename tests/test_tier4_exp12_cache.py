import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("zarr")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase0.data.io import save_latents_index, save_latents_zarr
from phase0.data.splits import SplitInfo, save_splits
from tier4.state_conditioned_directions import build_or_load_cache


def _make_splits(tmp_path: Path, train_speakers: list[int], eval_speakers: list[int], utt_rows: list[dict]) -> Path:
    splits_dir = tmp_path / "splits"
    train_utts = sorted([r["utterance_id"] for r in utt_rows if r["speaker_id"] in set(train_speakers)])
    eval_utts = sorted([r["utterance_id"] for r in utt_rows if r["speaker_id"] in set(eval_speakers)])
    splits = SplitInfo(
        train_speakers=train_speakers,
        eval_speakers=eval_speakers,
        train_utterances=train_utts,
        eval_utterances=eval_utts,
        train_utt_train=train_utts,
        train_utt_val=[],
    )
    save_splits(splits, splits_dir)
    return splits_dir


def test_build_or_load_cache_roundtrip(tmp_path: Path) -> None:
    # Two tiny utterances with deterministic latents.
    latents_dir = tmp_path / "latents.zarr"
    utt_rows = []

    def add_utt(utt_id: str, speaker_id: int, t: int) -> None:
        x = np.stack([np.full((3,), float(i), dtype=np.float32) for i in range(t)], axis=0)
        energy = np.ones((t,), dtype=np.float32)
        ts = np.arange(t, dtype=np.float32) / 12.5
        save_latents_zarr(x, energy, ts, speaker_id, utt_id, latents_dir)
        utt_rows.append(
            {
                "utterance_id": utt_id,
                "speaker_id": speaker_id,
                "n_frames": t,
                "duration_sec": float(t) / 12.5,
                "audio_path": str(tmp_path / f"{utt_id}.wav"),
            }
        )

    add_utt("utt_a", 1, 5)  # 4 deltas
    add_utt("utt_b", 2, 4)  # 3 deltas

    index_path = tmp_path / "latents_index.parquet"
    save_latents_index(utt_rows, index_path)

    splits_dir = _make_splits(tmp_path, train_speakers=[1], eval_speakers=[2], utt_rows=utt_rows)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    class _Logger:
        def info(self, *_a, **_k):
            pass

        def warning(self, *_a, **_k):
            pass

    logger = _Logger()

    cache = build_or_load_cache(
        latents_dir=latents_dir,
        latents_index=index_path,
        splits_dir=splits_dir,
        out_dir=out_dir,
        logger=logger,
        eps_mult=0.0,  # keep all
        max_utterances=None,
        max_deltas=None,
        overwrite=True,
    )

    z = np.array(cache["z"])
    du = np.array(cache["delta_unit"])
    dm = np.array(cache["delta_mag"])
    src = np.array(cache["delta_src_idx"])
    prev = np.array(cache["delta_prev_idx"])

    assert z.shape == (9, 3)  # 5 + 4 frames
    assert du.shape == (7, 3)  # 4 + 3 deltas
    assert dm.shape == (7,)
    assert src.shape == (7,)
    assert prev.shape == (7,)

    # Check first delta of utt_a uses source frame 0 and prev=-1.
    assert int(src[0]) == 0
    assert int(prev[0]) == -1

    # All deltas should be +1 in every dimension for this construction.
    # So unit direction is [1,1,1]/sqrt(3) and magnitude is sqrt(3).
    expected_dir = np.ones((3,), dtype=np.float32) / np.sqrt(3.0)
    assert np.allclose(du, expected_dir[None, :], atol=1e-6)
    assert np.allclose(dm, np.sqrt(3.0), atol=1e-6)

    # Load from cache without overwrite; should succeed.
    cache2 = build_or_load_cache(
        latents_dir=latents_dir,
        latents_index=index_path,
        splits_dir=splits_dir,
        out_dir=out_dir,
        logger=logger,
        eps_mult=0.0,
        max_utterances=None,
        max_deltas=None,
        overwrite=False,
    )
    assert int(cache2["stats"]["n_deltas_total"]) == 7


def test_shuffle_preserves_histogram() -> None:
    rng = np.random.default_rng(0)
    s = rng.integers(0, 10, size=(1000,), dtype=np.int32)
    s2 = s.copy()
    rng.shuffle(s2)
    assert np.array_equal(np.bincount(s, minlength=10), np.bincount(s2, minlength=10))

