from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pyarrow.dataset as ds
import torch
from torch.utils.data import IterableDataset

from phase0.data.io import LatentStore
from phase0.data.splits import load_splits
from phase0.utils.seed import get_rng


@dataclass(frozen=True)
class ZPair:
    z_prev: np.ndarray  # [D]
    dz: np.ndarray  # [D]


class ZPairIterableDataset(IterableDataset):
    def __init__(self, iterator_fn):
        super().__init__()
        self._iterator_fn = iterator_fn

    def __iter__(self):
        return self._iterator_fn()


def iter_zdyn_pairs(
    *,
    zdyn_dir: str | Path,
    zdyn_index_path: str | Path,
    splits_dir: str | Path,
    split: str,
    min_duration_sec: float,
    seed: int,
    max_pairs: Optional[int] = None,
    sample_prob: float = 1.0,
) -> Iterator[ZPair]:
    """
    Stream (z_t, Δz_t) pairs from exported z_dyn store.

    Notes:
    - Uses speaker split lists (Phase 0 convention) to filter utterances.
    - Applies per-pair Bernoulli subsampling via sample_prob for throughput control.
    """
    zdyn_dir = Path(zdyn_dir)
    store = LatentStore(zdyn_dir)
    rng = get_rng(int(seed))

    splits = load_splits(splits_dir)
    if split == "train":
        speakers = set(splits.train_speakers)
    elif split == "eval":
        speakers = set(splits.eval_speakers)
    else:
        raise ValueError("split must be train or eval")

    # Use pyarrow scanner to avoid loading full parquet into memory.
    dataset = ds.dataset(str(zdyn_index_path), format="parquet")
    scanner = dataset.scanner(
        columns=["utterance_id", "speaker_id", "n_frames", "duration_sec"],
        batch_size=4096,
    )

    yielded = 0
    for record_batch in scanner.to_batches():
        utt_col, spk_col, nframes_col, dur_col = record_batch.columns
        for i in range(record_batch.num_rows):
            speaker_id = int(spk_col[i].as_py())
            if speaker_id not in speakers:
                continue
            duration_sec = float(dur_col[i].as_py())
            if duration_sec < float(min_duration_sec):
                continue
            utt_id = str(utt_col[i].as_py())
            if utt_id not in store:
                continue
            try:
                z = store.get_latents(utt_id).astype(np.float32, copy=False)
            except Exception:
                continue

            if z.shape[0] < 2:
                continue

            z_prev = z[:-1]
            dz = z[1:] - z[:-1]

            if sample_prob >= 1.0:
                for t in range(z_prev.shape[0]):
                    yield ZPair(z_prev=z_prev[t], dz=dz[t])
                    yielded += 1
                    if max_pairs is not None and yielded >= int(max_pairs):
                        return
            else:
                for t in range(z_prev.shape[0]):
                    if float(rng.random()) > float(sample_prob):
                        continue
                    yield ZPair(z_prev=z_prev[t], dz=dz[t])
                    yielded += 1
                    if max_pairs is not None and yielded >= int(max_pairs):
                        return


def collate_zpairs(batch: list[ZPair]) -> dict[str, torch.Tensor]:
    z = np.stack([b.z_prev for b in batch], axis=0).astype(np.float32, copy=False)
    dz = np.stack([b.dz for b in batch], axis=0).astype(np.float32, copy=False)
    return {"z_prev": torch.from_numpy(z), "dz": torch.from_numpy(dz)}


def sample_eval_utterance_ids(
    *,
    zdyn_index_path: str | Path,
    splits_dir: str | Path,
    min_duration_sec: float,
    n_utterances: int,
    seed: int,
) -> list[str]:
    import pandas as pd

    splits = load_splits(splits_dir)
    eval_speakers = set(splits.eval_speakers)

    df = pd.read_parquet(str(zdyn_index_path), columns=["utterance_id", "speaker_id", "duration_sec"])
    df = df[df["speaker_id"].isin(eval_speakers)]
    df = df[df["duration_sec"] >= float(min_duration_sec)]
    utts = df["utterance_id"].astype(str).tolist()
    if not utts:
        return []

    rng = get_rng(int(seed))
    rng.shuffle(utts)
    return utts[: int(min(n_utterances, len(utts)))]

