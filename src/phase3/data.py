from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from phase0.data.io import LatentStore
from phase0.data.splits import load_splits

from .normalization import NormStats, normalize_x


@dataclass(frozen=True)
class UtteranceItem:
    utterance_id: str
    speaker_id: int
    n_frames: int


class Phase3UtteranceDataset(Dataset):
    """
    Returns full-utterance sequences for factorization training.

    Each item:
      - x: float32 [T, D]
      - mask: bool [T]
      - meta: utterance_id, speaker_id
    """

    def __init__(
        self,
        *,
        latents_dir: str | Path,
        latents_index_path: str | Path,
        splits_dir: str | Path,
        split: str,
        min_duration_sec: float = 0.0,
        norm_stats: Optional[NormStats] = None,
        max_utterances: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.latents_dir = Path(latents_dir)
        self.latent_store = LatentStore(self.latents_dir)
        self.norm_stats = norm_stats

        import pandas as pd

        df = pd.read_parquet(
            str(latents_index_path),
            columns=["utterance_id", "speaker_id", "n_frames", "duration_sec"],
        )
        df = df[df["duration_sec"] >= float(min_duration_sec)]

        splits = load_splits(splits_dir)
        if split == "train":
            speakers = set(splits.train_speakers)
        elif split == "eval":
            speakers = set(splits.eval_speakers)
        else:
            raise ValueError("split must be train or eval")

        df = df[df["speaker_id"].isin(speakers)]
        df = df.sort_values(["speaker_id", "utterance_id"])
        if max_utterances is not None:
            df = df.iloc[: int(max_utterances)]

        self.items: list[UtteranceItem] = [
            UtteranceItem(
                utterance_id=str(r.utterance_id),
                speaker_id=int(r.speaker_id),
                n_frames=int(r.n_frames),
            )
            for r in df.itertuples(index=False)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        x = self.latent_store.get_latents(item.utterance_id).astype(np.float32, copy=False)
        if self.norm_stats is not None:
            x = normalize_x(x, self.norm_stats).astype(np.float32, copy=False)
        mask = np.ones((x.shape[0],), dtype=bool)
        return {
            "x": torch.from_numpy(x),  # [T, D]
            "mask": torch.from_numpy(mask),  # [T]
            "utterance_id": item.utterance_id,
            "speaker_id": item.speaker_id,
        }


def collate_pad(batch: list[dict]) -> dict:
    """
    Pad variable-length sequences to [B, T_max, D] with a boolean mask.
    """
    if not batch:
        raise ValueError("Empty batch")

    x_list = [b["x"] for b in batch]
    lengths = [int(x.shape[0]) for x in x_list]
    d = int(x_list[0].shape[1])
    t_max = max(lengths)
    bsz = len(batch)

    x_pad = torch.zeros((bsz, t_max, d), dtype=torch.float32)
    mask = torch.zeros((bsz, t_max), dtype=torch.bool)
    for i, x in enumerate(x_list):
        t = int(x.shape[0])
        x_pad[i, :t] = x
        mask[i, :t] = True

    return {
        "x": x_pad,
        "mask": mask,
        "utterance_id": [b["utterance_id"] for b in batch],
        "speaker_id": torch.tensor([b["speaker_id"] for b in batch], dtype=torch.int32),
        "lengths": torch.tensor(lengths, dtype=torch.int32),
    }

