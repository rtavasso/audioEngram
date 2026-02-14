"""
Sequence-level dataset for CALM training.

Reads contiguous latent sequences from zarr store (pre-extracted by Exp 9)
and yields fixed-length segments for transformer training.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

from phase0.data.io import LatentStore, load_latents_index

logger = logging.getLogger("phase0")


class LatentSequenceDataset(IterableDataset):
    """
    Iterable dataset yielding fixed-length latent sequences [S, D].

    Reads utterance latents from zarr, filters by split, random-crops
    to seq_len frames. Shuffles utterance order each epoch.
    """

    def __init__(
        self,
        latents_dir: str | Path,
        latents_index_path: str | Path,
        splits_dir: str | Path,
        split: str = "train",
        seq_len: int = 64,
        min_utterance_len: int = 32,
        seed: int = 42,
    ):
        super().__init__()
        self.latent_store = LatentStore(latents_dir)
        self.seq_len = seq_len
        self.min_utterance_len = min_utterance_len
        self.seed = seed

        # Load utterance index
        index_df = load_latents_index(latents_index_path)

        # Load split speakers
        splits_dir = Path(splits_dir)
        if split == "train":
            speaker_file = splits_dir / "train_speakers.txt"
        else:
            speaker_file = splits_dir / "eval_speakers.txt"

        with open(speaker_file) as f:
            split_speakers = set(int(line.strip()) for line in f if line.strip())

        # Filter to split utterances with sufficient length
        mask = (
            index_df["speaker_id"].isin(split_speakers)
            & (index_df["n_frames"] >= min_utterance_len)
        )
        self.utterance_ids = index_df.loc[mask, "utterance_id"].tolist()
        logger.info(
            f"[calm_data] {split} split: {len(self.utterance_ids)} utterances "
            f"(seq_len={seq_len}, min_len={min_utterance_len})"
        )

    def __iter__(self):
        # Worker-aware seeding for multi-worker DataLoader
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_seed = self.seed + worker_info.id
        else:
            worker_seed = self.seed

        rng = np.random.default_rng(worker_seed)
        epoch = 0

        while True:
            # Shuffle utterance order each epoch
            order = rng.permutation(len(self.utterance_ids))

            for idx in order:
                utt_id = self.utterance_ids[idx]
                try:
                    x = self.latent_store.get_latents(utt_id)  # [T, D]
                except Exception:
                    continue

                T = x.shape[0]
                if T < self.seq_len:
                    # Pad short utterances with zeros on the left
                    pad = np.zeros((self.seq_len - T, x.shape[1]), dtype=np.float32)
                    x = np.concatenate([pad, x], axis=0)
                    T = self.seq_len

                # Random crop
                start = rng.integers(0, T - self.seq_len + 1)
                segment = x[start : start + self.seq_len]  # [S, D]
                yield torch.from_numpy(segment.astype(np.float32))

            epoch += 1
            # Re-seed per epoch to vary shuffle order
            rng = np.random.default_rng(worker_seed + epoch * 1000)


def build_calm_dataloader(
    latents_dir: str | Path,
    latents_index_path: str | Path,
    splits_dir: str | Path,
    split: str = "train",
    seq_len: int = 64,
    min_utterance_len: int = 32,
    batch_size: int = 16,
    num_workers: int = 2,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """Build DataLoader for CALM training."""
    dataset = LatentSequenceDataset(
        latents_dir=latents_dir,
        latents_index_path=latents_index_path,
        splits_dir=splits_dir,
        split=split,
        seq_len=seq_len,
        min_utterance_len=min_utterance_len,
        seed=seed,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
