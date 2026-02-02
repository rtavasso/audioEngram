"""
I/O utilities for zarr and parquet storage.

Handles saving and loading latent arrays and index tables.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import zarr


def save_latents_zarr(
    latents: np.ndarray,
    energy: np.ndarray,
    timestamps: np.ndarray,
    speaker_id: int,
    utterance_id: str,
    zarr_path: str | Path,
) -> None:
    """
    Save latents for a single utterance to zarr store.

    Args:
        latents: Latent array [T, D]
        energy: Per-frame energy [T]
        timestamps: Frame timestamps in seconds [T]
        speaker_id: Speaker ID
        utterance_id: Utterance ID (used as group key)
        zarr_path: Path to zarr store
    """
    zarr_path = Path(zarr_path)
    store = zarr.open(str(zarr_path), mode="a")

    # Create group for this utterance
    grp = store.create_group(utterance_id, overwrite=True)

    # Store arrays
    grp.array("x", latents, dtype="float32")
    grp.array("energy", energy, dtype="float32")
    grp.array("timestamps", timestamps, dtype="float32")

    # Store metadata as attributes
    grp.attrs["speaker_id"] = speaker_id
    grp.attrs["n_frames"] = latents.shape[0]
    grp.attrs["latent_dim"] = latents.shape[1]


def load_latents_zarr(
    utterance_id: str,
    zarr_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load latents for a single utterance from zarr store.

    Args:
        utterance_id: Utterance ID
        zarr_path: Path to zarr store

    Returns:
        Tuple of (latents [T, D], energy [T], timestamps [T], speaker_id)
    """
    store = zarr.open(str(zarr_path), mode="r")
    grp = store[utterance_id]

    latents = np.array(grp["x"])
    energy = np.array(grp["energy"])
    timestamps = np.array(grp["timestamps"])
    speaker_id = grp.attrs["speaker_id"]

    return latents, energy, timestamps, speaker_id


def get_zarr_utterance_ids(zarr_path: str | Path) -> list[str]:
    """Get all utterance IDs stored in a zarr store."""
    store = zarr.open(str(zarr_path), mode="r")
    return list(store.keys())


def save_latents_index(
    utterances: list[dict],
    output_path: str | Path,
) -> None:
    """
    Save latents index to parquet.

    Args:
        utterances: List of dicts with keys:
            - utterance_id
            - speaker_id
            - n_frames
            - duration_sec
            - audio_path
        output_path: Path to output parquet file
    """
    df = pd.DataFrame(utterances)
    df.to_parquet(str(output_path), index=False)


def load_latents_index(index_path: str | Path) -> pd.DataFrame:
    """
    Load latents index from parquet.

    Args:
        index_path: Path to parquet file

    Returns:
        DataFrame with utterance metadata
    """
    return pd.read_parquet(str(index_path))


def save_frames_index(
    frames: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save phase0 frame index to parquet.

    Args:
        frames: DataFrame with columns:
            - utterance_id
            - speaker_id
            - t (frame index)
            - pos_frac (position fraction t/T)
            - energy
            - split ('train' or 'eval')
            - is_high_energy (bool)
        output_path: Path to output parquet file
    """
    frames.to_parquet(str(output_path), index=False)


def load_frames_index(
    index_path: str | Path,
    columns: Optional[list[str]] = None,
    filters: Optional[list] = None,
) -> pd.DataFrame:
    """
    Load phase0 frame index from parquet.

    Args:
        index_path: Path to parquet file
        columns: Optional list of columns to load
        filters: Optional parquet filters (pyarrow engine)

    Returns:
        DataFrame with frame metadata
    """
    read_kwargs = {}
    if columns is not None:
        read_kwargs["columns"] = columns
    if filters is not None:
        read_kwargs["filters"] = filters

    try:
        return pd.read_parquet(str(index_path), **read_kwargs)
    except TypeError:
        # Some pandas/engine combos don't support `filters`.
        # If filters were requested, fail loudly so callers can fall back.
        if filters is not None:
            raise
        read_kwargs.pop("filters", None)
        return pd.read_parquet(str(index_path), **read_kwargs)


class LatentStore:
    """
    Convenient wrapper for accessing latents from zarr store.

    Caches open zarr store and provides frame-level access.
    """

    def __init__(self, zarr_path: str | Path):
        self.zarr_path = Path(zarr_path)
        self._store: Optional[zarr.Group] = None

    @property
    def store(self) -> zarr.Group:
        if self._store is None:
            self._store = zarr.open(str(self.zarr_path), mode="r")
        return self._store

    def get_latents(self, utterance_id: str) -> np.ndarray:
        """Get latent array for utterance [T, D]."""
        return np.array(self.store[utterance_id]["x"])

    def get_energy(self, utterance_id: str) -> np.ndarray:
        """Get energy array for utterance [T]."""
        return np.array(self.store[utterance_id]["energy"])

    def get_frame(self, utterance_id: str, t: int) -> np.ndarray:
        """Get single frame latent [D]."""
        return np.array(self.store[utterance_id]["x"][t])

    def get_context_window(
        self, utterance_id: str, t: int, window_size: int, lag: int
    ) -> np.ndarray:
        """
        Get context window for a target frame.

        Args:
            utterance_id: Utterance ID
            t: Target frame index
            window_size: Number of frames in context (W)
            lag: Lag between context end and target (L)

        Returns:
            Context array [window_size, D]
        """
        end = t - lag
        start = end - window_size
        return np.array(self.store[utterance_id]["x"][start:end])

    def __contains__(self, utterance_id: str) -> bool:
        return utterance_id in self.store
