"""
Phase 1 data streaming utilities.

Reads Phase 0 artifacts:
- frames parquet (row-wise metadata for valid target frames)
- latents zarr (utterance -> x[T, D])

Primary sample for a given lag k:
  context_flat = x[t-k-W+1 : t-k+1].flatten()
  delta        = x[t] - x[t-1]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds

from phase0.data.io import LatentStore
from phase0.features.context import get_context_flat
from phase0.features.normalization import compute_delta
from phase0.utils.seed import get_rng


@dataclass(frozen=True)
class Phase1Sample:
    context_flat: np.ndarray  # [W*D]
    delta: np.ndarray  # [D]
    speaker_id: int
    utterance_id: str
    t: int
    is_high_energy: bool
    pos_frac: float


def _as_python_scalar(v) -> object:
    # pyarrow Scalar -> python value
    if hasattr(v, "as_py"):
        return v.as_py()
    return v


def _default_row_filter(
    *,
    slice_name: str = "all",
) -> Callable[[bool, float], bool]:
    """
    Returns a predicate (is_high_energy, pos_frac) -> include.

    Mirrors Phase 0 slice semantics for convenience.
    """
    if slice_name == "all":
        return lambda is_high_energy, pos_frac: True
    if slice_name == "high_energy":
        return lambda is_high_energy, pos_frac: bool(is_high_energy)
    if slice_name == "utterance_medial":
        return lambda is_high_energy, pos_frac: (0.17 <= float(pos_frac) <= 0.83)
    raise ValueError(f"Unknown slice_name: {slice_name}")


class BufferedShuffle:
    """
    Small-memory shuffler for streaming iterators.

    Adds approximate shuffle without materializing the whole dataset.
    """

    def __init__(self, buffer_size: int, seed: int):
        if buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")
        self._buffer_size = int(buffer_size)
        self._rng = get_rng(int(seed))

    def __call__(self, it: Iterator[Phase1Sample]) -> Iterator[Phase1Sample]:
        buf: list[Phase1Sample] = []
        for item in it:
            if len(buf) < self._buffer_size:
                buf.append(item)
                continue
            j = int(self._rng.integers(0, len(buf)))
            yield buf[j]
            buf[j] = item

        # Drain remaining
        self._rng.shuffle(buf)
        yield from buf


def iter_phase1_samples(
    *,
    frames_index_path: str | Path,
    latents_dir: str | Path,
    split: str,
    window_size: int,
    horizon_k: int,
    slice_name: str = "all",
    max_samples: Optional[int] = None,
) -> Iterator[Phase1Sample]:
    """
    Stream Phase 1 samples in (approximate) parquet row order.

    Notes:
    - Keeps a single-utterance latent array in memory and swaps when utterance_id changes.
    - Applies validity checks for context extraction and delta computation.
    """
    frames_index_path = Path(frames_index_path)
    latent_store = LatentStore(latents_dir)
    row_ok = _default_row_filter(slice_name=slice_name)

    dataset = ds.dataset(str(frames_index_path), format="parquet")
    scanner = dataset.scanner(
        columns=[
            "utterance_id",
            "speaker_id",
            "t",
            "pos_frac",
            "is_high_energy",
            "split",
        ],
        filter=ds.field("split") == split,
        batch_size=8192,
    )

    current_utt_id: Optional[str] = None
    current_latents: Optional[np.ndarray] = None
    n_yielded = 0

    # Validity constraints:
    # - delta needs t >= 1
    # - context end is (t - k), inclusive; start must be >= 0 => t >= (W - 1) + k
    min_t = max(1, (window_size - 1) + horizon_k)

    for record_batch in scanner.to_batches():
        cols = record_batch.columns
        # Named schema access is slower; use indices matching scan columns
        utt_col, spk_col, t_col, pos_col, energy_col, _split_col = cols
        n_rows = record_batch.num_rows

        for i in range(n_rows):
            utt_id = str(_as_python_scalar(utt_col[i]))
            t = int(_as_python_scalar(t_col[i]))
            if t < min_t:
                continue

            is_high_energy = bool(_as_python_scalar(energy_col[i]))
            pos_frac = float(_as_python_scalar(pos_col[i]))
            if not row_ok(is_high_energy, pos_frac):
                continue

            if current_utt_id != utt_id:
                if utt_id not in latent_store:
                    current_utt_id = None
                    current_latents = None
                    continue
                try:
                    current_latents = latent_store.get_latents(utt_id)
                    current_utt_id = utt_id
                except Exception:
                    current_utt_id = None
                    current_latents = None
                    continue

            x = current_latents
            if x is None:
                continue
            if t >= x.shape[0]:
                continue

            try:
                ctx = get_context_flat(x, t, window_size, horizon_k).astype(np.float32, copy=False)
                dx = compute_delta(x, t).astype(np.float32, copy=False)
            except Exception:
                continue

            spk = int(_as_python_scalar(spk_col[i]))
            yield Phase1Sample(
                context_flat=ctx,
                delta=dx,
                speaker_id=spk,
                utterance_id=utt_id,
                t=t,
                is_high_energy=is_high_energy,
                pos_frac=pos_frac,
            )
            n_yielded += 1
            if max_samples is not None and n_yielded >= int(max_samples):
                return


def sample_eval_utterances(
    *,
    splits_dir: str | Path,
    latents_index_path: str | Path,
    n_utterances: int,
    seed: int,
) -> list[str]:
    """
    Sample eval utterance IDs for rollout diagnostics.

    Uses eval speaker IDs from Phase 0 splits dir, then filters latents index.
    """
    import pandas as pd

    splits_dir = Path(splits_dir)
    eval_speakers = []
    with open(splits_dir / "eval_speakers.txt") as f:
        for line in f:
            line = line.strip()
            if line:
                eval_speakers.append(int(line))
    eval_speaker_set = set(eval_speakers)

    df = pd.read_parquet(str(latents_index_path), columns=["utterance_id", "speaker_id"])
    eval_df = df[df["speaker_id"].isin(eval_speaker_set)]
    utts = eval_df["utterance_id"].astype(str).tolist()
    if not utts:
        return []

    rng = get_rng(seed)
    rng.shuffle(utts)
    return utts[: int(min(n_utterances, len(utts)))]
