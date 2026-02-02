import sys
from pathlib import Path

import pytest

# This is an end-to-end smoke test for the full Phase 0 pipeline, which depends
# on optional heavy deps (pandas, zarr, sklearn). Skip cleanly when not present
# (e.g., minimal CI environments).
pytest.importorskip("pandas")
pytest.importorskip("zarr")
pytest.importorskip("sklearn")

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from phase0.analysis.run_phase0 import run_full_analysis
from phase0.data.io import save_frames_index, save_latents_index, save_latents_zarr
from phase0.features.context import get_valid_frame_range


def _make_regime_latents(t: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a synthetic latent sequence where context predicts next-step delta.

    We build a regime signal s(t) that is visible in the mean-pooled context
    and drives a predictable delta component, while keeping most other dims
    as small noise.

    This is designed so `mean_pool_vq` with K=2 can reliably recover regimes.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((t, d), dtype=np.float32)

    block = 8
    ramp = 4  # soften regime transitions so "marker" dims don't spike deltas
    # Keep one dimension for the driven delta; make the remaining marker dims
    # dominate mean-pooled context so K=2 clustering is easy.
    marker_dims = min(6, max(1, d - 1))
    marker_amp = 6.0
    driven_dim = marker_dims if marker_dims < d else (d - 1)
    driven_step = 3.0

    def s(i: int) -> float:
        b = i // block
        base = 1.0 if (b % 2) == 0 else -1.0
        if ramp <= 0:
            return base
        r = i % block
        if r == 0:
            prev = 1.0 if ((b - 1) % 2) == 0 else -1.0
            return prev
        if r < ramp:
            prev = 1.0 if ((b - 1) % 2) == 0 else -1.0
            alpha = r / ramp
            return (1.0 - alpha) * prev + alpha * base
        return base

    # Initialize with small noise
    x[0] = rng.normal(0.0, 0.01, size=d).astype(np.float32)
    x[0, :marker_dims] += marker_amp * s(0)

    for i in range(1, t):
        si = s(i)
        # Marker dims: regime-visible but (mostly) stationary
        x[i, :marker_dims] = marker_amp * si

        # Driven dim: predictable delta conditioned on regime signal
        x[i, driven_dim] = x[i - 1, driven_dim] + driven_step * si + float(rng.normal(0.0, 0.01))

        # Remaining dims: small noise random walk
        for j in range(d):
            if j < marker_dims or j == driven_dim:
                continue
            x[i, j] = x[i - 1, j] + float(rng.normal(0.0, 0.01))

    energy = np.ones((t,), dtype=np.float32)
    timestamps = np.arange(t, dtype=np.float32) / 12.5
    return x, energy, timestamps


def test_phase0_pipeline_detects_structure(tmp_path: Path) -> None:
    """
    End-to-end smoke test: when Type 2 structure is present, train VR should
    beat the random baseline by a clear margin.
    """
    out_dir = tmp_path / "outputs" / "phase0"
    out_dir.mkdir(parents=True, exist_ok=True)

    latents_dir = out_dir / "latents.zarr"
    latents_index_path = out_dir / "latents_index.parquet"
    frames_index_path = out_dir / "phase0_frames.parquet"
    conditioning_dir = out_dir / "conditioning"
    metrics_path = out_dir / "metrics.json"
    tables_path = out_dir / "tables.csv"

    config = {
        "data": {"min_duration_sec": 0.0},
        "context": {"window_size": 4, "lags": [1]},
        "clustering": {
            "conditions": [{"name": "mean_vq2", "type": "mean_pool_vq", "k": 2}],
            "min_cluster_size": 1,
        },
        "slices": [{"name": "all", "filter": None}],
        "output": {
            "latents_dir": str(latents_dir),
            "latents_index": str(latents_index_path),
            "frames_index": str(frames_index_path),
            "conditioning_dir": str(conditioning_dir),
            "metrics_file": str(metrics_path),
            "tables_file": str(tables_path),
            "plots_dir": str(out_dir / "plots"),
        },
        "seed": 123,
    }

    config_path = tmp_path / "phase0.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    # Build a tiny dataset: 2 train speakers, 1 eval speaker, 1 utterance each.
    utterances = []
    frames_rows = []
    t_total, d = 64, 8
    window_size = config["context"]["window_size"]
    max_lag = max(config["context"]["lags"])
    first_valid, last_valid = get_valid_frame_range(t_total, window_size, max_lag)

    def add_utt(utt_id: str, speaker_id: int, split: str, seed: int) -> None:
        x, energy, timestamps = _make_regime_latents(t_total, d, seed=seed)
        save_latents_zarr(
            latents=x,
            energy=energy,
            timestamps=timestamps,
            speaker_id=speaker_id,
            utterance_id=utt_id,
            zarr_path=latents_dir,
        )
        utterances.append(
            {
                "utterance_id": utt_id,
                "speaker_id": speaker_id,
                "n_frames": t_total,
                "duration_sec": float(t_total / 12.5),
                "audio_path": "",
            }
        )
        for t in range(first_valid, last_valid):
            frames_rows.append(
                {
                    "utterance_id": utt_id,
                    "speaker_id": speaker_id,
                    "t": t,
                    "pos_frac": float(t / t_total),
                    "energy": float(energy[t]),
                    "split": split,
                    "is_high_energy": True,
                }
            )

    add_utt("utt_train_a", speaker_id=1, split="train", seed=1)
    add_utt("utt_train_b", speaker_id=2, split="train", seed=2)
    add_utt("utt_eval_c", speaker_id=999, split="eval", seed=3)

    save_latents_index(utterances, latents_index_path)
    save_frames_index(pd.DataFrame(frames_rows), frames_index_path)

    results = run_full_analysis(config_path)
    df = results["table"]

    row = df[(df["condition"] == "mean_vq2") & (df["lag"] == 1) & (df["slice"] == "all")].iloc[0]
    train_vr = float(row["train_variance_ratio"])
    random_vr = float(row["random_baseline"])

    # Expect meaningful separation from random on this synthetic dataset.
    assert random_vr > 0.9
    assert train_vr < 0.8
    assert train_vr < (random_vr - 0.1)
