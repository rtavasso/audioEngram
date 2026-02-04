import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase0.data.io import save_latents_zarr, save_latents_index
from phase1.injection_diag import run_injection_diagnostic


class _ZeroModel:
    def __init__(self, output_dim: int):
        self.output_dim = int(output_dim)

    def eval(self):
        return self

    def expected_mean(self, ctx: torch.Tensor) -> torch.Tensor:
        return torch.zeros((ctx.shape[0], self.output_dim), device=ctx.device, dtype=ctx.dtype)

    def sample_delta(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.expected_mean(ctx)

    def nll(self, ctx: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        # simple finite scalar per sample
        return 0.5 * (dx * dx).sum(dim=-1)


class _ZeroBaseline:
    def nll(self, dx: torch.Tensor) -> torch.Tensor:
        return 0.5 * (dx * dx).sum(dim=-1)


def _write_splits(splits_dir: Path, *, train_speakers: list[int], eval_speakers: list[int]) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train_speakers.txt").write_text("\n".join(str(x) for x in train_speakers) + "\n")
    (splits_dir / "eval_speakers.txt").write_text("\n".join(str(x) for x in eval_speakers) + "\n")


def test_injection_diagnostic_teacher_and_reset(tmp_path: Path) -> None:
    torch.manual_seed(0)

    latents_dir = tmp_path / "latents.zarr"
    latents_index = tmp_path / "latents_index.parquet"
    splits_dir = tmp_path / "splits"

    # Make a tiny store: one eval utterance with deterministic nonzero deltas.
    speaker_id = 123
    utt_id = "utt0"
    t, d = 80, 4
    x = (np.arange(t, dtype=np.float32).reshape(-1, 1) * np.ones((1, d), dtype=np.float32))  # Δ=1 along all dims
    energy = np.ones((t,), dtype=np.float32)
    timestamps = np.arange(t, dtype=np.float32) / 12.5
    save_latents_zarr(x, energy, timestamps, speaker_id=speaker_id, utterance_id=utt_id, zarr_path=latents_dir)

    save_latents_index(
        [
            {
                "utterance_id": utt_id,
                "speaker_id": speaker_id,
                "n_frames": t,
                "duration_sec": float(t) / 12.5,
                "audio_path": "dummy.flac",
            }
        ],
        latents_index,
    )

    _write_splits(splits_dir, train_speakers=[], eval_speakers=[speaker_id])

    model = _ZeroModel(output_dim=d)
    baseline = _ZeroBaseline()

    res = run_injection_diagnostic(
        model=model,
        baseline=baseline,
        latents_dir=latents_dir,
        latents_index_path=latents_index,
        splits_dir=splits_dir,
        horizon_k=1,
        window_size=4,
        k_steps=8,
        n_eval_utterances=1,
        segments_per_utt=2,
        max_frames_per_utt=60,
        seed=0,
        device=torch.device("cpu"),
        mode_inject_after_steps={
            "A_teacher": None,
            "C_one_shot": [1],
            "D_rollout": [],
        },
        sample_from_model=False,
    )

    a = res["modes"]["A_teacher"]["per_step"]
    assert all(float(s["state_err"]) == 0.0 for s in a), "Teacher forcing should have zero state error."

    c = res["modes"]["C_one_shot"]["per_step"]
    # After injecting at step 1, the *next* pre-step error should be reset.
    assert float(c[1]["state_err"]) == 0.0
    assert float(c[2]["state_err"]) > 0.0

    dmode = res["modes"]["D_rollout"]["per_step"]
    assert float(dmode[1]["state_err"]) > 0.0

