from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class NormStats:
    mean: np.ndarray  # [D]
    std: np.ndarray  # [D]
    n_frames: int

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "n_frames": int(self.n_frames),
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "NormStats":
        data = json.loads(Path(path).read_text())
        return cls(
            mean=np.array(data["mean"], dtype=np.float32),
            std=np.array(data["std"], dtype=np.float32),
            n_frames=int(data["n_frames"]),
        )


class OnlineMeanVarVec:
    """
    Welford online mean/variance for vectors.
    """

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.n = 0
        self.mean = np.zeros((self.dim,), dtype=np.float64)
        self.m2 = np.zeros((self.dim,), dtype=np.float64)

    def update_batch(self, x: np.ndarray) -> None:
        """
        Update stats with a batch x of shape [T, D] using a vectorized
        parallel-variance merge (Chan et al.) to avoid per-frame Python loops.
        """
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Expected [T,{self.dim}] got {x.shape}")
        xb = x.astype(np.float64, copy=False)
        n_b = int(xb.shape[0])
        if n_b == 0:
            return

        mean_b = xb.mean(axis=0)
        # Sum of squared deviations from mean_b
        diff = xb - mean_b
        m2_b = np.sum(diff * diff, axis=0)

        if self.n == 0:
            self.n = n_b
            self.mean = mean_b
            self.m2 = m2_b
            return

        n_a = int(self.n)
        mean_a = self.mean
        m2_a = self.m2

        delta = mean_b - mean_a
        n_t = n_a + n_b
        mean_t = mean_a + delta * (n_b / n_t)
        m2_t = m2_a + m2_b + (delta * delta) * (n_a * n_b / n_t)

        self.n = n_t
        self.mean = mean_t
        self.m2 = m2_t

    def finalize(self, min_std: float = 1e-4) -> NormStats:
        if self.n < 2:
            var = np.ones_like(self.mean, dtype=np.float64)
        else:
            var = self.m2 / (self.n - 1)
        std = np.sqrt(np.maximum(var, float(min_std) ** 2))
        return NormStats(
            mean=self.mean.astype(np.float32),
            std=std.astype(np.float32),
            n_frames=int(self.n),
        )


def normalize_x(x: np.ndarray, stats: NormStats) -> np.ndarray:
    return (x - stats.mean) / stats.std


def denormalize_x(xn: np.ndarray, stats: NormStats) -> np.ndarray:
    return xn * stats.std + stats.mean
