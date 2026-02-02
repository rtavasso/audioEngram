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
        # x: [T, D]
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Expected [T,{self.dim}] got {x.shape}")
        for row in x.astype(np.float64, copy=False):
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            delta2 = row - self.mean
            self.m2 += delta * delta2

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

