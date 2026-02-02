"""
Online statistics and simple baselines for Δx.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class OnlineMeanVar:
    n: int
    mean: np.ndarray
    m2: np.ndarray

    @classmethod
    def create(cls, dim: int) -> "OnlineMeanVar":
        return cls(n=0, mean=np.zeros((dim,), dtype=np.float64), m2=np.zeros((dim,), dtype=np.float64))

    def update(self, x: np.ndarray) -> None:
        # x: [D] float32/float64
        x64 = x.astype(np.float64, copy=False)
        self.n += 1
        delta = x64 - self.mean
        self.mean += delta / self.n
        delta2 = x64 - self.mean
        self.m2 += delta * delta2

    def finalize(self, min_var: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        if self.n < 2:
            var = np.full_like(self.mean, min_var, dtype=np.float64)
        else:
            var = self.m2 / (self.n - 1)
            var = np.maximum(var, min_var)
        return self.mean.astype(np.float32), var.astype(np.float32)


class DiagGaussianBaseline:
    """
    Unconditional p(Δx) baseline as diagonal Gaussian fit on train deltas.
    """

    def __init__(self, mean: np.ndarray, var: np.ndarray):
        self.mean = mean.astype(np.float32, copy=False)
        self.var = var.astype(np.float32, copy=False)
        self.log_var = np.log(self.var)
        self._cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def nll(self, delta: torch.Tensor) -> torch.Tensor:
        """
        delta: [B, D]
        returns: [B] nats
        """
        key = (str(delta.device), str(delta.dtype))
        cached = self._cache.get(key)
        if cached is None:
            mean = torch.from_numpy(self.mean).to(delta.device, dtype=delta.dtype)
            log_var = torch.from_numpy(self.log_var).to(delta.device, dtype=delta.dtype)
            cached = (mean, log_var)
            self._cache[key] = cached
        else:
            mean, log_var = cached
        var = torch.exp(log_var)
        z2 = (delta - mean) ** 2 / var
        log2pi = 1.8378770664093453
        return 0.5 * (z2 + log_var + log2pi).sum(dim=-1)
