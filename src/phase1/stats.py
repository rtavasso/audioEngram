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


@dataclass
class OnlineMeanVarScalar:
    """
    Numerically stable online mean/variance for a scalar stream.
    """

    n: int
    mean: float
    m2: float

    @classmethod
    def create(cls) -> "OnlineMeanVarScalar":
        return cls(n=0, mean=0.0, m2=0.0)

    def update(self, x: float) -> None:
        self.n += 1
        delta = float(x) - self.mean
        self.mean += delta / self.n
        delta2 = float(x) - self.mean
        self.m2 += delta * delta2

    def finalize(self, min_var: float = 1e-6) -> tuple[float, float]:
        if self.n < 2:
            return float(self.mean), float(min_var)
        var = self.m2 / (self.n - 1)
        return float(self.mean), float(max(var, min_var))


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


class FactorizedDirectionMagnitudeBaseline:
    """
    Unconditional baseline for the factorized (direction, magnitude) objective.

    Direction: uniform on the unit sphere in D dims.
    Magnitude: LogNormal fit to ||Δx|| on train.
    """

    def __init__(self, *, dim: int, logm_mean: float, logm_var: float):
        import math

        self.dim = int(dim)
        self.logm_mean = float(logm_mean)
        self.logm_var = float(logm_var)
        self.logm_sigma = float(math.sqrt(max(self.logm_var, 1e-12)))

        # Area(S^{D-1}) = 2 * pi^{D/2} / Gamma(D/2)
        self._log_sphere_area = float(math.log(2.0) + (self.dim / 2.0) * math.log(math.pi) - math.lgamma(self.dim / 2.0))

        self._cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def nll(self, delta: torch.Tensor) -> torch.Tensor:
        """
        delta: [B,D] -> nll: [B] in nats (direction + magnitude).
        """
        key = (str(delta.device), str(delta.dtype))
        cached = self._cache.get(key)
        if cached is None:
            log_area = torch.tensor(self._log_sphere_area, device=delta.device, dtype=delta.dtype)
            logm_mean = torch.tensor(self.logm_mean, device=delta.device, dtype=delta.dtype)
            logm_sigma = torch.tensor(self.logm_sigma, device=delta.device, dtype=delta.dtype)
            cached = (log_area, logm_mean, logm_sigma)
            self._cache[key] = cached
        else:
            log_area, logm_mean, logm_sigma = cached

        eps = 1e-8
        m = torch.linalg.vector_norm(delta, dim=-1).clamp_min(eps)  # [B]
        z = (torch.log(m) - logm_mean) / logm_sigma.clamp_min(eps)
        log2pi = 1.8378770664093453
        nll_mag = torch.log(m) + torch.log(logm_sigma.clamp_min(eps)) + 0.5 * (z * z + log2pi)

        # Uniform direction baseline: nll_dir = log Area(S^{D-1})
        nll_dir = log_area.expand_as(nll_mag)
        return nll_dir + nll_mag
