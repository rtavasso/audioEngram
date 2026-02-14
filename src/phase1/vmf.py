"""
Phase 1: vMF (direction) + LogNormal (magnitude) factorized dynamics model.

Models Δx as:
  d = Δx / ||Δx||  (unit direction)
  m = ||Δx||       (magnitude)

  d ~ vMF(mu_dir(ctx), kappa(ctx))
  m ~ LogNormal(mu_logm(ctx), sigma_logm(ctx))

This is intended to preserve *directional* predictability under rollout while
allowing magnitude uncertainty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def _mlp(in_dim: int, out_dim: int, hidden_dim: int, n_hidden_layers: int, dropout: float) -> nn.Module:
    if n_hidden_layers <= 1:
        return nn.Linear(in_dim, out_dim)
    layers: list[nn.Module] = []
    d = int(in_dim)
    for _ in range(int(n_hidden_layers) - 1):
        layers.append(nn.Linear(d, int(hidden_dim)))
        layers.append(nn.GELU())
        if dropout and float(dropout) > 0:
            layers.append(nn.Dropout(p=float(dropout)))
        d = int(hidden_dim)
    layers.append(nn.Linear(d, int(out_dim)))
    return nn.Sequential(*layers)


def _log_sphere_area(dim: int) -> float:
    # Area(S^{D-1}) = 2 * pi^{D/2} / Gamma(D/2)
    d = float(dim)
    return math.log(2.0) + (d / 2.0) * math.log(math.pi) - math.lgamma(d / 2.0)


def _log_vmf_normalizer_saddlepoint(kappa: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Differentiable approximation for log C_D(kappa) in D dimensions.

    Uses a large-order saddlepoint approximation for log I_v(kappa) with v = D/2 - 1,
    which is accurate for the high-dimensional regime we care about (D~512).
    """
    if dim < 2:
        raise ValueError("vMF requires dim >= 2")

    kappa = kappa.clamp_min(0.0)
    nu = (float(dim) / 2.0) - 1.0  # order
    nu_t = torch.tensor(nu, device=kappa.device, dtype=kappa.dtype)

    # For very small kappa, use the uniform-on-sphere limit: C(0) = 1/Area(S^{D-1})
    log_c0 = -_log_sphere_area(dim)

    # Avoid log(0) in the approximation; value doesn't matter when we take the small-kappa branch.
    k = kappa.clamp_min(1e-8)
    s = torch.sqrt(nu_t * nu_t + k * k)

    # log I_nu(k) ≈ s + nu*log(k/(nu+s)) - 0.5*log(2π s)
    log_iv = s + nu_t * torch.log(k / (nu_t + s)) - 0.5 * torch.log(2.0 * math.pi * s)

    # log C_D(k) = nu*log(k) - (D/2)*log(2π) - log I_nu(k)
    log_c = nu_t * torch.log(k) - (float(dim) / 2.0) * math.log(2.0 * math.pi) - log_iv

    return torch.where(kappa < 1e-4, torch.full_like(log_c, log_c0), log_c)


def vmf_nll(
    *,
    d_true: torch.Tensor,  # [B,D] unit
    mu_dir: torch.Tensor,  # [B,D] unit
    kappa: torch.Tensor,  # [B]
    eps: float = 1e-8,
) -> torch.Tensor:
    if d_true.ndim != 2 or mu_dir.ndim != 2:
        raise ValueError("d_true and mu_dir must be [B,D]")
    if d_true.shape != mu_dir.shape:
        raise ValueError("d_true and mu_dir shape mismatch")
    if kappa.ndim != 1 or kappa.shape[0] != d_true.shape[0]:
        raise ValueError("kappa must be [B]")

    # Re-normalize defensively.
    d = d_true / torch.linalg.vector_norm(d_true, dim=-1, keepdim=True).clamp_min(eps)
    mu = mu_dir / torch.linalg.vector_norm(mu_dir, dim=-1, keepdim=True).clamp_min(eps)
    dim = int(d.shape[1])

    log_c = _log_vmf_normalizer_saddlepoint(kappa, dim=dim)  # [B]
    dot = (mu * d).sum(dim=-1)  # [B]
    logp = log_c + kappa * dot
    return -logp


def lognormal_nll(
    *,
    m_true: torch.Tensor,  # [B]
    mu_logm: torch.Tensor,  # [B]
    sigma_logm: torch.Tensor,  # [B]
    eps: float = 1e-8,
) -> torch.Tensor:
    m = m_true.clamp_min(eps)
    sigma = sigma_logm.clamp_min(eps)
    z = (torch.log(m) - mu_logm) / sigma
    log2pi = 1.8378770664093453
    return torch.log(m) + torch.log(sigma) + 0.5 * (z * z + log2pi)


@dataclass(frozen=True)
class VmfLogNormalParams:
    mu_dir: torch.Tensor  # [B,D] unit
    log_kappa: torch.Tensor  # [B]
    mu_logm: torch.Tensor  # [B]
    log_sigma_logm: torch.Tensor  # [B]


class VmfLogNormal(nn.Module):
    """
    Factorized Δx model with the same (ctx -> params) interface as MDN.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        dropout: float,
        min_log_kappa: float,
        max_log_kappa: float,
        min_log_sigma_logm: float,
        max_log_sigma_logm: float,
        min_mu_logm: float = -5.0,
        max_mu_logm: float = 12.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        self._min_log_kappa = float(min_log_kappa)
        self._max_log_kappa = float(max_log_kappa)
        self._min_log_sigma_logm = float(min_log_sigma_logm)
        self._max_log_sigma_logm = float(max_log_sigma_logm)
        self._min_mu_logm = float(min_mu_logm)
        self._max_mu_logm = float(max_mu_logm)

        self.mu_dir = _mlp(self.input_dim, self.output_dim, int(hidden_dim), int(n_hidden_layers), float(dropout))
        self.log_kappa = _mlp(self.input_dim, 1, int(hidden_dim), max(1, int(n_hidden_layers)), float(dropout))
        self.mu_logm = _mlp(self.input_dim, 1, int(hidden_dim), max(1, int(n_hidden_layers)), float(dropout))
        self.log_sigma_logm = _mlp(self.input_dim, 1, int(hidden_dim), max(1, int(n_hidden_layers)), float(dropout))

        # Bias toward moderate concentration and magnitude uncertainty.
        if isinstance(self.log_kappa, nn.Linear):
            nn.init.constant_(self.log_kappa.bias, 3.0)
        if isinstance(self.log_sigma_logm, nn.Linear):
            nn.init.constant_(self.log_sigma_logm.bias, -0.5)

    def forward(self, ctx: torch.Tensor, *, eps: float = 1e-8) -> VmfLogNormalParams:
        mu_dir = self.mu_dir(ctx)
        mu_dir = mu_dir / torch.linalg.vector_norm(mu_dir, dim=-1, keepdim=True).clamp_min(eps)

        log_kappa = self.log_kappa(ctx).squeeze(-1)
        log_kappa = torch.clamp(log_kappa, self._min_log_kappa, self._max_log_kappa)

        mu_logm = self.mu_logm(ctx).squeeze(-1)
        mu_logm = torch.clamp(mu_logm, self._min_mu_logm, self._max_mu_logm)

        log_sigma_logm = self.log_sigma_logm(ctx).squeeze(-1)
        log_sigma_logm = torch.clamp(log_sigma_logm, self._min_log_sigma_logm, self._max_log_sigma_logm)

        return VmfLogNormalParams(mu_dir=mu_dir, log_kappa=log_kappa, mu_logm=mu_logm, log_sigma_logm=log_sigma_logm)

    def nll(self, ctx: torch.Tensor, dx: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
        p = self(ctx, eps=eps)
        m_true = torch.linalg.vector_norm(dx, dim=-1).clamp_min(eps)  # [B]
        d_true = dx / m_true.unsqueeze(-1)

        kappa = torch.exp(p.log_kappa)
        sigma_logm = torch.exp(p.log_sigma_logm)

        nll_dir = vmf_nll(d_true=d_true, mu_dir=p.mu_dir, kappa=kappa, eps=eps)
        nll_mag = lognormal_nll(m_true=m_true, mu_logm=p.mu_logm, sigma_logm=sigma_logm, eps=eps)
        return nll_dir + nll_mag

    def expected_mean(self, ctx: torch.Tensor) -> torch.Tensor:
        p = self(ctx)
        # E[m] for LogNormal = exp(mu + 0.5 sigma^2)
        sigma = torch.exp(p.log_sigma_logm)
        m_mean = torch.exp(p.mu_logm + 0.5 * (sigma * sigma))  # [B]
        return p.mu_dir * m_mean.unsqueeze(-1)

    def rollout_mean(self, ctx: torch.Tensor) -> torch.Tensor:
        """Deterministic Δx using median magnitude: exp(mu_logm) instead of E[m]=exp(mu+0.5*sigma^2)."""
        p = self(ctx)
        m_median = torch.exp(p.mu_logm)  # [B] — no sigma blow-up
        return p.mu_dir * m_median.unsqueeze(-1)

    @torch.no_grad()
    def sample_delta(self, ctx: torch.Tensor, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Sample Δx from the model. Intended for rollout diagnostics.
        """
        p = self(ctx)
        kappa = torch.exp(p.log_kappa)
        sigma = torch.exp(p.log_sigma_logm)

        # Sample magnitude: LogNormal
        eps = torch.randn((ctx.shape[0],), device=ctx.device, dtype=ctx.dtype, generator=generator)
        m = torch.exp(p.mu_logm + sigma * eps)  # [B]

        # Sample direction: Wood's algorithm (vectorized-ish per batch via python loop for robustness).
        d = _sample_vmf_batch(mu=p.mu_dir, kappa=kappa, generator=generator)
        return d * m.unsqueeze(-1)


@torch.no_grad()
def _sample_vmf_batch(
    *,
    mu: torch.Tensor,  # [B,D] unit
    kappa: torch.Tensor,  # [B]
    generator: Optional[torch.Generator],
    max_tries: int = 64,
) -> torch.Tensor:
    """
    Sample from vMF(mu, kappa) for each batch element (Wood, 1994).

    Implementation prioritizes clarity + stability over maximum throughput.
    """
    if mu.ndim != 2:
        raise ValueError("mu must be [B,D]")
    if kappa.ndim != 1 or kappa.shape[0] != mu.shape[0]:
        raise ValueError("kappa must be [B]")

    bsz, dim = int(mu.shape[0]), int(mu.shape[1])
    if dim < 2:
        raise ValueError("dim must be >= 2")

    device = mu.device
    dtype = mu.dtype

    out = torch.empty_like(mu)
    for i in range(bsz):
        mui = mu[i]
        ki = float(kappa[i].clamp_min(0.0).item())

        if ki < 1e-4:
            # Approximately uniform on the sphere.
            v = torch.randn((dim,), device=device, dtype=dtype, generator=generator)
            out[i] = v / torch.linalg.vector_norm(v).clamp_min(1e-8)
            continue

        # Wood's rejection sampling for w
        d1 = dim - 1.0
        b = (-2.0 * ki + math.sqrt(4.0 * ki * ki + d1 * d1)) / d1
        a = (d1 + 2.0 * ki + math.sqrt(4.0 * ki * ki + d1 * d1)) / 4.0
        d_const = 4.0 * a * b / (1.0 + b) - d1 * math.log(d1)

        w = None
        for _ in range(int(max_tries)):
            # torch.distributions sampling does not consistently support passing a Generator
            # across torch versions; rely on the global RNG seeded via phase0.utils.seed.
            eps = (
                torch.distributions.Beta(d1 / 2.0, d1 / 2.0)
                .sample((1,))
                .to(device=device, dtype=dtype)[0]
            )
            wi = (1.0 - (1.0 + b) * eps) / (1.0 - (1.0 - b) * eps)
            t = 2.0 * a * b / (1.0 - (1.0 - b) * eps)
            u = torch.rand((1,), device=device, dtype=dtype, generator=generator)[0]
            if d1 * torch.log(t) - t + d_const >= torch.log(u):
                w = wi
                break

        if w is None:
            # Fallback: use mean direction.
            out[i] = mui
            continue

        # Sample v uniformly from the unit sphere orthogonal to mu.
        v = torch.randn((dim,), device=device, dtype=dtype, generator=generator)
        v = v - (v @ mui) * mui
        v = v / torch.linalg.vector_norm(v).clamp_min(1e-8)

        x = w * mui + torch.sqrt(torch.clamp(1.0 - w * w, min=0.0)) * v
        out[i] = x / torch.linalg.vector_norm(x).clamp_min(1e-8)

    return out
