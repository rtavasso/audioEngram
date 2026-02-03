from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .memory import KMeansDeltaMemory


def _mlp(in_dim: int, out_dim: int, hidden_dim: int, n_layers: int, dropout: float) -> nn.Module:
    if n_layers <= 1:
        return nn.Linear(in_dim, out_dim)
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(nn.GELU())
        if dropout and dropout > 0:
            layers.append(nn.Dropout(p=float(dropout)))
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class DiagGaussianDelta:
    mu: torch.Tensor  # [B,D]
    log_sigma: torch.Tensor  # [D] or [B,D]


def diag_gaussian_nll(dz: torch.Tensor, params: DiagGaussianDelta) -> torch.Tensor:
    """
    Returns: [B] nats.
    """
    log2pi = 1.8378770664093453
    log_sigma = params.log_sigma
    while log_sigma.ndim < dz.ndim:
        log_sigma = log_sigma.unsqueeze(0)
    z2 = ((dz - params.mu) ** 2) * torch.exp(-2.0 * log_sigma)
    return 0.5 * (z2 + 2.0 * log_sigma + log2pi).sum(dim=-1)


def count_parameters(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))


class ParamDyn(nn.Module):
    """
    Parametric dynamics: mean network + global diagonal log_sigma.
    """

    def __init__(self, *, z_dim: int, hidden_dim: int, n_layers: int, dropout: float, min_log_sigma: float, max_log_sigma: float):
        super().__init__()
        self.net = _mlp(z_dim, z_dim, hidden_dim, n_layers, dropout)
        self._min_log_sigma = float(min_log_sigma)
        self._max_log_sigma = float(max_log_sigma)
        self.log_sigma = nn.Parameter(torch.zeros((z_dim,), dtype=torch.float32))

    def forward(self, z_prev: torch.Tensor) -> DiagGaussianDelta:
        mu = self.net(z_prev)
        log_sigma = torch.clamp(self.log_sigma, self._min_log_sigma, self._max_log_sigma)
        return DiagGaussianDelta(mu=mu, log_sigma=log_sigma)


class HybridDyn(nn.Module):
    """
    Hybrid dynamics: convex combination of param mean and memory mean, with learned gate.
    """

    def __init__(
        self,
        *,
        z_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        gate_hidden_dim: int,
        min_log_sigma: float,
        max_log_sigma: float,
        memory: KMeansDeltaMemory,
        memory_topk: int = 1,
        memory_temperature: float = 1.0,
    ):
        super().__init__()
        self.param = ParamDyn(
            z_dim=z_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            min_log_sigma=min_log_sigma,
            max_log_sigma=max_log_sigma,
        )
        # Simple gate: alpha(z) in (0,1) controlling weight on param vs memory mean.
        self.gate = _mlp(z_dim, 1, gate_hidden_dim, 2, 0.0)
        self.memory = memory
        self.memory_topk = int(memory_topk)
        self.memory_temperature = float(memory_temperature)

    def forward(self, z_prev: torch.Tensor) -> DiagGaussianDelta:
        p = self.param(z_prev)
        mu_mem = self.memory.predict_mean(z_prev, topk=self.memory_topk, temperature=self.memory_temperature)
        alpha = torch.sigmoid(self.gate(z_prev))  # [B,1]
        mu = alpha * p.mu + (1.0 - alpha) * mu_mem
        return DiagGaussianDelta(mu=mu, log_sigma=p.log_sigma)


class ResidualMemDyn(nn.Module):
    """
    Additive residual memory: μ(z) = μ_param(z) + μ_mem_residual(z).
    """

    def __init__(
        self,
        *,
        param: ParamDyn,
        residual_memory: KMeansDeltaMemory,
        memory_topk: int = 1,
        memory_temperature: float = 1.0,
    ):
        super().__init__()
        self.param = param
        self.residual_memory = residual_memory
        self.memory_topk = int(memory_topk)
        self.memory_temperature = float(memory_temperature)

    def forward(self, z_prev: torch.Tensor) -> DiagGaussianDelta:
        p = self.param(z_prev)
        r = self.residual_memory.predict_mean(z_prev, topk=self.memory_topk, temperature=self.memory_temperature)
        return DiagGaussianDelta(mu=p.mu + r, log_sigma=p.log_sigma)


class GatedResidualMemDyn(nn.Module):
    """
    Gated residual memory: μ(z) = μ_param(z) + g(z) * μ_mem_residual(z), g ∈ (0,1).
    """

    def __init__(
        self,
        *,
        param: ParamDyn,
        residual_memory: KMeansDeltaMemory,
        gate_hidden_dim: int,
        memory_topk: int = 1,
        memory_temperature: float = 1.0,
    ):
        super().__init__()
        self.param = param
        self.residual_memory = residual_memory
        self.gate = _mlp(param.log_sigma.numel(), 1, int(gate_hidden_dim), 2, 0.0)
        self.memory_topk = int(memory_topk)
        self.memory_temperature = float(memory_temperature)

    def forward(self, z_prev: torch.Tensor) -> DiagGaussianDelta:
        p = self.param(z_prev)
        r = self.residual_memory.predict_mean(z_prev, topk=self.memory_topk, temperature=self.memory_temperature)
        g = torch.sigmoid(self.gate(z_prev))  # [B,1]
        return DiagGaussianDelta(mu=p.mu + g * r, log_sigma=p.log_sigma)
