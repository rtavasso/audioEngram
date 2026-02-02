from __future__ import annotations

from dataclasses import dataclass

import torch

from .models import DiagGaussianParams, diag_gaussian_kl, diag_gaussian_nll


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # x: [B,T] or [B,T,*], mask: [B,T]
    m = mask.to(dtype=x.dtype)
    while m.ndim < x.ndim:
        m = m.unsqueeze(-1)
    num = (x * m).sum()
    den = m.sum().clamp_min(1.0)
    return num / den


def free_bits_kl(kl: torch.Tensor, *, free_bits_per_dim: float, n_dims: int) -> torch.Tensor:
    """
    Apply free bits per dim to KL that is summed over dims.
    kl: [B,T] in nats (sum over dims already)
    """
    if free_bits_per_dim <= 0:
        return kl
    fb = float(free_bits_per_dim) * float(n_dims)
    return torch.clamp(kl, min=fb)


@dataclass
class LossBreakdown:
    total: torch.Tensor
    recon: torch.Tensor
    kl: torch.Tensor
    dyn: torch.Tensor


def compute_losses(
    *,
    x: torch.Tensor,  # [B,T,D]
    mask: torch.Tensor,  # [B,T]
    x_hat: torch.Tensor,  # [B,T,D]
    q_rec: DiagGaussianParams,  # [B,T,z_rec]
    p_rec: DiagGaussianParams,  # [B,T,z_rec]
    dyn_params: DiagGaussianParams,  # [B,T-1,z_dyn]
    z_dyn_target: torch.Tensor,  # [B,T-1,z_dyn]
    recon_weight: float,
    beta: float,
    free_bits_per_dim: float,
    z_rec_dim: int,
    dyn_weight: float,
) -> LossBreakdown:
    # Recon MSE per timestep
    recon_per_t = ((x_hat - x) ** 2).mean(dim=-1)  # [B,T]
    recon = masked_mean(recon_per_t, mask)

    # KL per timestep
    kl_per_t = diag_gaussian_kl(q_rec, p_rec)  # [B,T]
    kl_per_t = free_bits_kl(kl_per_t, free_bits_per_dim=free_bits_per_dim, n_dims=z_rec_dim)
    kl = masked_mean(kl_per_t, mask)

    # Dynamics NLL for z_dyn[:,1:] given params from z_dyn[:,:-1]
    dyn_mask = mask[:, 1:]
    dyn_nll_per_t = diag_gaussian_nll(z_dyn_target, dyn_params)  # [B,T-1]
    dyn = masked_mean(dyn_nll_per_t, dyn_mask)

    total = float(recon_weight) * recon + float(beta) * kl + float(dyn_weight) * dyn
    return LossBreakdown(total=total, recon=recon, kl=kl, dyn=dyn)

