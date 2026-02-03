from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _mlp(in_dim: int, out_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> nn.Module:
    if num_layers <= 1:
        return nn.Linear(in_dim, out_dim)
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(nn.GELU())
        if dropout and dropout > 0:
            layers.append(nn.Dropout(p=float(dropout)))
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class DiagGaussianParams:
    mu: torch.Tensor  # [..., D]
    log_sigma: torch.Tensor  # [..., D]


def diag_gaussian_kl(q: DiagGaussianParams, p: DiagGaussianParams) -> torch.Tensor:
    """
    KL(q||p) for diagonal Gaussians, summed over last dim.
    Returns: [...], in nats.
    """
    # KL = log(sigma_p/sigma_q) + (sigma_q^2 + (mu_q-mu_p)^2) / (2 sigma_p^2) - 1/2
    log_sigma_q = q.log_sigma
    log_sigma_p = p.log_sigma
    sigma_q2 = torch.exp(2.0 * log_sigma_q)
    sigma_p2 = torch.exp(2.0 * log_sigma_p)
    diff2 = (q.mu - p.mu) ** 2
    term = (sigma_q2 + diff2) / (2.0 * sigma_p2)
    kl_per_dim = (log_sigma_p - log_sigma_q) + term - 0.5
    return kl_per_dim.sum(dim=-1)


def diag_gaussian_nll(x: torch.Tensor, p: DiagGaussianParams) -> torch.Tensor:
    """
    NLL of x under diagonal Gaussian p, summed over last dim.
    Returns: [...], in nats.
    """
    log2pi = 1.8378770664093453
    z2 = ((x - p.mu) ** 2) * torch.exp(-2.0 * p.log_sigma)
    return 0.5 * (z2 + 2.0 * p.log_sigma + log2pi).sum(dim=-1)


def sample_diag_gaussian(p: DiagGaussianParams) -> torch.Tensor:
    eps = torch.randn_like(p.mu)
    return p.mu + torch.exp(p.log_sigma) * eps


class DynEncoderGRU(nn.Module):
    """
    Causal state encoder e_dyn: x -> z_dyn.
    """

    def __init__(self, x_dim: int, hidden_dim: int, z_dyn_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=x_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=float(dropout) if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, z_dyn_dim)
        self.norm = nn.LayerNorm(z_dyn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.norm(self.proj(h))


class DynModelGRU(nn.Module):
    """
    Teacher-forced dynamics model for z_dyn: predicts next z_dyn as diag Gaussian.
    """

    def __init__(
        self,
        z_dyn_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        min_log_sigma: float = -6.0,
        max_log_sigma: float = 1.0,
    ):
        super().__init__()
        self.min_log_sigma = float(min_log_sigma)
        self.max_log_sigma = float(max_log_sigma)
        self.gru = nn.GRU(
            input_size=z_dyn_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=float(dropout) if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.mu = nn.Linear(hidden_dim, z_dyn_dim)
        self.log_sigma = nn.Linear(hidden_dim, z_dyn_dim)
        nn.init.constant_(self.log_sigma.bias, -1.0)

    def forward(self, z_dyn_prev: torch.Tensor) -> DiagGaussianParams:
        # z_dyn_prev: [B, T-1, D], predict params for z_dyn[:,1:]
        h, _ = self.gru(z_dyn_prev)
        mu = self.mu(h)
        log_sigma = torch.clamp(self.log_sigma(h), self.min_log_sigma, self.max_log_sigma)
        return DiagGaussianParams(mu=mu, log_sigma=log_sigma)


class PosteriorNet(nn.Module):
    def __init__(
        self,
        x_dim: int,
        z_dyn_dim: int,
        z_rec_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        min_log_sigma: float,
        max_log_sigma: float,
    ):
        super().__init__()
        self.min_log_sigma = float(min_log_sigma)
        self.max_log_sigma = float(max_log_sigma)
        in_dim = x_dim + z_dyn_dim
        self.mu = _mlp(in_dim, z_rec_dim, hidden_dim, num_layers, dropout)
        self.log_sigma = _mlp(in_dim, z_rec_dim, hidden_dim, num_layers, dropout)
        if isinstance(self.log_sigma, nn.Linear):
            nn.init.constant_(self.log_sigma.bias, -1.0)

    def forward(self, x: torch.Tensor, z_dyn: torch.Tensor) -> DiagGaussianParams:
        h = torch.cat([x, z_dyn], dim=-1)
        mu = self.mu(h)
        log_sigma = torch.clamp(self.log_sigma(h), self.min_log_sigma, self.max_log_sigma)
        return DiagGaussianParams(mu=mu, log_sigma=log_sigma)


class PriorNet(nn.Module):
    def __init__(
        self,
        z_dyn_dim: int,
        z_rec_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        min_log_sigma: float,
        max_log_sigma: float,
    ):
        super().__init__()
        self.min_log_sigma = float(min_log_sigma)
        self.max_log_sigma = float(max_log_sigma)
        self.mu = _mlp(z_dyn_dim, z_rec_dim, hidden_dim, num_layers, dropout)
        self.log_sigma = _mlp(z_dyn_dim, z_rec_dim, hidden_dim, num_layers, dropout)
        if isinstance(self.log_sigma, nn.Linear):
            nn.init.constant_(self.log_sigma.bias, -1.0)

    def forward(self, z_dyn: torch.Tensor) -> DiagGaussianParams:
        mu = self.mu(z_dyn)
        log_sigma = torch.clamp(self.log_sigma(z_dyn), self.min_log_sigma, self.max_log_sigma)
        return DiagGaussianParams(mu=mu, log_sigma=log_sigma)


class Reconstructor(nn.Module):
    def __init__(self, z_dyn_dim: int, z_rec_dim: int, x_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.net = _mlp(z_dyn_dim + z_rec_dim, x_dim, hidden_dim, num_layers, dropout)

    def forward(self, z_dyn: torch.Tensor, z_rec: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z_dyn, z_rec], dim=-1)
        return self.net(h)


@dataclass
class FactorizerOutput:
    z_dyn: torch.Tensor  # [B,T,z_dyn]
    q_rec: DiagGaussianParams  # [B,T,z_rec]
    p_rec: DiagGaussianParams  # [B,T,z_rec]
    z_rec_post: torch.Tensor  # [B,T,z_rec]
    z_rec_prior: torch.Tensor  # [B,T,z_rec]
    x_hat_post: torch.Tensor  # [B,T,x]
    x_hat_prior: torch.Tensor  # [B,T,x]
    x_hat_mixed: torch.Tensor  # [B,T,x]
    prior_mask: torch.Tensor  # [B,T,1] float in {0,1}


class Factorizer(nn.Module):
    def __init__(
        self,
        *,
        x_dim: int,
        z_dyn_dim: int,
        z_rec_dim: int,
        dyn_encoder_hidden: int,
        dyn_encoder_layers: int,
        dyn_encoder_dropout: float,
        dyn_model_hidden: int,
        dyn_model_layers: int,
        dyn_model_dropout: float,
        dyn_model_min_log_sigma: float,
        dyn_model_max_log_sigma: float,
        posterior_hidden: int,
        posterior_layers: int,
        posterior_dropout: float,
        posterior_min_log_sigma: float,
        posterior_max_log_sigma: float,
        prior_hidden: int,
        prior_layers: int,
        prior_dropout: float,
        prior_min_log_sigma: float,
        prior_max_log_sigma: float,
        recon_hidden: int,
        recon_layers: int,
        recon_dropout: float,
        z_dyn_layernorm: bool = True,
    ):
        super().__init__()
        self.x_dim = int(x_dim)
        self.z_dyn_dim = int(z_dyn_dim)
        self.z_rec_dim = int(z_rec_dim)

        self.e_dyn = DynEncoderGRU(
            x_dim=self.x_dim,
            hidden_dim=int(dyn_encoder_hidden),
            z_dyn_dim=self.z_dyn_dim,
            num_layers=int(dyn_encoder_layers),
            dropout=float(dyn_encoder_dropout),
        )
        if not bool(z_dyn_layernorm):
            # Replace the default LayerNorm with identity if requested.
            self.e_dyn.norm = nn.Identity()
        self.dyn = DynModelGRU(
            z_dyn_dim=self.z_dyn_dim,
            hidden_dim=int(dyn_model_hidden),
            num_layers=int(dyn_model_layers),
            dropout=float(dyn_model_dropout),
            min_log_sigma=float(dyn_model_min_log_sigma),
            max_log_sigma=float(dyn_model_max_log_sigma),
        )
        self.q = PosteriorNet(
            x_dim=self.x_dim,
            z_dyn_dim=self.z_dyn_dim,
            z_rec_dim=self.z_rec_dim,
            hidden_dim=int(posterior_hidden),
            num_layers=int(posterior_layers),
            dropout=float(posterior_dropout),
            min_log_sigma=float(posterior_min_log_sigma),
            max_log_sigma=float(posterior_max_log_sigma),
        )
        self.p = PriorNet(
            z_dyn_dim=self.z_dyn_dim,
            z_rec_dim=self.z_rec_dim,
            hidden_dim=int(prior_hidden),
            num_layers=int(prior_layers),
            dropout=float(prior_dropout),
            min_log_sigma=float(prior_min_log_sigma),
            max_log_sigma=float(prior_max_log_sigma),
        )
        self.g = Reconstructor(
            z_dyn_dim=self.z_dyn_dim,
            z_rec_dim=self.z_rec_dim,
            x_dim=self.x_dim,
            hidden_dim=int(recon_hidden),
            num_layers=int(recon_layers),
            dropout=float(recon_dropout),
        )

    def forward(self, x: torch.Tensor, *, prior_sample_prob: float = 0.0) -> FactorizerOutput:
        """
        x: [B,T,x_dim] (normalized if enabled)
        """
        z_dyn = self.e_dyn(x)
        q_rec = self.q(x, z_dyn)
        p_rec = self.p(z_dyn)

        z_rec_post = sample_diag_gaussian(q_rec)
        z_rec_prior = sample_diag_gaussian(p_rec)

        x_hat_post = self.g(z_dyn, z_rec_post)
        x_hat_prior = self.g(z_dyn, z_rec_prior)

        if prior_sample_prob and prior_sample_prob > 0:
            # Per-timestep Bernoulli mask
            b, t, _ = x.shape
            prior_mask = (torch.rand((b, t, 1), device=x.device) < float(prior_sample_prob)).to(x.dtype)
            z_rec_mix = prior_mask * z_rec_prior + (1.0 - prior_mask) * z_rec_post
        else:
            z_rec_mix = z_rec_post
            prior_mask = torch.zeros((x.shape[0], x.shape[1], 1), device=x.device, dtype=x.dtype)

        x_hat_mixed = self.g(z_dyn, z_rec_mix)
        return FactorizerOutput(
            z_dyn=z_dyn,
            q_rec=q_rec,
            p_rec=p_rec,
            z_rec_post=z_rec_post,
            z_rec_prior=z_rec_prior,
            x_hat_post=x_hat_post,
            x_hat_prior=x_hat_prior,
            x_hat_mixed=x_hat_mixed,
            prior_mask=prior_mask,
        )
