"""
Mixture Density Network for high-dimensional Δx prediction.

Models p(Δx | context) as a K-component diagonal Gaussian mixture.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MDNOutput:
    logit_pi: torch.Tensor  # [B, K]
    mu: torch.Tensor  # [B, K, D]
    log_sigma: torch.Tensor  # [B, K, D]


def _diag_gaussian_log_prob(x: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """
    Log probability for diagonal Gaussian.

    Shapes:
      x:         [B, D]
      mu:        [B, K, D]
      log_sigma: [B, K, D]
    Returns:
      log p(x | k): [B, K]
    """
    # [B, 1, D]
    x_ = x.unsqueeze(1)
    inv_sigma = torch.exp(-log_sigma)
    z = (x_ - mu) * inv_sigma
    log2pi = 1.8378770664093453  # log(2*pi)
    return -0.5 * (z * z).sum(dim=-1) - log_sigma.sum(dim=-1) - 0.5 * x.shape[-1] * log2pi


class MDN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_components: int = 8,
        hidden_dim: int = 1024,
        n_hidden_layers: int = 2,
        dropout: float = 0.0,
        min_log_sigma: float = -7.0,
        max_log_sigma: float = 2.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.n_components = int(n_components)
        self.min_log_sigma = float(min_log_sigma)
        self.max_log_sigma = float(max_log_sigma)

        layers: list[nn.Module] = []
        d = self.input_dim
        for _ in range(int(n_hidden_layers)):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.GELU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            d = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()

        # Heads
        self.pi_head = nn.Linear(d, self.n_components)
        self.mu_head = nn.Linear(d, self.n_components * self.output_dim)
        self.log_sigma_head = nn.Linear(d, self.n_components * self.output_dim)

        # Init: slightly conservative sigmas
        nn.init.constant_(self.log_sigma_head.bias, -1.0)

    def forward(self, context_flat: torch.Tensor) -> MDNOutput:
        h = self.backbone(context_flat)
        logit_pi = self.pi_head(h)
        mu = self.mu_head(h).view(-1, self.n_components, self.output_dim)
        log_sigma = self.log_sigma_head(h).view(-1, self.n_components, self.output_dim)
        log_sigma = torch.clamp(log_sigma, min=self.min_log_sigma, max=self.max_log_sigma)
        return MDNOutput(logit_pi=logit_pi, mu=mu, log_sigma=log_sigma)

    @torch.no_grad()
    def expected_mean(self, context_flat: torch.Tensor) -> torch.Tensor:
        out = self.forward(context_flat)
        pi = torch.softmax(out.logit_pi, dim=-1)  # [B, K]
        return (pi.unsqueeze(-1) * out.mu).sum(dim=1)  # [B, D]

    def nll(self, context_flat: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood per sample.
        Returns: [B]
        """
        out = self.forward(context_flat)
        log_pi = torch.log_softmax(out.logit_pi, dim=-1)  # [B, K]
        log_p_x_given_k = _diag_gaussian_log_prob(delta, out.mu, out.log_sigma)  # [B, K]
        log_p = torch.logsumexp(log_pi + log_p_x_given_k, dim=-1)  # [B]
        return -log_p


@torch.no_grad()
def sample_from_mdn(out: MDNOutput) -> torch.Tensor:
    """
    Sample Δx from MDN output.
    Returns: [B, D]
    """
    pi = torch.softmax(out.logit_pi, dim=-1)  # [B, K]
    cat = torch.distributions.Categorical(probs=pi)
    k = cat.sample()  # [B]
    # Gather component params
    b = torch.arange(out.mu.shape[0], device=out.mu.device)
    mu = out.mu[b, k]
    sigma = torch.exp(out.log_sigma[b, k])
    eps = torch.randn_like(mu)
    return mu + sigma * eps

