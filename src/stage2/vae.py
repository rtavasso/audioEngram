"""
AR-Friendly VAE wrapping Mimi encoder/decoder with a VAE bottleneck.

Architecture:
    Audio [B, 1, T] @ 24kHz
      -> MimiEncoder(project=False)        # pretrained, optionally frozen
      -> h [B, 512, T'] @ 12.5Hz           # raw encoder output (pre-RVQ)
      -> mu_proj:  Conv1d(512, D_vae, 1)   # trainable
      -> logvar_proj: Conv1d(512, D_vae, 1)
      -> reparameterize -> z [B, D_vae, T']
      -> dec_proj: MLP via 1x1 Convs       # trainable, nonlinear
      -> MimiDecoder                        # pretrained, optionally frozen
      -> Audio_hat [B, 1, T] @ 24kHz
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ARFriendlyVAE(nn.Module):
    """VAE bottleneck between Mimi encoder and decoder."""

    def __init__(
        self,
        mimi_encoder: nn.Module,
        mimi_decoder: nn.Module,
        latent_dim: int = 32,
        encoder_dim: int = 512,
        dec_hidden_dim: int = 256,
        freeze_encoder: bool = True,
        freeze_decoder: bool = False,
    ):
        super().__init__()
        self.mimi_encoder = mimi_encoder
        self.mimi_decoder = mimi_decoder
        self.latent_dim = latent_dim
        self.encoder_dim = encoder_dim
        self.dec_hidden_dim = dec_hidden_dim

        # VAE bottleneck projections
        self.mu_proj = nn.Conv1d(encoder_dim, latent_dim, 1)
        self.logvar_proj = nn.Conv1d(encoder_dim, latent_dim, 1)

        # Nonlinear decoder projection (per-timestep MLP via 1x1 convs)
        self.dec_proj = nn.Sequential(
            nn.Conv1d(latent_dim, dec_hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(dec_hidden_dim, encoder_dim, 1),
        )

        # Initialize logvar bias to -2.0 so initial sigma is small
        nn.init.zeros_(self.logvar_proj.weight)
        nn.init.constant_(self.logvar_proj.bias, -2.0)

        if freeze_encoder:
            for p in self.mimi_encoder.parameters():
                p.requires_grad = False

        if freeze_decoder:
            for p in self.mimi_decoder.parameters():
                p.requires_grad = False

    def encode(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode audio to VAE latent space.

        Args:
            audio: [B, 1, T] at 24kHz

        Returns:
            (z, mu, logvar) each [B, D_vae, T']
        """
        # project=False: raw encoder output, bypasses RVQ projections
        h = self.mimi_encoder(audio, project=False)  # [B, 512, T']
        mu = self.mu_proj(h)  # [B, D_vae, T']
        logvar = self.logvar_proj(h)  # [B, D_vae, T']
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, length: int | None = None) -> torch.Tensor:
        """
        Decode VAE latent to audio.

        Args:
            z: [B, D_vae, T']
            length: optional output audio length to trim to

        Returns:
            audio_hat [B, 1, T]
        """
        h_hat = self.dec_proj(z)  # [B, 512, T']
        audio_hat = self.mimi_decoder(h_hat)  # [B, 1, T]
        if length is not None:
            audio_hat = audio_hat[..., :length]
        return audio_hat

    def forward(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Full forward pass: encode + decode.

        Args:
            audio: [B, 1, T] at 24kHz

        Returns:
            dict with keys: audio_hat, z, mu, logvar
        """
        length = audio.shape[-1]
        z, mu, logvar = self.encode(audio)
        audio_hat = self.decode(z, length=length)
        return {
            "audio_hat": audio_hat,
            "z": z,
            "mu": mu,
            "logvar": logvar,
        }

    @torch.no_grad()
    def extract_latents(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract deterministic latents (mu) for inference.

        Args:
            audio: [B, 1, T] at 24kHz

        Returns:
            mu [B, D_vae, T']
        """
        h = self.mimi_encoder(audio, project=False)
        return self.mu_proj(h)

    def bottleneck_parameters(self):
        """Parameters of the trainable bottleneck (mu_proj, logvar_proj, dec_proj)."""
        yield from self.mu_proj.parameters()
        yield from self.logvar_proj.parameters()
        yield from self.dec_proj.parameters()

    def decoder_parameters(self):
        """Parameters of the Mimi decoder (if unfrozen)."""
        yield from self.mimi_decoder.parameters()

    def trainable_parameters(self):
        """All parameters that require grad."""
        for p in self.parameters():
            if p.requires_grad:
                yield p
