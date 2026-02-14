"""
Multi-Scale STFT Discriminator for VAE-GAN training.

Follows EnCodec/Mimi MS-STFTD design:
- N sub-discriminators at different STFT resolutions
- Each operates on complex STFT (real+imag stacked as 2 channels)
- Weight normalization on all Conv2d layers
- Returns logits + intermediate features for feature matching loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class STFTDiscriminator(nn.Module):
    """Single-scale STFT discriminator with weight-normalized convolutions."""

    def __init__(self, n_fft: int = 1024, hop_length: int | None = None, channels: int = 32):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4

        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(2, channels, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels * 2, (3, 9), stride=(2, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels * 2, channels * 4, (3, 9), stride=(2, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels * 4, channels * 8, (3, 9), stride=(2, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels * 8, channels * 8, (3, 3), padding=(1, 1))),
            weight_norm(nn.Conv2d(channels * 8, 1, (3, 3), padding=(1, 1))),
        ])

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            audio: [B, 1, T] or [B, T]

        Returns:
            (logits, features) where logits is [B, 1, F', T'] and
            features is a list of intermediate activations.
        """
        x = audio.squeeze(1) if audio.dim() == 3 else audio
        window = torch.hann_window(self.n_fft, device=x.device, dtype=x.dtype)
        spec = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=window, return_complex=True,
        )
        # Stack real and imaginary as 2 channels: [B, 2, F, T']
        x = torch.stack([spec.real, spec.imag], dim=1)

        features = []
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i < len(self.convs) - 1:
                x = F.leaky_relu(x, 0.2)
                features.append(x)
        return x, features


class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-scale STFT discriminator with N sub-discriminators."""

    def __init__(
        self,
        n_ffts: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048),
        channels: int = 32,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(n_fft=n, hop_length=n // 4, channels=channels)
            for n in n_ffts
        ])

    def forward(
        self, audio: torch.Tensor,
    ) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        """
        Args:
            audio: [B, 1, T]

        Returns:
            List of (logits, features) per scale.
        """
        return [d(audio) for d in self.discriminators]
