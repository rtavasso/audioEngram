"""
WavLM semantic distillation loss for VAE latents.

Teacher:
  frozen WavLM features from ground-truth audio.

Student:
  a train-only linear projection that maps VAE latents (D_latent)
  to WavLM feature dim (typically 768 for wavlm-base-plus).

Loss:
  1 - cosine_similarity(projected_latent, wavlm_teacher_feature)

WavLM expects 16kHz input, so we resample from 24kHz.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torchaudio.functional as AF

logger = logging.getLogger("phase0")


class WavLMDistillation(nn.Module):
    """WavLM-based semantic distillation loss."""

    def __init__(
        self,
        latent_dim: int,
        model_name: str = "microsoft/wavlm-base-plus",
        layer: int = 7,
        source_sr: int = 24000,
        target_sr: int = 16000,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        from transformers import WavLMModel

        self.wavlm = WavLMModel.from_pretrained(model_name)
        self.wavlm.eval()
        for p in self.wavlm.parameters():
            p.requires_grad = False

        self.layer = layer
        self.source_sr = source_sr
        self.target_sr = target_sr
        self._device = torch.device(device)
        self.wavlm.to(self._device)
        self.wavlm_dim = int(self.wavlm.config.hidden_size)

        # Train-only projection head: latent_dim -> wavlm_dim.
        # No bias to discourage a degenerate bias-only solution when latents collapse.
        self.latent_proj = nn.Linear(int(latent_dim), self.wavlm_dim, bias=False)
        self.latent_proj.to(self._device)

        # Pre-convert to fp16 on GPU to save memory (~180MB vs ~360MB)
        if self._device.type == "cuda":
            self.wavlm.half()

    @torch.no_grad()
    def _extract(self, audio_24k: torch.Tensor) -> torch.Tensor:
        """Extract WavLM hidden states from audio (no grad).

        Args:
            audio_24k: [B, T] at 24kHz

        Returns:
            Hidden states [B, T', D] from the specified layer.
        """
        audio_16k = AF.resample(audio_24k, self.source_sr, self.target_sr)
        audio_16k = audio_16k.to(device=self._device, dtype=self.wavlm.dtype)

        out = self.wavlm(audio_16k, output_hidden_states=True)
        return out.hidden_states[self.layer]

    def forward(self, audio_24k: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """
        Compute WavLM distillation loss.

        Args:
            audio_24k: [B, 1, T] or [B, T] original audio at 24kHz
            latents: [B, D_latent, T_latent] (e.g. posterior mean `mu`)

        Returns:
            Scalar loss = mean(1 - cos(projected_latent, wavlm_teacher)).
        """
        x = audio_24k.squeeze(1) if audio_24k.dim() == 3 else audio_24k
        z = latents.to(self._device)

        # Teacher features: [B, T_w, D_w] (frozen).
        feat_teacher = self._extract(x)  # no grad

        # Match teacher temporal resolution to latent sequence length.
        # Interpolate over time on [B, D_w, T_w] -> [B, D_w, T_z].
        t_z = z.shape[-1]
        feat_teacher_t = feat_teacher.transpose(1, 2).float()
        feat_teacher_t = nn.functional.interpolate(
            feat_teacher_t,
            size=t_z,
            mode="linear",
            align_corners=False,
        )
        feat_teacher = feat_teacher_t.transpose(1, 2)  # [B, T_z, D_w]

        # Student projection: [B, D_z, T_z] -> [B, T_z, D_w]
        z_bt = z.transpose(1, 2).float()
        feat_student = self.latent_proj(z_bt)

        # Cosine distillation objective (maximize cosine similarity).
        cos = nn.functional.cosine_similarity(feat_student, feat_teacher, dim=-1)
        return (1.0 - cos).mean()
