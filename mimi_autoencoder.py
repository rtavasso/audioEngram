"""
Mimi Autoencoder - Continuous latent audio codec extracted from Moshi.

This module provides the Mimi encoder/decoder operating on continuous
representations, bypassing the RVQ quantization layer.

Usage:
    from mimi_autoencoder import MimiAutoencoder, load_mimi_autoencoder

    # Load pretrained model
    autoencoder = load_mimi_autoencoder(device='cuda')

    # Encode audio to continuous latent
    latent = autoencoder.encode(wav)  # [B, 1, T] -> [B, 512, T']

    # Decode back to audio
    recon = autoencoder.decode(latent)  # [B, 512, T'] -> [B, 1, T]

Specs:
    - Input: 24 kHz mono audio
    - Latent: 512-dim @ 12.5 Hz (1920x temporal compression)
    - Causal with ~80ms latency
"""

import sys
from pathlib import Path

# Handle path conflict: local moshi/ directory shadows installed package
# Add the correct moshi package path before other imports
_moshi_pkg_path = Path(__file__).parent / "moshi" / "moshi"
if _moshi_pkg_path.exists():
    sys.path.insert(0, str(_moshi_pkg_path))

import torch
import torch.nn as nn
from typing import Optional


class MimiEncoder(nn.Module):
    """Encoder that projects audio to continuous latent space."""

    def __init__(self, mimi):
        super().__init__()
        self.encoder = mimi.encoder
        self.encoder_transformer = mimi.encoder_transformer
        self.downsample = getattr(mimi, 'downsample', None)
        # Store reference for frame rate conversion and quantizer projections
        self._mimi = mimi
        self.frame_size = mimi.frame_size

        # Extract quantizer projections for proper continuous path
        # The quantizer has input_proj (512->256) and output_proj (256->512)
        # For continuous latents, we apply: input_proj -> output_proj (skip VQ)
        self.rvq_first = mimi.quantizer.rvq_first
        self.rvq_rest = mimi.quantizer.rvq_rest

    def _to_framerate(self, x: torch.Tensor) -> torch.Tensor:
        return self._mimi._to_framerate(x)

    def forward(self, x: torch.Tensor, project: bool = True) -> torch.Tensor:
        """
        Encode audio waveform to continuous latent.

        Args:
            x: Audio tensor of shape [B, 1, T] at 24kHz
            project: If True, apply quantizer's input/output projections
                     to match the decoder's expected distribution.

        Returns:
            Continuous latent of shape [B, 512, T'] at 12.5Hz
            where T' = ceil(T / 1920)
        """
        from moshi.modules.conv import pad_for_conv1d
        x = pad_for_conv1d(x, self.frame_size, self.frame_size)
        emb = self.encoder(x)
        if self.encoder_transformer is not None:
            (emb,) = self.encoder_transformer(emb)
        emb = self._to_framerate(emb)

        if project:
            # Apply quantizer projections without actual quantization
            # This produces latents in the decoder's expected space
            emb_first = self.rvq_first.output_proj(self.rvq_first.input_proj(emb))
            emb_rest = self.rvq_rest.output_proj(self.rvq_rest.input_proj(emb))
            emb = emb_first + emb_rest

        return emb


class MimiDecoder(nn.Module):
    """Decoder that reconstructs audio from continuous latent."""

    def __init__(self, mimi):
        super().__init__()
        self.decoder = mimi.decoder
        self.decoder_transformer = mimi.decoder_transformer
        self.upsample = getattr(mimi, 'upsample', None)
        # Store the bound method for frame rate conversion
        self._mimi = mimi

    def _to_encoder_framerate(self, x: torch.Tensor) -> torch.Tensor:
        return self._mimi._to_encoder_framerate(x)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Decode continuous latent to audio waveform.

        Args:
            emb: Continuous latent of shape [B, 512, T'] at 12.5Hz

        Returns:
            Audio tensor of shape [B, 1, T] at 24kHz
            Note: Output may have padding; trim to desired length.
        """
        emb = self._to_encoder_framerate(emb)
        if self.decoder_transformer is not None:
            (emb,) = self.decoder_transformer(emb)
        out = self.decoder(emb)
        return out


class MimiAutoencoder(nn.Module):
    """
    Mimi as a continuous autoencoder without quantization.

    Extracts the pretrained encoder and decoder from Mimi, bypassing
    the RVQ quantization layer to work directly with continuous latents.

    Attributes:
        encoder: MimiEncoder module
        decoder: MimiDecoder module
        sample_rate: Audio sample rate (24000 Hz)
        frame_rate: Latent frame rate (12.5 Hz)
        latent_dim: Latent dimension (512)
        frame_size: Samples per latent frame (1920)
    """

    def __init__(self, mimi):
        """
        Initialize from a loaded Mimi model.

        Args:
            mimi: A MimiModel instance from moshi.models.loaders.get_mimi()
        """
        super().__init__()
        self.encoder = MimiEncoder(mimi)
        self.decoder = MimiDecoder(mimi)
        self.sample_rate = mimi.sample_rate      # 24000
        self.frame_rate = mimi.frame_rate        # 12.5
        self.latent_dim = mimi.dimension         # 512
        self.frame_size = mimi.frame_size        # 1920

    def encode(self, x: torch.Tensor, project: bool = True) -> torch.Tensor:
        """
        Encode audio to continuous latent.

        Args:
            x: Audio tensor [B, 1, T] at 24kHz
            project: If True (default), apply quantizer projections to produce
                     latents compatible with the decoder. If False, return raw
                     encoder output (useful for feature extraction).

        Returns:
            Latent tensor [B, 512, T'] at 12.5Hz
        """
        return self.encoder(x, project=project)

    def decode(self, emb: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """
        Decode continuous latent to audio.

        Args:
            emb: Latent tensor [B, 512, T'] at 12.5Hz
            length: Optional output length to trim to

        Returns:
            Audio tensor [B, 1, T] at 24kHz
        """
        out = self.decoder(emb)
        if length is not None:
            out = out[..., :length]
        return out

    def forward(self, x: torch.Tensor, project: bool = True) -> torch.Tensor:
        """
        Encode and decode (reconstruction).

        Args:
            x: Audio tensor [B, 1, T] at 24kHz
            project: If True (default), apply quantizer projections for
                     proper reconstruction. If False, use raw encoder output.

        Returns:
            Reconstructed audio [B, 1, T] at 24kHz (trimmed to input length)
        """
        length = x.shape[-1]
        emb = self.encode(x, project=project)
        return self.decode(emb, length=length)

    def get_latent_length(self, audio_length: int) -> int:
        """Calculate latent sequence length for given audio length."""
        import math
        return math.ceil(audio_length / self.frame_size)

    def get_audio_length(self, latent_length: int) -> int:
        """Calculate audio length for given latent sequence length."""
        return latent_length * self.frame_size


def load_mimi_autoencoder(
    checkpoint_path: Optional[str] = None,
    device: str = 'cpu',
    dtype: Optional[torch.dtype] = None,
) -> MimiAutoencoder:
    """
    Load pretrained Mimi autoencoder.

    Args:
        checkpoint_path: Path to checkpoint. If None, downloads from HuggingFace.
        device: Device to load model on ('cpu', 'cuda', etc.)
        dtype: Optional dtype (e.g., torch.bfloat16)

    Returns:
        MimiAutoencoder instance

    Example:
        >>> autoencoder = load_mimi_autoencoder(device='cuda')
        >>> wav = torch.randn(1, 1, 24000 * 5)  # 5 seconds
        >>> latent = autoencoder.encode(wav)
        >>> recon = autoencoder.decode(latent, length=wav.shape[-1])
    """
    from moshi.models import loaders

    if checkpoint_path is None:
        from huggingface_hub import hf_hub_download
        checkpoint_path = hf_hub_download(
            "kyutai/moshiko-pytorch-bf16",
            "tokenizer-e351c8d8-checkpoint125.safetensors"
        )

    mimi = loaders.get_mimi(checkpoint_path, device=device)

    if dtype is not None:
        mimi = mimi.to(dtype)

    return MimiAutoencoder(mimi)


if __name__ == "__main__":
    # Quick test
    print("Loading Mimi autoencoder...")
    autoencoder = load_mimi_autoencoder(device='cpu')

    print(f"Sample rate: {autoencoder.sample_rate} Hz")
    print(f"Frame rate: {autoencoder.frame_rate} Hz")
    print(f"Latent dim: {autoencoder.latent_dim}")
    print(f"Frame size: {autoencoder.frame_size} samples")
    print(f"Compression: {autoencoder.frame_size}x temporal")

    # Test encode/decode
    duration = 2.0  # seconds
    wav = torch.randn(1, 1, int(autoencoder.sample_rate * duration))
    print(f"\nInput shape: {wav.shape}")

    latent = autoencoder.encode(wav)
    print(f"Latent shape: {latent.shape}")

    recon = autoencoder.decode(latent, length=wav.shape[-1])
    print(f"Output shape: {recon.shape}")

    print("\nDone!")
