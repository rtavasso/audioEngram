"""
PocketMimiVAE: VAE wrapping pocket-tts MimiModel with a 32D bottleneck.

Architecture:
    Audio [B, 1, T] @ 24kHz
      -> MimiModel.encode_to_latent()   # pretrained encoder (frozen)
      -> h [B, 512, T'] @ 12.5Hz
      -> mu_proj: Conv1d(512, D_vae, 1)
      -> logvar_proj: Conv1d(512, D_vae, 1)
      -> reparameterize -> z [B, D_vae, T']
      -> dec_proj: MLP via 1x1 Convs [D_vae -> hidden -> 512]
      -> MimiModel.decode_from_latent() # pretrained decoder (fine-tuned)
      -> Audio_hat [B, 1, T'] @ 24kHz

Key difference from ARFriendlyVAE: pocket-tts MimiModel uses stateful
streaming modules. decode_from_latent() requires a mimi_state dict from
init_states(). Fresh states must be created per forward pass.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger("phase0")


class PocketMimiVAE(nn.Module):
    """VAE bottleneck between pocket-tts Mimi encoder and decoder."""

    def __init__(
        self,
        mimi,  # pocket_tts MimiModel
        latent_dim: int = 32,
        dec_hidden_dim: int = 256,
        freeze_encoder: bool = True,
        freeze_decoder: bool = False,
    ):
        super().__init__()
        self.mimi = mimi
        self.latent_dim = latent_dim
        self.encoder_dim = mimi.dimension  # 512
        self.dec_hidden_dim = dec_hidden_dim

        # VAE bottleneck projections
        self.mu_proj = nn.Conv1d(self.encoder_dim, latent_dim, 1)
        self.logvar_proj = nn.Conv1d(self.encoder_dim, latent_dim, 1)

        # Nonlinear decoder projection (per-timestep MLP via 1x1 convs)
        self.dec_proj = nn.Sequential(
            nn.Conv1d(latent_dim, dec_hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(dec_hidden_dim, self.encoder_dim, 1),
        )

        # Initialize logvar bias for small initial sigma
        nn.init.zeros_(self.logvar_proj.weight)
        nn.init.constant_(self.logvar_proj.bias, -2.0)

        # Freeze encoder components
        if freeze_encoder:
            for p in self.mimi.encoder.parameters():
                p.requires_grad = False
            for p in self.mimi.encoder_transformer.parameters():
                p.requires_grad = False
            if hasattr(self.mimi, 'downsample'):
                for p in self.mimi.downsample.parameters():
                    p.requires_grad = False
            for p in self.mimi.quantizer.parameters():
                p.requires_grad = False

        if freeze_decoder:
            for p in self.mimi.decoder.parameters():
                p.requires_grad = False
            for p in self.mimi.decoder_transformer.parameters():
                p.requires_grad = False
            if hasattr(self.mimi, 'upsample'):
                for p in self.mimi.upsample.parameters():
                    p.requires_grad = False

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def _init_mimi_state(
        self,
        batch_size: int,
        sequence_length: int = 512,
        device: torch.device | None = None,
    ) -> dict:
        """Create fresh streaming state for mimi, with tensors on the right device."""
        from pocket_tts.modules.stateful_module import init_states

        state = init_states(self.mimi, batch_size=batch_size, sequence_length=sequence_length)

        if device is None:
            device = next(self.mimi.decoder.parameters()).device
        for module_name, module_state in state.items():
            for key, val in module_state.items():
                if isinstance(val, torch.Tensor):
                    module_state[key] = val.to(device)

        return state

    def _encode_single(self, audio_single: torch.Tensor) -> torch.Tensor:
        """Encode a single audio sample through Mimi encoder.

        Args:
            audio_single: [1, 1, T] at 24kHz

        Returns:
            h: [1, 512, T'] encoder output
        """
        # encode_to_latent(model_state=None) builds CPU streaming buffers in
        # pocket-tts. For CUDA inputs we mirror the same logic but pass a
        # device-aligned state dict.
        from pocket_tts.modules.conv import pad_for_conv1d
        with torch.autocast(device_type=audio_single.device.type, enabled=False):
            audio_single = audio_single.float()
            x = pad_for_conv1d(audio_single, self.mimi.frame_size, self.mimi.frame_size)
            seq_len = max(8, int(x.shape[-1] // self.mimi.frame_size) + 8)
            mimi_state = self._init_mimi_state(
                1,
                sequence_length=seq_len,
                device=audio_single.device,
            )
            h = self.mimi.encoder(x, model_state=mimi_state)
            (h,) = self.mimi.encoder_transformer(h, model_state=None)
            if self.mimi.encoder_frame_rate != self.mimi.frame_rate:
                h = self.mimi.downsample(h, model_state=mimi_state)
        return h

    def encode(
        self,
        audio: torch.Tensor,
        return_h: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode audio to VAE latent space.

        Args:
            audio: [B, 1, T] at 24kHz

        Returns:
            (z, mu, logvar) each [B, D_vae, T']
        """
        # Process each sample individually (Mimi streaming state + RoPE
        # offset broadcasting requires B=1 for non-streaming mode)
        hs = [self._encode_single(audio[i:i+1]) for i in range(audio.shape[0])]
        h = torch.cat(hs, dim=0)  # [B, 512, T']

        mu = self.mu_proj(h)
        logvar = self.logvar_proj(h)
        z = self._reparameterize(mu, logvar)
        if return_h:
            return z, mu, logvar, h
        return z, mu, logvar

    def _decode_single(self, h_hat_single: torch.Tensor) -> torch.Tensor:
        """Decode a single latent through Mimi decoder.

        Args:
            h_hat_single: [1, 512, T']

        Returns:
            audio_hat: [1, 1, T]
        """
        # Run decoder outside autocast to avoid dtype mismatches with
        # streaming state (KV cache initialized as float32).
        with torch.autocast(device_type=h_hat_single.device.type, enabled=False):
            h_hat_single = h_hat_single.float()
            # Keep cache length aligned with current latent length.
            seq_len = max(8, int(h_hat_single.shape[-1]) + 8)
            mimi_state = self._init_mimi_state(
                1,
                sequence_length=seq_len,
                device=h_hat_single.device,
            )
            return self.mimi.decode_from_latent(h_hat_single, mimi_state)

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

        # Process each sample individually (RoPE offset broadcasting requires B=1)
        outs = [self._decode_single(h_hat[i:i+1]) for i in range(h_hat.shape[0])]
        audio_hat = torch.cat(outs, dim=0)

        if length is not None:
            audio_hat = audio_hat[..., :length]
        return audio_hat

    def reconstruct_with_pretrained_mimi(self, audio: torch.Tensor, length: int | None = None) -> torch.Tensor:
        """
        Reconstruct audio through the pretrained Mimi path only (no VAE bottleneck).

        Args:
            audio: [B, 1, T] at 24kHz
            length: optional output audio length to trim to

        Returns:
            audio_hat [B, 1, T]
        """
        hs = [self._encode_single(audio[i:i+1]) for i in range(audio.shape[0])]
        outs = [self._decode_single(h) for h in hs]
        audio_hat = torch.cat(outs, dim=0)
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
        z, mu, logvar, h = self.encode(audio, return_h=True)
        audio_hat = self.decode(z, length=length)
        return {
            "audio_hat": audio_hat,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "h_enc": h,
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
        hs = [self._encode_single(audio[i:i+1]) for i in range(audio.shape[0])]
        h = torch.cat(hs, dim=0)
        return self.mu_proj(h)

    def bottleneck_parameters(self):
        """Parameters of the trainable bottleneck (mu_proj, logvar_proj, dec_proj)."""
        yield from self.mu_proj.parameters()
        yield from self.logvar_proj.parameters()
        yield from self.dec_proj.parameters()

    def decoder_parameters(self):
        """Parameters of the Mimi decoder (if unfrozen)."""
        yield from self.mimi.decoder.parameters()
        yield from self.mimi.decoder_transformer.parameters()
        if hasattr(self.mimi, 'upsample'):
            yield from self.mimi.upsample.parameters()

    def trainable_parameters(self):
        """All parameters that require grad."""
        for p in self.parameters():
            if p.requires_grad:
                yield p


def build_pocket_mimi_vae(
    *,
    latent_dim: int = 32,
    dec_hidden_dim: int = 256,
    freeze_encoder: bool = True,
    freeze_decoder: bool = False,
    device: str | torch.device = "cpu",
    seanet_ratios: list[int] | None = None,
    transformer_num_layers: int | None = None,
    transformer_context: int | None = None,
    allow_partial_pretrained_load: bool = False,
    load_pretrained_mimi: bool = True,
) -> PocketMimiVAE:
    """
    Build PocketMimiVAE from pocket-tts config + pretrained weights.

    Loads the b6369a24 config, builds MimiModel, loads pretrained weights,
    and wraps in PocketMimiVAE.

    By default this function requires a full strict weight load from the
    pretrained checkpoint. If architecture overrides are requested, set
    allow_partial_pretrained_load=True to opt into partial loading.
    """
    from pocket_tts.modules import mimi_transformer
    from pocket_tts.modules.dummy_quantizer import DummyQuantizer
    from pocket_tts.modules.seanet import SEANetDecoder, SEANetEncoder
    from pocket_tts.models.mimi import MimiModel
    from pocket_tts.utils.config import load_config
    from pocket_tts.utils.utils import download_if_necessary
    from pocket_tts.utils.weights_loading import get_mimi_state_dict

    # Load pocket-tts config
    config_dir = Path(__file__).parent.parent.parent / "pocket-tts" / "pocket_tts" / "config"
    config = load_config(config_dir / "b6369a24.yaml")
    mimi_config = config.mimi.model_dump()

    # Apply architecture overrides
    arch_overridden = False
    if seanet_ratios is not None:
        logger.info(f"Overriding seanet ratios: {mimi_config['seanet']['ratios']} -> {seanet_ratios}")
        mimi_config["seanet"]["ratios"] = seanet_ratios
        arch_overridden = True
    if transformer_num_layers is not None:
        logger.info(f"Overriding transformer num_layers: {mimi_config['transformer']['num_layers']} -> {transformer_num_layers}")
        mimi_config["transformer"]["num_layers"] = transformer_num_layers
        arch_overridden = True
    if transformer_context is not None:
        logger.info(f"Overriding transformer context: {mimi_config['transformer']['context']} -> {transformer_context}")
        mimi_config["transformer"]["context"] = transformer_context
        arch_overridden = True
    if arch_overridden and not allow_partial_pretrained_load:
        raise ValueError(
            "Architecture overrides are enabled but allow_partial_pretrained_load=False. "
            "Disable overrides to load pretrained Mimi weights strictly."
        )

    # Build Mimi components
    encoder = SEANetEncoder(**mimi_config["seanet"])
    decoder = SEANetDecoder(**mimi_config["seanet"])
    encoder_transformer = mimi_transformer.ProjectedTransformer(**mimi_config["transformer"])
    decoder_transformer = mimi_transformer.ProjectedTransformer(**mimi_config["transformer"])
    quantizer = DummyQuantizer(**mimi_config["quantizer"])

    mimi = MimiModel(
        encoder,
        decoder,
        quantizer,
        channels=mimi_config["channels"],
        sample_rate=mimi_config["sample_rate"],
        frame_rate=mimi_config["frame_rate"],
        encoder_frame_rate=mimi_config["sample_rate"] / encoder.hop_length,
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    )

    # Load pretrained weights. The pocket-tts config stores mimi weights
    # either as a separate mimi.weights_path or bundled in the full TTS
    # weights_path (with "mimi." prefix).
    def _load_filtered(state_dict):
        """Load weights, filtering out shape mismatches when arch is overridden."""
        if not arch_overridden:
            mimi.load_state_dict(state_dict, strict=True)
            return
        if not allow_partial_pretrained_load:
            raise RuntimeError(
                "Partial pretrained load requested while allow_partial_pretrained_load=False."
            )
        model_state = mimi.state_dict()
        compatible = {}
        skipped_shape = []
        for k, v in state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                compatible[k] = v
            elif k in model_state:
                skipped_shape.append(k)
        result = mimi.load_state_dict(compatible, strict=False)
        logger.info(
            f"Partial weight load — loaded: {len(compatible)}, "
            f"shape mismatch: {len(skipped_shape)}, "
            f"missing: {len(result.missing_keys)}, "
            f"unexpected: {len(result.unexpected_keys)}"
        )

    loaded = False
    if load_pretrained_mimi:
        if config.mimi.weights_path is not None:
            logger.info(f"Loading Mimi weights from {config.mimi.weights_path}")
            weights_file = download_if_necessary(config.mimi.weights_path)
            mimi_state = get_mimi_state_dict(weights_file)
            _load_filtered(mimi_state)
            loaded = True
        elif config.weights_path is not None:
            # Full TTS safetensors — extract mimi.* keys
            import safetensors
            logger.info(f"Loading Mimi weights from full TTS model: {config.weights_path}")
            try:
                weights_file = download_if_necessary(config.weights_path)
            except Exception:
                if config.weights_path_without_voice_cloning is not None:
                    weights_file = download_if_necessary(config.weights_path_without_voice_cloning)
                else:
                    raise
            mimi_state = {}
            with safetensors.safe_open(weights_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("mimi."):
                        new_key = key.removeprefix("mimi.")
                        mimi_state[new_key] = f.get_tensor(key)
            _load_filtered(mimi_state)
            loaded = True

    if loaded:
        logger.info("Mimi weights loaded successfully")
    elif not load_pretrained_mimi:
        logger.info("Skipping pretrained Mimi load (expected to load from checkpoint)")
    else:
        logger.warning("No Mimi weights path in config — using random initialization")

    vae = PocketMimiVAE(
        mimi=mimi,
        latent_dim=latent_dim,
        dec_hidden_dim=dec_hidden_dim,
        freeze_encoder=freeze_encoder,
        freeze_decoder=freeze_decoder,
    )

    return vae.to(device)
