"""
CALM model: backbone transformer + short-context transformer + consistency head.

Architecture (from Rouard et al. 2025):
  - CausalBackboneTransformer: processes noised latent sequence with causal mask
  - ShortContextTransformer: attends to K recent clean frames per position
  - ConsistencyHead: TrigFlow-parameterized MLP for one-step denoising
  - CALM: top-level module wrapping all three + EMA teacher head
"""

from __future__ import annotations

import copy
import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger("phase0")


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, S, D] -> [B, S, D] with positional encoding added."""
        return x + self.pe[:, :x.size(1)]


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding (for consistency head)
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embedding for continuous timestep t in [0, pi/2]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] -> [B, dim]."""
        t = t.float()
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)  # [B, half]
        return torch.cat([args.sin(), args.cos()], dim=-1)  # [B, dim]


# ---------------------------------------------------------------------------
# Causal Backbone Transformer
# ---------------------------------------------------------------------------

class CausalBackboneTransformer(nn.Module):
    """
    Standard causal (decoder-only) transformer on noised latent sequences.

    Input:  x_noised [B, S, D=latent_dim]
    Output: z_long   [B, S, d_model]  (per-position conditioning)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, latent_dim] noised latent sequence
        Returns:
            z_long: [B, S, d_model]
        """
        S = x.size(1)
        h = self.input_proj(x)
        h = self.pos_enc(h)

        # Causal mask: upper-triangular True = masked
        mask = nn.Transformer.generate_square_subsequent_mask(
            S, device=x.device, dtype=x.dtype
        )
        h = self.transformer(h, mask=mask, is_causal=True)
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# Short-Context Transformer
# ---------------------------------------------------------------------------

class ShortContextTransformer(nn.Module):
    """
    Small transformer that attends to K recent *clean* frames per position.

    For efficiency: batch all positions into [B*S, K, D], run transformer,
    take the last output token per window.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        short_ctx_k: int = 10,
    ):
        super().__init__()
        self.short_ctx_k = short_ctx_k
        self.input_proj = nn.Linear(latent_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=short_ctx_k + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x_clean: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x_clean: [B, S, D] full clean latent sequence
            positions: optional int tensor of target positions (default: 1..S-1)

        Returns:
            z_short: [B, P, d_model] where P = len(positions) or S-1
        """
        B, S, D = x_clean.shape
        K = self.short_ctx_k

        if positions is None:
            # Default: predict for positions 1..S-1 (each uses context from preceding frames)
            positions = torch.arange(1, S, device=x_clean.device)
        P = len(positions)

        # Build windows: for each target position s, take x_clean[max(0,s-K):s]
        windows = []
        for s in positions:
            s = int(s)
            start = max(0, s - K)
            window = x_clean[:, start:s]  # [B, min(K, s), D]
            # Pad to K if shorter
            pad_len = K - window.size(1)
            if pad_len > 0:
                pad = torch.zeros(B, pad_len, D, device=x_clean.device, dtype=x_clean.dtype)
                window = torch.cat([pad, window], dim=1)  # [B, K, D]
            windows.append(window)

        # Stack: [B, P, K, D] -> reshape to [B*P, K, D]
        windows = torch.stack(windows, dim=1)  # [B, P, K, D]
        windows_flat = windows.reshape(B * P, K, D)

        h = self.input_proj(windows_flat)  # [B*P, K, d_model]
        h = self.pos_enc(h)

        # Causal mask for the short context window
        mask = nn.Transformer.generate_square_subsequent_mask(
            K, device=x_clean.device, dtype=x_clean.dtype
        )
        h = self.transformer(h, mask=mask, is_causal=True)

        # Take last token output from each window
        h_last = h[:, -1]  # [B*P, d_model]
        z_short = self.output_proj(h_last)  # [B*P, d_model]
        return z_short.reshape(B, P, -1)  # [B, P, d_model]

    def forward_single(self, recent_clean: torch.Tensor) -> torch.Tensor:
        """
        Inference helper: process a single position from pre-sliced context.

        Args:
            recent_clean: [B, K', D] where K' <= K (last K clean frames)

        Returns:
            z_short: [B, 1, d_model]
        """
        B, Kp, D = recent_clean.shape
        K = self.short_ctx_k

        # Pad to K if needed
        if Kp < K:
            pad = torch.zeros(B, K - Kp, D, device=recent_clean.device, dtype=recent_clean.dtype)
            recent_clean = torch.cat([pad, recent_clean], dim=1)

        h = self.input_proj(recent_clean)  # [B, K, d_model]
        h = self.pos_enc(h)

        mask = nn.Transformer.generate_square_subsequent_mask(
            K, device=recent_clean.device, dtype=recent_clean.dtype
        )
        h = self.transformer(h, mask=mask, is_causal=True)
        h_last = h[:, -1:]  # [B, 1, d_model]
        return self.output_proj(h_last)  # [B, 1, d_model]


# ---------------------------------------------------------------------------
# Consistency Head (TrigFlow MLP)
# ---------------------------------------------------------------------------

class ConsistencyHead(nn.Module):
    """
    TrigFlow-parameterized consistency model head.

    f_phi(x_t, t, Z) = cos(t)*x_t + sin(t)*F_phi(x_t, t, Z)

    Boundary condition: f(x, 0) = x (identity at t=0).
    At t=pi/2: f(x_t, pi/2) = F_phi (pure network prediction).
    """

    def __init__(
        self,
        latent_dim: int = 32,
        d_cond: int = 256,
        hidden_dim: int = 512,
        n_layers: int = 4,
        t_embed_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Timestep embedding
        self.t_embed = SinusoidalTimestepEmbedding(t_embed_dim)
        self.t_proj = nn.Linear(t_embed_dim, hidden_dim)

        # Conditioning projection
        self.cond_proj = nn.Linear(d_cond, hidden_dim)

        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # Hidden layers
        layers = []
        for _ in range(n_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        self.backbone = nn.Sequential(*layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute raw network output F_phi (before TrigFlow skip connection).

        Args:
            x_t:    [N, latent_dim] noised target
            t:      [N] timestep in [0, pi/2]
            z_cond: [N, d_cond] conditioning vector

        Returns:
            F_phi: [N, latent_dim]
        """
        t_emb = self.t_embed(t)        # [N, t_embed_dim]
        t_h = self.t_proj(t_emb)       # [N, hidden_dim]
        z_h = self.cond_proj(z_cond)    # [N, hidden_dim]
        x_h = self.input_proj(x_t)     # [N, hidden_dim]

        h = x_h + t_h + z_h
        h = self.backbone(h)
        return self.output_proj(h)      # [N, latent_dim]

    def predict_clean(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full TrigFlow prediction: f(x_t, t, Z) = cos(t)*x_t + sin(t)*F(x_t, t, Z).

        Args:
            x_t:    [N, latent_dim]
            t:      [N]
            z_cond: [N, d_cond]

        Returns:
            x_hat: [N, latent_dim] predicted clean sample
        """
        F_phi = self.forward(x_t, t, z_cond)
        cos_t = torch.cos(t).unsqueeze(-1)  # [N, 1]
        sin_t = torch.sin(t).unsqueeze(-1)  # [N, 1]
        return cos_t * x_t + sin_t * F_phi


# ---------------------------------------------------------------------------
# CALM: top-level module
# ---------------------------------------------------------------------------

class CALM(nn.Module):
    """
    Continuous Audio Language Model.

    Wraps backbone transformer, short-context transformer, consistency head,
    and EMA teacher. Provides training forward and autoregressive generation.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        d_model: int = 256,
        n_backbone_layers: int = 4,
        n_short_ctx_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        short_ctx_k: int = 10,
        head_hidden_dim: int = 512,
        head_n_layers: int = 4,
        ema_decay: float = 0.999,
        normalize_latents: bool = True,
        norm_momentum: float = 0.01,
        norm_eps: float = 1e-5,
        use_noise_injection: bool = True,
        use_short_context: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.ema_decay = ema_decay
        self.normalize_latents = normalize_latents
        self.norm_momentum = norm_momentum
        self.norm_eps = norm_eps
        self.use_noise_injection = use_noise_injection
        self.use_short_context = use_short_context

        self.backbone = CausalBackboneTransformer(
            latent_dim=latent_dim,
            d_model=d_model,
            n_layers=n_backbone_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.short_ctx = ShortContextTransformer(
            latent_dim=latent_dim,
            d_model=d_model,
            n_layers=n_short_ctx_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            short_ctx_k=short_ctx_k,
        )

        self.head = ConsistencyHead(
            latent_dim=latent_dim,
            d_cond=d_model,
            hidden_dim=head_hidden_dim,
            n_layers=head_n_layers,
        )

        # EMA copy of head (teacher)
        self.head_ema = copy.deepcopy(self.head)
        for p in self.head_ema.parameters():
            p.requires_grad = False

        # Latent centering/normalization (paper: "Center and normalize")
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_var", torch.ones(latent_dim))
        self.register_buffer("latent_stats_initialized", torch.tensor(False))

        # Adaptive weight w_psi for consistency loss
        self.w_psi = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    @torch.no_grad()
    def _update_latent_stats(self, x_raw: torch.Tensor) -> None:
        if not self.normalize_latents:
            return
        # x_raw: [B, S, D]
        batch_mean = x_raw.float().mean(dim=(0, 1))
        batch_var = x_raw.float().var(dim=(0, 1), unbiased=False).clamp_min(self.norm_eps)

        if not bool(self.latent_stats_initialized.item()):
            self.latent_mean.copy_(batch_mean)
            self.latent_var.copy_(batch_var)
            self.latent_stats_initialized.fill_(True)
            return

        m = float(self.norm_momentum)
        self.latent_mean.mul_(1.0 - m).add_(batch_mean, alpha=m)
        self.latent_var.mul_(1.0 - m).add_(batch_var, alpha=m)

    def _normalize_latents(self, x_raw: torch.Tensor) -> torch.Tensor:
        if not self.normalize_latents:
            return x_raw
        mean = self.latent_mean.view(1, 1, -1).to(device=x_raw.device, dtype=x_raw.dtype)
        var = self.latent_var.view(1, 1, -1).to(device=x_raw.device, dtype=x_raw.dtype)
        return (x_raw - mean) / (var + self.norm_eps).sqrt()

    def _denormalize_latents(self, x_norm: torch.Tensor) -> torch.Tensor:
        if not self.normalize_latents:
            return x_norm
        mean = self.latent_mean.view(1, 1, -1).to(device=x_norm.device, dtype=x_norm.dtype)
        var = self.latent_var.view(1, 1, -1).to(device=x_norm.device, dtype=x_norm.dtype)
        return x_norm * (var + self.norm_eps).sqrt() + mean

    def compute_conditioning(
        self, x_clean: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute conditioning vectors Z for each target position.

        Args:
            x_clean: [B, S, D] clean latent sequence

        Returns:
            Z:         [B, S-1, d_model] combined conditioning
            x_targets: [B, S-1, D] target frames (positions 1..S-1)
        """
        B, S, D = x_clean.shape

        if self.training:
            self._update_latent_stats(x_clean)
        x_clean = self._normalize_latents(x_clean)

        # Noise injection for backbone input
        if self.use_noise_injection:
            k = torch.rand(B, S, 1, device=x_clean.device, dtype=x_clean.dtype)
            eps_noise = torch.randn_like(x_clean)
            x_noised = k.sqrt() * eps_noise + (1 - k).sqrt() * x_clean
        else:
            x_noised = x_clean

        # Backbone on full noised sequence, shifted: use positions 0..S-2 to predict 1..S-1
        z_long = self.backbone(x_noised[:, :-1])  # [B, S-1, d_model]

        # Short-context on clean sequence
        if self.use_short_context:
            z_short = self.short_ctx(x_clean)  # [B, S-1, d_model] (positions 1..S-1)
        else:
            z_short = torch.zeros_like(z_long)

        Z = z_long + z_short  # [B, S-1, d_model]
        x_targets = x_clean[:, 1:]  # [B, S-1, D]

        return Z, x_targets

    def consistency_loss(
        self,
        x_targets: torch.Tensor,
        Z: torch.Tensor,
        head_batch_mult: int = 4,
    ) -> tuple[torch.Tensor, float]:
        """
        Denoising prediction loss: train f(x_t, t, Z) to reconstruct x_clean.

        The student head predicts clean x from noised x_t via TrigFlow:
            f(x_t, t, Z) = cos(t)*x_t + sin(t)*F_phi(x_t, t, Z)
        Supervised directly against the clean target x_clean.

        The EMA head is maintained for generation quality but not used in the loss.

        Args:
            x_targets: [B, S-1, D] clean target frames (normalized)
            Z:         [B, S-1, d_cond] conditioning
            head_batch_mult: number of noise samples per target

        Returns:
            (loss, raw_mse) where raw_mse is the unweighted denoising MSE for logging.
        """
        B, P, D = x_targets.shape
        N = B * P
        d_cond = Z.size(-1)

        # Flatten
        x_flat = x_targets.reshape(N, D)
        z_flat = Z.reshape(N, d_cond)

        # Repeat for head batch multiplier
        M = head_batch_mult
        x_rep = x_flat.repeat(M, 1)       # [N*M, D]
        z_rep = z_flat.repeat(M, 1)       # [N*M, d_cond]
        NM = N * M

        # Sample t ~ U(0, pi/2), clamp away from 0 for numerical stability.
        # Keep t / trig ops in fp32 for stability under AMP.
        t = torch.rand(NM, device=x_rep.device, dtype=torch.float32) * (math.pi / 2 - 1e-4) + 1e-4
        eps = torch.randn(NM, D, device=x_rep.device, dtype=torch.float32)
        x_rep_f = x_rep.float()

        # Noised targets: x_t = cos(t)*x + sin(t)*eps
        cos_t = torch.cos(t).unsqueeze(-1)  # [NM, 1]
        sin_t = torch.sin(t).unsqueeze(-1)  # [NM, 1]
        x_t = (cos_t * x_rep_f + sin_t * eps).to(dtype=x_rep.dtype)

        # Student: full TrigFlow prediction f = cos(t)*x_t + sin(t)*F
        z_in = z_rep.to(dtype=x_t.dtype)
        F_student = self.head(x_t, t, z_in)           # [NM, D]
        f_student = cos_t * x_t.float() + sin_t * F_student.float()  # [NM, D] predicted clean

        # Target is the actual clean sample
        diff = f_student - x_rep_f                     # [NM, D]
        diff_sq = (diff ** 2).mean(dim=-1, keepdim=True)  # [NM, 1] per-sample MSE

        # Adaptive weight (fp32, clamped)
        with torch.autocast(device_type=x_rep.device.type, enabled=False):
            w = self.w_psi(t.unsqueeze(-1))            # [NM, 1]
        w = w.clamp(min=-10.0, max=10.0)

        # Loss: e^w * MSE - w  (adaptive weighting)
        loss = (torch.exp(w) * diff_sq - w).mean()

        raw_mse = float(diff_sq.mean().item())
        return loss, raw_mse

    @torch.no_grad()
    def update_ema(self) -> None:
        """Update EMA teacher head parameters."""
        for p_ema, p_student in zip(
            self.head_ema.parameters(), self.head.parameters()
        ):
            p_ema.data.mul_(self.ema_decay).add_(p_student.data, alpha=1.0 - self.ema_decay)

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        n_steps: int,
        temperature: float = 0.8,
    ) -> torch.Tensor:
        """
        Autoregressive generation via one-step consistency sampling.

        Args:
            prompt: [1, P, D] prefix latent sequence
            n_steps: number of new frames to generate
            temperature: noise temperature (0.8 = reduced variance)

        Returns:
            generated: [n_steps, D] newly generated frames
        """
        if n_steps <= 0:
            return prompt.new_empty((0, self.latent_dim))

        device = prompt.device
        K = self.short_ctx.short_ctx_k

        prompt_norm = self._normalize_latents(prompt)

        # Build sequence as a list of [D] tensors
        generated = list(prompt_norm[0])

        for _ in range(n_steps):
            seq = torch.stack(generated).unsqueeze(0)  # [1, L, D]

            # Backbone (no noise at inference) — last position conditioning
            z_long = self.backbone(seq)[:, -1:]  # [1, 1, d_model]

            # Short-context: last K clean frames
            if self.use_short_context:
                recent = seq[:, -K:]  # [1, min(K,L), D]
                z_short = self.short_ctx.forward_single(recent)  # [1, 1, d_model]
            else:
                z_short = torch.zeros_like(z_long)

            Z = (z_long + z_short).squeeze(1)  # [1, d_model]

            # One-step consistency sampling from t=pi/2
            eps = torch.randn(1, self.latent_dim, device=device) * math.sqrt(temperature)
            t = torch.full((1,), math.pi / 2, device=device)

            # At t=pi/2: f(eps, pi/2, Z) = cos(pi/2)*eps + sin(pi/2)*F = F
            # So prediction is just F_phi(eps, pi/2, Z)
            F_pred = self.head_ema(eps, t, Z)  # [1, D]
            generated.append(F_pred.squeeze(0))

        prompt_len = prompt.size(1)
        out_norm = torch.stack(generated[prompt_len:]).unsqueeze(0)  # [1, n_steps, D]
        out_raw = self._denormalize_latents(out_norm).squeeze(0)
        return out_raw  # [n_steps, D]

    def param_count(self) -> dict[str, int]:
        """Return parameter counts per component."""
        def _count(module):
            return sum(p.numel() for p in module.parameters())

        return {
            "backbone": _count(self.backbone),
            "short_ctx": _count(self.short_ctx),
            "head": _count(self.head),
            "w_psi": _count(self.w_psi),
            "total_trainable": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }
