"""
Loss functions for AR-Friendly VAE training.

- reconstruction_loss: Multi-scale STFT (L1 mag + log-mag) + time-domain L1
- kl_loss: Standard VAE KL vs N(0,I)
- smoothness_loss: Penalizes sharp latent jumps
- predictability_loss: MSE of a small predictor on z deltas
- temporal_prediction_loss_from_delta: short-context next-step prediction loss (cosine + MSE)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _stft_mag(x: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
    """Compute STFT magnitude. x: [B, T] -> [B, F, T']."""
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
                       window=window, return_complex=True)
    return spec.abs()  # [B, F, T']


def reconstruction_loss(
    audio: torch.Tensor,
    audio_hat: torch.Tensor,
    stft_windows: tuple[int, ...] = (256, 512, 1024, 2048),
) -> torch.Tensor:
    """
    Multi-scale STFT loss (L1 on magnitude + log-magnitude) + time-domain L1.

    Args:
        audio: [B, 1, T] ground-truth
        audio_hat: [B, 1, T] reconstruction

    Returns:
        Scalar loss (mean over batch).
    """
    # Squeeze channel dim for STFT
    x = audio.squeeze(1)  # [B, T]
    x_hat = audio_hat.squeeze(1)  # [B, T]

    # Time-domain L1
    loss = F.l1_loss(x_hat, x)

    # Multi-scale STFT
    for n_fft in stft_windows:
        hop = n_fft // 4
        mag = _stft_mag(x, n_fft, hop)
        mag_hat = _stft_mag(x_hat, n_fft, hop)
        # L1 on magnitude
        loss = loss + F.l1_loss(mag_hat, mag)
        # L1 on log-magnitude
        log_mag = torch.log(mag.clamp_min(1e-5))
        log_mag_hat = torch.log(mag_hat.clamp_min(1e-5))
        loss = loss + F.l1_loss(log_mag_hat, log_mag)

    return loss


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL divergence vs N(0,I).

    Args:
        mu: [B, D, T']
        logvar: [B, D, T']

    Returns:
        Scalar KL (sum over D*T', mean over batch).
    """
    # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum(dim=(1, 2)).mean()  # sum over D,T'; mean over B


def smoothness_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Penalize sharp jumps in latent trajectory.

    Args:
        z: [B, D, T']

    Returns:
        Scalar loss: mean ||z[:,:,t] - z[:,:,t-1]||^2
    """
    dz = z[:, :, 1:] - z[:, :, :-1]  # [B, D, T'-1]
    return dz.pow(2).mean()


def predictability_loss(
    z: torch.Tensor,
    predictor: torch.nn.Module,
    window_size: int = 8,
) -> torch.Tensor:
    """
    MSE between predictor's delta prediction and actual delta.

    Predictor takes z[:,:,t-W:t] flattened, predicts z[:,:,t] - z[:,:,t-1].
    Gradients flow back through z to the encoder via reparameterization.

    Args:
        z: [B, D, T']
        predictor: Module mapping [B, W*D] -> [B, D]
        window_size: context window W

    Returns:
        Scalar MSE loss.
    """
    B, D, T = z.shape
    if T <= window_size:
        return z.new_tensor(0.0)

    # Collect windows and targets
    # z_windows: for each t in [W, T), take z[:,:,t-W:t] -> [B, D, W]
    # targets: z[:,:,t] - z[:,:,t-1]
    windows = []
    targets = []
    for t in range(window_size, T):
        windows.append(z[:, :, t - window_size:t])  # [B, D, W]
        targets.append(z[:, :, t] - z[:, :, t - 1])  # [B, D]

    # Stack and reshape
    windows = torch.stack(windows, dim=0)  # [N, B, D, W] where N = T - W
    targets = torch.stack(targets, dim=0)  # [N, B, D]

    N = windows.shape[0]
    # Merge N and B for batched forward pass
    windows_flat = windows.reshape(N * B, D * window_size)  # [N*B, W*D]
    targets_flat = targets.reshape(N * B, D)  # [N*B, D]

    pred = predictor(windows_flat)  # [N*B, D]
    return F.mse_loss(pred, targets_flat)


def temporal_prediction_loss_from_delta(
    z: torch.Tensor,
    delta_pred: torch.Tensor,
    *,
    alpha: float = 0.5,
    start_t: int = 0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Temporal prediction loss for next-step latent dynamics.

    Implements:
        L_pred = alpha * (1 - cos_sim(delta_pred, delta_true))
               + (1 - alpha) * || z_{t+1} - z_hat_{t+1} ||^2

    where:
        delta_true = z_{t+1} - z_t
        z_hat_{t+1} = z_t + delta_pred

    Args:
        z: [B, D, T] latent sequence
        delta_pred: [B, D, N] predicted deltas for a contiguous t-range
        alpha: mix between cosine-on-deltas and MSE-on-next-latent
        start_t: index of the first delta_pred timestep within z (t = start_t)
        eps: numerical epsilon for cosine similarity

    Returns:
        (loss_total, loss_cos, loss_mse)
    """
    if z.ndim != 3:
        raise ValueError(f"Expected z with shape [B, D, T], got {tuple(z.shape)}")
    if delta_pred.ndim != 3:
        raise ValueError(f"Expected delta_pred with shape [B, D, N], got {tuple(delta_pred.shape)}")

    B, D, T = z.shape
    _, Dp, N = delta_pred.shape
    if Dp != D:
        raise ValueError(f"delta_pred D mismatch: z has D={D}, delta_pred has D={Dp}")
    if N <= 0 or T <= 1:
        zero = z.new_tensor(0.0)
        return zero, zero, zero
    if start_t < 0:
        raise ValueError(f"start_t must be >= 0, got {start_t}")
    if start_t + N + 1 > T:
        raise ValueError(
            f"Invalid shapes/range: start_t={start_t}, N={N}, T={T} "
            f"(need start_t + N + 1 <= T)"
        )
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    z_t = z[:, :, start_t:start_t + N]
    z_tp1 = z[:, :, start_t + 1:start_t + N + 1]
    delta_true = z_tp1 - z_t

    # Cosine similarity over the latent dimension, averaged over batch+time.
    cos = F.cosine_similarity(delta_pred, delta_true, dim=1, eps=eps)  # [B, N]
    l_cos = (1.0 - cos).mean()

    z_hat_tp1 = z_t + delta_pred
    l_mse = F.mse_loss(z_hat_tp1, z_tp1)

    loss = float(alpha) * l_cos + (1.0 - float(alpha)) * l_mse
    return loss, l_cos, l_mse
