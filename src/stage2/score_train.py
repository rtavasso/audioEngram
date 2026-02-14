"""
Score model training loop on latent frames from zarr.

Trains a ScoreNetwork via denoising score matching on individual latent frames
(no temporal structure needed for the score model itself).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

from phase0.data.io import LatentStore
from phase0.utils.seed import get_rng

from .score_model import (
    ScoreNetwork,
    ConditionalScoreNetwork,
    denoising_score_matching_loss,
    denoising_score_matching_loss_conditional,
)


logger = logging.getLogger("phase0")


class LatentFrameDataset(IterableDataset):
    """Stream individual latent frames from zarr store."""

    def __init__(
        self,
        latents_dir: str | Path,
        utterance_ids: list[str],
        seed: int = 42,
    ):
        self.store = LatentStore(latents_dir)
        self.utterance_ids = utterance_ids
        self.seed = seed

    def __iter__(self):
        rng = get_rng(self.seed)
        utt_ids = list(self.utterance_ids)
        rng.shuffle(utt_ids)

        for utt_id in utt_ids:
            if utt_id not in self.store:
                continue
            x = self.store.get_latents(utt_id).astype(np.float32, copy=False)
            # Shuffle frames within utterance
            indices = np.arange(x.shape[0])
            rng.shuffle(indices)
            for idx in indices:
                yield torch.from_numpy(x[idx])  # [D]


class ConditionalLatentFrameDataset(IterableDataset):
    """
    Stream (context_flat, target_z) pairs for conditional score modeling.

    Given an utterance latent sequence x[0:T], for a target time t the
    conditioning context is the window:
        x[t-k-W+1 : t-k+1]   (inclusive end), flattened to [W*D]
    and the target is z_clean = x[t].
    """

    def __init__(
        self,
        latents_dir: str | Path,
        utterance_ids: list[str],
        window_size: int,
        horizon_k: int,
        max_frames_per_utt: Optional[int] = None,
        seed: int = 42,
    ):
        self.store = LatentStore(latents_dir)
        self.utterance_ids = utterance_ids
        self.window_size = int(window_size)
        self.horizon_k = int(horizon_k)
        self.max_frames_per_utt = None if max_frames_per_utt is None else int(max_frames_per_utt)
        self.seed = int(seed)

    def __iter__(self):
        rng = get_rng(self.seed)
        utt_ids = list(self.utterance_ids)
        rng.shuffle(utt_ids)

        min_t = max(0, (self.window_size - 1) + self.horizon_k)

        for utt_id in utt_ids:
            if utt_id not in self.store:
                continue
            x = self.store.get_latents(utt_id).astype(np.float32, copy=False)  # [T, D]
            t_total = int(x.shape[0])
            if t_total <= min_t:
                continue

            # Optionally subsample per-utterance to reduce I/O cost
            n_avail = t_total - min_t
            if self.max_frames_per_utt is not None and self.max_frames_per_utt > 0:
                n_use = int(min(n_avail, self.max_frames_per_utt))
            else:
                n_use = int(n_avail)

            # Sample indices without replacement when possible; otherwise just shuffle.
            if n_use < n_avail:
                rel = rng.choice(np.arange(n_avail, dtype=np.int64), size=n_use, replace=False)
            else:
                rel = np.arange(n_avail, dtype=np.int64)
                rng.shuffle(rel)

            for r in rel:
                t = int(min_t + int(r))
                end = t - self.horizon_k
                start = end - self.window_size + 1
                if start < 0 or end < 0:
                    continue

                ctx = x[start : end + 1].reshape(-1)  # [W*D]
                z = x[t]  # [D]
                yield (torch.from_numpy(ctx), torch.from_numpy(z))


def train_score_model(
    *,
    latents_dir: str | Path,
    utterance_ids: list[str],
    out_dir: Path,
    latent_dim: int = 512,
    hidden_dim: int = 1024,
    n_layers: int = 4,
    sigma_min: float = 0.01,
    sigma_max: float = 1.0,
    batch_size: int = 256,
    num_workers: int = 0,
    max_steps: int = 20000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip_norm: float = 1.0,
    log_every: int = 100,
    save_every: int = 5000,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
    output_skip: bool = False,
    loss_weighting: str = "sigma2",
) -> tuple[ScoreNetwork, str]:
    """
    Train a score model via denoising score matching.

    Returns:
        (trained ScoreNetwork, path to final checkpoint)
    """
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = ScoreNetwork(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        output_skip=bool(output_skip),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    dataset = LatentFrameDataset(latents_dir, utterance_ids, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    logger.info(
        f"[score_train] dim={latent_dim} hidden={hidden_dim} layers={n_layers} "
        f"sigma=[{sigma_min},{sigma_max}] batch={batch_size} steps={max_steps}"
    )

    model.train()
    step = 0
    data_iter = iter(loader)
    loss_accum = 0.0
    log_count = 0

    while step < max_steps:
        try:
            z_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            z_batch = next(data_iter)

        z_batch = z_batch.to(device, non_blocking=True)  # [B, D]

        optimizer.zero_grad(set_to_none=True)
        loss = denoising_score_matching_loss(
            model, z_batch, sigma_min=sigma_min, sigma_max=sigma_max, loss_weighting=str(loss_weighting),
        )
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        step += 1
        loss_accum += float(loss.item())
        log_count += 1

        if log_every and step % log_every == 0:
            avg_loss = loss_accum / max(log_count, 1)
            logger.info(f"[score_train] step={step}/{max_steps} loss={avg_loss:.4f}")
            loss_accum = 0.0
            log_count = 0

        if save_every and step % save_every == 0:
            _save_score_checkpoint(model, step, ckpt_dir / f"score_step{step}.pt",
                                   latent_dim, hidden_dim, n_layers, bool(output_skip))

    final_path = ckpt_dir / "score_final.pt"
    _save_score_checkpoint(model, step, final_path, latent_dim, hidden_dim, n_layers, bool(output_skip))
    logger.info(f"[score_train] Done. Final checkpoint: {final_path}")

    return model, str(final_path)


def _save_score_checkpoint(
    model: ScoreNetwork, step: int, path: Path,
    latent_dim: int, hidden_dim: int, n_layers: int, output_skip: bool,
) -> None:
    torch.save({
        "model": model.state_dict(),
        "step": step,
        "model_kwargs": {
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "output_skip": bool(output_skip),
        },
    }, path)


def load_score_checkpoint(
    path: str | Path,
    device: torch.device = torch.device("cpu"),
) -> tuple[ScoreNetwork, dict]:
    """Load a score model from checkpoint."""
    ckpt = torch.load(str(path), map_location=device)
    model_kwargs = dict(ckpt.get("model_kwargs", {}))
    # Backward compatibility: older checkpoints used an output skip connection.
    model_kwargs.setdefault("output_skip", True)
    model = ScoreNetwork(**model_kwargs)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt


def train_conditional_score_model(
    *,
    latents_dir: str | Path,
    utterance_ids: list[str],
    out_dir: Path,
    latent_dim: int,
    cond_dim: int,
    window_size: int,
    horizon_k: int,
    hidden_dim: int = 1024,
    n_layers: int = 4,
    sigma_min: float = 0.01,
    sigma_max: float = 1.0,
    batch_size: int = 256,
    num_workers: int = 0,
    max_steps: int = 20000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip_norm: float = 1.0,
    log_every: int = 100,
    save_every: int = 5000,
    max_frames_per_utt: Optional[int] = None,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
    output_skip: bool = False,
    loss_weighting: str = "sigma2",
) -> tuple[ConditionalScoreNetwork, str]:
    """Train a conditional score model p(z | context_flat) via denoising score matching."""
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = ConditionalScoreNetwork(
        latent_dim=int(latent_dim),
        cond_dim=int(cond_dim),
        hidden_dim=int(hidden_dim),
        n_layers=int(n_layers),
        output_skip=bool(output_skip),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    dataset = ConditionalLatentFrameDataset(
        latents_dir=latents_dir,
        utterance_ids=utterance_ids,
        window_size=int(window_size),
        horizon_k=int(horizon_k),
        max_frames_per_utt=max_frames_per_utt,
        seed=int(seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
    )

    logger.info(
        f"[score_train_cond] dim={latent_dim} cond_dim={cond_dim} hidden={hidden_dim} layers={n_layers} "
        f"sigma=[{sigma_min},{sigma_max}] batch={batch_size} steps={max_steps} "
        f"W={window_size} k={horizon_k}"
    )

    model.train()
    step = 0
    data_iter = iter(loader)
    loss_accum = 0.0
    log_count = 0

    while step < int(max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        cond_batch, z_batch = batch  # [B,C], [B,D]
        cond_batch = cond_batch.to(device, non_blocking=True)
        z_batch = z_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = denoising_score_matching_loss_conditional(
            model,
            z_batch,
            cond_batch,
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            loss_weighting=str(loss_weighting),
        )
        loss.backward()
        if grad_clip_norm and float(grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        optimizer.step()

        step += 1
        loss_accum += float(loss.item())
        log_count += 1

        if log_every and step % int(log_every) == 0:
            avg_loss = loss_accum / max(log_count, 1)
            logger.info(f"[score_train_cond] step={step}/{max_steps} loss={avg_loss:.4f}")
            loss_accum = 0.0
            log_count = 0

        if save_every and step % int(save_every) == 0:
            _save_cond_score_checkpoint(
                model, step, ckpt_dir / f"score_cond_step{step}.pt",
                latent_dim=int(latent_dim),
                cond_dim=int(cond_dim),
                hidden_dim=int(hidden_dim),
                n_layers=int(n_layers),
                window_size=int(window_size),
                horizon_k=int(horizon_k),
                output_skip=bool(output_skip),
            )

    final_path = ckpt_dir / "score_cond_final.pt"
    _save_cond_score_checkpoint(
        model, step, final_path,
        latent_dim=int(latent_dim),
        cond_dim=int(cond_dim),
        hidden_dim=int(hidden_dim),
        n_layers=int(n_layers),
        window_size=int(window_size),
        horizon_k=int(horizon_k),
        output_skip=bool(output_skip),
    )
    logger.info(f"[score_train_cond] Done. Final checkpoint: {final_path}")
    model.eval()
    return model, str(final_path)


def _save_cond_score_checkpoint(
    model: ConditionalScoreNetwork,
    step: int,
    path: Path,
    *,
    latent_dim: int,
    cond_dim: int,
    hidden_dim: int,
    n_layers: int,
    window_size: int,
    horizon_k: int,
    output_skip: bool,
) -> None:
    torch.save({
        "model": model.state_dict(),
        "step": int(step),
        "model_kwargs": {
            "latent_dim": int(latent_dim),
            "cond_dim": int(cond_dim),
            "hidden_dim": int(hidden_dim),
            "n_layers": int(n_layers),
            "output_skip": bool(output_skip),
        },
        "conditioning": {
            "window_size": int(window_size),
            "horizon_k": int(horizon_k),
        },
    }, path)


def load_conditional_score_checkpoint(
    path: str | Path,
    device: torch.device = torch.device("cpu"),
) -> tuple[ConditionalScoreNetwork, dict]:
    """Load a conditional score model from checkpoint."""
    ckpt = torch.load(str(path), map_location=device)
    model_kwargs = dict(ckpt.get("model_kwargs", {}))
    model_kwargs.setdefault("output_skip", False)
    model = ConditionalScoreNetwork(**model_kwargs)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt
