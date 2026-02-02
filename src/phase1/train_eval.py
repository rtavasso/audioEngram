"""
Phase 1 training and evaluation routines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

from phase0.features.context import get_context_flat
from phase0.features.normalization import compute_delta
from phase0.utils.logging import get_logger
from phase0.utils.seed import set_seed

from .data import Phase1Sample, iter_phase1_samples, BufferedShuffle, sample_eval_utterances
from .mdn import MDN, sample_from_mdn
from .metrics import MetricAgg
from .stats import OnlineMeanVar, DiagGaussianBaseline


class Phase1IterableDataset(IterableDataset):
    def __init__(self, iterator_fn):
        super().__init__()
        self._iterator_fn = iterator_fn

    def __iter__(self):
        return self._iterator_fn()


def _device_from_config(device: str) -> torch.device:
    d = str(device).lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


@dataclass
class Phase1RunResult:
    horizon_k: int
    slice_name: str
    train: dict
    eval: dict
    rollout: Optional[dict] = None


def fit_unconditional_baseline(
    *,
    frames_index_path: str | Path,
    latents_dir: str | Path,
    window_size: int,
    horizon_k: int,
    slice_name: str,
    max_samples: Optional[int],
) -> DiagGaussianBaseline:
    """
    Fit unconditional diag-Gaussian baseline p(Δx) on train split.
    """
    mv = None
    for s in iter_phase1_samples(
        frames_index_path=frames_index_path,
        latents_dir=latents_dir,
        split="train",
        window_size=window_size,
        horizon_k=horizon_k,
        slice_name=slice_name,
        max_samples=max_samples,
    ):
        if mv is None:
            mv = OnlineMeanVar.create(dim=int(s.delta.shape[0]))
        mv.update(s.delta)

    if mv is None:
        raise RuntimeError("No samples found to fit unconditional baseline")

    mean, var = mv.finalize()
    return DiagGaussianBaseline(mean=mean, var=var)


@torch.no_grad()
def _eval_loop(
    *,
    model: MDN,
    loader: DataLoader,
    baseline: DiagGaussianBaseline,
    device: torch.device,
) -> dict:
    model.eval()
    agg = MetricAgg()
    dim = model.output_dim

    for batch in loader:
        ctx = batch["context"].to(device)
        dx = batch["delta"].to(device)
        nll = model.nll(ctx, dx)
        nll_b = baseline.nll(dx)
        pred = model.expected_mean(ctx)
        agg.update(nll=nll, nll_baseline=nll_b, pred_mean=pred, target=dx)

    return agg.finalize(dim=dim)


def _batchify(samples: list[Phase1Sample]) -> dict:
    # Convert to contiguous float32 tensors
    ctx = np.stack([s.context_flat for s in samples], axis=0).astype(np.float32, copy=False)
    dx = np.stack([s.delta for s in samples], axis=0).astype(np.float32, copy=False)
    return {
        "context": torch.from_numpy(ctx),
        "delta": torch.from_numpy(dx),
    }


def _collate_fn(samples: list[Phase1Sample]) -> dict:
    return _batchify(samples)


def train_and_eval_for_k(
    *,
    frames_index_path: str | Path,
    latents_dir: str | Path,
    splits_dir: str | Path,
    latents_index_path: str | Path,
    out_dir: str | Path,
    horizon_k: int,
    window_size: int,
    slice_name: str,
    seed: int,
    device: torch.device,
    n_components: int,
    hidden_dim: int,
    n_hidden_layers: int,
    dropout: float,
    min_log_sigma: float,
    max_log_sigma: float,
    batch_size: int,
    num_workers: int,
    max_steps: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    log_every: int,
    eval_every: int,
    save_every: int,
    shuffle_buffer: int,
    max_train_samples: Optional[int],
    max_eval_samples: Optional[int],
    rollout_enabled: bool,
    rollout_n_eval_utterances: int,
    rollout_max_frames_per_utt: int,
    rollout_sample_from_mixture: bool,
) -> Phase1RunResult:
    logger = get_logger()
    set_seed(seed)

    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Baseline (train-fit)
    baseline = fit_unconditional_baseline(
        frames_index_path=frames_index_path,
        latents_dir=latents_dir,
        window_size=window_size,
        horizon_k=horizon_k,
        slice_name=slice_name,
        max_samples=max_train_samples,
    )

    input_dim = window_size * 512  # Mimi latents are 512-dim in Phase 0 artifacts
    output_dim = 512

    model = MDN(
        input_dim=input_dim,
        output_dim=output_dim,
        n_components=n_components,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        dropout=dropout,
        min_log_sigma=min_log_sigma,
        max_log_sigma=max_log_sigma,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train iterator with buffered shuffle
    base_it = lambda: iter_phase1_samples(
        frames_index_path=frames_index_path,
        latents_dir=latents_dir,
        split="train",
        window_size=window_size,
        horizon_k=horizon_k,
        slice_name=slice_name,
        max_samples=max_train_samples,
    )
    shuffler = BufferedShuffle(buffer_size=shuffle_buffer, seed=seed + 10_000 + horizon_k)
    train_it = lambda: shuffler(base_it())
    train_ds = Phase1IterableDataset(train_it)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # Eval loaders (no shuffle)
    eval_train_ds = Phase1IterableDataset(
        lambda: iter_phase1_samples(
            frames_index_path=frames_index_path,
            latents_dir=latents_dir,
            split="train",
            window_size=window_size,
            horizon_k=horizon_k,
            slice_name=slice_name,
            max_samples=max_eval_samples,
        )
    )
    eval_eval_ds = Phase1IterableDataset(
        lambda: iter_phase1_samples(
            frames_index_path=frames_index_path,
            latents_dir=latents_dir,
            split="eval",
            window_size=window_size,
            horizon_k=horizon_k,
            slice_name=slice_name,
            max_samples=max_eval_samples,
        )
    )
    eval_train_loader = DataLoader(
        eval_train_ds,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=_collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    eval_eval_loader = DataLoader(
        eval_eval_ds,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    logger.info(
        f"[phase1] k={horizon_k} slice={slice_name} device={device.type} "
        f"batch={batch_size} steps={max_steps}"
    )

    step = 0
    model.train()
    train_iter = iter(train_loader)

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ctx = batch["context"].to(device)
        dx = batch["delta"].to(device)

        opt.zero_grad(set_to_none=True)
        nll = model.nll(ctx, dx).mean()
        nll.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        opt.step()

        step += 1

        if log_every and step % log_every == 0:
            logger.info(f"[phase1] k={horizon_k} step={step}/{max_steps} train_nll={float(nll.item()):.4f}")

        if save_every and step % save_every == 0:
            ckpt_path = ckpt_dir / f"mdn_k{horizon_k}_step{step}.pt"
            torch.save(
                {"model": model.state_dict(), "step": step, "horizon_k": horizon_k, "window_size": window_size},
                ckpt_path,
            )

        if eval_every and step % eval_every == 0:
            train_metrics = _eval_loop(model=model, loader=eval_train_loader, baseline=baseline, device=device)
            eval_metrics = _eval_loop(model=model, loader=eval_eval_loader, baseline=baseline, device=device)
            logger.info(
                f"[phase1] k={horizon_k} step={step} "
                f"eval_dnll={eval_metrics['dnll']:.4f} nll={eval_metrics['nll']:.4f} "
                f"baseline={eval_metrics['nll_baseline']:.4f}"
            )

    # Final checkpoint
    final_ckpt = ckpt_dir / f"mdn_k{horizon_k}_final.pt"
    torch.save(
        {"model": model.state_dict(), "step": step, "horizon_k": horizon_k, "window_size": window_size},
        final_ckpt,
    )

    # Final eval
    train_metrics = _eval_loop(model=model, loader=eval_train_loader, baseline=baseline, device=device)
    eval_metrics = _eval_loop(model=model, loader=eval_eval_loader, baseline=baseline, device=device)

    rollout_metrics = None
    if rollout_enabled:
        rollout_metrics = rollout_context_gap(
            model=model,
            baseline=baseline,
            splits_dir=splits_dir,
            latents_index_path=latents_index_path,
            latents_dir=latents_dir,
            window_size=window_size,
            horizon_k=horizon_k,
            n_eval_utterances=rollout_n_eval_utterances,
            max_frames_per_utt=rollout_max_frames_per_utt,
            device=device,
            seed=seed + 20_000 + horizon_k,
            sample_from_mixture=rollout_sample_from_mixture,
        )

    return Phase1RunResult(
        horizon_k=horizon_k,
        slice_name=slice_name,
        train=train_metrics,
        eval=eval_metrics,
        rollout=rollout_metrics,
    )


@torch.no_grad()
def rollout_context_gap(
    *,
    model: MDN,
    baseline: DiagGaussianBaseline,
    splits_dir: str | Path,
    latents_index_path: str | Path,
    latents_dir: str | Path,
    window_size: int,
    horizon_k: int,
    n_eval_utterances: int,
    max_frames_per_utt: int,
    device: torch.device,
    seed: int,
    sample_from_mixture: bool,
) -> dict:
    """
    Secondary diagnostic: compare teacher-forced context vs rollout-corrupted context.

    For each eval utterance:
    - Build x̂ by iteratively sampling/using mean Δx̂ from the model.
    - Evaluate NLL of true Δx under p(Δx | context) using:
        a) context from true x (teacher-forced)
        b) context from x̂ (rollout-corrupted)
    """
    from phase0.data.io import LatentStore as _LatentStore

    logger = get_logger()
    utt_ids = sample_eval_utterances(
        splits_dir=splits_dir,
        latents_index_path=latents_index_path,
        n_utterances=n_eval_utterances,
        seed=seed,
    )

    if not utt_ids:
        return {"n_utterances": 0}

    store = _LatentStore(latents_dir)

    agg_teacher = MetricAgg()
    agg_rollout = MetricAgg()
    dim = model.output_dim
    min_t = max(1, (window_size - 1) + horizon_k)

    for utt_id in utt_ids:
        if utt_id not in store:
            continue
        x = store.get_latents(utt_id).astype(np.float32, copy=False)
        t_total = int(min(x.shape[0], max_frames_per_utt))
        if t_total <= min_t + 1:
            continue

        x_hat = x[:min_t].copy()
        # Extend x_hat to length t_total for in-place fills
        x_hat = np.concatenate([x_hat, np.zeros((t_total - min_t, x.shape[1]), dtype=np.float32)], axis=0)

        for t in range(min_t, t_total):
            # Teacher-forced context
            ctx_true = get_context_flat(x, t, window_size, horizon_k).astype(np.float32, copy=False)
            dx_true = compute_delta(x, t).astype(np.float32, copy=False)

            # Rollout context (from x_hat)
            ctx_hat = get_context_flat(x_hat, t, window_size, horizon_k).astype(np.float32, copy=False)

            ctx_true_t = torch.from_numpy(ctx_true).unsqueeze(0).to(device)
            ctx_hat_t = torch.from_numpy(ctx_hat).unsqueeze(0).to(device)
            dx_true_t = torch.from_numpy(dx_true).unsqueeze(0).to(device)

            nll_true = model.nll(ctx_true_t, dx_true_t)
            nll_hat = model.nll(ctx_hat_t, dx_true_t)
            nll_b = baseline.nll(dx_true_t)

            pred_true = model.expected_mean(ctx_true_t)
            pred_hat = model.expected_mean(ctx_hat_t)

            agg_teacher.update(nll=nll_true, nll_baseline=nll_b, pred_mean=pred_true, target=dx_true_t)
            agg_rollout.update(nll=nll_hat, nll_baseline=nll_b, pred_mean=pred_hat, target=dx_true_t)

            # Advance rollout state with sampled or mean delta under ctx_hat
            out = model(ctx_hat_t)
            if sample_from_mixture:
                dx_hat = sample_from_mdn(out)[0].detach().cpu().numpy()
            else:
                dx_hat = model.expected_mean(ctx_hat_t)[0].detach().cpu().numpy()
            x_hat[t] = x_hat[t - 1] + dx_hat.astype(np.float32, copy=False)

    teacher = agg_teacher.finalize(dim=dim)
    rollout = agg_rollout.finalize(dim=dim)
    logger.info(
        f"[phase1-rollout] k={horizon_k} teacher_nll={teacher['nll']:.4f} "
        f"rollout_nll={rollout['nll']:.4f} gap={rollout['nll']-teacher['nll']:.4f}"
    )
    return {
        "n_utterances": int(n_eval_utterances),
        "teacher_forced": teacher,
        "rollout_context": rollout,
        "gap_nll": float(rollout["nll"] - teacher["nll"]),
        "gap_dnll": float(rollout["dnll"] - teacher["dnll"]),
    }


def write_results(
    results: list[Phase1RunResult],
    metrics_path: str | Path,
    tables_path: str | Path,
) -> None:
    import pandas as pd

    rows = []
    for r in results:
        row = {
            "horizon_k": r.horizon_k,
            "slice": r.slice_name,
            "train_nll": r.train.get("nll"),
            "train_nll_baseline": r.train.get("nll_baseline"),
            "train_dnll": r.train.get("dnll"),
            "eval_nll": r.eval.get("nll"),
            "eval_nll_baseline": r.eval.get("nll_baseline"),
            "eval_dnll": r.eval.get("dnll"),
            "eval_direction_cos": r.eval.get("direction_cosine"),
            "eval_logmag_r2": r.eval.get("logmag_r2"),
        }
        if r.rollout:
            row["rollout_gap_nll"] = r.rollout.get("gap_nll")
            row["rollout_gap_dnll"] = r.rollout.get("gap_dnll")
        rows.append(row)

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)

    df = pd.DataFrame(rows).sort_values(["slice", "horizon_k"])
    tables_path = Path(tables_path)
    tables_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(tables_path), index=False)
