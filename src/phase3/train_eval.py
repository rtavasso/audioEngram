from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from phase0.utils.logging import get_logger
from phase0.utils.seed import set_seed

from .data import Phase3UtteranceDataset, collate_pad
from .losses import compute_losses
from .models import Factorizer, diag_gaussian_kl
from .normalization import NormStats, OnlineMeanVarVec


def _device_from_config(device: str) -> torch.device:
    d = str(device).lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


@dataclass
class Phase3EvalMetrics:
    recon: float
    kl: float
    dyn: float
    total: float
    n_batches: int


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(dtype=x.dtype)
    while m.ndim < x.ndim:
        m = m.unsqueeze(-1)
    num = (x * m).sum()
    den = m.sum().clamp_min(1.0)
    return num / den


def _masked_var_per_dim(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Returns per-dim variance over (B,T) masked positions.
    x: [B,T,D], mask: [B,T]
    """
    m = mask.to(dtype=x.dtype).unsqueeze(-1)  # [B,T,1]
    den = m.sum().clamp_min(1.0)
    mean = (x * m).sum(dim=(0, 1)) / den.squeeze(-1)  # [D]
    mean2 = ((x * x) * m).sum(dim=(0, 1)) / den.squeeze(-1)  # [D]
    var = mean2 - mean * mean
    return torch.clamp(var, min=0.0)


def compute_or_load_norm_stats(
    *,
    latents_dir: str | Path,
    latents_index: str | Path,
    splits_dir: str | Path,
    stats_file: str | Path,
    x_dim: int,
    min_duration_sec: float,
    max_train_utterances: Optional[int],
) -> NormStats:
    stats_path = Path(stats_file)
    if stats_path.exists():
        return NormStats.load(stats_path)

    logger = get_logger()
    logger.info(f"[phase3] Computing x normalization stats (train) -> {stats_path}")
    ds = Phase3UtteranceDataset(
        latents_dir=latents_dir,
        latents_index_path=latents_index,
        splits_dir=splits_dir,
        split="train",
        min_duration_sec=min_duration_sec,
        norm_stats=None,
        max_utterances=max_train_utterances,
    )
    mv = OnlineMeanVarVec(dim=x_dim)
    for i in range(len(ds)):
        x = ds[i]["x"].numpy()
        mv.update_batch(x)
        if (i + 1) % 100 == 0:
            logger.info(f"[phase3] Stats pass: {i+1}/{len(ds)} utterances")

    stats = mv.finalize()
    stats.save(stats_path)
    return stats


@torch.no_grad()
def evaluate(
    *,
    model: Factorizer,
    loader: DataLoader,
    device: torch.device,
    recon_weight: float,
    beta: float,
    free_bits_per_dim: float,
    z_rec_dim: int,
    dyn_weight: float,
    prior_sample_prob: float,
    max_batches: int,
) -> Phase3EvalMetrics:
    model.eval()
    logger = get_logger()
    sums = {"recon": 0.0, "kl": 0.0, "dyn": 0.0, "total": 0.0}
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        out = model(x, prior_sample_prob=prior_sample_prob)
        dyn_params = model.dyn(out.z_dyn[:, :-1])
        losses = compute_losses(
            x=x,
            mask=mask,
            x_hat=out.x_hat_mixed,
            q_rec=out.q_rec,
            p_rec=out.p_rec,
            dyn_params=dyn_params,
            z_dyn_target=out.z_dyn[:, 1:],
            recon_weight=recon_weight,
            beta=beta,
            free_bits_per_dim=free_bits_per_dim,
            z_rec_dim=z_rec_dim,
            dyn_weight=dyn_weight,
        )
        sums["recon"] += float(losses.recon.item())
        sums["kl"] += float(losses.kl.item())
        sums["dyn"] += float(losses.dyn.item())
        sums["total"] += float(losses.total.item())
        n += 1
        if n >= max_batches:
            break
    if n == 0:
        logger.warning("[phase3] Eval loader produced 0 batches")
        return Phase3EvalMetrics(recon=float("nan"), kl=float("nan"), dyn=float("nan"), total=float("nan"), n_batches=0)

    return Phase3EvalMetrics(
        recon=sums["recon"] / n,
        kl=sums["kl"] / n,
        dyn=sums["dyn"] / n,
        total=sums["total"] / n,
        n_batches=n,
    )


def train(
    *,
    config: dict,
    resume_checkpoint: Optional[str | Path] = None,
) -> None:
    logger = get_logger()
    device = _device_from_config(config["train"]["device"])
    set_seed(int(config["train"]["seed"]))

    out_dir = Path(config["output"]["out_dir"])
    ckpt_dir = Path(config["output"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_dim = int(config["model"]["x_dim"])
    z_dyn_dim = int(config["model"]["z_dyn_dim"])
    z_rec_dim = int(config["model"]["z_rec_dim"])

    norm_stats = None
    if config.get("normalization", {}).get("enabled", True):
        norm_stats = compute_or_load_norm_stats(
            latents_dir=config["data"]["latents_dir"],
            latents_index=config["data"]["latents_index"],
            splits_dir=config["data"]["splits_dir"],
            stats_file=config["normalization"]["stats_file"],
            x_dim=x_dim,
            min_duration_sec=float(config["data"]["min_duration_sec"]),
            max_train_utterances=config["normalization"].get("max_train_utterances"),
        )
        logger.info(f"[phase3] Using normalization stats: n_frames={norm_stats.n_frames}")

    train_ds = Phase3UtteranceDataset(
        latents_dir=config["data"]["latents_dir"],
        latents_index_path=config["data"]["latents_index"],
        splits_dir=config["data"]["splits_dir"],
        split="train",
        min_duration_sec=float(config["data"]["min_duration_sec"]),
        norm_stats=norm_stats,
    )
    eval_ds = Phase3UtteranceDataset(
        latents_dir=config["data"]["latents_dir"],
        latents_index_path=config["data"]["latents_index"],
        splits_dir=config["data"]["splits_dir"],
        split="eval",
        min_duration_sec=float(config["data"]["min_duration_sec"]),
        norm_stats=norm_stats,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["train"]["num_workers"]),
        collate_fn=collate_pad,
        pin_memory=(device.type == "cuda"),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pad,
        pin_memory=(device.type == "cuda"),
    )

    m = config["model"]
    model = Factorizer(
        x_dim=x_dim,
        z_dyn_dim=z_dyn_dim,
        z_rec_dim=z_rec_dim,
        dyn_encoder_hidden=int(m["dyn_encoder"]["hidden_dim"]),
        dyn_encoder_layers=int(m["dyn_encoder"]["num_layers"]),
        dyn_encoder_dropout=float(m["dyn_encoder"]["dropout"]),
        dyn_model_hidden=int(m["dyn_model"]["hidden_dim"]),
        dyn_model_layers=int(m["dyn_model"]["num_layers"]),
        dyn_model_dropout=float(m["dyn_model"]["dropout"]),
        dyn_model_min_log_sigma=float(m["dyn_model"]["min_log_sigma"]),
        dyn_model_max_log_sigma=float(m["dyn_model"]["max_log_sigma"]),
        posterior_hidden=int(m["posterior"]["hidden_dim"]),
        posterior_layers=int(m["posterior"]["num_layers"]),
        posterior_dropout=float(m["posterior"]["dropout"]),
        posterior_min_log_sigma=float(m["posterior"]["min_log_sigma"]),
        posterior_max_log_sigma=float(m["posterior"]["max_log_sigma"]),
        prior_hidden=int(m["prior"]["hidden_dim"]),
        prior_layers=int(m["prior"]["num_layers"]),
        prior_dropout=float(m["prior"]["dropout"]),
        prior_min_log_sigma=float(m["prior"]["min_log_sigma"]),
        prior_max_log_sigma=float(m["prior"]["max_log_sigma"]),
        recon_hidden=int(m["reconstructor"]["hidden_dim"]),
        recon_layers=int(m["reconstructor"]["num_layers"]),
        recon_dropout=float(m["reconstructor"]["dropout"]),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    step = 0
    if resume_checkpoint is not None:
        ckpt = torch.load(str(resume_checkpoint), map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step = int(ckpt.get("step", 0))
        logger.info(f"[phase3] Resumed from {resume_checkpoint} at step {step}")

    log_path = Path(config["output"]["train_log_jsonl"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    recon_weight = float(config["loss"]["recon_weight"])
    beta_final = float(config["loss"]["beta_final"])
    beta_warmup = int(config["loss"]["beta_warmup_steps"])
    free_bits_per_dim = float(config["loss"]["free_bits_per_dim"])
    dyn_weight = float(config["loss"]["dyn_weight"])
    prior_sample_prob = float(config["loss"]["prior_sample_prob"])

    max_steps = int(config["train"]["max_steps"])
    log_every = int(config["train"]["log_every"])
    eval_every = int(config["train"]["eval_every"])
    save_every = int(config["train"]["save_every"])
    eval_batches = int(config["train"]["eval_batches"])
    grad_clip = float(config["train"]["grad_clip_norm"])

    logger.info(
        f"[phase3] device={device.type} train_utts={len(train_ds)} eval_utts={len(eval_ds)} "
        f"batch={config['train']['batch_size']} max_steps={max_steps}"
    )

    train_iter = iter(train_loader)
    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x = batch["x"].to(device)
        mask = batch["mask"].to(device)

        # Beta warmup
        if beta_warmup > 0:
            beta = beta_final * min(1.0, float(step + 1) / float(beta_warmup))
        else:
            beta = beta_final

        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(x, prior_sample_prob=prior_sample_prob)
        dyn_params = model.dyn(out.z_dyn[:, :-1])
        losses = compute_losses(
            x=x,
            mask=mask,
            x_hat=out.x_hat_mixed,
            q_rec=out.q_rec,
            p_rec=out.p_rec,
            dyn_params=dyn_params,
            z_dyn_target=out.z_dyn[:, 1:],
            recon_weight=recon_weight,
            beta=beta,
            free_bits_per_dim=free_bits_per_dim,
            z_rec_dim=z_rec_dim,
            dyn_weight=dyn_weight,
        )
        losses.total.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        opt.step()

        step += 1

        if log_every and step % log_every == 0:
            # Diagnostics to catch collapse/cheating early.
            with torch.no_grad():
                # z_dyn activity
                z_dyn_var = _masked_var_per_dim(out.z_dyn, mask)
                z_dyn_var_mean = float(z_dyn_var.mean().item())
                z_dyn_l2 = torch.linalg.vector_norm(out.z_dyn, dim=-1)  # [B,T]
                z_dyn_l2_mean = float(_masked_mean(z_dyn_l2, mask).item())

                # z_rec activity
                z_rec_post_l2 = torch.linalg.vector_norm(out.z_rec_post, dim=-1)
                z_rec_prior_l2 = torch.linalg.vector_norm(out.z_rec_prior, dim=-1)
                z_rec_post_l2_mean = float(_masked_mean(z_rec_post_l2, mask).item())
                z_rec_prior_l2_mean = float(_masked_mean(z_rec_prior_l2, mask).item())

                # Raw KL (before free-bits clamp) for visibility
                kl_raw = diag_gaussian_kl(out.q_rec, out.p_rec)  # [B,T]
                kl_raw_mean = float(_masked_mean(kl_raw, mask).item())

                # Sigma stats
                q_log_sigma_mean = float(_masked_mean(out.q_rec.log_sigma, mask).item())
                p_log_sigma_mean = float(_masked_mean(out.p_rec.log_sigma, mask).item())

                # Recon gap: posterior vs prior-only reconstruction
                recon_post = ((out.x_hat_post - x) ** 2).mean(dim=-1)  # [B,T]
                recon_prior = ((out.x_hat_prior - x) ** 2).mean(dim=-1)  # [B,T]
                recon_post_mean = float(_masked_mean(recon_post, mask).item())
                recon_prior_mean = float(_masked_mean(recon_prior, mask).item())
                recon_prior_over_post = float(recon_prior_mean / max(recon_post_mean, 1e-12))

                # How often we forced prior during mixed recon
                prior_frac = float(_masked_mean(out.prior_mask.squeeze(-1), mask).item())

            row = {
                "step": step,
                "beta": beta,
                "loss_total": float(losses.total.item()),
                "loss_recon": float(losses.recon.item()),
                "loss_kl": float(losses.kl.item()),
                "loss_dyn": float(losses.dyn.item()),
                "kl_raw": kl_raw_mean,
                "z_dyn_var_mean": z_dyn_var_mean,
                "z_dyn_l2_mean": z_dyn_l2_mean,
                "z_rec_post_l2_mean": z_rec_post_l2_mean,
                "z_rec_prior_l2_mean": z_rec_prior_l2_mean,
                "q_log_sigma_mean": q_log_sigma_mean,
                "p_log_sigma_mean": p_log_sigma_mean,
                "recon_post": recon_post_mean,
                "recon_prior": recon_prior_mean,
                "recon_prior_over_post": recon_prior_over_post,
                "prior_frac": prior_frac,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            logger.info(
                f"[phase3] step={step} total={row['loss_total']:.4f} recon={row['loss_recon']:.4f} "
                f"kl={row['loss_kl']:.4f} dyn={row['loss_dyn']:.4f} "
                f"kl_raw={row['kl_raw']:.3f} z_dyn_var={row['z_dyn_var_mean']:.3e} "
                f"prior_frac={row['prior_frac']:.2f} prior/post={row['recon_prior_over_post']:.2f}"
            )

        if save_every and step % save_every == 0:
            ckpt_path = ckpt_dir / f"phase3_step{step}.pt"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "config": config}, ckpt_path)

        if eval_every and step % eval_every == 0:
            metrics = evaluate(
                model=model,
                loader=eval_loader,
                device=device,
                recon_weight=recon_weight,
                beta=beta,
                free_bits_per_dim=free_bits_per_dim,
                z_rec_dim=z_rec_dim,
                dyn_weight=dyn_weight,
                prior_sample_prob=prior_sample_prob,
                max_batches=eval_batches,
            )
            logger.info(
                f"[phase3] eval step={step} total={metrics.total:.4f} recon={metrics.recon:.4f} "
                f"kl={metrics.kl:.4f} dyn={metrics.dyn:.4f} batches={metrics.n_batches}"
            )
            Path(config["output"]["eval_metrics_file"]).write_text(
                json.dumps({"step": step, **metrics.__dict__}, indent=2)
            )

    # Final checkpoint
    final_ckpt = ckpt_dir / "phase3_final.pt"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "config": config}, final_ckpt)
