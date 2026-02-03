#!/usr/bin/env python3
"""
Phase 4.5: Fine-tune param dynamics with K-step rollout loss.

Motivation:
Teacher-forced one-step training can look great but explode under free-running rollout.
This script directly optimizes stability by training on unrolled rollouts in z_dyn space.

Workflow:
  1) Ensure you have outputs/phase4/checkpoints/param_final.pt from scripts/31_phase4_train_eval.py
  2) Run this script to fine-tune param with rollout loss and refit residual memory.

Usage:
  uv run python scripts/32_phase4_rollout_finetune.py --config configs/phase4.yaml
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from phase0.data.io import LatentStore
from phase0.utils.logging import setup_logging, get_logger
from phase0.utils.seed import set_seed

from phase4.data import (
    ZPairIterableDataset,
    collate_zpairs,
    iter_zdyn_pairs,
    iter_rollout_segments,
    collate_rollout_segments,
    sample_eval_utterance_ids,
)
from phase4.memory import KMeansDeltaMemory
from phase4.models import ParamDyn, DiagGaussianDelta, diag_gaussian_nll


def _device_from_config(device: str) -> torch.device:
    d = str(device).lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _clip_l2(x: torch.Tensor, max_l2: float) -> torch.Tensor:
    if not max_l2 or max_l2 <= 0:
        return x
    l2 = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(max_l2 / l2, max=1.0)
    return x * scale


def _linear_schedule(step: int, start: float, end: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return float(end)
    a = min(max(float(step) / float(warmup_steps), 0.0), 1.0)
    return float(start + a * (end - start))


@torch.no_grad()
def _fit_unconditional_baseline(cfg: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sums = None
    sums2 = None
    n = 0
    for p in iter_zdyn_pairs(
        zdyn_dir=cfg["data"]["zdyn_dir"],
        zdyn_index_path=cfg["data"]["zdyn_index"],
        splits_dir=cfg["data"]["splits_dir"],
        split="train",
        min_duration_sec=float(cfg["data"]["min_duration_sec"]),
        seed=int(cfg["train"]["seed"]),
        max_pairs=int(cfg["memory"]["max_fit_pairs"]),
        sample_prob=1.0,
    ):
        v = p.dz.astype(np.float64, copy=False)
        if sums is None:
            sums = np.zeros_like(v)
            sums2 = np.zeros_like(v)
        sums += v
        sums2 += v * v
        n += 1
    if n == 0 or sums is None or sums2 is None:
        raise RuntimeError("No training pairs found for baseline fit.")
    mean = (sums / n).astype(np.float32, copy=False)
    var = (sums2 / n - (mean.astype(np.float64) ** 2)).astype(np.float32, copy=False)
    var = np.maximum(var, 1e-8).astype(np.float32, copy=False)
    return torch.from_numpy(mean).to(device), torch.from_numpy(var).to(device)


@torch.no_grad()
def _eval_teacher_forced(
    *,
    name: str,
    predict_fn,
    loader: DataLoader,
    baseline_mean: torch.Tensor,
    baseline_var: torch.Tensor,
    device: torch.device,
    max_batches: int,
) -> dict:
    log2pi = 1.8378770664093453

    def baseline_nll(dz: torch.Tensor) -> torch.Tensor:
        var = baseline_var
        while var.ndim < dz.ndim:
            var = var.unsqueeze(0)
        mean = baseline_mean
        while mean.ndim < dz.ndim:
            mean = mean.unsqueeze(0)
        z2 = ((dz - mean) ** 2) / var
        return 0.5 * (z2 + torch.log(var) + log2pi).sum(dim=-1)

    nll_sum = 0.0
    nllb_sum = 0.0
    cos_sum = 0.0
    n = 0
    ns = 0
    for batch in loader:
        z_prev = batch["z_prev"].to(device)
        dz = batch["dz"].to(device)
        pred: DiagGaussianDelta = predict_fn(z_prev)
        nll = diag_gaussian_nll(dz, pred)
        nllb = baseline_nll(dz)

        mu = pred.mu
        num = (mu * dz).sum(dim=-1)
        den = torch.linalg.vector_norm(mu, dim=-1) * torch.linalg.vector_norm(dz, dim=-1)
        cos = num / torch.clamp(den, min=1e-12)

        nll_sum += float(nll.mean().item())
        nllb_sum += float(nllb.mean().item())
        cos_sum += float(cos.mean().item())
        n += 1
        ns += int(z_prev.shape[0])
        if max_batches and n >= int(max_batches):
            break

    return {
        "model": name,
        "n_batches": n,
        "n_samples": ns,
        "nll": nll_sum / max(n, 1),
        "nll_baseline": nllb_sum / max(n, 1),
        "dnll": (nll_sum - nllb_sum) / max(n, 1),
        "direction_cos": cos_sum / max(n, 1),
    }


@torch.no_grad()
def _eval_rollout(
    *,
    name: str,
    predict_fn,
    store: LatentStore,
    utt_ids: list[str],
    max_frames: int,
    baseline_mean: torch.Tensor,
    baseline_var: torch.Tensor,
    device: torch.device,
    step_alpha: float,
    clip_dz_l2: float,
) -> dict:
    log2pi = 1.8378770664093453

    def baseline_nll(dz: torch.Tensor) -> torch.Tensor:
        var = baseline_var
        while var.ndim < dz.ndim:
            var = var.unsqueeze(0)
        mean = baseline_mean
        while mean.ndim < dz.ndim:
            mean = mean.unsqueeze(0)
        z2 = ((dz - mean) ** 2) / var
        return 0.5 * (z2 + torch.log(var) + log2pi).sum(dim=-1)

    tf_nll_sum = 0.0
    ro_nll_sum = 0.0
    tf_b_sum = 0.0
    ro_b_sum = 0.0
    n_steps = 0
    n_clipped = 0
    dz_l2_sum = 0.0
    max_z_hat_l2 = 0.0
    any_nonfinite = False
    first_nonfinite = None

    for utt in utt_ids:
        z = store.get_latents(utt).astype(np.float32, copy=False)
        if z.shape[0] < 2:
            continue
        t_max = min(int(max_frames), int(z.shape[0]))
        z_true = torch.from_numpy(z[:t_max]).to(device)
        dz_true = z_true[1:] - z_true[:-1]

        z_hat = z_true[0].clone()
        max_z_hat_l2 = max(max_z_hat_l2, float(torch.linalg.vector_norm(z_hat).item()))

        for t in range(t_max - 1):
            dz_t = dz_true[t : t + 1]

            pred_tf = predict_fn(z_true[t : t + 1])
            tf_nll_sum += float(diag_gaussian_nll(dz_t, pred_tf).item())
            tf_b_sum += float(baseline_nll(dz_t).item())

            pred_ro = predict_fn(z_hat.unsqueeze(0))
            ro_nll_sum += float(diag_gaussian_nll(dz_t, pred_ro).item())
            ro_b_sum += float(baseline_nll(dz_t).item())

            dz_hat = pred_ro.mu.squeeze(0) * float(step_alpha)
            if clip_dz_l2 and clip_dz_l2 > 0:
                l2 = float(torch.linalg.vector_norm(dz_hat).item())
                dz_l2_sum += l2
                if l2 > float(clip_dz_l2):
                    dz_hat = _clip_l2(dz_hat, float(clip_dz_l2))
                    n_clipped += 1

            z_hat = z_hat + dz_hat
            max_z_hat_l2 = max(max_z_hat_l2, float(torch.linalg.vector_norm(z_hat).item()))
            if not torch.isfinite(z_hat).all():
                any_nonfinite = True
                first_nonfinite = t + 1
                break

            n_steps += 1

        if any_nonfinite:
            break

    if n_steps == 0:
        return {"model": name, "n_steps": 0, "rollout_nonfinite": True}

    tf_nll = tf_nll_sum / n_steps
    ro_nll = ro_nll_sum / n_steps
    tf_dnll = (tf_nll_sum - tf_b_sum) / n_steps
    ro_dnll = (ro_nll_sum - ro_b_sum) / n_steps
    return {
        "model": name,
        "n_steps": n_steps,
        "rollout_nonfinite": any_nonfinite,
        "first_nonfinite_step": first_nonfinite,
        "max_z_hat_l2": max_z_hat_l2,
        "rollout_step_alpha": float(step_alpha),
        "rollout_clip_dz_l2": float(clip_dz_l2),
        "rollout_n_clipped": int(n_clipped),
        "rollout_mean_dz_l2_preclip": float(dz_l2_sum / max(n_steps, 1)),
        "teacher_forced_nll": tf_nll,
        "rollout_nll": ro_nll,
        "teacher_forced_dnll": tf_dnll,
        "rollout_dnll": ro_dnll,
        "rollout_gap_nll": ro_nll - tf_nll,
        "rollout_gap_dnll": ro_dnll - tf_dnll,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4.5 rollout fine-tune")
    parser.add_argument("--config", type=str, default="configs/phase4.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/phase4/checkpoints/param_final.pt",
        help="Param checkpoint to fine-tune",
    )
    args = parser.parse_args()

    setup_logging(name="phase4-rollout")
    logger = get_logger()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = _device_from_config(cfg["train"]["device"])
    set_seed(int(cfg["train"]["seed"]))

    out_dir = Path(cfg["output"]["out_dir"])
    ckpt_dir = Path(cfg["output"]["checkpoints_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load memory (for residual refit later)
    mem_path = Path(cfg["memory"]["output_npz"])
    memory = KMeansDeltaMemory.load(mem_path, device=device)
    z_dim = memory.dim

    # Load param model
    base_ckpt = torch.load(args.checkpoint, map_location=device)
    mcfg = cfg["model"]
    param = ParamDyn(
        z_dim=z_dim,
        hidden_dim=int(mcfg["hidden_dim"]),
        n_layers=int(mcfg["n_layers"]),
        dropout=float(mcfg["dropout"]),
        min_log_sigma=float(mcfg["min_log_sigma"]),
        max_log_sigma=float(mcfg["max_log_sigma"]),
    ).to(device)
    param.load_state_dict(base_ckpt["model"])
    param.eval()

    # Baseline stats and eval loaders
    baseline_mean, baseline_var = _fit_unconditional_baseline(cfg, device)
    eval_pairs = ZPairIterableDataset(
        lambda: iter_zdyn_pairs(
            zdyn_dir=cfg["data"]["zdyn_dir"],
            zdyn_index_path=cfg["data"]["zdyn_index"],
            splits_dir=cfg["data"]["splits_dir"],
            split="eval",
            min_duration_sec=float(cfg["data"]["min_duration_sec"]),
            seed=int(cfg["train"]["seed"]) + 7,
            max_pairs=None,
            sample_prob=1.0,
        )
    )
    eval_loader = DataLoader(
        eval_pairs,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=0,
        collate_fn=collate_zpairs,
        pin_memory=(device.type == "cuda"),
    )

    store = LatentStore(Path(cfg["data"]["zdyn_dir"]))
    utt_ids = sample_eval_utterance_ids(
        zdyn_index_path=cfg["data"]["zdyn_index"],
        splits_dir=cfg["data"]["splits_dir"],
        min_duration_sec=float(cfg["data"]["min_duration_sec"]),
        n_utterances=int(cfg["eval"]["rollout_eval_utts"]),
        seed=int(cfg["train"]["seed"]) + 321,
    )

    eval_cfg = cfg.get("eval", {})
    rollout_max_frames = int(eval_cfg.get("rollout_max_frames", 2000))
    rollout_step_alpha = float(eval_cfg.get("rollout_step_alpha", 1.0))
    rollout_clip = float(eval_cfg.get("rollout_clip_dz_l2", 5.0))

    pre_tf = _eval_teacher_forced(
        name="param_pre",
        predict_fn=param,
        loader=eval_loader,
        baseline_mean=baseline_mean,
        baseline_var=baseline_var,
        device=device,
        max_batches=int(eval_cfg.get("max_batches", 200)),
    )
    pre_ro = _eval_rollout(
        name="param_pre",
        predict_fn=param,
        store=store,
        utt_ids=utt_ids,
        max_frames=rollout_max_frames,
        baseline_mean=baseline_mean,
        baseline_var=baseline_var,
        device=device,
        step_alpha=rollout_step_alpha,
        clip_dz_l2=rollout_clip,
    )

    # Rollout fine-tune
    rtcfg = cfg.get("rollout_train", {})
    if not bool(rtcfg.get("enabled", True)):
        logger.info("[phase4.5] rollout_train.disabled; exiting after pre-eval")
        return 0

    k = int(rtcfg.get("k", 16))
    batch_size = int(rtcfg.get("batch_size", 256))
    max_steps = int(rtcfg.get("max_steps", 5000))
    segments_per_utt = int(rtcfg.get("segments_per_utt", 8))
    lr = float(rtcfg.get("lr", 2e-4))
    grad_clip = float(rtcfg.get("grad_clip_norm", 1.0))
    log_every = int(rtcfg.get("log_every", 100))
    step_alpha = float(rtcfg.get("step_alpha", rollout_step_alpha))
    clip_dz_l2 = float(rtcfg.get("clip_dz_l2", rollout_clip))
    rollout_weight = float(rtcfg.get("rollout_weight", 1.0))
    teacher_weight = float(rtcfg.get("teacher_weight", 0.0))
    state_weight = float(rtcfg.get("state_weight", 0.0))

    sched_p_start = float(rtcfg.get("sched_teacher_prob_start", 0.0))
    sched_p_end = float(rtcfg.get("sched_teacher_prob_end", 0.0))
    sched_warmup = int(rtcfg.get("sched_teacher_warmup_steps", 0))
    z_noise_std = float(rtcfg.get("z_noise_std", 0.0))

    seg_ds = ZPairIterableDataset(
        lambda: iter_rollout_segments(
            zdyn_dir=cfg["data"]["zdyn_dir"],
            zdyn_index_path=cfg["data"]["zdyn_index"],
            splits_dir=cfg["data"]["splits_dir"],
            split="train",
            min_duration_sec=float(cfg["data"]["min_duration_sec"]),
            k=k,
            segments_per_utt=segments_per_utt,
            seed=int(cfg["train"]["seed"]) + 1001,
            max_segments=None,
        )
    )
    seg_loader = DataLoader(
        seg_ds,
        batch_size=batch_size,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=collate_rollout_segments,
        pin_memory=(device.type == "cuda"),
    )

    opt = torch.optim.AdamW(param.parameters(), lr=lr, weight_decay=float(cfg["train"]["weight_decay"]))
    param.train()
    logger.info(
        f"[phase4.5] Fine-tuning param: k={k} batch={batch_size} steps={max_steps} lr={lr:g} "
        f"w_rollout={rollout_weight:g} w_teacher={teacher_weight:g} w_state={state_weight:g} "
        f"sched_p={sched_p_start:g}->{sched_p_end:g}@{sched_warmup} noise_std={z_noise_std:g}"
    )

    seg_iter = iter(seg_loader)
    for step in range(1, max_steps + 1):
        try:
            batch = next(seg_iter)
        except StopIteration:
            seg_iter = iter(seg_loader)
            batch = next(seg_iter)

        z0 = batch["z0"].to(device)  # [B,D]
        dz_seq = batch["dz_seq"].to(device)  # [B,K,D]
        # Ground-truth states for the segment.
        z_true = z0.unsqueeze(1) + torch.cumsum(dz_seq, dim=1)  # [B,K,D] = z_{t+1..t+K}
        z_true_prev = torch.cat([z0.unsqueeze(1), z_true[:, :-1]], dim=1)  # [B,K,D] = z_{t..t+K-1}

        z_hat = z0  # [B,D]

        opt.zero_grad(set_to_none=True)
        loss_roll = 0.0
        loss_tf = 0.0
        loss_state = 0.0
        n_used_true = 0

        p_teacher = _linear_schedule(step, sched_p_start, sched_p_end, sched_warmup)
        for i in range(k):
            dz_true = dz_seq[:, i]  # [B,D]

            # Scheduled sampling: choose model input state for this step.
            if p_teacher > 0:
                use_true = (torch.rand((z_hat.shape[0], 1), device=device) < p_teacher).to(dtype=z_hat.dtype)
                z_in = use_true * z_true_prev[:, i] + (1.0 - use_true) * z_hat
                n_used_true += int(use_true.sum().item())
            else:
                z_in = z_hat

            pred_roll = param(z_in)
            loss_roll = loss_roll + diag_gaussian_nll(dz_true, pred_roll).mean()

            if teacher_weight and teacher_weight > 0:
                pred_tf = param(z_true_prev[:, i])
                loss_tf = loss_tf + diag_gaussian_nll(dz_true, pred_tf).mean()

            dz_hat = pred_roll.mu * float(step_alpha)
            if clip_dz_l2 and clip_dz_l2 > 0:
                dz_hat = _clip_l2(dz_hat, float(clip_dz_l2))
            z_hat = z_hat + dz_hat
            if z_noise_std and z_noise_std > 0:
                z_hat = z_hat + float(z_noise_std) * torch.randn_like(z_hat)

            if state_weight and state_weight > 0:
                # Align rolled-out state to ground-truth state.
                loss_state = loss_state + torch.mean((z_hat - z_true[:, i]) ** 2)

        loss_roll = loss_roll / float(k)
        loss_tf = loss_tf / float(k) if teacher_weight and teacher_weight > 0 else loss_tf
        loss_state = loss_state / float(k) if state_weight and state_weight > 0 else loss_state

        loss = float(rollout_weight) * loss_roll + float(teacher_weight) * loss_tf + float(state_weight) * loss_state
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(param.parameters(), max_norm=grad_clip)
        opt.step()

        if log_every and step % log_every == 0:
            with torch.no_grad():
                log_sigma_mean = float(torch.clamp(param.log_sigma, float(mcfg["min_log_sigma"]), float(mcfg["max_log_sigma"])).mean().item())
                true_dz_l2 = float(torch.linalg.vector_norm(dz_seq.reshape(-1, dz_seq.shape[-1]), dim=-1).mean().item())
            logger.info(
                f"[phase4.5] step={step}/{max_steps} loss={float(loss.item()):.4f} "
                f"L_roll={float(loss_roll.item()):.4f} L_tf={float(loss_tf.item()):.4f} L_state={float(loss_state.item()):.4f} "
                f"p_teacher={p_teacher:.3f} used_true={n_used_true}/{(z0.shape[0]*k)} "
                f"log_sigma_mean={log_sigma_mean:.3f} true_dz_l2={true_dz_l2:.3f}"
            )

    # Save finetuned checkpoint
    out_ckpt = ckpt_dir / "param_rollout_finetuned.pt"
    torch.save({"model": param.state_dict(), "config": cfg}, out_ckpt)
    logger.info(f"[phase4.5] Wrote checkpoint: {out_ckpt}")

    # Re-evaluate param
    param.eval()
    post_tf = _eval_teacher_forced(
        name="param_post",
        predict_fn=param,
        loader=eval_loader,
        baseline_mean=baseline_mean,
        baseline_var=baseline_var,
        device=device,
        max_batches=int(eval_cfg.get("max_batches", 200)),
    )
    post_ro = _eval_rollout(
        name="param_post",
        predict_fn=param,
        store=store,
        utt_ids=utt_ids,
        max_frames=rollout_max_frames,
        baseline_mean=baseline_mean,
        baseline_var=baseline_var,
        device=device,
        step_alpha=rollout_step_alpha,
        clip_dz_l2=rollout_clip,
    )

    # Fit residual memory with updated param
    logger.info("[phase4.5] Fitting residual memory from finetuned param")
    counts = torch.zeros((memory.n_clusters,), dtype=torch.int64, device=device)
    sum_r = torch.zeros((memory.n_clusters, z_dim), dtype=torch.float64, device=device)
    sum_r2 = torch.zeros((memory.n_clusters, z_dim), dtype=torch.float64, device=device)

    fit_pairs = int(cfg["memory"]["max_fit_pairs"])
    n = 0
    with torch.no_grad():
        for p in iter_zdyn_pairs(
            zdyn_dir=cfg["data"]["zdyn_dir"],
            zdyn_index_path=cfg["data"]["zdyn_index"],
            splits_dir=cfg["data"]["splits_dir"],
            split="train",
            min_duration_sec=float(cfg["data"]["min_duration_sec"]),
            seed=int(cfg["train"]["seed"]) + 2002,
            max_pairs=fit_pairs,
            sample_prob=1.0,
        ):
            z_prev = torch.from_numpy(p.z_prev.astype(np.float32, copy=False)).to(device)
            dz = torch.from_numpy(p.dz.astype(np.float32, copy=False)).to(device)
            mu = param(z_prev.unsqueeze(0)).mu.squeeze(0)
            r = (dz - mu).to(dtype=torch.float64)
            idx = memory.nearest_index(z_prev.unsqueeze(0)).squeeze(0)
            counts[idx] += 1
            sum_r[idx] += r
            sum_r2[idx] += r * r
            n += 1
            if n >= fit_pairs:
                break

    counts_f = counts.to(dtype=torch.float64).clamp_min(1.0).unsqueeze(-1)
    res_mean = (sum_r / counts_f).to(dtype=torch.float32)
    res_var = (sum_r2 / counts_f - (res_mean.to(dtype=torch.float64) ** 2)).to(dtype=torch.float32).clamp_min(1e-8)
    resid_path = out_dir / "zdyn_memory_residual_rollout_finetuned.npz"
    np.savez_compressed(
        str(resid_path),
        centroids=memory.centroids.detach().cpu().numpy(),
        res_mean=res_mean.detach().cpu().numpy(),
        res_var=res_var.detach().cpu().numpy(),
        counts=counts.detach().cpu().numpy(),
    )
    resid_mem = KMeansDeltaMemory.load(resid_path, device=device)

    log_sigma_shared = torch.clamp(param.log_sigma, float(mcfg["min_log_sigma"]), float(mcfg["max_log_sigma"]))

    def resid_add_predict(zp: torch.Tensor) -> DiagGaussianDelta:
        mu = param(zp).mu + resid_mem.predict_mean(
            zp, topk=int(cfg["memory"].get("topk", 8)), temperature=float(cfg["memory"].get("temperature", 1.0))
        )
        return DiagGaussianDelta(mu=mu, log_sigma=log_sigma_shared)

    resid_tf = _eval_teacher_forced(
        name="resid_add_post",
        predict_fn=resid_add_predict,
        loader=eval_loader,
        baseline_mean=baseline_mean,
        baseline_var=baseline_var,
        device=device,
        max_batches=int(eval_cfg.get("max_batches", 200)),
    )
    resid_ro = _eval_rollout(
        name="resid_add_post",
        predict_fn=resid_add_predict,
        store=store,
        utt_ids=utt_ids,
        max_frames=rollout_max_frames,
        baseline_mean=baseline_mean,
        baseline_var=baseline_var,
        device=device,
        step_alpha=rollout_step_alpha,
        clip_dz_l2=rollout_clip,
    )

    summary = {
        "pre": {"teacher_forced": pre_tf, "rollout": pre_ro},
        "post": {"teacher_forced": post_tf, "rollout": post_ro},
        "post_resid_add": {"teacher_forced": resid_tf, "rollout": resid_ro, "residual_memory_npz": str(resid_path)},
        "checkpoints": {"base": str(args.checkpoint), "finetuned": str(out_ckpt)},
        "rollout_train": {
            "k": k,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "step_alpha": step_alpha,
            "clip_dz_l2": clip_dz_l2,
            "rollout_weight": rollout_weight,
            "teacher_weight": teacher_weight,
            "state_weight": state_weight,
            "sched_teacher_prob_start": sched_p_start,
            "sched_teacher_prob_end": sched_p_end,
            "sched_teacher_warmup_steps": sched_warmup,
            "z_noise_std": z_noise_std,
        },
    }
    out_json = out_dir / "phase4_rollout_finetune_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    logger.info(f"[phase4.5] Wrote summary: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
