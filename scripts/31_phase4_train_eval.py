#!/usr/bin/env python3
"""
Phase 4: Train/evaluate param vs memory vs hybrid dynamics models for z_dyn.

Outputs:
  - outputs/phase4/metrics.json
  - outputs/phase4/train_log.jsonl
  - outputs/phase4/checkpoints/*.pt

Usage:
  uv run python scripts/31_phase4_train_eval.py --config configs/phase4.yaml
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

from phase4.data import ZPairIterableDataset, collate_zpairs, iter_zdyn_pairs, sample_eval_utterance_ids
from phase4.memory import KMeansDeltaMemory
from phase4.models import ParamDyn, HybridDyn, DiagGaussianDelta, diag_gaussian_nll, count_parameters


def _device_from_config(device: str) -> torch.device:
    d = str(device).lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


@torch.no_grad()
def _eval_teacher_forced(
    *,
    model_name: str,
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
    nb = 0
    for batch in loader:
        z_prev = batch["z_prev"].to(device)
        dz = batch["dz"].to(device)

        pred: DiagGaussianDelta = predict_fn(z_prev)
        nll = diag_gaussian_nll(dz, pred)  # [B]
        nllb = baseline_nll(dz)  # [B]

        # Direction cosine of predicted mean vs target
        mu = pred.mu
        num = (mu * dz).sum(dim=-1)
        den = torch.linalg.vector_norm(mu, dim=-1) * torch.linalg.vector_norm(dz, dim=-1)
        cos = num / torch.clamp(den, min=1e-12)

        nll_sum += float(nll.mean().item())
        nllb_sum += float(nllb.mean().item())
        cos_sum += float(cos.mean().item())
        n += 1
        nb += int(z_prev.shape[0])
        if max_batches and n >= int(max_batches):
            break

    return {
        "model": model_name,
        "n_batches": n,
        "n_samples": nb,
        "nll": nll_sum / max(n, 1),
        "nll_baseline": nllb_sum / max(n, 1),
        "dnll": (nll_sum - nllb_sum) / max(n, 1),
        "direction_cos": cos_sum / max(n, 1),
    }


@torch.no_grad()
def _eval_rollout(
    *,
    model_name: str,
    predict_fn,
    store: LatentStore,
    utt_ids: list[str],
    max_frames: int,
    baseline_mean: torch.Tensor,
    baseline_var: torch.Tensor,
    device: torch.device,
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

    # Teacher-forced vs free-running:
    # - teacher forced: z_prev = z_true[t]
    # - rollout:       z_prev = z_hat[t] where z_hat[t+1] = z_hat[t] + mu(z_hat[t])
    tf_nll_sum = 0.0
    ro_nll_sum = 0.0
    tf_b_sum = 0.0
    ro_b_sum = 0.0
    n_steps = 0

    for utt in utt_ids:
        z = store.get_latents(utt).astype(np.float32, copy=False)
        if z.shape[0] < 2:
            continue
        t_max = min(int(max_frames), int(z.shape[0]))

        z_true = torch.from_numpy(z[:t_max]).to(device)
        dz_true = z_true[1:] - z_true[:-1]  # [T-1,D]

        z_hat = z_true[0].clone()

        for t in range(t_max - 1):
            dz_t = dz_true[t : t + 1]  # [1,D]

            # Teacher-forced
            pred_tf = predict_fn(z_true[t : t + 1])
            tf_nll_sum += float(diag_gaussian_nll(dz_t, pred_tf).item())
            tf_b_sum += float(baseline_nll(dz_t).item())

            # Rollout
            pred_ro = predict_fn(z_hat.unsqueeze(0))
            ro_nll_sum += float(diag_gaussian_nll(dz_t, pred_ro).item())
            ro_b_sum += float(baseline_nll(dz_t).item())
            z_hat = z_hat + pred_ro.mu.squeeze(0)

            n_steps += 1

    if n_steps == 0:
        return {"model": model_name, "n_steps": 0, "rollout_gap_dnll": float("nan")}

    tf_nll = tf_nll_sum / n_steps
    ro_nll = ro_nll_sum / n_steps
    tf_dnll = (tf_nll_sum - tf_b_sum) / n_steps
    ro_dnll = (ro_nll_sum - ro_b_sum) / n_steps
    return {
        "model": model_name,
        "n_steps": n_steps,
        "teacher_forced_nll": tf_nll,
        "rollout_nll": ro_nll,
        "teacher_forced_dnll": tf_dnll,
        "rollout_dnll": ro_dnll,
        "rollout_gap_nll": ro_nll - tf_nll,
        "rollout_gap_dnll": ro_dnll - tf_dnll,
    }


def _fit_unconditional_baseline(
    *,
    cfg: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Fit unconditional diag Gaussian on train deltas.
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
        max_pairs=cfg["memory"]["max_fit_pairs"],
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 train/eval for z_dyn dynamics with memory")
    parser.add_argument("--config", type=str, default="configs/phase4.yaml")
    args = parser.parse_args()

    setup_logging(name="phase4")
    logger = get_logger()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = _device_from_config(cfg["train"]["device"])
    set_seed(int(cfg["train"]["seed"]))

    out_dir = Path(cfg["output"]["out_dir"])
    ckpt_dir = Path(cfg["output"]["checkpoints_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load memory
    mem_path = Path(cfg["memory"]["output_npz"])
    if not mem_path.exists():
        raise FileNotFoundError(f"Memory file not found: {mem_path}. Run scripts/30_phase4_fit_memory.py first.")
    memory = KMeansDeltaMemory.load(mem_path, device=device)
    z_dim = memory.dim
    logger.info(f"[phase4] device={device.type} z_dim={z_dim} clusters={memory.n_clusters}")

    # Baseline stats
    baseline_mean, baseline_var = _fit_unconditional_baseline(cfg=cfg, device=device)

    # Data
    train_it = lambda: iter_zdyn_pairs(
        zdyn_dir=cfg["data"]["zdyn_dir"],
        zdyn_index_path=cfg["data"]["zdyn_index"],
        splits_dir=cfg["data"]["splits_dir"],
        split="train",
        min_duration_sec=float(cfg["data"]["min_duration_sec"]),
        seed=int(cfg["train"]["seed"]),
        max_pairs=None,
        sample_prob=1.0,
    )
    train_ds = ZPairIterableDataset(train_it)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=collate_zpairs,
        pin_memory=(device.type == "cuda"),
    )

    eval_it = lambda: iter_zdyn_pairs(
        zdyn_dir=cfg["data"]["zdyn_dir"],
        zdyn_index_path=cfg["data"]["zdyn_index"],
        splits_dir=cfg["data"]["splits_dir"],
        split="eval",
        min_duration_sec=float(cfg["data"]["min_duration_sec"]),
        seed=int(cfg["train"]["seed"]) + 1,
        max_pairs=None,
        sample_prob=1.0,
    )
    eval_ds = ZPairIterableDataset(eval_it)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=0,
        collate_fn=collate_zpairs,
        pin_memory=(device.type == "cuda"),
    )

    # Models
    mcfg = cfg["model"]
    param = ParamDyn(
        z_dim=z_dim,
        hidden_dim=int(mcfg["hidden_dim"]),
        n_layers=int(mcfg["n_layers"]),
        dropout=float(mcfg["dropout"]),
        min_log_sigma=float(mcfg["min_log_sigma"]),
        max_log_sigma=float(mcfg["max_log_sigma"]),
    ).to(device)
    hybrid = HybridDyn(
        z_dim=z_dim,
        hidden_dim=int(mcfg["hidden_dim"]),
        n_layers=int(mcfg["n_layers"]),
        dropout=float(mcfg["dropout"]),
        gate_hidden_dim=int(mcfg["gate_hidden_dim"]),
        min_log_sigma=float(mcfg["min_log_sigma"]),
        max_log_sigma=float(mcfg["max_log_sigma"]),
        memory=memory,
    ).to(device)

    logger.info(f"[phase4] param params={count_parameters(param):,}")
    logger.info(f"[phase4] hybrid params={count_parameters(hybrid):,} (+memory table {memory.n_clusters}x{z_dim})")

    opt = torch.optim.AdamW(
        list(param.parameters()) + list(hybrid.parameters()),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    log_path = Path(cfg["output"]["train_log_jsonl"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Train both param and hybrid jointly (shared batches) to compare fairly.
    step = 0
    max_steps = int(cfg["train"]["max_steps"])
    log_every = int(cfg["train"]["log_every"])
    grad_clip = float(cfg["train"]["grad_clip_norm"])

    train_iter = iter(train_loader)
    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        z_prev = batch["z_prev"].to(device)
        dz = batch["dz"].to(device)

        param.train()
        hybrid.train()
        opt.zero_grad(set_to_none=True)

        p_params = param(z_prev)
        h_params = hybrid(z_prev)
        loss_p = diag_gaussian_nll(dz, p_params).mean()
        loss_h = diag_gaussian_nll(dz, h_params).mean()
        loss = loss_p + loss_h
        loss.backward()

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(list(param.parameters()) + list(hybrid.parameters()), max_norm=grad_clip)
        opt.step()
        step += 1

        if log_every and step % log_every == 0:
            row = {
                "step": step,
                "loss_param": float(loss_p.item()),
                "loss_hybrid": float(loss_h.item()),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            logger.info(f"[phase4] step={step} loss_param={row['loss_param']:.4f} loss_hybrid={row['loss_hybrid']:.4f}")

    # Save checkpoints
    torch.save({"model": param.state_dict(), "config": cfg}, ckpt_dir / "param_final.pt")
    torch.save({"model": hybrid.state_dict(), "config": cfg}, ckpt_dir / "hybrid_final.pt")

    # Evaluate teacher-forced on eval
    def mem_predict(zp: torch.Tensor) -> DiagGaussianDelta:
        mu = memory.predict_mean(zp)
        # Use per-cluster variance for NLL (converted to log_sigma)
        var = memory.predict_var(zp)
        log_sigma = 0.5 * torch.log(var.clamp_min(1e-8))
        return DiagGaussianDelta(mu=mu, log_sigma=log_sigma)

    param_metrics = _eval_teacher_forced(
        model_name="param",
        predict_fn=param,
        loader=eval_loader,
        baseline_mean=baseline_mean,
        baseline_var=baseline_var,
        device=device,
        max_batches=int(cfg["eval"]["max_batches"]),
    )
    hybrid_metrics = _eval_teacher_forced(
        model_name="hybrid",
        predict_fn=hybrid,
        loader=eval_loader,
        baseline_mean=baseline_mean,
        baseline_var=baseline_var,
        device=device,
        max_batches=int(cfg["eval"]["max_batches"]),
    )
    mem_metrics = _eval_teacher_forced(
        model_name="memory",
        predict_fn=mem_predict,
        loader=eval_loader,
        baseline_mean=baseline_mean,
        baseline_var=baseline_var,
        device=device,
        max_batches=int(cfg["eval"]["max_batches"]),
    )

    # Rollout eval on a small subset of eval utterances
    store = LatentStore(Path(cfg["data"]["zdyn_dir"]))
    utt_ids = sample_eval_utterance_ids(
        zdyn_index_path=cfg["data"]["zdyn_index"],
        splits_dir=cfg["data"]["splits_dir"],
        min_duration_sec=float(cfg["data"]["min_duration_sec"]),
        n_utterances=int(cfg["eval"]["rollout_eval_utts"]),
        seed=int(cfg["train"]["seed"]) + 123,
    )
    rollout_max_frames = int(cfg["eval"]["rollout_max_frames"])

    mem_roll = _eval_rollout(
        model_name="memory",
        predict_fn=mem_predict,
        store=store,
        utt_ids=utt_ids,
        max_frames=rollout_max_frames,
        baseline_mean=baseline_mean,
        baseline_var=baseline_var,
        device=device,
    )
    param_roll = _eval_rollout(
        model_name="param",
        predict_fn=param,
        store=store,
        utt_ids=utt_ids,
        max_frames=rollout_max_frames,
        baseline_mean=baseline_mean,
        baseline_var=baseline_var,
        device=device,
    )
    hybrid_roll = _eval_rollout(
        model_name="hybrid",
        predict_fn=hybrid,
        store=store,
        utt_ids=utt_ids,
        max_frames=rollout_max_frames,
        baseline_mean=baseline_mean,
        baseline_var=baseline_var,
        device=device,
    )

    metrics = {
        "teacher_forced": [mem_metrics, param_metrics, hybrid_metrics],
        "rollout": [mem_roll, param_roll, hybrid_roll],
    }
    Path(cfg["output"]["metrics_json"]).write_text(json.dumps(metrics, indent=2))
    logger.info(f"[phase4] Wrote metrics: {cfg['output']['metrics_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

