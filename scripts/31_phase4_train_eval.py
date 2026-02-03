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
from phase4.models import (
    ParamDyn,
    HybridDyn,
    ResidualMemDyn,
    GatedResidualMemDyn,
    DiagGaussianDelta,
    diag_gaussian_nll,
    count_parameters,
)


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
    gate_sum = 0.0
    gate_sq_sum = 0.0
    gate_n = 0
    for batch in loader:
        z_prev = batch["z_prev"].to(device)
        dz = batch["dz"].to(device)

        pred: DiagGaussianDelta = predict_fn(z_prev)
        nll = diag_gaussian_nll(dz, pred)  # [B]
        nllb = baseline_nll(dz)  # [B]

        # Optional gate diagnostics (for hybrid/gated residual models).
        if hasattr(predict_fn, "gate"):
            try:
                gate_raw = predict_fn.gate(z_prev)  # [B,1] or [B]
                gate = torch.sigmoid(gate_raw).reshape(-1)
                gate_sum += float(gate.sum().item())
                gate_sq_sum += float((gate * gate).sum().item())
                gate_n += int(gate.numel())
            except Exception:
                pass

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

    out = {
        "model": model_name,
        "n_batches": n,
        "n_samples": nb,
        "nll": nll_sum / max(n, 1),
        "nll_baseline": nllb_sum / max(n, 1),
        "dnll": (nll_sum - nllb_sum) / max(n, 1),
        "direction_cos": cos_sum / max(n, 1),
    }
    if gate_n > 0:
        gate_mean = gate_sum / gate_n
        gate_var = max(gate_sq_sum / gate_n - gate_mean * gate_mean, 0.0)
        out["gate_mean"] = gate_mean
        out["gate_std"] = float(gate_var**0.5)
    return out


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

    # Teacher-forced vs free-running:
    # - teacher forced: z_prev = z_true[t]
    # - rollout:       z_prev = z_hat[t] where z_hat[t+1] = z_hat[t] + mu(z_hat[t])
    tf_nll_sum = 0.0
    ro_nll_sum = 0.0
    tf_b_sum = 0.0
    ro_b_sum = 0.0
    n_steps = 0
    n_clipped = 0
    dz_l2_sum = 0.0

    global_max_z_hat_l2 = 0.0
    global_first_nonfinite_step = None
    any_nonfinite = False

    for utt in utt_ids:
        z = store.get_latents(utt).astype(np.float32, copy=False)
        if z.shape[0] < 2:
            continue
        t_max = min(int(max_frames), int(z.shape[0]))

        z_true = torch.from_numpy(z[:t_max]).to(device)
        dz_true = z_true[1:] - z_true[:-1]  # [T-1,D]

        z_hat = z_true[0].clone()
        max_z_hat_l2 = float(torch.linalg.vector_norm(z_hat).item())
        first_nonfinite_step = None

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

            dz_hat = pred_ro.mu.squeeze(0)
            if step_alpha != 1.0:
                dz_hat = dz_hat * float(step_alpha)
            if clip_dz_l2 and clip_dz_l2 > 0:
                l2 = float(torch.linalg.vector_norm(dz_hat).item())
                dz_l2_sum += l2
                if l2 > float(clip_dz_l2):
                    dz_hat = dz_hat * (float(clip_dz_l2) / max(l2, 1e-12))
                    n_clipped += 1
            z_hat = z_hat + dz_hat
            max_z_hat_l2 = max(max_z_hat_l2, float(torch.linalg.vector_norm(z_hat).item()))
            if not torch.isfinite(z_hat).all():
                first_nonfinite_step = t + 1
                break

            n_steps += 1

        global_max_z_hat_l2 = max(global_max_z_hat_l2, max_z_hat_l2)
        if first_nonfinite_step is not None:
            any_nonfinite = True
            if global_first_nonfinite_step is None or first_nonfinite_step < global_first_nonfinite_step:
                global_first_nonfinite_step = first_nonfinite_step

    if n_steps == 0:
        return {
            "model": model_name,
            "n_steps": 0,
            "rollout_nonfinite": True,
            "first_nonfinite_step": 0,
            "max_z_hat_l2": float("nan"),
            "rollout_gap_dnll": float("nan"),
        }

    tf_nll = tf_nll_sum / n_steps
    ro_nll = ro_nll_sum / n_steps
    tf_dnll = (tf_nll_sum - tf_b_sum) / n_steps
    ro_dnll = (ro_nll_sum - ro_b_sum) / n_steps
    return {
        "model": model_name,
        "n_steps": n_steps,
        "rollout_nonfinite": any_nonfinite,
        "first_nonfinite_step": global_first_nonfinite_step,
        "max_z_hat_l2": global_max_z_hat_l2,
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
    memory_topk = int(cfg["memory"].get("topk", 1))
    memory_temp = float(cfg["memory"].get("temperature", 1.0))
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
        memory_topk=memory_topk,
        memory_temperature=memory_temp,
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

    # Train both param and hybrid(delta-memory) jointly (shared batches) to compare fairly.
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
            with torch.no_grad():
                gate_mean = float(torch.sigmoid(hybrid.gate(z_prev)).mean().item())
            row = {
                "step": step,
                "loss_param": float(loss_p.item()),
                "loss_hybrid": float(loss_h.item()),
                "hybrid_gate_mean": gate_mean,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            logger.info(
                f"[phase4] step={step} loss_param={row['loss_param']:.4f} loss_hybrid={row['loss_hybrid']:.4f} "
                f"hybrid_gate_mean={row['hybrid_gate_mean']:.3f}"
            )

    # Save checkpoints
    torch.save({"model": param.state_dict(), "config": cfg}, ckpt_dir / "param_final.pt")
    torch.save({"model": hybrid.state_dict(), "config": cfg}, ckpt_dir / "hybrid_final.pt")

    # Fit residual memory: r = Δz - μ_param(z)
    # Default to enabled so older configs still run the improved Phase 4b ablation.
    residual_enabled = bool(cfg["memory"].get("residual_enabled", True))
    residual_memory = None
    residual_path = Path(cfg["memory"].get("residual_output_npz", out_dir / "zdyn_memory_residual.npz"))
    if residual_enabled:
        logger.info("[phase4] Fitting residual memory statistics: r = Δz - μ_param(z)")
        # Hard-assign to nearest centroid for stats (retrieval can still be soft top-k).
        counts = torch.zeros((memory.n_clusters,), dtype=torch.int64, device=device)
        sum_r = torch.zeros((memory.n_clusters, z_dim), dtype=torch.float64, device=device)
        sum_r2 = torch.zeros((memory.n_clusters, z_dim), dtype=torch.float64, device=device)

        fit_pairs = int(cfg["memory"]["max_fit_pairs"])
        n = 0
        param.eval()
        with torch.no_grad():
            it = iter_zdyn_pairs(
                zdyn_dir=cfg["data"]["zdyn_dir"],
                zdyn_index_path=cfg["data"]["zdyn_index"],
                splits_dir=cfg["data"]["splits_dir"],
                split="train",
                min_duration_sec=float(cfg["data"]["min_duration_sec"]),
                seed=int(cfg["train"]["seed"]) + 999,
                max_pairs=fit_pairs,
                sample_prob=1.0,
            )
            for p in it:
                z_prev_np = p.z_prev.astype(np.float32, copy=False)
                dz_np = p.dz.astype(np.float32, copy=False)
                z_prev = torch.from_numpy(z_prev_np).to(device)
                dz = torch.from_numpy(dz_np).to(device)
                mu = param(z_prev.unsqueeze(0)).mu.squeeze(0)
                r = (dz - mu).to(dtype=torch.float64)
                idx = memory.nearest_index(z_prev.unsqueeze(0)).squeeze(0)
                counts[idx] += 1
                sum_r[idx] += r
                sum_r2[idx] += r * r
                n += 1
                if n >= fit_pairs:
                    break

        if int(counts.sum().item()) == 0:
            logger.warning("[phase4] Residual memory fit found 0 pairs; disabling residual memory.")
            residual_enabled = False
        else:
            counts_f = counts.to(dtype=torch.float64).clamp_min(1.0).unsqueeze(-1)
            res_mean = (sum_r / counts_f).to(dtype=torch.float32)
            res_var = (sum_r2 / counts_f - (res_mean.to(dtype=torch.float64) ** 2)).to(dtype=torch.float32).clamp_min(1e-8)

            residual_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                str(residual_path),
                centroids=memory.centroids.detach().cpu().numpy(),
                res_mean=res_mean.detach().cpu().numpy(),
                res_var=res_var.detach().cpu().numpy(),
                counts=counts.detach().cpu().numpy(),
            )
            logger.info(f"[phase4] Wrote residual memory npz: {residual_path}")
            residual_memory = KMeansDeltaMemory.load(residual_path, device=device)

    # Optional: train a gated residual memory model (gate only; param frozen)
    gated_resid = None
    if residual_enabled and residual_memory is not None:
        gated_resid = GatedResidualMemDyn(
            param=param,
            residual_memory=residual_memory,
            gate_hidden_dim=int(mcfg["gate_hidden_dim"]),
            memory_topk=memory_topk,
            memory_temperature=memory_temp,
        ).to(device)
        for p in gated_resid.param.parameters():
            p.requires_grad = False
        logger.info(f"[phase4] gated_resid params={count_parameters(gated_resid):,} (gate only; param frozen)")

        gate_steps = int(cfg["train"].get("gate_steps", 5000))
        if gate_steps > 0:
            gate_opt = torch.optim.AdamW(
                gated_resid.gate.parameters(),
                lr=float(cfg["train"]["lr"]),
                weight_decay=float(cfg["train"]["weight_decay"]),
            )
            train_iter = iter(train_loader)
            for s in range(gate_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                z_prev = batch["z_prev"].to(device)
                dz = batch["dz"].to(device)
                gated_resid.train()
                gate_opt.zero_grad(set_to_none=True)
                pred = gated_resid(z_prev)
                loss_gate = diag_gaussian_nll(dz, pred).mean()
                loss_gate.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(gated_resid.gate.parameters(), max_norm=grad_clip)
                gate_opt.step()
                if log_every and (s + 1) % log_every == 0:
                    with torch.no_grad():
                        g_mean = float(torch.sigmoid(gated_resid.gate(z_prev)).mean().item())
                    logger.info(
                        f"[phase4] gate step={s+1}/{gate_steps} loss_gate={float(loss_gate.item()):.4f} gate_mean={g_mean:.3f}"
                    )

        torch.save({"model": gated_resid.state_dict(), "config": cfg}, ckpt_dir / "gated_resid_final.pt")

    # Evaluate teacher-forced on eval
    log_sigma_shared = torch.clamp(param.log_sigma, float(mcfg["min_log_sigma"]), float(mcfg["max_log_sigma"]))

    def mem_predict(zp: torch.Tensor) -> DiagGaussianDelta:
        mu = memory.predict_mean(zp, topk=memory_topk, temperature=memory_temp)
        return DiagGaussianDelta(mu=mu, log_sigma=log_sigma_shared)

    def resid_add_predict(zp: torch.Tensor) -> DiagGaussianDelta:
        if residual_memory is None:
            raise RuntimeError("residual_memory not available")
        mu = param(zp).mu + residual_memory.predict_mean(zp, topk=memory_topk, temperature=memory_temp)
        return DiagGaussianDelta(mu=mu, log_sigma=log_sigma_shared)

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

    extra_tf = []
    if residual_memory is not None:
        extra_tf.append(
            _eval_teacher_forced(
                model_name="resid_add",
                predict_fn=resid_add_predict,
                loader=eval_loader,
                baseline_mean=baseline_mean,
                baseline_var=baseline_var,
                device=device,
                max_batches=int(cfg["eval"]["max_batches"]),
            )
        )
    if gated_resid is not None:
        extra_tf.append(
            _eval_teacher_forced(
                model_name="gated_resid",
                predict_fn=gated_resid,
                loader=eval_loader,
                baseline_mean=baseline_mean,
                baseline_var=baseline_var,
                device=device,
                max_batches=int(cfg["eval"]["max_batches"]),
            )
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
    # Defaults are chosen to prevent NaN divergence in free-running rollouts while
    # minimally affecting typical steps (Δz l2 ~ O(1) in our runs).
    rollout_step_alpha = float(cfg.get("eval", {}).get("rollout_step_alpha", 1.0))
    rollout_clip_dz_l2 = float(cfg.get("eval", {}).get("rollout_clip_dz_l2", 5.0))

    mem_roll = _eval_rollout(
        model_name="memory",
        predict_fn=mem_predict,
        store=store,
        utt_ids=utt_ids,
        max_frames=rollout_max_frames,
        baseline_mean=baseline_mean,
        baseline_var=baseline_var,
        device=device,
        step_alpha=rollout_step_alpha,
        clip_dz_l2=rollout_clip_dz_l2,
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
        step_alpha=rollout_step_alpha,
        clip_dz_l2=rollout_clip_dz_l2,
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
        step_alpha=rollout_step_alpha,
        clip_dz_l2=rollout_clip_dz_l2,
    )

    extra_roll = []
    if residual_memory is not None:
        extra_roll.append(
            _eval_rollout(
                model_name="resid_add",
                predict_fn=resid_add_predict,
                store=store,
                utt_ids=utt_ids,
                max_frames=rollout_max_frames,
                baseline_mean=baseline_mean,
                baseline_var=baseline_var,
                device=device,
                step_alpha=rollout_step_alpha,
                clip_dz_l2=rollout_clip_dz_l2,
            )
        )
    if gated_resid is not None:
        extra_roll.append(
            _eval_rollout(
                model_name="gated_resid",
                predict_fn=gated_resid,
                store=store,
                utt_ids=utt_ids,
                max_frames=rollout_max_frames,
                baseline_mean=baseline_mean,
                baseline_var=baseline_var,
                device=device,
                step_alpha=rollout_step_alpha,
                clip_dz_l2=rollout_clip_dz_l2,
            )
        )

    metrics = {
        "teacher_forced": [mem_metrics, param_metrics, hybrid_metrics, *extra_tf],
        "rollout": [mem_roll, param_roll, hybrid_roll, *extra_roll],
    }
    Path(cfg["output"]["metrics_json"]).write_text(json.dumps(metrics, indent=2))
    logger.info(f"[phase4] Wrote metrics: {cfg['output']['metrics_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
