#!/usr/bin/env python3
"""
Stage 2 - Experiment 6B: Supervised rollout projector (train on model errors).

Motivation:
Score-based denoisers trained on isotropic Gaussian noise can fail to help
rollouts because the actual dynamics-model error distribution is structured
(especially angular drift) and context-dependent.

This experiment trains a projector network on the dynamics model's own rollout
errors:
  - Freeze a Phase 1 dynamics model (MDN or vMF).
  - Sample rollout segments (context window + K future states).
  - Rollout the dynamics model, and after each step train a projector to map
      (context_flat, z_hat) -> z_true
    via z_corr = z_hat + projector(...).
  - Evaluate injection diagnostic with/without projector correction.

Usage:
  uv run python scripts/tier2_exp6b_projector_correction.py \
      --config configs/tier2_exp6b_projector_correction.yaml \
      --checkpoint outputs/phase1/checkpoints/mdn_k1_final.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import yaml

from experiment import register_run, finalize_run
from phase0.data.io import LatentStore, load_latents_index
from phase0.utils.logging import setup_logging
from phase0.utils.seed import set_seed, get_rng
from phase1.checkpoints import load_phase1_checkpoint
from phase1.data import sample_eval_utterances
from phase1.train_eval import _device_from_config, fit_unconditional_baseline, fit_factorized_baseline
from stage2.projector_model import ContextStateProjector, apply_projector


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _iter_rollout_segments_from_zarr(
    *,
    latents_dir: str | Path,
    latents_index_path: str | Path,
    splits_dir: str | Path,
    split: str,
    window_size: int,
    horizon_k: int,
    k_steps: int,
    segments_per_utt: int,
    seed: int,
    max_segments: int | None = None,
    min_duration_sec: float = 3.0,
):
    """
    Yield {z_window [W,D], z_seq [K,D]} for training projector.

    z_seq[s] is the *true* state at step s (starting at time t0+W).
    """
    from phase1.data import iter_rollout_segments

    for seg in iter_rollout_segments(
        latents_dir=latents_dir,
        latents_index_path=latents_index_path,
        splits_dir=splits_dir,
        split=split,
        window_size=window_size,
        horizon_k=horizon_k,
        k_steps=k_steps,
        segments_per_utt=segments_per_utt,
        seed=seed,
        max_segments=max_segments,
        min_duration_sec=min_duration_sec,
    ):
        z_window = seg["z_window"].astype(np.float32, copy=False)  # [W,D]
        z_seq_full = seg["z_seq"].astype(np.float32, copy=False)   # [K+1,D]
        z_seq = z_seq_full[:k_steps]  # [K,D] targets for each step
        yield {"z_window": z_window, "z_seq": z_seq}


def _collate_segments(items: list[dict]) -> dict:
    z_window = np.stack([it["z_window"] for it in items], axis=0).astype(np.float32, copy=False)  # [B,W,D]
    z_seq = np.stack([it["z_seq"] for it in items], axis=0).astype(np.float32, copy=False)  # [B,K,D]
    return {"z_window": torch.from_numpy(z_window), "z_seq": torch.from_numpy(z_seq)}


def train_projector(
    *,
    projector: ContextStateProjector,
    dynamics_model: object,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
    logger,
) -> dict:
    train_cfg = cfg["projector_train"]
    data_cfg = cfg["data"]
    inj_cfg = cfg["injection"]

    window_size = int(inj_cfg["window_size"])
    horizon_k = int(inj_cfg.get("horizon_k", 1))
    unroll_k = int(train_cfg.get("unroll_k", 4))

    batch_size = int(train_cfg["batch_size"])
    max_steps = int(train_cfg["max_steps"])
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])
    grad_clip_norm = float(train_cfg["grad_clip_norm"])
    log_every = int(train_cfg["log_every"])
    save_every = int(train_cfg["save_every"])
    segments_per_utt = int(train_cfg.get("segments_per_utt", 2))
    max_segments = train_cfg.get("max_segments")
    min_duration_sec = float(train_cfg.get("min_duration_sec", 3.0))

    ckpt_dir = out_dir / "projector" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Freeze dynamics model
    if hasattr(dynamics_model, "eval"):
        dynamics_model.eval()
    for p in getattr(dynamics_model, "parameters", lambda: [])():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=weight_decay)

    seed = int(cfg.get("seed", 42))
    rng = get_rng(seed + 60_000)

    def make_loader(epoch_seed: int):
        it = lambda: _iter_rollout_segments_from_zarr(
            latents_dir=data_cfg["latents_dir"],
            latents_index_path=data_cfg["latents_index"],
            splits_dir=data_cfg["splits_dir"],
            split="train",
            window_size=window_size,
            horizon_k=horizon_k,
            k_steps=unroll_k,
            segments_per_utt=segments_per_utt,
            seed=epoch_seed,
            max_segments=None if max_segments is None else int(max_segments),
            min_duration_sec=min_duration_sec,
        )

        class _DS(torch.utils.data.IterableDataset):
            def __iter__(self):
                return it()

        return torch.utils.data.DataLoader(
            _DS(),
            batch_size=batch_size,
            num_workers=int(train_cfg.get("num_workers", 0)),
            collate_fn=_collate_segments,
            pin_memory=(device.type == "cuda"),
        )

    step = 0
    loss_accum = 0.0
    loss_count = 0
    projector.train()

    epoch = 0
    loader = make_loader(epoch_seed=int(rng.integers(0, 2**31 - 1)))
    data_iter = iter(loader)

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            loader = make_loader(epoch_seed=int(rng.integers(0, 2**31 - 1)))
            data_iter = iter(loader)
            batch = next(data_iter)

        z_window = batch["z_window"].to(device, non_blocking=True)  # [B,W,D]
        z_seq = batch["z_seq"].to(device, non_blocking=True)  # [B,K,D]
        B, W, D = z_window.shape
        K = z_seq.shape[1]

        ctx_window = z_window.clone()
        z_prev = ctx_window[:, -1, :]  # [B,D]

        opt.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)

        for s in range(K):
            ctx_flat = ctx_window.reshape(B, -1)
            # Dynamics prediction: Δz_hat
            if hasattr(dynamics_model, "rollout_mean"):
                dx_hat = dynamics_model.rollout_mean(ctx_flat)
            else:
                dx_hat = dynamics_model.expected_mean(ctx_flat)
            z_hat = z_prev + dx_hat  # [B,D]

            z_true = z_seq[:, s, :]  # [B,D]
            z_corr = apply_projector(projector=projector, ctx_flat=ctx_flat, z_hat=z_hat, scale=1.0)

            # MSE on corrected state
            total_loss = total_loss + torch.mean((z_corr - z_true) ** 2)

            # Advance rollout using corrected state
            z_prev = z_corr
            if s < K - 1:
                ctx_window = torch.cat([ctx_window[:, 1:, :], z_corr.unsqueeze(1)], dim=1)

        total_loss = total_loss / float(max(K, 1))
        total_loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=grad_clip_norm)
        opt.step()

        step += 1
        loss_accum += float(total_loss.item())
        loss_count += 1

        if log_every and step % log_every == 0:
            logger.info(f"[exp6b] step={step}/{max_steps} loss={loss_accum / max(loss_count,1):.6f}")
            loss_accum = 0.0
            loss_count = 0

        if save_every and step % save_every == 0:
            ckpt_path = ckpt_dir / f"projector_step{step}.pt"
            _save_projector_checkpoint(projector, step, ckpt_path, cfg)

    final_path = ckpt_dir / "projector_final.pt"
    _save_projector_checkpoint(projector, step, final_path, cfg)
    logger.info(f"[exp6b] Projector training done. Final checkpoint: {final_path}")
    projector.eval()
    return {"final_checkpoint": str(final_path), "step": step}


def _save_projector_checkpoint(projector: ContextStateProjector, step: int, path: Path, cfg: dict) -> None:
    torch.save(
        {
            "model": projector.state_dict(),
            "step": int(step),
            "model_kwargs": {
                "latent_dim": int(projector.latent_dim),
                "context_dim": int(projector.context_dim),
                "hidden_dim": int(cfg["projector_model"]["hidden_dim"]),
                "n_layers": int(cfg["projector_model"]["n_layers"]),
                "dropout": float(cfg["projector_model"].get("dropout", 0.0)),
            },
        },
        path,
    )


def load_projector_checkpoint(path: str | Path, *, device: torch.device) -> tuple[ContextStateProjector, dict]:
    ckpt = torch.load(str(path), map_location=device)
    kwargs = ckpt["model_kwargs"]
    model = ContextStateProjector(
        latent_dim=int(kwargs["latent_dim"]),
        context_dim=int(kwargs["context_dim"]),
        hidden_dim=int(kwargs["hidden_dim"]),
        n_layers=int(kwargs["n_layers"]),
        dropout=float(kwargs.get("dropout", 0.0)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def run_projector_corrected_injection_diagnostic(
    *,
    model,
    baseline,
    projector: ContextStateProjector,
    projector_scale: float,
    latents_dir: str | Path,
    latents_index_path: str | Path,
    splits_dir: str | Path,
    horizon_k: int,
    window_size: int,
    k_steps: int,
    n_eval_utterances: int,
    segments_per_utt: int,
    max_frames_per_utt: int,
    seed: int,
    device: torch.device,
) -> dict:
    store = LatentStore(latents_dir)
    rng = get_rng(seed)

    utt_ids = sample_eval_utterances(
        splits_dir=splits_dir,
        latents_index_path=latents_index_path,
        n_utterances=n_eval_utterances,
        seed=seed + 123,
    )
    if not utt_ids:
        return {"n_utterances": 0, "modes": {}}

    min_t = max(1, (window_size - 1) + horizon_k)

    modes = {"D_rollout": {}, "D_corrected": {}}
    for mode in modes:
        modes[mode] = {
            "steps": [{
                "n": 0, "nll_sum": 0.0, "cos_sum": 0.0,
                "state_err_sum": 0.0, "nll_baseline_sum": 0.0,
            } for _ in range(k_steps)]
        }

    n_utts_used = 0
    for utt_id in utt_ids:
        if utt_id not in store:
            continue
        x_true_np = store.get_latents(utt_id).astype(np.float32, copy=False)
        t_total = min(x_true_np.shape[0], max_frames_per_utt)
        if t_total <= min_t + k_steps + 1:
            continue

        t0_max = t_total - k_steps - 1
        if t0_max <= min_t:
            continue

        starts = rng.choice(
            np.arange(min_t, t0_max, dtype=np.int64),
            size=min(segments_per_utt, t0_max - min_t),
            replace=False,
        )
        if starts.size == 0:
            continue

        x_true = torch.from_numpy(x_true_np[:t_total]).to(device=device, dtype=torch.float32)

        for t0 in starts:
            t0 = int(t0)

            for mode_name in ["D_rollout", "D_corrected"]:
                use_correction = (mode_name == "D_corrected")
                x_prev = x_true[t0 - 1].clone()

                end0 = t0 - horizon_k
                start0 = end0 - window_size + 1
                ctx_window = x_true[start0:end0 + 1].clone()  # [W, D]

                for s in range(k_steps):
                    t = t0 + s

                    ctx_flat = ctx_window.reshape(1, -1)  # [1, W*D]
                    dx_true = (x_true[t] - x_true[t - 1]).unsqueeze(0)  # [1, D]

                    nll = model.nll(ctx_flat, dx_true)
                    nll_b = baseline.nll(dx_true)
                    pred = model.expected_mean(ctx_flat)

                    eps = 1e-8
                    cos = float((pred * dx_true).sum() / (
                        pred.norm().clamp_min(eps) * dx_true.norm().clamp_min(eps)
                    ).item())

                    state_err = float((x_prev - x_true[t - 1]).norm().item())

                    agg = modes[mode_name]["steps"][s]
                    agg["n"] += 1
                    agg["nll_sum"] += float(nll.sum().item())
                    agg["nll_baseline_sum"] += float(nll_b.sum().item())
                    agg["cos_sum"] += cos
                    agg["state_err_sum"] += state_err

                    if hasattr(model, "rollout_mean"):
                        dx_hat = model.rollout_mean(ctx_flat)
                    else:
                        dx_hat = model.expected_mean(ctx_flat)

                    x_hat = x_prev + dx_hat.squeeze(0)

                    if use_correction:
                        x_curr = apply_projector(
                            projector=projector,
                            ctx_flat=ctx_flat,
                            z_hat=x_hat.unsqueeze(0),
                            scale=float(projector_scale),
                        ).squeeze(0)
                    else:
                        x_curr = x_hat

                    x_prev = x_curr

                    if s < k_steps - 1:
                        new_end = t0 + s + 1 - horizon_k
                        if s + 1 < horizon_k:
                            new_frame = x_true[new_end]
                        else:
                            new_frame = x_curr
                        ctx_window = torch.cat([ctx_window[1:], new_frame.unsqueeze(0)], dim=0)

        n_utts_used += 1
        if n_utts_used >= n_eval_utterances:
            break

    result_modes = {}
    for mode_name, mdata in modes.items():
        out_steps = []
        for s, agg in enumerate(mdata["steps"]):
            n = agg["n"]
            if n == 0:
                out_steps.append({"step": s + 1, "n": 0})
                continue
            out_steps.append({
                "step": s + 1,
                "n": n,
                "nll": agg["nll_sum"] / n,
                "nll_baseline": agg["nll_baseline_sum"] / n,
                "dnll": (agg["nll_sum"] - agg["nll_baseline_sum"]) / n,
                "cos": agg["cos_sum"] / n,
                "state_err": agg["state_err_sum"] / n,
            })

        n_all = sum(a["n"] for a in mdata["steps"])
        if n_all > 0:
            result_modes[mode_name] = {
                "n": n_all,
                "nll": sum(a["nll_sum"] for a in mdata["steps"]) / n_all,
                "cos": sum(a["cos_sum"] for a in mdata["steps"]) / n_all,
                "state_err": sum(a["state_err_sum"] for a in mdata["steps"]) / n_all,
                "per_step": out_steps,
            }
        else:
            result_modes[mode_name] = {"n": 0, "per_step": out_steps}

    return {
        "n_utterances": n_utts_used,
        "projector_scale": float(projector_scale),
        "modes": result_modes,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 2 Exp6B: supervised rollout projector")
    p.add_argument("--config", type=str, default="configs/tier2_exp6b_projector_correction.yaml")
    p.add_argument("--checkpoint", type=str, default=None, help="Phase 1 checkpoint path")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--projector-checkpoint", type=str, default=None, help="Load a pre-trained projector and skip training")
    args = p.parse_args()

    if os.environ.get("NO_TORCH_COMPILE"):
        os.environ["TORCH_COMPILE_DISABLE"] = "1"

    logger = setup_logging(name="phase0")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_id = args.run_id or _default_run_id()
    out_root = Path(cfg["output"]["out_dir"])
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    run = register_run(
        experiment="exp6b_projector_correction",
        run_id=run_id,
        config_path=args.config,
        config=cfg,
        cli_args=sys.argv[1:],
        out_dir=out_dir,
        log_name="phase0",
    )

    set_seed(int(cfg.get("seed", 42)))
    device = _device_from_config(cfg["train"]["device"])

    data_cfg = cfg["data"]
    latents_dir = data_cfg["latents_dir"]
    latents_index = data_cfg["latents_index"]
    splits_dir = data_cfg["splits_dir"]

    ckpt_path = args.checkpoint or data_cfg.get("phase1_checkpoint")
    if not ckpt_path:
        logger.error("No Phase 1 checkpoint provided. Use --checkpoint or set data.phase1_checkpoint in config.")
        finalize_run(run, status="failed")
        return 1

    logger.info(f"[exp6b] Loading Phase 1 checkpoint: {ckpt_path}")
    phase1_model, ckpt = load_phase1_checkpoint(ckpt_path, device=device)
    model_type = str(ckpt.get("model_type", "mdn")).lower()
    window_size = int(ckpt.get("window_size", cfg["injection"]["window_size"]))

    # Fit baseline matched to model objective
    frames_index = data_cfg.get("frames_index", str(Path(latents_dir).parent / "phase0_frames.parquet"))
    if model_type == "vmf":
        baseline = fit_factorized_baseline(
            frames_index_path=frames_index,
            latents_dir=latents_dir,
            window_size=window_size,
            horizon_k=1,
            slice_name="all",
            max_samples=None,
        )
    else:
        baseline = fit_unconditional_baseline(
            frames_index_path=frames_index,
            latents_dir=latents_dir,
            window_size=window_size,
            horizon_k=1,
            slice_name="all",
            max_samples=None,
        )

    # Projector model
    proj_cfg = cfg["projector_model"]
    D = getattr(phase1_model, "output_dim", None)
    if D is None:
        D = ckpt.get("output_dim")
    if D is None:
        mk = ckpt.get("model_kwargs") or {}
        if isinstance(mk, dict):
            D = mk.get("output_dim")
    if D is None:
        # Last resort: infer from the latent store itself.
        store = LatentStore(latents_dir)
        try:
            any_utt = next(iter(store.store.keys()))
            D = int(store.get_latents(str(any_utt)).shape[1])
        except Exception as e:
            raise RuntimeError("Could not infer latent_dim D for projector.") from e

    D = int(D)
    ctx_dim = int(window_size) * int(D)

    if args.projector_checkpoint:
        projector, proj_ckpt = load_projector_checkpoint(args.projector_checkpoint, device=device)
        logger.info(f"[exp6b] Loaded projector from {args.projector_checkpoint}")
        proj_final = args.projector_checkpoint
    else:
        projector = ContextStateProjector(
            latent_dim=D,
            context_dim=ctx_dim,
            hidden_dim=int(proj_cfg["hidden_dim"]),
            n_layers=int(proj_cfg["n_layers"]),
            dropout=float(proj_cfg.get("dropout", 0.0)),
        ).to(device)
        logger.info(f"[exp6b] Training projector: D={D} ctx_dim={ctx_dim}")
        train_result = train_projector(
            projector=projector,
            dynamics_model=phase1_model,
            cfg=cfg,
            device=device,
            out_dir=out_dir,
            logger=logger,
        )
        proj_final = train_result["final_checkpoint"]

    # Scale sweep
    scale_list = cfg.get("projector_sweep", {}).get("scale", [0.0, 0.25, 0.5, 1.0, 2.0])
    inj_cfg = cfg["injection"]

    results = []
    for scale in scale_list:
        logger.info(f"[exp6b] Eval sweep: scale={scale}")
        r = run_projector_corrected_injection_diagnostic(
            model=phase1_model,
            baseline=baseline,
            projector=projector,
            projector_scale=float(scale),
            latents_dir=latents_dir,
            latents_index_path=latents_index,
            splits_dir=splits_dir,
            horizon_k=int(inj_cfg.get("horizon_k", 1)),
            window_size=window_size,
            k_steps=int(inj_cfg["k_steps"]),
            n_eval_utterances=int(inj_cfg["n_eval_utterances"]),
            segments_per_utt=int(inj_cfg["segments_per_utt"]),
            max_frames_per_utt=int(inj_cfg["max_frames_per_utterance"]),
            seed=int(cfg.get("seed", 42)),
            device=device,
        )
        results.append(r)

        for mode_name, mdata in r.get("modes", {}).items():
            if mdata.get("n", 0) > 0:
                logger.info(
                    f"  {mode_name}: cos={mdata.get('cos', float('nan')):.4f} "
                    f"state_err={mdata.get('state_err', float('nan')):.4f}"
                )

    with open(out_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    rows = []
    for r in results:
        scale = r.get("projector_scale")
        for mode_name, mdata in r.get("modes", {}).items():
            if mdata.get("n", 0) > 0:
                rows.append({
                    "scale": scale,
                    "mode": mode_name,
                    "cos": mdata.get("cos"),
                    "state_err": mdata.get("state_err"),
                    "nll": mdata.get("nll"),
                })
    summary = pd.DataFrame(rows)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(str(summary_path), index=False)
    logger.info(f"[exp6b] Summary: {summary_path}")

    km = {"projector_checkpoint": str(proj_final), "n_sweep_configs": len(results)}
    corr_rows = [r for r in rows if r.get("mode") == "D_corrected"]
    if corr_rows:
        best = min(corr_rows, key=lambda x: x.get("state_err", float("inf")))
        km["best_corrected_state_err"] = best.get("state_err")
        km["best_corrected_cos"] = best.get("cos")
        km["best_scale"] = best.get("scale")

    finalize_run(run, key_metrics=km)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
