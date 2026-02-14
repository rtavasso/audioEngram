#!/usr/bin/env python3
"""
Stage 2 - Experiment 6: Score-based manifold correction.

1. Train score model on Mimi latents (denoising score matching)
2. Load existing Phase 1 MDN k=1 checkpoint
3. Run modified injection diagnostic: after each rollout step, apply
   score-based correction before updating context
4. Sweep correction hyperparams (Langevin or Tweedie)
5. Write results

Usage:
  uv run python scripts/tier2_exp6_score_correction.py \
      --config configs/tier2_exp6_score_correction.yaml \
      --checkpoint outputs/tier1/exp1_vmf/<RUN>/checkpoints/vmf_k1_final.pt
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


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@torch.no_grad()
def run_corrected_injection_diagnostic(
    *,
    model,
    baseline,
    score_model,
    score_model_conditional: bool,
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
    correction_method: str,
    correction_n_steps: int,
    correction_step_size: float,
    correction_sigma: float,
    correction_tweedie_scale: float,
) -> dict:
    """
    Run injection diagnostic with score-based correction applied after each rollout step.

    Runs two modes:
        D_rollout: pure rollout (no correction, baseline)
        D_corrected: rollout + correction after each step
    """
    from phase0.features.context import get_context_flat
    from phase0.features.normalization import compute_delta
    from stage2.score_correction import langevin_correction, tweedie_correction

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

    # Accumulators per mode per step
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

                # Build initial context window
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

                    # Cosine similarity
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

                    # Advance rollout
                    if hasattr(model, "rollout_mean"):
                        dx_hat = model.rollout_mean(ctx_flat)
                    else:
                        dx_hat = model.expected_mean(ctx_flat)

                    x_curr = x_prev + dx_hat.squeeze(0)

                    # Apply correction if enabled
                    if use_correction:
                        method = str(correction_method).lower()
                        x_curr_2d = x_curr.unsqueeze(0)  # [1, D]
                        if score_model_conditional:
                            sigma_t = x_curr_2d.new_full((1, 1), float(correction_sigma))
                            if method == "tweedie":
                                score = score_model(x_curr_2d, sigma_t, ctx_flat)
                                x_curr_2d = x_curr_2d + (
                                    float(correction_tweedie_scale) * (float(correction_sigma) ** 2)
                                ) * score
                            elif method == "langevin":
                                for _ in range(int(correction_n_steps)):
                                    score = score_model(x_curr_2d, sigma_t, ctx_flat)
                                    x_curr_2d = x_curr_2d + float(correction_step_size) * score
                            else:
                                raise ValueError(f"Unknown correction_method: {correction_method}")
                        else:
                            if method == "tweedie":
                                x_curr_2d = tweedie_correction(
                                    x_curr_2d,
                                    score_model,
                                    sigma=float(correction_sigma),
                                    scale=float(correction_tweedie_scale),
                                )
                            elif method == "langevin":
                                x_curr_2d = langevin_correction(
                                    x_curr_2d,
                                    score_model,
                                    n_steps=int(correction_n_steps),
                                    step_size=float(correction_step_size),
                                    sigma=float(correction_sigma),
                                )
                            else:
                                raise ValueError(f"Unknown correction_method: {correction_method}")
                        x_curr = x_curr_2d.squeeze(0)

                    x_prev = x_curr

                    # Update context window
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

    # Finalize
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
        "correction": {
            "method": str(correction_method),
            "n_steps": correction_n_steps,
            "step_size": correction_step_size,
            "sigma": correction_sigma,
            "tweedie_scale": correction_tweedie_scale,
        },
        "modes": result_modes,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 2 Exp6: score-based correction")
    p.add_argument("--config", type=str, default="configs/tier2_exp6_score_correction.yaml")
    p.add_argument("--checkpoint", type=str, default=None, help="Phase 1 checkpoint path")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--score-max-steps", type=int, default=None, help="Override score training steps")
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
        experiment="exp6_score_correction", run_id=run_id, config_path=args.config,
        config=cfg, cli_args=sys.argv[1:], out_dir=out_dir, log_name="phase0",
    )

    set_seed(int(cfg.get("seed", 42)))
    device = _device_from_config(cfg["train"]["device"])

    data_cfg = cfg["data"]
    latents_dir = data_cfg["latents_dir"]
    latents_index = data_cfg["latents_index"]
    splits_dir = data_cfg["splits_dir"]

    # 1. Load Phase 1 checkpoint
    ckpt_path = args.checkpoint or data_cfg.get("phase1_checkpoint")
    if not ckpt_path:
        logger.error("No Phase 1 checkpoint provided. Use --checkpoint or set data.phase1_checkpoint in config.")
        finalize_run(run, status="failed")
        return 1

    logger.info(f"[exp6] Loading Phase 1 checkpoint: {ckpt_path}")
    phase1_model, ckpt = load_phase1_checkpoint(ckpt_path, device=device)
    model_type = str(ckpt.get("model_type", "mdn")).lower()
    window_size = int(ckpt.get("window_size", cfg["injection"]["window_size"]))

    # Fit baseline
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

    # 2. Train score model (unconditional or conditional)
    score_cfg = cfg["score_model"]
    score_train_cfg = cfg["score_train"]
    score_max_steps = args.score_max_steps or int(score_train_cfg["max_steps"])
    score_conditional = bool(score_cfg.get("conditional", False))
    output_skip = bool(score_cfg.get("output_skip", False))
    loss_weighting = str(score_train_cfg.get("loss_weighting", "sigma2"))

    # Get utterance IDs for score training (train split)
    idx = load_latents_index(latents_index)
    from phase0.data.splits import load_splits
    splits = load_splits(splits_dir)
    train_speaker_set = set(splits.train_speakers)
    train_utts = idx[idx["speaker_id"].isin(train_speaker_set)]["utterance_id"].astype(str).tolist()
    logger.info(f"[exp6] {len(train_utts)} train utterances for score model")

    logger.info(f"[exp6] Training score model... (conditional={score_conditional})")
    if score_conditional:
        from stage2.score_train import train_conditional_score_model

        latent_dim = int(score_cfg["latent_dim"])
        cond_dim = int(window_size) * int(latent_dim)
        max_frames_per_utt = score_train_cfg.get("max_frames_per_utt", None)

        score_model, score_ckpt_path = train_conditional_score_model(
            latents_dir=latents_dir,
            utterance_ids=train_utts,
            out_dir=out_dir / "score_model",
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            window_size=window_size,
            horizon_k=int(cfg["injection"].get("horizon_k", 1)),
            hidden_dim=int(score_cfg["hidden_dim"]),
            n_layers=int(score_cfg["n_layers"]),
            sigma_min=float(score_cfg["sigma_min"]),
            sigma_max=float(score_cfg["sigma_max"]),
            batch_size=int(score_train_cfg["batch_size"]),
            num_workers=int(score_train_cfg.get("num_workers", 0)),
            max_steps=score_max_steps,
            lr=float(score_train_cfg["lr"]),
            weight_decay=float(score_train_cfg["weight_decay"]),
            grad_clip_norm=float(score_train_cfg["grad_clip_norm"]),
            log_every=int(score_train_cfg["log_every"]),
            save_every=int(score_train_cfg["save_every"]),
            max_frames_per_utt=None if max_frames_per_utt is None else int(max_frames_per_utt),
            seed=int(cfg.get("seed", 42)),
            device=device,
            output_skip=output_skip,
            loss_weighting=loss_weighting,
        )
    else:
        from stage2.score_train import train_score_model

        score_model, score_ckpt_path = train_score_model(
            latents_dir=latents_dir,
            utterance_ids=train_utts,
            out_dir=out_dir / "score_model",
            latent_dim=int(score_cfg["latent_dim"]),
            hidden_dim=int(score_cfg["hidden_dim"]),
            n_layers=int(score_cfg["n_layers"]),
            sigma_min=float(score_cfg["sigma_min"]),
            sigma_max=float(score_cfg["sigma_max"]),
            batch_size=int(score_train_cfg["batch_size"]),
            num_workers=int(score_train_cfg.get("num_workers", 0)),
            max_steps=score_max_steps,
            lr=float(score_train_cfg["lr"]),
            weight_decay=float(score_train_cfg["weight_decay"]),
            grad_clip_norm=float(score_train_cfg["grad_clip_norm"]),
            log_every=int(score_train_cfg["log_every"]),
            save_every=int(score_train_cfg["save_every"]),
            seed=int(cfg.get("seed", 42)),
            device=device,
            output_skip=output_skip,
            loss_weighting=loss_weighting,
        )
    score_model.eval()

    # 3. Sweep correction hyperparams
    sweep_cfg = cfg["correction_sweep"]
    method_list = sweep_cfg.get("method", ["langevin"])
    if isinstance(method_list, str):
        method_list = [method_list]
    n_steps_list = sweep_cfg.get("n_steps", [3])
    step_size_list = sweep_cfg.get("step_size", [0.01])
    sigma_list = sweep_cfg.get("sigma", [0.1])
    tweedie_scale_list = sweep_cfg.get("tweedie_scale", [1.0])
    inj_cfg = cfg["injection"]

    sweep_results = []

    for method in method_list:
        method = str(method).lower()
        if method not in ("langevin", "tweedie"):
            raise ValueError(f"correction_sweep.method must be langevin or tweedie (got {method})")

        if method == "tweedie":
            iter_space = [(None, None, float(sigma), float(scale)) for sigma in sigma_list for scale in tweedie_scale_list]
        else:
            iter_space = [
                (int(n_steps), float(step_size), float(sigma), 1.0)
                for n_steps in n_steps_list
                for step_size in step_size_list
                for sigma in sigma_list
            ]

        for n_steps, step_size, sigma, tweedie_scale in iter_space:
            if method == "tweedie":
                logger.info(f"[exp6] Correction sweep: method=tweedie sigma={sigma} scale={tweedie_scale}")
                n_steps_i = 0
                step_size_f = 0.0
            else:
                logger.info(f"[exp6] Correction sweep: method=langevin n_steps={n_steps} step_size={step_size} sigma={sigma}")
                n_steps_i = int(n_steps)
                step_size_f = float(step_size)

            result = run_corrected_injection_diagnostic(
                model=phase1_model,
                baseline=baseline,
                score_model=score_model,
                score_model_conditional=score_conditional,
                latents_dir=latents_dir,
                latents_index_path=latents_index,
                splits_dir=splits_dir,
                horizon_k=int(inj_cfg.get("horizon_k", 1)),
                window_size=window_size,
                k_steps=int(inj_cfg["k_steps"]),
                n_eval_utterances=int(inj_cfg["n_eval_utterances"]),
                segments_per_utt=int(inj_cfg["segments_per_utt"]),
                max_frames_per_utt=int(inj_cfg["max_frames_per_utterance"]),
                seed=42,
                device=device,
                correction_method=method,
                correction_n_steps=n_steps_i,
                correction_step_size=step_size_f,
                correction_sigma=float(sigma),
                correction_tweedie_scale=float(tweedie_scale),
            )

            sweep_results.append(result)

            for mode_name, mdata in result.get("modes", {}).items():
                if mdata.get("n", 0) > 0:
                    logger.info(
                        f"  {mode_name}: cos={mdata.get('cos', float('nan')):.4f} "
                        f"state_err={mdata.get('state_err', float('nan')):.4f}"
                    )

    # Write all results
    with open(out_dir / "sweep_results.json", "w") as f:
        json.dump(sweep_results, f, indent=2)

    # Build summary table
    summary_rows = []
    for r in sweep_results:
        corr = r.get("correction", {})
        for mode_name, mdata in r.get("modes", {}).items():
            if mdata.get("n", 0) > 0:
                summary_rows.append({
                    "method": corr.get("method"),
                    "n_steps": corr.get("n_steps"),
                    "step_size": corr.get("step_size"),
                    "sigma": corr.get("sigma"),
                    "tweedie_scale": corr.get("tweedie_scale"),
                    "mode": mode_name,
                    "cos": mdata.get("cos"),
                    "state_err": mdata.get("state_err"),
                    "nll": mdata.get("nll"),
                })

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(str(summary_path), index=False)
    logger.info(f"[exp6] Summary: {summary_path}")

    # Key metrics for manifest
    km = {
        "score_checkpoint": score_ckpt_path,
        "score_conditional": score_conditional,
        "n_sweep_configs": len(sweep_results),
    }
    # Find best corrected config by state_err
    corrected_rows = [r for r in summary_rows if r.get("mode") == "D_corrected"]
    if corrected_rows:
        best = min(corrected_rows, key=lambda x: x.get("state_err", float("inf")))
        km["best_corrected_state_err"] = best.get("state_err")
        km["best_corrected_cos"] = best.get("cos")
        km["best_method"] = best.get("method")
        km["best_sigma"] = best.get("sigma")
        km["best_tweedie_scale"] = best.get("tweedie_scale")
        km["best_n_steps"] = best.get("n_steps")
        km["best_step_size"] = best.get("step_size")

    finalize_run(run, key_metrics=km)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
