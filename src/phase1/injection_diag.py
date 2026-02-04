"""
Tier 1 Experiment 2: Teacher-forcing injection diagnostic.

Evaluates how model performance degrades under rollout-corrupted context, and
how much it can be "reset" via periodic injections of ground-truth states.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from phase0.data.io import LatentStore
from phase0.features.context import get_context_flat
from phase0.features.normalization import compute_delta
from phase0.utils.seed import get_rng


@dataclass
class StepAgg:
    n: int
    nll_sum: float
    nll_baseline_sum: float
    cos_sum: float
    mag_ratio_sum: float
    state_err_sum: float

    @classmethod
    def create(cls, k_steps: int) -> list["StepAgg"]:
        return [cls(n=0, nll_sum=0.0, nll_baseline_sum=0.0, cos_sum=0.0, mag_ratio_sum=0.0, state_err_sum=0.0) for _ in range(int(k_steps))]


def _cos_and_mag_ratio(pred_mean: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    p = pred_mean
    y = target
    p_norm = torch.linalg.vector_norm(p, dim=-1).clamp_min(eps)
    y_norm = torch.linalg.vector_norm(y, dim=-1).clamp_min(eps)
    cos = (p * y).sum(dim=-1) / (p_norm * y_norm)
    mag_ratio = p_norm / y_norm
    return cos, mag_ratio


@torch.no_grad()
def run_injection_diagnostic(
    *,
    model: object,
    baseline: object,
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
    mode_inject_after_steps: dict[str, Optional[list[int]]],
    sample_from_model: bool,
) -> dict:
    """
    mode_inject_after_steps maps mode -> list of rollout step indices (1-based) after which to inject x_true,
    or None to indicate pure teacher forcing (always x_true).
    """
    from phase1.data import sample_eval_utterances

    store = LatentStore(latents_dir)
    rng = get_rng(int(seed))

    utt_ids = sample_eval_utterances(
        splits_dir=splits_dir,
        latents_index_path=latents_index_path,
        n_utterances=int(n_eval_utterances),
        seed=int(seed) + 123,
    )
    if not utt_ids:
        return {"n_utterances": 0, "modes": {}}

    # Per-mode, per-step accumulators
    per_mode_steps: dict[str, list[StepAgg]] = {m: StepAgg.create(k_steps) for m in mode_inject_after_steps.keys()}

    min_t = max(1, (int(window_size) - 1) + int(horizon_k))

    n_utts_used = 0
    for utt_id in utt_ids:
        if utt_id not in store:
            continue
        x_true = store.get_latents(utt_id).astype(np.float32, copy=False)
        t_total = int(min(int(x_true.shape[0]), int(max_frames_per_utt)))
        if t_total <= (min_t + int(k_steps) + 1):
            continue

        t0_max = t_total - int(k_steps) - 1
        if t0_max <= min_t:
            continue

        starts = rng.choice(np.arange(min_t, t0_max, dtype=np.int64), size=int(min(segments_per_utt, t0_max - min_t)), replace=False)
        for t0 in starts.tolist():
            t0 = int(t0)
            d = int(x_true.shape[1])

            for mode, inject_after in mode_inject_after_steps.items():
                teacher_forcing = inject_after is None
                inject_set = set(int(s) for s in (inject_after or []))

                # Build x_hat up to the segment (prefix is true), then roll out K steps.
                x_hat = np.concatenate(
                    [x_true[:t0].copy(), np.zeros((int(k_steps), d), dtype=np.float32)],
                    axis=0,
                )

                for s in range(1, int(k_steps) + 1):
                    t = t0 + (s - 1)

                    if teacher_forcing:
                        # Pure teacher forcing: x_hat == x_true always.
                        x_hat[t - 1] = x_true[t - 1]

                    ctx_true = get_context_flat(x_true, t, int(window_size), int(horizon_k)).astype(np.float32, copy=False)
                    dx_true = compute_delta(x_true, t).astype(np.float32, copy=False)

                    ctx_hat = (
                        ctx_true
                        if teacher_forcing
                        else get_context_flat(x_hat, t, int(window_size), int(horizon_k)).astype(np.float32, copy=False)
                    )

                    ctx_hat_t = torch.from_numpy(ctx_hat).unsqueeze(0).to(device)
                    ctx_true_t = torch.from_numpy(ctx_true).unsqueeze(0).to(device)
                    dx_true_t = torch.from_numpy(dx_true).unsqueeze(0).to(device)

                    # Evaluate NLL of true delta under (teacher vs rollout-corrupted) context.
                    nll_hat = model.nll(ctx_hat_t, dx_true_t)  # [1]
                    nll_b = baseline.nll(dx_true_t)  # [1]
                    pred_hat = model.expected_mean(ctx_hat_t)  # [1,D]

                    cos, mag_ratio = _cos_and_mag_ratio(pred_hat, dx_true_t)

                    # State error before the step (how far the rollout context has drifted)
                    state_err = float(np.linalg.norm(x_hat[t - 1] - x_true[t - 1]))

                    agg = per_mode_steps[mode][s - 1]
                    agg.n += 1
                    agg.nll_sum += float(nll_hat.item())
                    agg.nll_baseline_sum += float(nll_b.item())
                    agg.cos_sum += float(cos.item())
                    agg.mag_ratio_sum += float(mag_ratio.item())
                    agg.state_err_sum += float(state_err)

                    if teacher_forcing:
                        x_hat[t] = x_true[t]
                        continue

                    # Advance rollout state using sampled or mean delta.
                    if sample_from_model and hasattr(model, "sample_delta"):
                        dx_hat = model.sample_delta(ctx_hat_t)[0].detach().cpu().numpy().astype(np.float32, copy=False)
                    else:
                        dx_hat = pred_hat[0].detach().cpu().numpy().astype(np.float32, copy=False)
                    x_hat[t] = x_hat[t - 1] + dx_hat

                    # Optional correction after this step.
                    if s in inject_set:
                        x_hat[t] = x_true[t]

        n_utts_used += 1
        if n_utts_used >= int(n_eval_utterances):
            break

    # Finalize
    modes_out: dict[str, dict] = {}
    for mode, steps in per_mode_steps.items():
        out_steps = []
        for s, agg in enumerate(steps, start=1):
            if agg.n == 0:
                out_steps.append(
                    {
                        "step": s,
                        "n": 0,
                        "nll": float("nan"),
                        "nll_baseline": float("nan"),
                        "dnll": float("nan"),
                        "cos": float("nan"),
                        "mag_ratio": float("nan"),
                        "state_err": float("nan"),
                    }
                )
                continue

            n = float(agg.n)
            nll = agg.nll_sum / n
            nll_b = agg.nll_baseline_sum / n
            out_steps.append(
                {
                    "step": s,
                    "n": int(agg.n),
                    "nll": float(nll),
                    "nll_baseline": float(nll_b),
                    "dnll": float(nll - nll_b),
                    "cos": float(agg.cos_sum / n),
                    "mag_ratio": float(agg.mag_ratio_sum / n),
                    "state_err": float(agg.state_err_sum / n),
                }
            )

        # Overall aggregates across steps (weighted by n per step)
        n_all = sum(int(a.n) for a in steps)
        if n_all == 0:
            modes_out[mode] = {"n": 0, "per_step": out_steps}
            continue

        nll_all = sum(a.nll_sum for a in steps) / n_all
        nll_b_all = sum(a.nll_baseline_sum for a in steps) / n_all
        modes_out[mode] = {
            "n": int(n_all),
            "nll": float(nll_all),
            "nll_baseline": float(nll_b_all),
            "dnll": float(nll_all - nll_b_all),
            "cos": float(sum(a.cos_sum for a in steps) / n_all),
            "mag_ratio": float(sum(a.mag_ratio_sum for a in steps) / n_all),
            "state_err": float(sum(a.state_err_sum for a in steps) / n_all),
            "per_step": out_steps,
        }

    return {
        "n_utterances": int(min(n_eval_utterances, len(utt_ids))),
        "horizon_k": int(horizon_k),
        "window_size": int(window_size),
        "k_steps": int(k_steps),
        "modes": modes_out,
    }

