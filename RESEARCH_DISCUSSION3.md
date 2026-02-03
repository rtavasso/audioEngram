# Research Discussion 3: Phase 0→3 Findings, Failure Modes, and Current Status

**Date:** 2026-02-03  
**Project:** `engramAudio` (LibriSpeech train-clean-100, Mimi continuous latents @ 12.5 Hz)  
**Goal:** Determine whether predictable structure exists in the representation and, if so, whether we can transform it into an **AR-friendly state** (`z_dyn`) plus an **innovation/residual** (`z_rec`) that stabilizes rollouts.

This document is a consolidated narrative of what we ran, what happened, what broke, what we fixed, and what the results currently imply. It is intended to be a “paper backbone”: clear experimental intent, concrete numbers, and explicit failure modes.

---

## 1. Executive Summary

1. **Phase 0 clustering failed** to explain next-step dynamics variance in raw Mimi latents, but that did *not* imply “no structure exists”.
2. **Phase 1 probabilistic prediction succeeded**: a learned MDN predicts one-step latent deltas significantly better than an unconditional baseline, and predictability decays smoothly with lag `k`.
3. The predictable component is primarily **directional** (high cosine similarity) rather than **magnitude** (negative log-magnitude R² across horizons).
4. **Autoregressive rollout is catastrophically unstable** in raw Mimi latents (and remains unstable at long horizons even after Phase 3).
5. **Phase 3 produces a representation (`z_dyn`) that is substantially more predictable than raw Mimi latents** under the Phase 1 lagged-context diagnostic (stronger ΔNLL per-dim and higher direction cosine at k=1).
6. **However, the desired state/innovation split is not yet achieved**: in our pragmatic Phase 3 training setup, `z_rec` tends to be weakly used for reconstruction (posterior vs prior recon nearly identical), despite enforcing a non-zero target KL.

The current strongest takeaway: **“A predictable state exists and can be learned (z_dyn), and it improves the lagged-context predictability metrics.”** The remaining gap is turning that into **stable long-horizon generation** and a **meaningful innovation channel**.

---

## 2. Phase 0 (Recap): Why the Initial Failure Was Ambiguous

**Phase 0 intent:** test whether coarse context clustering creates predictive equivalence classes for next-step dynamics (Engram-style key/value retrieval plausibility check).

**Outcome (high level):**
- Clustering-based conditioning explained ~0% of next-step dynamics variance in raw latent space (variance ratios ~1.0).

**Key interpretation issue (from RESEARCH_DISCUSSION.md):**
- A representation can still contain predictable structure even if **(a)** the key discards the wrong information, **(b)** the conditional distribution is multimodal (mean predictor fails), or **(c)** “junk” dimensions dominate Euclidean variance.

This motivated Phase 1: a *direct* probabilistic conditional entropy test with a learned predictor.

---

## 3. Phase 1: Predictor Baseline (Lagged-Context Primary Diagnostic)

### 3.1 Diagnostic definition

We model next-step delta under a stale context:
- Target: `Δx_t = x_t - x_{t-1}`
- Context: `c_{t,k} = concat(x_{t-k-W+1 : t-k})` (window size `W`, lag horizon `k`)
- Train an MDN for `p(Δx_t | c_{t,k})`
- Baseline: unconditional diagonal Gaussian `p(Δx_t)`

Report:
- `NLL` and `ΔNLL = NLL_model - NLL_baseline` (nats / frame; and for some runs, nats / dim)
- Direction metric: cosine similarity between predicted mean delta and true delta
- Magnitude metric: R² on log magnitude (typically negative here)
- Secondary diagnostic: **rollout-context gap** (replace true context with model-generated context)

### 3.2 Phase 1 on raw Mimi latents (historical run)

Using `k ∈ {1,2,4,8}` and slice `all`, we observed:
- **Eval ΔNLL improves** (negative) across horizons, decaying with `k`:
  - `k=1: eval ΔNLL ≈ -185.44`
  - `k=2: eval ΔNLL ≈ -131.07`
  - `k=4: eval ΔNLL ≈ -84.85`
  - `k=8: eval ΔNLL ≈ -66.79`
- **Direction is predictive** at short horizon:
  - `k=1: eval cosine ≈ 0.611`
  - decays quickly by `k=8`
- **Magnitude is not predictive**:
  - log-magnitude R² < 0 at all horizons
- **Rollout instability** is severe:
  - `k=4` yielded `inf` rollout gap, `k=8` produced enormous values (catastrophic divergence).

### 3.3 Train/eval discrepancy: what it actually was

We observed an anomaly where *mean* train ΔNLL could be positive while eval ΔNLL was negative. This was investigated by re-evaluating a fixed checkpoint on 200k samples and logging quantiles + worst offenders.

**Root cause:** rare, catastrophic outliers in the train split dominated the mean conditional NLL. Typical samples (median/most quantiles) showed train and eval were consistent (ΔNLL negative); a handful of frames produced astronomically large NLL due to large `||Δx||` under a diagonal Gaussian mixture with a minimum σ clamp.

**Lesson:** for this experiment class, always log robust summaries and worst-case examples; means alone are not trustworthy.

---

## 4. Phase 3: RSSM-Style Factorizer (Single-Rate)

### 4.1 Phase 3 intent (from RESEARCH_DISCUSSION.md)

We want:
- `z_dyn,t`: predictable state that is AR-friendly
- `z_rec,t`: innovation/residual that captures unpredictable detail and is controlled by a KL “budget”

The goal is a representation that is both:
- **Predictable** (stable rollouts in state space)
- **Expressive** (reconstruction quality preserved by allocating unpredictable detail into `z_rec`)

### 4.2 What we implemented

Single-rate factorizer on frozen Mimi latents:
- `z_dyn = e_dyn(x)` (causal GRU encoder)
- `p(z_dyn,t | z_dyn,<t)` teacher-forced dynamics model
- `q(z_rec|x,z_dyn)` posterior, `p(z_rec|z_dyn)` prior
- Decoder reconstructs `x` from `(z_dyn, z_rec)`
- Training includes a fraction of timesteps decoded using `z_rec ~ p(z_rec|z_dyn)` (“prior sampling”) to prevent decoder from ignoring the prior.

**Pragmatic stability choices:**
- Fixed dynamics σ (by default) to avoid the “σ collapse ⇒ extremely negative NLL dominates training” failure mode.
- Extensive logging to detect:
  - `z_dyn` collapse (`z_dyn_var_mean`, `z_dyn_delta_l2_mean`)
  - “all info in one channel” behavior (`recon_prior_over_post`, `kl_raw`)
  - dynamics predictability (`dyn_mse`)

### 4.3 Phase 3 failure modes observed and mitigations applied

**Failure mode A: dynamics σ collapse / loss domination**
- Symptom: `loss_dyn` becomes hugely negative; `z_dyn` variance collapses; training “wins” by making dynamics likelihood arbitrarily high via σ clamp.
- Fix: default to fixed σ=1 for dynamics (`min_log_sigma = max_log_sigma = 0`) and log dyn MSE separately.

**Failure mode B: logging bug made σ stats nonsensical**
- Symptom: `q_log_sigma_mean` printed values like `-55` despite clamp bounds.
- Fix: log per-dim mean correctly (avoid accidental sum over dimensions); clamp `kl_raw` to ≥0 in logs.

**Failure mode C: one-channel solutions**
1) `z_dyn` collapse (constant state, dyn easy) while `z_rec` carries everything  
2) `z_rec` unused (posterior≈prior, recon_prior≈recon_post) while `z_dyn` carries almost everything

We added a pragmatic control to address (2):
- **Target-KL mode**: enforce a non-zero KL capacity for `z_rec` (nats per timestep), rather than only β-weighting with free-bits.

This ensures `z_rec` has a measurable information budget, but it does *not* guarantee that information corresponds to reconstruction-relevant innovation (see §6).

---

## 5. Phase 1-on-Phase 3: Predictability of `z_dyn`

### 5.1 Experimental setup

We exported `z_dyn` sequences to `outputs/phase3/zdyn.zarr` and ran the Phase 1 lagged-context MDN predictor on `Δz_dyn`.

### 5.2 Results (Phase 1 predictor on `z_dyn`)

Slice: `all`, horizons `k ∈ {1,2,4,8}`.

Key metrics (eval):

| k | eval ΔNLL (nats/frame) | eval ΔNLL/dim | eval cos(direction) | eval logmag R² | rollout gap ΔNLL |
|---:|---:|---:|---:|---:|---:|
| 1 | -61.11 | -0.477 | 0.708 | -0.098 | 68.7 |
| 2 | -30.37 | -0.237 | 0.440 | -5.734 | — |
| 4 | -17.44 | -0.136 | 0.152 | -39.242 | — |
| 8 | -12.54 | -0.098 | 0.061 | -66.951 | 2.2e11 |

**Interpretation:**
- `z_dyn` is **highly predictable at k=1**, with *stronger* direction cosine (0.708) than the earlier raw Mimi run (≈0.61).
- Predictability decays smoothly with lag, consistent with a finite-timescale predictable component.
- Magnitude predictability remains poor (log-magnitude R² negative).
- Rollout instability remains the major unsolved problem: long-horizon rollout gap still explodes (k=8 catastrophic).

**Conclusion:** Phase 3 produced a representation whose *one-step conditional structure* is stronger and cleaner, but the project still needs explicit long-horizon stabilization (rollout-based training and/or a meaningful innovation model).

---

## 6. Where We Are Relative to the Phase 3 “State/Innovation Split” Goal

### 6.1 What is working
- We can learn a state-like representation (`z_dyn`) that is:
  - non-collapsed (non-trivial variance and movement),
  - teacher-forced predictable (low dyn MSE),
  - and measurably more predictable under lagged-context than raw Mimi latents.

### 6.2 What is not yet working (and why it matters)
- Despite enforcing a non-zero target KL, `z_rec` is not reliably **reconstruction-relevant**:
  - empirically, recon from posterior vs prior can remain nearly identical (`recon_prior_over_post ≈ 1`), suggesting the decoder doesn’t need `z_rec` for the current loss mix/capacity regime.
- Without a meaningful innovation channel:
  - we cannot claim the “magnitude/noise lives in innovation” story,
  - and we do not gain a mechanism for stable diversity under rollout.

### 6.3 Why rollout is still unstable
The lagged-context predictor and the Phase 3 teacher-forced dynamics both optimize *local* prediction. They do not directly optimize **multi-step free-running stability**. The catastrophic rollout gap at longer horizons is consistent with the earlier Phase 1 rollout failure: small errors compound rapidly when the representation is not explicitly trained for rollouts.

---

## 7. Practical Failure Modes (Checklist) and the Knobs That Address Them

### 7.1 `z_dyn` collapse (constant state)
- Symptoms:
  - `z_dyn_var_mean → 0`, `z_dyn_delta_l2_mean → 0`
  - `dyn_mse → 0` trivially
- Knobs:
  - reduce `dyn_weight`
  - increase `prior_sample_prob`
  - reduce `z_rec_dim` and/or increase KL pressure on `z_rec` (in beta mode)

### 7.2 `z_rec` unused (no innovation usefulness)
- Symptoms:
  - `recon_prior_over_post ≈ 1.0` persistently
  - `kl_raw` may be >0 (target-KL) but recon doesn’t change
- Knobs:
  - increase `loss.kl.target_final` (more capacity for residual that matters)
  - decrease `prior_sample_prob` (let posterior demonstrate value early)
  - consider architectural constraint: additive decoder `x̂ = g(z_dyn) + r(z_dyn,z_rec)` with small residual head

### 7.3 Rollout instability (catastrophic divergence)
- Symptoms:
  - rollout gap NLL blows up to inf/1e10+ at moderate horizon
- Knobs (most important future work):
  - add rollout reconstruction loss (multi-step, no teacher forcing)
  - multi-horizon dynamics objectives
  - explicit stochastic innovation model for magnitude/detail (so the model can represent uncertainty rather than “pretend it is deterministic”)

---

## 8. What We Would Claim Today (and What We Would Not)

**We can credibly claim:**
- Mimi continuous latents contain significant predictable structure at this frame rate.
- A learned predictor reveals structure missed by clustering-based Phase 0 keys.
- A learned state representation (`z_dyn`) can be trained that increases short-horizon predictability and directional alignment.

**We cannot credibly claim yet:**
- Stable long-horizon AR rollouts in this representation (still catastrophic at longer horizons).
- A robust, meaningful separation of predictable state vs unpredictable innovation (`z_rec`) that improves generation.

---

## 9. Concrete Next Experiments (Paper-Relevant)

1. **Rollout-trained Phase 3 objective**
   - Curriculum on rollout horizon K (start 1–2, grow to 8–16).
   - Decode with `z_rec ~ p(z_rec|z_dyn)` during rollout, and penalize reconstruction to ground-truth `x`.
2. **Direct comparison: Phase 1 on raw Mimi vs Phase 1 on `z_dyn`**
   - Already partially done; package as a single table with matched metrics.
3. **Make `z_rec` meaningfully reconstructive**
   - Increase target KL capacity and verify that `recon_prior_over_post` becomes > 1 (but not huge).
   - If needed, add decoder structure that forces residual semantics.
4. **Cross-encoder generality (optional)**
   - Repeat Phase 1 and Phase 3 on a second encoder (EnCodec/DAC) to test whether Mimi is unusually “innovation-heavy”.

---

## Appendix A: Commands / Artifacts (Repro Pointers)

Phase 0:
- `uv run python scripts/01_make_speaker_splits.py`
- `uv run python scripts/02_infer_latents.py --device cuda`
- `uv run python scripts/03_build_phase0_dataset.py`

Phase 1 on any latents store:
- Set `configs/phase1.yaml`:
  - `data.latents_dir` = zarr store
  - `data.frames_index` = `outputs/phase0/phase0_frames.parquet`
  - `data.splits_dir` = `outputs/phase0/splits`
  - `data.latents_index` = corresponding index parquet (needed for rollout)
- Run:
  - `uv run python scripts/10_phase1_predictor_baseline.py --config configs/phase1.yaml --slice all`

Phase 3:
- `uv run python scripts/20_phase3_train_factorizer.py --config configs/phase3.yaml`
- `uv run python scripts/21_phase3_export_zdyn.py --config configs/phase3.yaml --checkpoint outputs/phase3/checkpoints/phase3_final.pt --split all`

