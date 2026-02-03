# Research Discussion 4: Phase 4/4.5 (Engram Integration + Rollout Training) Results

**Date:** 2026-02-03  
**Scope:** Phase 4 (Engram integration) and Phase 4.5 (rollout-trained dynamics) on exported Phase 3 `z_dyn`  
**Primary question:** *If Phase 3 yields an AR-predictable state (`z_dyn`), does an Engram-style memory help, and what does it take to make free-running rollouts stable?*

This document is a phase-focused narrative arc intended to be paper-ready: what we tried, what failed, what we changed, and what the results imply.

---

## 1. Setup and Evaluation Protocol

### 1.1 Data
- Representation: exported Phase 3 predictable state `z_dyn` (single-rate) stored in `outputs/phase3/zdyn.zarr`.
- Split: speaker-disjoint train/eval (Phase 0 `outputs/phase0/splits`).
- Training target: one-step transition `Δz_t = z_{t+1} - z_t`.

### 1.2 Metrics (decision-grade)

We evaluate two regimes:

1) **Teacher-forced (one-step):** condition on true `z_t`.
   - Metric: `ΔNLL = NLL[p(Δz | z_t)] - NLL[p(Δz)]` (negative is better than baseline).
   - Direction metric: cosine similarity between predicted mean `μ(z_t)` and true `Δz_t`.

2) **Free-running rollout:** condition on model-generated `ẑ_t`.
   - Rollout rule: `ẑ_{t+1} = ẑ_t + μ(ẑ_t)` (mean rollout).
   - Metric: compute NLL of the *true* `Δz_t` under the model conditioned on `ẑ_t` and compare to unconditional baseline (`rollout_dnll`).
   - Diagnostics: non-finite rollouts, max `||ẑ||₂`, and (later) clip statistics.

**Why this matters:** teacher-forced metrics can look excellent even when the model diverges immediately in rollout. Rollout metrics are the downstream proxy for AR generation viability.

---

## 2. Phase 4 Attempt 1: Naive Memory over `z_t` (Negative Result)

### 2.1 Hypothesis
If `z_dyn` contains reusable motifs, then nearest-neighbor / k-means lookup keyed by `z_t` should predict `Δz_t` well.

### 2.2 Implementation (Phase 4)
- Fit `k`-means on keys `z_t` (train split).
- For each cluster, store `E[Δz | cluster]` and `Var[Δz | cluster]`.
- Evaluate a “memory-only” predictor by nearest-centroid lookup; compare against a parametric MLP dynamics model.

### 2.3 Outcome
- **Memory-only** performed badly (in early runs, catastrophically) due to unreliable per-cluster variance and brittle hard assignment.
- **Parametric model** performed well teacher-forced, but could still diverge in free-running rollouts.

**Conclusion:** naive “k-means memory predicts the whole delta” is not a reliable Engram test.

---

## 3. Phase 4 Attempt 2: Memory as Residual + Soft Retrieval (Positive Teacher-Forced Result)

We revised Phase 4 to better match how memory tends to help in practice: **store exceptions/residuals**, not the entire prediction.

### 3.1 Key idea: residual memory
Train a parametric model `μ_param(z)` first. Then fit memory on residuals:
```
r(z_t) = Δz_t - μ_param(z_t)
```
At inference, combine:
```
μ(z_t) = μ_param(z_t) + μ_mem_residual(z_t)
```

### 3.2 Softer retrieval
Replace hard nearest-centroid with **soft top‑k** weighting:
```
w_i ∝ softmax(-||z - c_i||² / τ)
μ_mem = Σ_i w_i μ_i
```

### 3.3 Results (teacher-forced)
On eval one-step prediction:
- `param`: `dnll ≈ -44.54`, direction cos ≈ `0.695`
- `hybrid` (gated blend of param+memory): `dnll ≈ -46.15`, cos ≈ `0.702`, **gate_mean ≈ 0.25**
- `resid_add`: `dnll ≈ -47.98`, cos ≈ `0.717` (best)
- `gated_resid`: essentially identical to `resid_add`, with **gate_mean ≈ 1.0** (gate saturates)

**Interpretation:** Engram-style memory *does* help in teacher-forced prediction when used as a residual corrector. The gate saturating to ~1 suggests the residual memory is broadly useful under one-step training.

### 3.4 Results (rollout before rollout-training)
Even with improved memory design, rollout remained poor:
- Memory variants were numerically stable (no NaNs; `||ẑ||` remained near the data manifold).
- However, `rollout_dnll` stayed **positive** (worse than unconditional baseline), indicating that free-running conditioning still destroys predictive advantage.
- The pure parametric model could explode off-manifold (NaNs or enormous NLL) without explicit stabilization.

**Conclusion:** residual memory helps one-step and improves numerical stability, but it does not fix the core compounding-error problem.

---

## 4. Phase 4.5: Rollout-Trained Dynamics (Major Stability Breakthrough)

### 4.1 Motivation
We were measuring rollout behavior without training for it. Teacher-forced one-step NLL is not the right objective if the downstream use case is autoregressive rollouts.

### 4.2 Method: K-step rollout loss (K=16)
Fine-tune the parametric dynamics model with an unrolled loss:
- Sample rollout segments `(z0, Δz[0:K])` from train utterances.
- Roll out `ẑ` for `K` steps using the model’s own predictions.
- At each step, score the *true* next-step delta under the model conditioned on the rolled-out state.

This directly optimizes: “be accurate on-distribution *and* remain accurate on the model’s induced state distribution.”

### 4.3 Pre vs Post results (param model)

**Teacher-forced:**
- `dnll` got worse: `-44.54 → -33.83`
- Direction cosine got better: `0.695 → 0.713`

**Rollout:**
- Catastrophic failure pre-finetune:
  - `rollout_dnll ≈ +761,868` (even with clip stabilizer)
  - large `||ẑ||₂` and frequent clipping
- Stable, near-baseline post-finetune:
  - `rollout_dnll ≈ -0.987` (slightly better than unconditional baseline under rollout)
  - `max ||ẑ||₂ ≈ 10.25`, `n_clipped = 0`, mean `||Δẑ||` extremely small

**Interpretation:** rollout training converts an exploding free-running model into a stable one. This is strong evidence that the rollout failure mode is largely an *objective mismatch* problem, not purely a representation problem.

### 4.4 Residual memory after rollout training
After rollout fine-tune, we refit residual memory and evaluated `resid_add`:
- Teacher-forced: small improvement vs `param_post` (`dnll` changes by < 1 nat/frame).
- Rollout: essentially identical to `param_post` (`rollout_dnll ≈ -0.98`).

**Interpretation:** once the dynamics model is rollout-trained, residual memory provides little incremental benefit under this evaluation protocol. Memory was most useful as a patch for the one-step-trained model, not as a substitute for rollout training.

---

## 5. What We Learned (Paper-Relevant Claims)

1. **Engram-style memory does not help as a direct predictor of `Δz`** when keyed by raw `z_t` (naive memory).
2. **Memory helps in the correct regime:** as a residual corrector on top of a parametric model, improving teacher-forced NLL and direction alignment.
3. **Rollout instability is the dominant bottleneck**, and memory alone does not solve it.
4. **Rollout-trained objectives fix the catastrophic divergence** and yield a stable free-running model whose rollout performance approaches the unconditional baseline (and slightly beats it here).
5. After rollout training, **memory adds little** (at least with this simple key and residual design), suggesting that the main lever is objective alignment rather than retrieval.

---

## 6. Remaining Gaps / Next Experiments

### 6.1 Rollout advantage gap
Even post-finetune, the model loses most of its teacher-forced advantage once it conditions on its own states:
- teacher-forced `dnll ≈ -35`
- rollout `dnll ≈ -1`

This suggests additional work is needed to keep the model’s induced state distribution close to the data distribution and/or to better model uncertainty.

### 6.2 Ablations for maximal insight
1. **Sweep rollout horizon K**: `{4, 8, 16, 32}` to map the stability timescale.
2. **Mixed objective**: `L = L_1step + λ L_rollout` to recover one-step NLL without reintroducing rollout explosion.
3. **Uncertainty modeling**: mean rollouts are extremely conservative post-finetune (`||Δẑ||` very small). Evaluate whether the model is “playing it safe” via mean shrinkage and whether learned σ is compensating.

---

## Appendix: Artifacts / Commands

Phase 4 memory + eval:
- `uv run python scripts/30_phase4_fit_memory.py --config configs/phase4.yaml`
- `uv run python scripts/31_phase4_train_eval.py --config configs/phase4.yaml`

Phase 4.5 rollout fine-tune:
- `uv run python scripts/32_phase4_rollout_finetune.py --config configs/phase4.yaml`
- Output summary: `outputs/phase4/phase4_rollout_finetune_summary.json`

