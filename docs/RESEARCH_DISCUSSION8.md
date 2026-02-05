# Research Discussion 8: Tier 1 Initial Results → Paper Narrative (Preliminary)

**Date:** 2026-02-05  
**Participants:** Riley + Codex  
**Context:** First-pass readout of Tier 1 experiments (Exp1 vMF, Exp2 injection diagnostic, Exp3 representation comparison) and how to frame the story for a paper.

---

## Executive Summary (What the Tier 1 results say so far)

Tier 1 was designed to answer three questions:

1. **Does direction/magnitude factorization (vMF × LogNormal) improve rollout stability?**  
   **Preliminary answer:** *No, not in the current implementation.*  
   The vMF model produces teacher-forced direction cosines that look “reasonable” at short horizons (e.g., `k=1` cosine ≈ `0.73`), but **it is worse than a baseline in ΔNLL** on eval for all tested horizons (ΔNLL ≈ `+2.1` to `+2.2` nats/frame). See `outputs/tier1/exp1_vmf/tables.csv`.

2. **Is the rollout gap “recoverable” (distribution shift you can correct) or “catastrophic” (one-step error already kills you)?**  
   **Preliminary answer:** *Catastrophic, and it happens almost immediately.*  
   In the injection diagnostic, teacher forcing (Mode A) is stable across steps, but any mode that rolls out predicted state (Modes B/C/D) **blows up by step 2–3**, producing enormous errors (`state_err ~ 5e13`) and then `Inf/NaN`. See `outputs/tier1/exp2_injection/metrics.json`.

3. **Is the gap representation-specific (Mimi) or more fundamental?**  
   **Preliminary answer:** *Not yet answered.*  
   We only have Mimi12 results logged under Tier 1 outputs. EnCodec did not populate in the initial run; this was likely an environment/library availability issue (previous implementation depended on torchaudio’s EnCodec bundle). See `outputs/tier1/exp3_rep_compare/summary.csv` and `outputs/tier1/exp3_rep_compare/mimi12/phase1/tables.csv`.

**Key diagnosis for the paper narrative:** the Tier 1 results strongly suggest that **magnitude calibration dominates failure under rollout**. Direction appears predictably aligned under teacher forcing, but rollouts explode because the predicted step magnitudes are too large (consistent with the observed absurd `mag_ratio` values in Exp2).

---

## 1. Tier 1 Results (Observed)

### 1.1 Experiment 1: vMF direction + LogNormal magnitude (teacher-forced metrics)

**Source:** `outputs/tier1/exp1_vmf/tables.csv`

Highlights:
- **Eval ΔNLL is positive** for all horizons (worse than baseline):
  - `k=1`: ΔNLL ≈ `+2.124`
  - `k=2`: ΔNLL ≈ `+2.164`
  - `k=4`: ΔNLL ≈ `+2.203`
  - `k=8`: ΔNLL ≈ `+2.195`
- **Eval direction cosine starts high and collapses with horizon**:
  - `k=1`: cosine ≈ `0.727`
  - `k=2`: cosine ≈ `0.431`
  - `k=4`: cosine ≈ `0.147`
  - `k=8`: cosine ≈ `0.0587`

Interpretation:
- The factorized model is **not outperforming** a simple unconditional baseline on likelihood (at least as currently implemented/trained).
- Short-horizon direction predictability is present at `k=1`, but it degrades rapidly with horizon—consistent with either (a) insufficient context for longer-horizon direction, or (b) instability/scale issues in the magnitude head that contaminate learning.

### 1.2 Experiment 2: Injection diagnostic (rollout brittleness)

**Source:** `outputs/tier1/exp2_injection/metrics.json`

What happens per mode:
- **Mode A (teacher forcing):** stable; `state_err=0` by construction; ΔNLL is finite across steps.
- **Modes B (periodic), C (one-shot), D (pure rollout):** **catastrophic blow-up almost immediately**.
  - In B and D, step 2 already jumps to NLL ~ `8e22` and `state_err ~ 5e13`, and then hits `Inf/NaN` by step 3.
  - In C, the blow-up is delayed slightly, but still occurs by step 3–4.

The most important empirical takeaway:
- This is **not** “gradual drift.” It’s **an unstable dynamical system** under the current rollout rule: once we feed the model its own predicted state, the process leaves the data manifold essentially instantly.

### 1.3 Experiment 3: Representation comparison (incomplete)

**Sources:**
- `outputs/tier1/exp3_rep_compare/summary.csv`
- `outputs/tier1/exp3_rep_compare/mimi12/phase1/tables.csv`

What we have:
- Only Mimi12 is present. The summary reports a best eval ΔNLL at `k=1` of about `-185.58` nats/frame (better than baseline).

What we do *not* have yet:
- A side-by-side EnCodec run in Tier 1 outputs, so we cannot adjudicate whether the rollout gap is Mimi-specific.

---

## 2. A Unifying Diagnosis: Magnitude dominates rollout failure

Tier 1’s initial failure mode looks like **step-size explosion**:

- In Exp2, `mag_ratio` in teacher forcing is on the order of `1e12`, and in rollout modes it becomes `Infinity` right before the divergence. See `outputs/tier1/exp2_injection/metrics.json`.

This pattern is consistent with a very specific modeling issue:

> If we compute the rollout update using a deterministic “expected mean” Δz, and the magnitude model is LogNormal with a large predicted σ, then **E[m] = exp(μ + ½σ²)** can become astronomically large even when μ is moderate.

This “LogNormal mean blow-up” is a plausible proximate mechanism that:
- doesn’t necessarily destroy teacher-forced direction cosine at `k=1`, and
- can still cause immediate rollout divergence (huge step magnitudes create huge off-manifold contexts).

**Implication for paper framing:** the direction/magnitude factorization hypothesis may still be right in spirit (“direction is predictable; magnitude is not”), but the *current* implementation’s rollout rule is effectively “commit to the worst possible magnitude summary statistic.”

---

## 3. Paper Narrative: How to tell this story without overclaiming

### 3.1 Core problem statement (paper intro)

Autoregressive generation in continuous latent spaces (e.g., audio codecs) suffers from a **teacher-forcing / rollout gap**:
- Under teacher forcing, models can predict next-step deltas well.
- Under rollout, the model conditions on its own predictions; small errors compound, often catastrophically.

This is an instance of **compounding error under distribution shift**, but the audio setting makes it sharper:
- 512-D latents, high innovation rate, and strong sensitivity of downstream decoding quality to off-manifold trajectories.

### 3.2 Hypothesis (factorization)

The motivating empirical claim (from earlier phases and onboarding) is:
- **Direction** of Δz is more predictable than **magnitude**.

So we try to factor uncertainty:
```
Δz = m · d
d ~ vMF(μ_dir(ctx), κ(ctx))
m ~ LogNormal(μ_logm(ctx), σ_logm(ctx))
```

### 3.3 What Tier 1 shows (as of 2026-02-04/05)

Tier 1 gives a “negative” but extremely informative result:

1. A naive factorized model can have decent short-horizon direction cosine under teacher forcing yet still be **worse in ΔNLL** than a baseline (`outputs/tier1/exp1_vmf/tables.csv`).
2. Rollout diagnostics show the system is **unstable within 1–2 steps** when conditioned on predicted state (`outputs/tier1/exp2_injection/metrics.json`).
3. The failure signature points to **magnitude miscalibration / step-size explosion** as a key culprit, not merely directional uncertainty.

### 3.4 “So what?” contribution framing

Even before a final “win,” this yields a paper-relevant point:

> In continuous AR latent generation, it’s not enough to factor uncertainty; you must also choose rollout summaries and training targets that are **stable under heavy-tailed magnitude uncertainty**.

This is a crisp, falsifiable statement that can be:
- supported by injection diagnostics (catastrophe timing),
- supported by magnitude diagnostics (e.g., `||Δz_pred||/||Δz_true||` curves), and
- strengthened by a representation comparison once EnCodec runs are in.

---

## 4. What we should run next to complete Tier 1 (and strengthen the narrative)

These are the minimal next experiments that turn the Tier 1 story into a clean paper arc:

1. **Complete Exp3 (representation comparison) with EnCodec.**  
   The representation story is central: if EnCodec is more predictable/stable, we can claim the gap is partly representation-induced.

2. **Fix the “LogNormal mean blow-up” and re-run Exp2 + rollout metrics for Exp1.**  
   Concrete options (choose one consistent with “slow but correct”):
   - Use **median magnitude** for deterministic rollouts: `m_det = exp(μ_logm)` (not `E[m]`).
   - Or use **mode**: `m_det = exp(μ_logm - σ²)` for extra conservatism.
   - Clamp `log_sigma_logm` much tighter, or regularize σ to prevent extreme `½σ²`.

3. **Report the success criterion directly:** cosine at `K=16` rollout for vMF (after fixing magnitude).  
   This matches the onboarding bar and gives a single headline metric.

4. **Add one perceptual sanity check** (even informal): does “direction-correct but slow” decode better than “stationary” or “exploded”?  
   This can be a qualitative appendix or a small listening panel.

---

## 5. Suggested Figures / Tables for the paper

1. **Teacher forcing vs rollout gap (NLL or ΔNLL over steps):**  
   Exp2 per-step curves for modes A/B/C/D (illustrates catastrophe timing).

2. **Direction vs magnitude diagnostics:**  
   Plot per-step cosine and magnitude ratio for rollout (the intended “Mode B” signature: cosine high, magnitude ratio < 1).

3. **Representation comparison table:**  
   Best eval ΔNLL and rollout stability metrics for Mimi12 vs EnCodec (and any higher-rate variant).

4. **Ablation table:**  
   vMF rollout using `E[m]` vs `median(m)` vs `mode(m)` for deterministic rollouts (this isolates the step-size hypothesis).

---

## Appendix: Exact result file locations (for reproducibility)

- Exp1 (vMF): `outputs/tier1/exp1_vmf/metrics.json`, `outputs/tier1/exp1_vmf/tables.csv`
- Exp2 (Injection): `outputs/tier1/exp2_injection/metrics.json`, `outputs/tier1/exp2_injection/plots/`
- Exp3 (Rep compare): `outputs/tier1/exp3_rep_compare/summary.csv`, `outputs/tier1/exp3_rep_compare/mimi12/phase1/tables.csv`

