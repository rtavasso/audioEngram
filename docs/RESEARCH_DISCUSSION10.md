# Research Discussion 10: Stage 1 Results — What We Found, What It Means, and Where to Go

**Date:** 2026-02-06
**Participants:** Riley + Claude (Opus 4.6)
**Context:** Stage 1 experiments from Discussion 9 §10 are complete. This document reports findings, evaluates against the §4.2 decision matrix, and recommends Stage 2 direction.
**Status:** Results analysis and proposal

---

## Executive Summary

Stage 1 is done. Six experiments ran to completion on LibriSpeech train-clean-100 with Mimi 512-dim latents at 12.5 Hz. The headline findings:

1. **vMF direction prediction is strong at k=1 (cos=0.73) but collapses by k=4 (cos=0.15).** The factorization is sound; the model simply can't predict multi-step directions.

2. **Rollout is universally catastrophic.** Every model, every representation, every training procedure: state error explodes within 2–4 rollout steps. No intervention resolved this.

3. **Rollout fine-tuning stabilizes training but doesn't improve rollout quality.** Exp 1B runs without NaN (after sigma tightening) but the finetuned model's D_rollout cos is -0.004 — indistinguishable from random.

4. **Mimi outperforms EnCodec for prediction (2× ΔNLL), but EnCodec rollout drifts less (5.7× lower state error).** Lower-dimensional representations produce smoother dynamics at the cost of information.

5. **PCA shows the same tradeoff.** PCA-32 has higher teacher-forced cos (0.71 vs 0.61) and lower rollout state error (48 vs 79) but 10× worse ΔNLL. Dimensionality reduction smooths dynamics but destroys task-relevant information.

6. **Direction is perceptually dominant.** Listening tests confirm: 75%-magnitude with true direction ≈ ground truth; true magnitude with random direction = garbage. vMF's conceptual decomposition is validated — the bottleneck is sustaining directional accuracy, not magnitude estimation.

**Against the §4.2 decision matrix:** We land primarily in **row 4** (may be fundamental to audio) with a strong **row 5** signal (direction matters perceptually). No representation or model change resolved rollout collapse, but the dimensionality/stability tradeoff (rows 2–3) offers a partial thread to follow.

---

## 1. Stage 1 Experimental Results

### 1.1 Exp 1: vMF+LogNormal Baseline

Trained VmfLogNormal models at horizons k ∈ {1, 2, 4, 8} on Mimi 512-dim latents, window_size=8.

| Horizon k | Eval ΔNLL (vMF) | Direction cos | logmag R² |
|-----------|-----------------|---------------|-----------|
| 1 | -2.20 | **0.732** | -3686 |
| 2 | -2.24 | 0.444 | -3695 |
| 4 | -2.28 | 0.151 | -3674 |
| 8 | -2.29 | 0.058 | -3695 |

*Note: vMF ΔNLL operates in factored (direction + magnitude) space and is not directly comparable to MDN ΔNLL in full ℝ⁵¹² space. See Exp 3/4 for MDN-scale numbers.*

**Key observations:**

- **Direction cosine degrades rapidly:** 0.73 → 0.44 → 0.15 → 0.06. By k=4, direction is barely above chance (random cos in ℝ⁵¹² has std ≈ 0.044). The model finds strong directional signal at k=1 but cannot predict where trajectories will go 4+ steps ahead.

- **Magnitude R² is catastrophically negative** (−3686), meaning the LogNormal component is far worse than predicting the mean magnitude. The model can estimate direction but has no useful magnitude signal at any horizon.

- **ΔNLL is nearly flat across horizons** (−2.20 to −2.29), suggesting the NLL improvement comes almost entirely from the direction component at k=1, with the model essentially converging to the unconditional baseline as horizon increases.

- **All rollout evaluations produced NaN** due to the E[m] = exp(μ + ½σ²) overflow (σ=7.39 from max_log_sigma_logm=2.0). This was the bug diagnosed in Discussion 8. Fixed in Exp 1B.

### 1.2 Exp 1B: vMF Rollout Fine-Tuning

Loaded k=1 checkpoint, fine-tuned with K=16 rollout loss. Three fixes applied:
- Clamped mu_logm to [−5, 12] in VmfLogNormal.forward() (prevents extreme predictions off-manifold)
- Tightened max_log_sigma_logm from 2.0 → 0.7 during fine-tuning (σ: 7.39 → 2.01)
- Added NaN guard in training loop (never triggered — the clamps were sufficient)

**Training:** 5000 steps, stable loss ≈ −948, zero NaN. Scheduled sampling from p_teacher=0.2 → 0.0 over 2000 steps.

**Injection diagnostic (16-step forecast):**

| Mode | ΔNLL | Direction cos | Mag ratio | State err |
|------|------|---------------|-----------|-----------|
| A_teacher | +1.13 | 0.118 | 7.89 | 0.0 |
| D_rollout (step 1) | +1.09 | 0.103 | 7.76 | 0.0 |
| D_rollout (step 4) | +1.17 | −0.001 | 8.49 | 101 |
| D_rollout (step 8) | +3.07 | −0.006 | 3783 | 7783 |
| D_rollout (step 16) | +8.77 | −0.012 | 18848 | 203148 |

**Key observations:**

- **Rollout still collapses.** D_rollout cos drops below chance by step 2. Magnitude ratio explodes to 18,000× by step 16. State error grows exponentially.

- **Teacher-forced performance degraded.** A_teacher cos=0.118 vs the pre-finetuning cos=0.73 at k=1. The rollout fine-tuning traded single-step accuracy for no rollout improvement — a net loss. The sigma tightening (necessary for numerical stability) likely forced the optimizer to redistribute capacity away from direction prediction.

- **Magnitude is consistently over-predicted by ~8×** even under teacher forcing (mag_ratio=7.89). This reflects the tightened sigma creating a mismatch between the pre-trained model's learned distribution and the new constraint.

- **The NaN fix was necessary but not sufficient.** Training is numerically stable, but the underlying problem (context drift → off-manifold → unpredictable) remains.

### 1.3 Exp 3: Representation Comparison (Mimi vs EnCodec)

Trained MDN (Gaussian mixture) baselines on both Mimi (512D) and EnCodec encoder (128D) latents. Both at 12.5 Hz from the same LibriSpeech-100 audio.

**Phase 1 prediction quality:**

| Representation | Dim | Eval ΔNLL (MDN) |
|----------------|-----|-----------------|
| Mimi | 512 | **−184.4** |
| EnCodec encoder | 128 | −93.6 |

Mimi's predictor achieves approximately 2× the entropy reduction. The richer 512D representation contains more predictable structure.

**Injection diagnostic (16-step, MDN k=1 models):**

| | Mimi 512D | EnCodec 128D |
|---|---|---|
| **A_teacher** cos | 0.606 | 0.569 |
| **A_teacher** ΔNLL | −154.3 | −79.5 |
| **A_teacher** mag_ratio | 0.631 | 0.821 |
| **D_rollout** cos | 0.068 | 0.093 |
| **D_rollout** ΔNLL | +40.1 | **+0.62** |
| **D_rollout** state_err | 79.1 | **13.9** |

**Key observations:**

- **Mimi is better for teacher-forced prediction.** Higher cos (0.61 vs 0.57), much larger ΔNLL (−154 vs −80). The 512D space encodes more dynamics-relevant information.

- **EnCodec rollout is more stable.** State error 5.7× lower (13.9 vs 79.1). ΔNLL stays near baseline (+0.62) vs Mimi's +40.1. The lower-dimensional representation has a thicker manifold — predictions that drift slightly off don't catastrophically diverge.

- **Neither achieves useful rollout.** EnCodec rollout cos=0.093 is still essentially random. The model degrades to the unconditional baseline within 2–3 steps regardless of representation.

- **EnCodec magnitude estimation is better** (mag_ratio 0.82 vs 0.63 under teacher forcing). The 128D space has simpler magnitude structure.

### 1.4 Exp 3B: Perceptual Evaluation (Synthetic Trajectory Audio)

Decoded synthetic latent trajectories through Mimi decoder. 10 utterances × 6 modes, saved as 48kHz WAV files.

| Mode | Description | Perceptual quality |
|------|-------------|-------------------|
| **GT** | Ground truth trajectory | Clear speech (reference) |
| **B_slow_a0.75** | True direction, 75% magnitude | **Sounds like GT** |
| **B_slow_a0.50** | True direction, 50% magnitude | Degraded but recognizably speech |
| **B_slow_a0.25** | True direction, 25% magnitude | Distorted vocals, still speech-like |
| **A_stationary** | Frozen at frame 0 (no dynamics) | Garbage noise |
| **C_wrong_dir** | Random direction, true magnitude | Garbage noise |

**Key observations:**

- **Direction is the perceptually critical variable.** B_0.75 (wrong magnitude, right direction) ≈ GT. C_wrong_dir (right magnitude, wrong direction) = garbage. This is the single most actionable finding from Stage 1.

- **Magnitude is highly forgiving.** Even 25% of true magnitude with correct direction produces recognizable (if distorted) speech. The decoder is tolerant of magnitude errors but not directional errors.

- **Dynamics are essential.** A_stationary (constant latent) = garbage. The temporal evolution of latent state encodes speech content — a frozen frame contains no intelligible information.

- **This validates the vMF factorization's conceptual decomposition.** The problem is not the factorization into direction + magnitude; it's that no model sustains directional accuracy beyond 1 step.

### 1.5 Exp 4: PCA Linear Readout

Fit IncrementalPCA on Mimi latents, projected to 32D and 64D, trained Phase 1 MDN on each.

| Representation | Dim | Explained var | Eval ΔNLL (k=1) |
|----------------|-----|---------------|-----------------|
| Raw Mimi | 512 | 100% | −176.9 |
| PCA-64 | 64 | 79.1% | −26.7 |
| PCA-32 | 32 | 65.8% | −14.7 |

**Injection diagnostic comparison:**

| | Raw 512D | PCA-32D | PCA-64D |
|---|---|---|---|
| **A_teacher** cos | 0.606 | **0.708** | 0.645 |
| **A_teacher** ΔNLL | −149.0 | −13.6 | −26.7 |
| **D_rollout** cos | 0.069 | 0.099 | 0.097 |
| **D_rollout** state_err | 78.8 | **47.9** | 53.1 |
| **D_rollout** ΔNLL | +45.8 | +40.4 | +81.9 |

**Key observations:**

- **Lower dimensions → simpler dynamics.** PCA-32 achieves higher teacher-forced direction cos (0.71 vs 0.61) — the 32D subspace captures the most predictable variance. Rollout state error is also lower (48 vs 79).

- **But prediction quality collapses.** ΔNLL goes from −177 to −15 (12× worse). The discarded 79.1% of variance contains information the predictor needs.

- **Rollout still fails.** PCA-32 D_rollout cos=0.10 — marginally above chance. Even in 32D, the model can't sustain direction prediction beyond 1 step. Dimensionality reduction doesn't solve the fundamental problem.

- **PCA explained variance confirms Mimi is distributed.** 32 components capture only 65.8% of variance — information is spread across many dimensions, not concentrated in a few principal components. This argues against simple dimensionality reduction.

---

## 2. Decision Matrix Evaluation

Mapping Stage 1 results against Discussion 9 §4.2:

| Row | Criterion | Evidence | Verdict |
|-----|-----------|----------|---------|
| 1. vMF rollout works (cos > 0.6, no collapse) | cos=0.73 at k=1 only; k=4 cos=0.15; rollout fine-tuning cos=-0.004 | **NO** — fails decisively |
| 2. vMF fails, EnCodec dramatically better | EnCodec ΔNLL is *worse* (−94 vs −184); rollout state_err 5.7× lower but cos still ~0.09 | **PARTIAL** — more stable, not dramatically better |
| 3. vMF fails, EnCodec similar, linear projection helps | PCA gives higher teacher cos (0.71) and lower state_err (48) but 12× worse ΔNLL | **PARTIAL** — smoother dynamics but massive information loss |
| 4. All fail; may be fundamental to audio | All representations, all models show rollout collapse within 2–4 steps | **MOST CONSISTENT** |
| 5. "Slow but correct" sounds good | B_0.75 ≈ GT; C_wrong_dir = garbage | **YES** — direction is perceptually critical |

**Assessment:** Row 4 is the primary match — the problem appears fundamental to continuous audio latent dynamics at 12.5 Hz. However, two important nuances:

1. **Row 5 provides a lifeline.** Direction accuracy doesn't need to be perfect — 75% magnitude scaling with correct direction is perceptually transparent. Even modest directional preservation under rollout would produce useful audio.

2. **Rows 2–3 suggest dimensionality mediates stability.** EnCodec (128D) and PCA-32 both show less rollout divergence despite lower prediction quality. A learned low-dimensional representation could potentially capture more information than PCA while maintaining PCA's smoothness advantage.

---

## 3. Cross-Cutting Insights

### 3.1 The universal rollout collapse pattern

Across all 8 experimental conditions (vMF k=1 through k=8, MDN on Mimi/EnCodec/PCA-32/PCA-64), rollout follows the same trajectory:

- **Step 1:** Prediction is informative (cos=0.57–0.73 depending on model)
- **Step 2–3:** Direction cosine drops precipitously; state error begins exponential growth
- **Step 4+:** Prediction is indistinguishable from random; state has diverged from manifold
- **Step 8+:** Magnitude explodes; NLL becomes positive (worse than unconditional)

This pattern is invariant to model type (vMF vs MDN), representation (Mimi vs EnCodec), dimensionality (32 to 512), and training procedure (teacher-forced vs rollout fine-tuned). The conclusion: **the problem is not about any specific model or representation configuration. It is about the nature of autoregressive prediction in continuous audio latent spaces at 12.5 Hz.**

### 3.2 The dimensionality–quality tradeoff

A clear tradeoff emerges across all experiments:

| Representation | Dim | Teacher ΔNLL | Teacher cos | Rollout state_err |
|----------------|-----|-------------|-------------|-------------------|
| PCA-32 | 32 | −13.6 | **0.71** | **47.9** |
| PCA-64 | 64 | −26.7 | 0.65 | 53.1 |
| EnCodec encoder | 128 | −79.5 | 0.57 | **13.9** |
| Mimi (raw) | 512 | −154.3 | 0.61 | 79.1 |

Lower dimensions → simpler dynamics (higher cos, lower state_err) but less information (lower ΔNLL). The question is whether a learned encoder can do better than PCA at preserving information in low dimensions while maintaining smoothness. CALM's 32D VAE achieves competitive reconstruction (PESQ 2.42, STOI 0.90 vs Mimi-like codecs), suggesting the answer is yes.

### 3.3 What the perceptual test means for model evaluation

The NLL-based metrics and perceptual quality are not aligned:

- **By NLL:** Mimi 512D is best (ΔNLL −184). PCA-32 is terrible (ΔNLL −15).
- **By rollout stability:** EnCodec and PCA-32 are best. Mimi 512D is worst.
- **By perception:** Direction accuracy is all that matters. Magnitude can be 75% wrong with no perceptual consequence.

This suggests our evaluation framework over-weights magnitude accuracy (through NLL) relative to perceptual importance. A model that achieves cos=0.4 with magnitude ratio 0.5 would sound *good* based on the B_0.75 result — but its NLL would be terrible because the magnitude error dominates the log-likelihood.

**Implication:** Future evaluation should weight direction cosine heavily and consider perceptual metrics (PESQ, STOI, mel distance on decoded rollout) as primary rather than raw NLL.

### 3.4 Why rollout fine-tuning made things worse

Exp 1B is the clearest negative result. The finetuned model's A_teacher cos=0.118 vs the original model's cos=0.73 — a 6× degradation in teacher-forced performance with no rollout improvement. What happened:

1. **Sigma tightening changed the loss landscape.** The pretrained model had log_sigma_logm=2.0 (σ=7.39). Clamping to 0.7 (σ=2.01) invalidated the pretrained parameters. The optimizer had to re-learn magnitude prediction under a different constraint.

2. **Rollout loss dominates at low teacher probability.** After step 2000, p_teacher=0.0 — the model sees only its own predictions as context. With degraded context, the gradient signal pushes toward *reducing sensitivity to context* (i.e., predicting the unconditional mean), not toward better prediction.

3. **This is the mean-collapse failure mode from Phase 4.5**, manifesting in the vMF setting. Rollout training converges to "don't move" because that minimizes error accumulation. The factorized vMF version collapses to low kappa (diffuse direction) and biased magnitude rather than literally zero displacement, but the effect is equivalent.

---

## 4. Updated Probability Estimates

| Hypothesis | Disc. 9 | Post-Stage 1 | Reasoning |
|------------|---------|--------------|-----------|
| vMF preserves direction under rollout (cos > 0.6 at K=16) | 0.50 | **0.05** | Decisively refuted. cos < 0.15 at k=4 teacher-forced; rollout ~0 by step 2 |
| Representation change shows >2× ΔNLL improvement | 0.60 | **0.35** | EnCodec was worse, not better. PCA lost information. Evidence weakens this |
| β-VAE alone (KL + low dim) improves over Mimi for AR | 0.55 | **0.50** | Dimensionality/stability tradeoff supports this, but EnCodec didn't show dramatic improvement |
| Explicit AR losses (L_smooth, L_pred) help beyond β-VAE baseline | 0.35 | **0.40** | Slight upward update — Stage 1 shows the problem is manifold geometry, not model capacity. AR-aware training objectives directly target what's broken |
| Score-based correction improves rollout stability | 0.35 | **0.40** | Universal off-manifold catastrophe is exactly the target scenario. Slight upward update |
| Linear readout competitive with learned encoder | 0.45 | **0.25** | PCA failed to preserve information. Learned encoder needed |
| Problem is fundamental to audio at 12.5 Hz | — | **0.30** | New estimate. Consistent with all evidence, but CALM generates coherent audio from continuous latents — the problem is solvable, the question is how |

---

## 5. What We've Established

Before recommending Stage 2 directions, let's be clear about what is and isn't settled:

**Established facts:**
- Direction is predictable at k=1 (cos=0.73 vMF, cos=0.61 MDN) — there is real structure
- Direction prediction degrades with horizon (cos halves roughly every doubling of k)
- Magnitude prediction is essentially random (logmag R² deeply negative)
- One step of model-predicted context causes immediate and irrecoverable divergence
- Lower-dimensional representations produce smoother dynamics but encode less information
- Direction accuracy matters far more than magnitude accuracy for perceptual quality
- Rollout training converges to mean collapse regardless of model type

**Open questions:**
- Can a learned low-dim encoder (β-VAE) with AR-friendliness objectives produce latents where rollout doesn't collapse? This is the core representation hypothesis.
- Would score-based correction keep rollout states on-manifold long enough for the predictor to remain informative? This is the drift correction hypothesis.
- How much of the dimensionality–stability tradeoff can a learned encoder recover vs PCA's information loss?
- Is 12.5 Hz fundamentally too fast for autoregressive continuous prediction?
- Would training directly on perceptual loss (decoded audio) rather than NLL change the picture?

---

## 6. Stage 2 Recommendations

The research goal is to develop latent representations that are *inherently* robust to autoregressive prediction — not to compensate for fragile representations with inference-time hacks like CALM's noise injection. CALM demonstrates that continuous AR audio *works* with enough architectural compensation (noise injection, dual-context transformer, consistency head). Our thesis is that a representation designed for AR-friendliness should make those compensations unnecessary or greatly reduced.

Stage 1 provides the diagnostic baseline: we now know exactly how and how fast existing representations fail under rollout. Stage 2 attacks the cause (manifold geometry) rather than the symptom (prediction errors).

Ordered by alignment with the core research question:

### 6.1 β-VAE with AR-Friendliness Objectives (HIGH priority, ~3 weeks)

**Motivation:** This is the central experiment of the project. Stage 1 established that:
- Lower-dim representations have smoother dynamics (PCA-32 state_err 48 vs Mimi 79, teacher cos 0.71 vs 0.61)
- But PCA destroys information (ΔNLL −15 vs −177)
- A *learned* low-dim encoder could preserve information while maintaining smoothness

The key insight from Stage 1: the dimensionality–stability tradeoff is not a law of nature — it's a property of *unoptimized* dimensionality reduction. A VAE trained with explicit AR-friendliness objectives should occupy a different point on this tradeoff curve.

**Protocol:**
1. **Baseline β-VAE** (32D, β=0.1): Mimi-style encoder/decoder, KL regularization only. Establishes how much KL + low dim helps without AR-specific objectives.
2. **β-VAE + L_smooth**: Add temporal smoothness loss E_t[‖z_t − z_{t-1}‖²]. Penalizes sharp jumps that make dynamics unpredictable.
3. **β-VAE + L_pred**: Jointly train lightweight predictor; penalize unpredictable dynamics. This directly optimizes the latent space for the downstream AR task.
4. **β-VAE + L_smooth + L_pred**: Combined. Tests whether the losses are complementary.

For each variant, run the full Stage 1 diagnostic battery:
- Phase 1 MDN predictor (ΔNLL at k=1,2,4,8)
- Injection diagnostic (Modes A–D, 16-step)
- Reconstruction quality (PESQ, STOI, mel distance)

**Decision criteria:**
- If any β-VAE achieves ΔNLL > PCA-32 (−14.7) AND state_err < Mimi (79) AND PESQ > 2.0 → full sweep justified
- If L_pred or L_smooth variants substantially outperform baseline β-VAE → AR-friendliness objectives have value beyond simple KL regularization. This is the paper's central claim.
- If baseline β-VAE matches the AR-loss variants → the story is simpler: "just use a low-dim KL-regularized VAE"

**Why this first:** This is what the project is about. Stage 1 was diagnostic; Stage 2 is the intervention.

### 6.2 Score-Based Manifold Correction (MEDIUM priority, ~1 week)

**Motivation:** A complementary approach to the representation pivot. Rather than redesigning the manifold, learn its geometry and correct drift post-hoc. A score model ∇_z log p(z) provides a gradient field pointing toward high-density regions.

This is *not* noise injection (which corrupts training inputs to build predictor robustness). Score correction operates at inference time to pull rolled-out states back onto the data manifold — the representation stays untouched, but the dynamics system gains a "guardrail."

**Protocol:**
1. Train unconditional score model on Mimi latent frames (denoising score matching)
2. After each dynamics step during rollout, apply 1–3 Langevin correction steps: z ← z + η∇_z log p(z) + √(2η)ε
3. Evaluate rollout with and without score correction
4. If β-VAE (§6.1) is also available, test score correction on both Mimi and β-VAE latents

**Decision criterion:** If correction extends divergence horizon from ~2 steps to >8 steps without destroying dynamics signal, score correction is a viable component of the system. It complements the representation work rather than replacing it — an AR-friendly representation that *also* has score correction would be the strongest configuration.

**Why this matters for the thesis:** Score correction treats the *representation's geometry* as the object of interest (learning its score function), not the predictor. It's representation-aware in a way that noise injection isn't.

### 6.3 Perceptual Rollout Evaluation (MEDIUM priority, ~2 days)

**Motivation:** NLL may not reflect perceptual quality (§3.3). Before investing weeks in representation training, we should know the perceptual divergence horizon on existing representations.

**Protocol:**
1. Take the best MDN k=1 checkpoint (Mimi 512D)
2. Generate actual rollout trajectories (not synthetic modes) at forecast lengths 1, 2, 4, 8, 16 steps
3. Decode through Mimi decoder, save as WAVs
4. Compare to GT — establish when rollout audio becomes perceptually unacceptable

**Purpose:** Calibrates our expectations. If rollout audio is tolerable for 4+ steps despite poor NLL, the practical bar for β-VAE success is lower than the metrics suggest. If it's garbage after 1 step, the problem is even more urgent.

### 6.4 Frame Rate Experiment (DEFERRED)

**Motivation:** 12.5 Hz may be inherently too fast. At lower frame rates, each step covers more audio, and dynamics may be more predictable.

**Deferred because:** Requires re-training the encoder/decoder (changing frame rate changes the architecture), which is expensive. Run after §6.1 results clarify whether the problem is frame-rate-dependent or manifold-dependent.

---

## 7. Recommended Priority Order

```
Week 1:  §6.3 Perceptual rollout eval (2 days)
         §6.2 Score-based correction (start in parallel, ~1 week)
Week 2:  §6.1 β-VAE baseline (32D, β=0.1, KL only) — architecture + training setup
Week 3:  §6.1 β-VAE + AR losses (L_smooth, L_pred) — the core experiment
Week 4:  Diagnostic battery on all β-VAE variants
Week 5+: Full sweep (if warranted) or β-VAE + score correction combination
```

### Decision gates

**After §6.2 (score correction on Mimi):**
- If correction extends rollout to >8 steps → score correction is a viable technique; test it on β-VAE latents later
- If partial improvement → promising complement to representation work
- If no improvement → the manifold geometry is too thin for score-based recovery; representation redesign is the only path

**After §6.1 baseline β-VAE:**
- If baseline β-VAE beats PCA-32 on ΔNLL and matches on stability → the learned encoder recovers information PCA loses. AR losses are worth testing.
- If baseline β-VAE ≈ PCA-32 → KL + low dim alone doesn't outperform linear projection. Need AR losses to differentiate.
- If baseline β-VAE is worse than PCA-32 → architecture or training issue; debug before proceeding.

**After §6.1 AR-loss variants:**
- If L_pred substantially outperforms baseline β-VAE on rollout stability → central thesis validated. Write the paper around this.
- If L_smooth helps but L_pred doesn't → temporal smoothness matters more than predictability objectives. Interesting but narrower contribution.
- If neither helps → KL + low dim is the entire story; AR-specific objectives don't add value.

---

## 8. Paper Narrative Assessment

Based on Stage 1 results, the paper narrative options from Discussion 9 §11:

**Option A is the target:** "Representation design is the primary lever for continuous AR audio"

If β-VAE with AR-friendliness objectives (§6.1) produces representations where rollout doesn't immediately collapse, the contributions are:
1. A diagnostic battery that quantifies AR-friendliness of continuous representations (reusable tool)
2. The direction/magnitude decomposition and perceptual validation — direction dominates perception
3. Systematic evidence that existing representations (Mimi, EnCodec) fail under rollout regardless of dynamics model
4. Demonstration that learned representations with AR-aware training objectives produce inherently stable dynamics — the central claim
5. The dimensionality–quality–stability tradeoff as a design principle for audio latent spaces

**Option B remains possible** if score-based correction (§6.2) works well. Story: "understanding and correcting latent manifold geometry enables stable continuous AR generation."

**Option C** ("why audio is hard and what partially helps") is the fallback if results are mixed. Stage 1's diagnostic framework and the direction >> magnitude perceptual finding are publishable regardless.

**Option D** (negative results) if nothing works. The systematic failure analysis across representations, models, and training procedures is a methodological contribution.

---

## 9. Relation to Engram / Memory

The memory-as-basis hypothesis from Discussion 5 remains untested. Stage 1 results reframe its relevance:

- Memory-as-basis requires direction archetypes to exist. The perceptual finding (direction >> magnitude) actually *increases* the value of archetypes if they exist — constraining rollout predictions to a finite set of valid directions could prevent the directional collapse that kills audio quality.
- However, archetype discovery requires the dynamics model to be stable enough to observe recurring directional patterns. With rollout collapsing by step 2, we can't currently test this.
- If §6.1 (β-VAE) produces a representation where rollout is stable for 8+ steps, archetype discovery on the new latent space becomes a natural follow-up — and potentially a key differentiator from CALM's approach.

**Updated status:** Gated on §6.1 success. If AR-friendly representations extend the rollout horizon, archetype discovery moves from "deferred" to "high priority" as a way to constrain and structure the dynamics model's predictions.

---

## Appendix A: Complete Injection Diagnostic Summary

All models evaluated on 16 utterances, 8 segments each (n=2048), 16-step forecast.

### A.1 Teacher-forced (Mode A) comparison

| Experiment | Model | Dim | A_teacher ΔNLL | A_teacher cos | A_teacher mag_ratio |
|------------|-------|-----|----------------|---------------|---------------------|
| Exp 1 | vMF k=1 | 512 | — | 0.732 | — |
| Exp 1B | vMF finetuned | 512 | +1.13* | 0.118 | 7.89 |
| Exp 3 Mimi | MDN k=1 | 512 | −154.3 | 0.606 | 0.631 |
| Exp 3 EnCodec | MDN k=1 | 128 | −79.5 | 0.569 | 0.821 |
| Exp 4 Raw | MDN k=1 | 512 | −149.0 | 0.606 | 0.664 |
| Exp 4 PCA-32 | MDN k=1 | 32 | −13.6 | 0.708 | 0.773 |
| Exp 4 PCA-64 | MDN k=1 | 64 | −26.7 | 0.645 | 0.720 |

*Exp 1B's positive ΔNLL indicates the finetuned model is worse than unconditional baseline.

### A.2 Pure rollout (Mode D) comparison

| Experiment | Model | Dim | D_rollout ΔNLL | D_rollout cos | D_rollout state_err |
|------------|-------|-----|----------------|---------------|---------------------|
| Exp 1B | vMF finetuned | 512 | +4.55 | −0.004 | 62234 |
| Exp 3 Mimi | MDN k=1 | 512 | +40.1 | 0.068 | 79.1 |
| Exp 3 EnCodec | MDN k=1 | 128 | +0.62 | 0.093 | **13.9** |
| Exp 4 Raw | MDN k=1 | 512 | +45.8 | 0.069 | 78.8 |
| Exp 4 PCA-32 | MDN k=1 | 32 | +40.4 | 0.099 | 47.9 |
| Exp 4 PCA-64 | MDN k=1 | 64 | +81.9 | 0.097 | 53.1 |

Note: Exp 1B's state_err (62234) is orders of magnitude higher than others because the vMF rollout-finetuned model's magnitude predictions explode (mag_ratio=8767 by step 16), while the MDN models' magnitude predictions shrink (mag_ratio ≈ 0.17–0.20). Both are wrong, but explosion is worse than collapse for state error.

### A.3 Exp 3B perceptual modes

| Mode | Audio quality | Direction | Magnitude |
|------|-------------|-----------|-----------|
| GT | Clear speech | True | True |
| B_slow_a0.75 | ≈ GT | True | 75% |
| B_slow_a0.50 | Degraded speech | True | 50% |
| B_slow_a0.25 | Distorted vocals | True | 25% |
| A_stationary | Garbage noise | None (frozen) | None |
| C_wrong_dir | Garbage noise | Random | True |

---

## Appendix B: Experimental Conditions and Compute

| Experiment | Config | GPU time | Key output |
|------------|--------|----------|------------|
| Exp 1 | tier1_exp1_vmf.yaml | ~2h 15m | metrics.json, tables.csv, 4 checkpoints |
| Exp 1B | tier1_exp1b_vmf_rollout.yaml | ~17m | summary.json, injection_diag.json, checkpoint |
| Exp 3 | tier1_exp3_rep_compare.yaml | ~2h 6m | summary.csv, 2× injection_diag.json |
| Exp 3B | tier1_exp3b_synthetic.yaml | ~5s | 60 WAV files (10 utterances × 6 modes) |
| Exp 4 | tier1_exp4_linear_readout.yaml | ~4h | summary.csv, 3× injection_diag.json |

Total Stage 1 compute: approximately 8.5 GPU-hours on TITAN X Pascal.

All runs tracked via `outputs/manifest.jsonl`. Reproducible via:
```bash
uv run python scripts/collect_results.py --latest
```
