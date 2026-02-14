# Research Discussion 12: Paradigm Shift — Discrete Directions for Stable Continuous AR Audio

**Date:** 2026-02-06  
**Participants:** Riley + Claude (Opus 4.6)  
**Context:** Post-Stage 2 analysis. Two stages of systematic experimentation have established a robust 2-step rollout ceiling for continuous AR prediction across all representations and models tested. This document proposes a paradigm shift: rather than continuing to optimize continuous prediction, exploit the direction/magnitude decomposition by quantizing directions to a single codebook while keeping magnitudes continuous.  
**Status:** Proposal for new experimental direction

---

## Executive Summary

Eleven discussions and two experimental stages have converged on a clear picture: **continuous autoregressive prediction of audio latent vectors hits a fundamental 2-step rollout ceiling that no representation change, training procedure, or inference-time correction has broken.** Ten experimental conditions across 32D to 512D representations, linear to learned encoders, with and without AR-friendliness objectives, with and without score correction — all collapse by step 2.

But three findings from this same body of work point toward a different approach:

1. **Direction dominates perception.** 75% magnitude with true direction ≈ ground truth; true magnitude with random direction = garbage. The perceptually critical variable is direction, not magnitude. (Stage 1, Exp 3B)

2. **k=1 prediction is perceptually lossless.** The dynamics model's single-step prediction produces audio indistinguishable from ground truth. The problem is purely error accumulation, not individual step quality. (Stage 2, Exp 7)

3. **Continuous vectors drift off-manifold with no recovery.** This is the structural cause of the 2-step ceiling. Discrete tokens don't have this problem — they land on valid codebook entries by construction.

**The proposal:** Factor latent dynamics into a **single discrete direction codebook** (no RVQ, no iterative unrolling) plus a **continuous magnitude scalar**. The AR model predicts one categorical distribution + one scalar per frame. Directions "snap" to valid manifold points, eliminating the off-manifold drift that kills rollout. Magnitudes remain continuous because they're perceptually forgiving.

This approach is latency-compatible with CALM's real-time pipeline: single-codebook direction lookup adds no sequential depth beyond what CALM's continuous prediction already requires, and avoids the iterative codebook unrolling that makes RVQ-based approaches slower to first audio.

**The key unknown:** Does the direction space of audio latent dynamics compress well into a manageable codebook (K ≤ 1024)? If directions cluster into a small set of archetypes, this works. If the direction space is uniformly distributed across the hypersphere, the codebook would need to be impractically large. This is a cheap, decisive experiment.

---

## 1. Why We're Changing Direction

### 1.1 The case against continuing on the current path

Discussion 11 §6 recommended three Stage 3 experiments: β-VAE hyperparameter sweep, end-to-end rollout training, and frame rate reduction. We are proposing to **deprioritize all three** in favor of the direction quantization approach. The reasoning:

**β-VAE hyperparameter sweep (Discussion 11 §6.1):** The Stage 2 β-VAE already demonstrated that learned representations improve prediction quality (3× ΔNLL over PCA-32) without breaking the rollout ceiling. The baseline β-VAE's marginal k=2 survival (rollout_nll=-38.85, gap=8.3) is the best result across all conditions, but it's still marginal — and it was produced by the simplest variant (no AR losses), suggesting that hyperparameter optimization within this paradigm has limited upside. Estimated probability of breaking k=4 with a sweep: **0.15** (down from Discussion 11's 0.35, based on the pattern that all interventions within the continuous prediction paradigm produce marginal, inconsistent improvements).

**End-to-end rollout training (Discussion 11 §6.2):** Conceptually sound but high-risk. The same mean-collapse failure mode that killed Exp 1B and Phase 4.5 rollout training would likely recur: backpropagating rollout loss through the encoder incentivizes constant latents (zero dynamics = zero rollout error). Reconstruction loss provides a countervailing gradient, but the equilibrium may still favor reduced dynamics. Estimated probability of breaking k=4: **0.20**.

**Frame rate reduction (Discussion 11 §6.4):** Requires retraining the encoder/decoder, which is expensive. And even if lower frame rates make per-step prediction easier, the 2-step ceiling would likely persist at the new rate — you'd get coarser granularity with the same horizon, not a longer horizon. More importantly, reducing frame rate increases latency per step, working against the real-time constraint. Estimated probability of fundamentally changing the picture: **0.15**.

**The meta-pattern:** Every intervention within the "make continuous prediction work better" paradigm has produced marginal, inconsistent results. The 2-step ceiling is invariant to representation dimensionality (32–512), representation type (raw, PCA, learned), training objectives (reconstruction, KL, smoothness, predictability), training procedure (teacher forcing, rollout fine-tuning), and inference-time correction (score matching). This is not a parameter to be optimized — it's a structural property of continuous AR in this domain.

### 1.2 The case for the paradigm shift

The direction quantization approach doesn't try to make continuous prediction work better. It changes the prediction problem:

| | Current paradigm | Proposed paradigm |
|---|---|---|
| **Prediction target** | 32D continuous vector (β-VAE Δz) | 1 categorical (direction index) + 1 scalar (magnitude) |
| **Off-manifold risk** | High — continuous vectors drift with no recovery | **Zero for direction** — codebook entries are by construction valid directions |
| **Error accumulation** | Compound: each step's continuous error feeds into the next | **Bounded for direction** — wrong index is still a valid direction. Only magnitude accumulates continuously, and magnitude errors are perceptually forgiving |
| **Prediction difficulty** | High — 32D continuous conditional density | **Lower** — classification over K categories + scalar regression |
| **Latency** | One forward pass → 32D output | One forward pass → softmax over K + scalar. **No sequential depth increase** |

The structural advantage: discrete tokens have a built-in "snap to manifold" property that continuous vectors lack. This is why RVQ-based models (SoundStorm, VALL-E) don't suffer from the same rollout collapse — each predicted token is a valid codebook entry regardless of what errors preceded it. Our approach captures this advantage for the perceptually critical variable (direction) while avoiding RVQ's latency penalty (iterative codebook unrolling across multiple quantization levels).

### 1.3 Relation to the real-time constraint

This research builds on CALM for real-time speech generation. The binding constraint: **any proposed approach must not add latency to the generation pipeline.** CALM's pipeline is: encode → AR predict next continuous latent → consistency model decode to audio. The latency is one backbone forward pass + one consistency step.

**Why direction quantization is compatible:**

- Single codebook, no unrolling. The AR model outputs a softmax over K directions + a scalar magnitude. This replaces CALM's continuous output head (which predicts parameters for the consistency model's input). Sequential depth is unchanged.
- The codebook lookup (index → unit vector) is a single gather operation — negligible latency.
- Magnitude prediction is a single scalar — simpler than predicting a full continuous vector.
- The consistency head still operates on the resulting continuous latent (after direction lookup + magnitude scaling), so the decoding pathway is unchanged.

**Why RVQ would not be compatible:** RVQ requires iterating over multiple codebook levels sequentially — each level conditions on the sum of all previous levels. For Mimi's 8 RVQ levels, this means 8 sequential prediction steps before emitting one audio frame. A single direction codebook avoids this entirely.

**Why chunk-wise flow matching needs careful evaluation:** Generating 4 frames at once via flow matching means audio cannot be emitted until the entire chunk is complete. Even if distilled to a single function evaluation, the computation covers 4 frames simultaneously rather than 1, potentially adding latency. The latency implications are not fatal but require concrete profiling before committing. We defer this as a contingency (§6.4) rather than a primary direction.

---

## 2. The Direction Quantization Hypothesis

### 2.1 Core idea

Given a sequence of latent vectors z_1, z_2, ..., z_T from the β-VAE encoder, compute frame-to-frame deltas Δz_t = z_t - z_{t-1} and decompose:

- **Direction:** d_t = Δz_t / ‖Δz_t‖ ∈ S^{D-1} (unit hypersphere in D dimensions)
- **Magnitude:** m_t = ‖Δz_t‖ ∈ ℝ₊

Learn a codebook C = {c_1, ..., c_K} of K direction archetypes on S^{D-1} via spherical k-means on the training set. At inference, the AR model predicts:

1. A categorical distribution p(k | context) over K direction archetypes
2. A continuous magnitude m_t | context

The next latent is reconstructed as: z_{t+1} = z_t + m_t · c_{k*}, where k* is sampled from the categorical.

### 2.2 Why this might work

**Structural argument:** The 2-step rollout ceiling is caused by continuous direction errors accumulating and pushing states off-manifold. Quantized directions eliminate this failure mode for the critical variable. A wrong direction is still a *valid* direction — the state moves to a different but on-manifold location, rather than drifting into undefined space. This changes error accumulation from "exponential divergence" to "random walk on the manifold."

**Perceptual argument:** Exp 3B established that direction accuracy drives perception. Magnitude can be 25% of ground truth and still produce recognizable speech. This means: (a) the direction codebook must be high-quality (quantization error directly affects audio quality), but (b) the magnitude model can be simple and approximate.

**Information-theoretic argument:** If speech dynamics at 12.5 Hz have concentrated directional structure (i.e., most frame-to-frame transitions fall into a moderate number of patterns), then the direction codebook compresses this efficiently. The entropy of the direction distribution is the key quantity — if it's low enough for K ≤ 1024 to capture, the categorical prediction is tractable. If directions are uniformly distributed on S^31, even K = 10^6 wouldn't suffice and the approach fails.

**Precedent from other domains:** VQ-VAE demonstrated that learned discrete codebooks can capture continuous structure efficiently. Neural audio codecs (EnCodec, SoundStream, Mimi) use VQ extensively. The novel element here is applying VQ specifically to the *direction of change* rather than to the state itself, motivated by the empirical finding that direction is the perceptually critical and dynamically unstable variable.

### 2.3 Why this might fail

**Risk 1: Direction space is too high-entropy.** If frame-to-frame audio dynamics in 32D are effectively uniformly distributed on S^31, no codebook of practical size captures them. The angular quantization error would be too large, and decoded audio from quantized directions would be noticeably degraded.

**Mitigation:** This is testable cheaply (§4.1, Exp 8). If K=4096 still produces audible degradation, we know the direction space doesn't compress.

**Risk 2: Categorical prediction is not easier than continuous prediction.** If the conditional distribution p(direction_index | context) is nearly uniform over many archetypes at each step, the classification problem is as hard as the continuous prediction problem it replaces.

**Mitigation:** Measure top-1 and top-5 accuracy of a simple classifier on direction indices. If top-5 accuracy is high (>50%), the distribution is concentrated enough for the approach to work.

**Risk 3: Magnitude accumulation still causes problems.** Even with perfect directions, if magnitude errors accumulate, the state drifts in the "speed" of traversal. A trajectory that should be fast becomes slow or vice versa.

**Mitigation:** Exp 3B showed magnitude errors are perceptually forgiving. Even 75% magnitude produces near-GT audio. And magnitude is a 1D continuous variable — much easier to predict and constrain than a 32D vector.

**Risk 4: The codebook fragments the dynamics.** By forcing directions into discrete categories, we lose the smooth interpolation that continuous representations provide. Transitions between phonemes or acoustic events might sound choppy or unnatural.

**Mitigation:** If the codebook is large enough (K ≥ 256), the quantization is fine-grained enough that interpolation between archetypes (via the categorical distribution's uncertainty) provides smoothness. Also testable via reconstruction quality (§4.1).

### 2.4 Relation to Mimi's RVQ

It's important to distinguish this approach from Mimi's RVQ:

| | Mimi RVQ | Direction VQ (proposed) |
|---|---|---|
| **What is quantized** | The full latent state z_t | The direction of change Δz_t/‖Δz_t‖ |
| **Number of codebooks** | 8 (hierarchical, residual) | **1** (flat) |
| **Inference** | Sequential: predict level 1, then level 2 conditioned on 1, ... | **Single step**: predict one categorical + one scalar |
| **Latency** | ~8× single-level latency | **1× single-level latency** |
| **Magnitude** | Implicit in the multi-level residuals | **Explicit continuous scalar** |
| **AR error accumulation** | Each level's token is valid → stable | Each direction index is valid → stable. Magnitude is continuous but 1D and perceptually forgiving |

The key architectural difference: a single codebook means a single prediction step. This is the critical property for real-time latency.

---

## 3. What Needs to Be True (Hypotheses to Test)

The approach requires three things to be true, each independently testable:

### H1: Direction space compresses into a practical codebook

**Quantitative criterion:** Spherical k-means on β-VAE 32D direction vectors (Δz/‖Δz‖) achieves mean angular quantization error < 30° at K ≤ 1024, AND reconstructed audio from quantized directions (with GT magnitudes) has PESQ > 2.0 and is perceptually acceptable on listening.

**Why 30°:** In 32D, random directions have expected angular distance of ~90°. 30° represents substantial preservation of directional information. But this threshold should be calibrated against perceptual quality — the real criterion is "does it sound good."

**If false:** Direction space is too distributed for single-codebook VQ. Consider: (a) hierarchical direction codebooks (2 levels, still faster than 8-level RVQ), (b) product quantization of the direction space, or (c) abandon the direction quantization approach.

### H2: Quantized-direction audio is perceptually acceptable

**Quantitative criterion:** Audio decoded from β-VAE latents reconstructed with quantized directions + GT magnitudes is perceptually close to GT (within the quality range of B_slow_a0.75 from Exp 3B). Formal metric: mel distance between quantized-direction audio and GT audio < 1.5× the mel distance of the β-VAE's own reconstruction error.

**If false but H1 is true:** The decoder is sensitive to fine directional differences that the codebook loses. Consider: (a) larger K, (b) fine-tuning the decoder on quantized-direction inputs, (c) using the β-VAE decoder specifically rather than Mimi.

### H3: Discrete direction prediction survives multi-step rollout

**Quantitative criterion:** An AR model predicting (direction_index, magnitude) from context achieves finite, meaningful NLL at k=4 rollout (320ms). Direction accuracy: top-1 rollout accuracy > 20% at k=4 (above chance = 1/K). Audio quality: rollout WAVs at k=4 are perceptually better than the continuous rollout baseline from Exp 7.

**If false:** Discrete prediction helps with off-manifold drift but the conditional distribution over directions is too uncertain for accurate multi-step prediction. The approach provides partial benefit (stable rollout with low directional accuracy) but may not produce high-quality audio. Consider: (a) beam search / sampling strategies over the categorical, (b) combining with CALM's noise injection for additional robustness.

---

## 4. Experimental Plan

### 4.1 Exp 8: Direction Codebook Feasibility (HIGH priority, ~3 days)

**This is the decisive experiment.** Everything else is gated on its outcome.

**Protocol:**

1. **Extract directions from β-VAE latents.** Use the baseline β-VAE (32D, β=0.01) checkpoint from Exp 5. Encode LibriSpeech train-clean-100 → z sequences. Compute Δz_t = z_t - z_{t-1}. Compute d_t = Δz_t / ‖Δz_t‖ and m_t = ‖Δz_t‖. Filter out near-zero magnitudes (‖Δz_t‖ < ε, e.g., ε = 0.01 × median magnitude) — these represent silence/pauses where direction is undefined.

2. **Spherical k-means.** Run k-means on the unit hypersphere for K ∈ {64, 256, 1024, 4096}. Use cosine distance, initialize with k-means++. Record:
   - **Quantization error:** mean angular distance between each d_t and its nearest archetype (degrees)
   - **Codebook utilization:** fraction of K archetypes that are assigned ≥0.1% of training directions. If utilization is low (e.g., <50% at K=1024), the effective codebook is smaller than K
   - **Entropy of assignment distribution:** H = -Σ p_k log p_k. Compare to log(K) (uniform). Low H/log(K) ratio = concentrated structure = good for classification

3. **Reconstruct with quantized directions.** For each K: replace true d_t with nearest codebook entry c_{k*}, keep true m_t, reconstruct z_{t+1} = z_t + m_t · c_{k*}. Decode through the β-VAE decoder → Mimi decoder → audio. Also decode with GT directions + GT magnitudes as reference.

4. **Perceptual evaluation.** Listen to 10 eval utterances × 4 K values + GT reference. Record quality notes. Compute mel distance and L1 against GT audio.

5. **Magnitude distribution analysis.** Characterize m_t distribution: fit LogNormal, Gamma, empirical histogram. Report mean, median, std, skew. This informs the magnitude prediction model design (§4.3).

6. **Near-zero magnitude handling analysis.** Characterize the fraction of frames with ‖Δz_t‖ < ε. These are frames where the latent state barely changes (silence, sustained sounds). For these frames, direction is noise — the model should predict "no change" rather than a direction + near-zero magnitude. Report the fraction of near-zero frames as a function of ε threshold.

**Compute estimate:** ~1–2 hours. K-means on ~600K direction vectors in 32D is fast. Decoding through β-VAE + Mimi for 10 utterances × 5 conditions is ~30 seconds.

**Decision criteria:**

| K | Quantization error | Perceptual quality | Codebook utilization | Verdict |
|---|---|---|---|---|
| 256 | < 30° | ≈ GT | > 50% | **Strong go** — small codebook suffices |
| 1024 | < 30° | ≈ GT | > 50% | **Go** — moderate codebook, tractable classification |
| 4096 | < 30° | ≈ GT | > 50% | **Marginal** — large codebook, classification may be hard |
| 4096 | > 45° | Degraded | — | **No go** — direction space doesn't compress |

**Additional signal:** If quantization error is low but perceptual quality is degraded, the issue is error accumulation through the reconstruction z_{t+1} = z_t + m_t · c_{k*} — small per-frame errors compound over the utterance. In this case, measure cumulative trajectory divergence: ‖z_T^{quantized} - z_T^{GT}‖ / T for varying utterance length T. If it grows linearly (not exponentially), the approach may still work with periodic re-anchoring.

### 4.2 Exp 9: Near-Zero Magnitude Handling (embedded in Exp 8)

The decomposition Δz = m · d is ill-defined when m ≈ 0. These frames need special treatment.

**Protocol (run as part of Exp 8):**

1. Compute the fraction of training frames with m_t < ε for ε ∈ {0.001, 0.01, 0.1, 0.5} × median(m).
2. For these frames, analyze: are they clustered in time (silence segments)? Or scattered throughout (micro-pauses, sustained vowels)?
3. Design decision: either (a) add a special "no change" token to the codebook (index 0 = zero displacement), or (b) predict a binary "change/no-change" variable before direction+magnitude.

**Decision:** If >10% of frames are near-zero, a "no change" token is needed. If <2%, the issue can be handled by floor-clamping magnitude during training.

### 4.3 Exp 10: Discrete Direction AR Model (HIGH priority, ~1 week)

**Gated on Exp 8 success.** Only proceed if a codebook K* is identified where quantized-direction reconstruction is perceptually acceptable.

**Protocol:**

1. **Quantize training set.** Assign each training direction d_t to its nearest codebook entry → index sequence i_1, i_2, ..., i_T ∈ {1, ..., K*}. Record magnitudes m_1, ..., m_T.

2. **Train factored AR model.** Architecture: transformer or MLP backbone (match the Phase 1 MDN's capacity for fair comparison) with two output heads:
   - **Direction head:** linear → softmax over K* classes. Loss: cross-entropy.
   - **Magnitude head:** linear → LogNormal or Gaussian parameters (μ, σ). Loss: negative log-likelihood.
   - **Context:** Previous W=8 frames of (z_t, or equivalently the sequence of direction indices + magnitudes + cumulative state).
   - If Exp 9 identifies a "no change" token, include it as class 0 in the direction head.

3. **Teacher-forced evaluation.** Report:
   - Direction: top-1 accuracy, top-5 accuracy, cross-entropy loss
   - Magnitude: NLL, R², median absolute error
   - Compare to continuous baselines: Phase 1 MDN direction cosine and magnitude R² on the same β-VAE latents

4. **Rollout evaluation.** The critical test. Roll out the model for K ∈ {1, 2, 4, 8, 16} steps:
   - At each step: sample direction index from predicted categorical, sample magnitude from predicted distribution. Compute z_{t+1} = z_t + m_t · c_{k*}.
   - Feed predicted state back as context for next step.
   - Report: direction top-1 accuracy at each rollout step, magnitude error, cumulative state error, trajectory divergence.
   - **Key comparison:** Does direction accuracy degrade more slowly than the continuous baseline's direction cosine? If top-1 accuracy at k=4 is >20% while continuous cos at k=4 is ~0.09 (effectively random), the discrete approach wins.

5. **Perceptual rollout evaluation.** Decode rolled-out trajectories through β-VAE decoder → Mimi decoder → audio at k=1, 2, 4, 8. Listen and compare to Exp 7 (continuous rollout on Mimi) and GT. This is the bottom-line test.

6. **Ablation: sampling strategy.** Compare:
   - Argmax (greedy) direction selection
   - Sampling from the full categorical
   - Top-p sampling (nucleus sampling on direction logits)
   - Effect on rollout stability and audio quality

**Compute estimate:** ~1 day for model training, ~1 day for evaluation and rollout. Total ~2–3 days.

**Decision criteria:**

| Outcome | Interpretation | Next step |
|---|---|---|
| Rollout direction top-1 > 20% at k=4 AND audio quality better than continuous rollout | **Central hypothesis confirmed** | Full pipeline integration (§4.4) |
| Rollout direction degrades similarly to continuous, but state doesn't explode | **Partial win** — on-manifold drift without catastrophic divergence | May still be useful; investigate sampling strategies, larger K |
| Rollout direction degrades similarly, state explodes (categorical entropy too high) | **Approach doesn't help** — the prediction problem is equally hard in discrete space | Reconsider chunk-wise generation (§6.4) or re-anchoring strategies (§6.3) |
| Teacher-forced accuracy is poor (<40% top-1) | **Codebook doesn't capture predictable structure** — directions may cluster geometrically but not dynamically | Investigate temporal codebook learning (§5.2) |

### 4.4 Exp 11: CALM-Style Pipeline Integration (MEDIUM priority, ~1–2 weeks)

**Gated on Exp 10 success.** Only proceed if discrete direction AR shows meaningful rollout improvement.

**Protocol:**

1. **Replace CALM's prediction head** with factored direction-index + magnitude prediction. The backbone transformer remains unchanged.
2. **Evaluate end-to-end:** Given text/phoneme conditioning, generate latent trajectory via factored AR, decode via consistency model, produce audio.
3. **Ablate CALM's compensating mechanisms:**
   - Does the system still need noise injection? If direction quantization provides its own "snap to manifold" robustness, noise injection may be redundant.
   - Does the system still need the dual-context (clean short-context) transformer? If rollout states stay on-manifold, the short-context pathway may be unnecessary.
   - Does the consistency head simplify? With discrete directions constraining the output space, the consistency model may need fewer sampling steps.
4. **Latency benchmark.** Measure wall-clock time per frame: (a) CALM baseline, (b) CALM + factored prediction. Verify no latency regression.

**Decision criteria:** If factored prediction matches CALM's audio quality while reducing or eliminating the need for noise injection and/or dual-context architecture, the paper's central claim is validated: direction quantization addresses the root cause of rollout instability, not just the symptoms.

---

## 5. Contingencies and Alternative Directions

### 5.1 If the direction space doesn't compress (Exp 8 fails)

If K=4096 still produces >45° quantization error or audibly degraded reconstruction, the direction space in 32D is too distributed for a flat codebook. Options:

**5.1a Product quantization of directions.** Split the 32D direction vector into M subspaces (e.g., M=4 × 8D) and VQ each subspace independently. Total codebook: K^M combinations from M small codebooks of size K each. This is still a single prediction step if M is small (predict M categoricals in parallel, not sequentially). Latency cost: M parallel softmaxes instead of 1.

**5.1b Residual direction quantization (2 levels max).** Quantize with K=1024, then quantize the residual direction error with a second codebook of K=256. Two sequential steps, still far fewer than Mimi's 8 levels. Latency cost: 2 sequential predictions instead of 1 — acceptable for real-time if each is fast.

**5.1c Lower-dimensional direction space.** If 32D directions are too distributed, train a β-VAE with 16D or even 8D bottleneck. The direction space of S^7 is far more compressible than S^31. Trade: lower reconstruction quality (more information loss at 8D).

**5.1d Abandon direction quantization.** If no variant produces acceptable results, return to Discussion 11 §6 plan (β-VAE sweep, end-to-end rollout training) or evaluate chunk-wise generation (§6.4).

### 5.2 If codebook is geometrically good but temporally unpredictive (Exp 10 teacher-forced accuracy < 40%)

The codebook captures the spatial structure of directions but the *temporal* sequence of codebook indices is hard to predict. This could happen if geometrically nearby directions correspond to very different acoustic events.

**5.2a Temporal codebook learning.** Instead of k-means on individual directions, learn the codebook end-to-end with a prediction loss: optimize C to minimize both quantization error and conditional entropy H(i_t | context). This makes the codebook "prediction-aware" — it groups directions not by geometric proximity but by dynamic similarity.

**5.2b Conditional codebook.** Use different codebooks for different contexts (e.g., phonetic environment, pitch range). This reduces the effective K per prediction step at the cost of learning multiple codebooks.

### 5.3 If discrete rollout is better but still insufficient for k=4+ (Exp 10 partial win)

Direction quantization helps (state stays on-manifold, no explosion) but directional accuracy still degrades because the classification is uncertain.

**5.3a Scheduled re-anchoring.** Emit audio from predicted latents for N steps, then re-anchor from a clean encoder observation. This is essentially CALM's approach, but with a longer viable prediction window (if direction VQ extends the horizon from k=2 to, say, k=4, the re-anchoring period doubles).

**5.3b Beam search over direction sequences.** Maintain top-B direction sequences and score them jointly. Select the most coherent trajectory. Latency cost: B× compute per step — only viable if B is small (2–4).

**5.3c Hybrid: discrete direction + continuous residual.** Predict the direction codebook index (for manifold stability), then predict a small continuous correction vector. This gives the snap-to-manifold benefit of discrete prediction plus the expressiveness of continuous. Adds one continuous prediction head but at much lower dimensionality (the correction is small).

---

## 6. Previously Proposed Experiments — Updated Status

### 6.1 β-VAE Hyperparameter Sweep (Discussion 11 §6.1)

**Status: DEPRIORITIZED.** The evidence suggests the 2-step ceiling is a structural property of continuous AR, not a hyperparameter to optimize. If direction quantization fails (§5.1d), this becomes the fallback.

**Updated probability of breaking k=4:** 0.15

### 6.2 End-to-End Rollout Training (Discussion 11 §6.2)

**Status: DEPRIORITIZED.** High risk of mean collapse. If direction quantization succeeds, end-to-end training could be applied to the factored model (train encoder + direction codebook + dynamics model jointly) — but this is a later optimization, not a first-order experiment.

**Updated probability of breaking k=4:** 0.20

### 6.3 Re-Anchoring Strategies

**Status: CONTINGENCY (§5.3a).** If direction quantization extends the rollout horizon but not enough for practical use, re-anchoring becomes the pragmatic complement. The longer the autonomous rollout window, the less frequently re-anchoring is needed, and the lower the effective latency.

### 6.4 Chunk-Wise Generation

**Status: CONTINGENCY.** If direction quantization fails and continuous AR with re-anchoring is the only viable path, chunk-wise flow matching on β-VAE latents could generate 2–4 frames simultaneously. Latency implications need profiling — a consistency-distilled chunk model might be fast enough for real-time. But this is a fundamentally different generation paradigm, and we should exhaust the AR approach first.

---

## 7. Updated Probability Estimates

| Hypothesis | Disc. 11 | Current | Reasoning |
|---|---|---|---|
| β-VAE hyperparameter sweep breaks k=4 ceiling | 0.35 | **0.15** | Pattern of marginal improvements within continuous paradigm |
| End-to-end rollout training breaks k=4 ceiling | 0.30 | **0.20** | High risk of mean collapse; unclear if better than direct representation optimization |
| Frame rate reduction fundamentally changes the picture | 0.25 | **0.15** | Likely moves the ceiling, not removes it; hurts latency |
| Direction space compresses to K ≤ 1024 (H1) | — | **0.50** | Novel hypothesis. Speech dynamics may have concentrated directional structure, or may not. Genuinely uncertain |
| Quantized-direction audio is perceptually acceptable (H2) | — | **0.60** | Conditional on H1. If quantization error is low, the β-VAE decoder should handle it — it already tolerates significant magnitude variation |
| Discrete direction AR survives k=4 rollout (H3) | — | **0.40** | Conditional on H1 ∧ H2. The structural advantage (on-manifold by construction) is real, but the classification problem might be equally hard |
| Full pipeline (direction VQ + magnitude + CALM backbone) produces competitive audio | — | **0.25** | Compound probability: requires H1 ∧ H2 ∧ H3 ∧ successful integration. But if all components work, the result is strong |
| Problem is fundamental to AR audio at 12.5 Hz (any paradigm) | 0.50 | **0.35** | Slight downward revision — we haven't tested the discrete direction paradigm yet, which is qualitatively different from all prior conditions. CALM demonstrates the problem is solvable with enough compensation |

---

## 8. Recommended Priority Order

```
Week 1:     §4.1 Exp 8: Direction codebook feasibility (decisive experiment)
            §4.2 Exp 9: Near-zero magnitude handling (embedded in Exp 8)
            DECISION GATE: evaluate Exp 8 against §4.1 criteria
Week 2:     §4.3 Exp 10: Discrete direction AR model (if Exp 8 succeeds)
            DECISION GATE: evaluate Exp 10 against §4.3 criteria
Week 3–4:   §4.4 Exp 11: CALM-style pipeline integration (if Exp 10 succeeds)
            OR §5.1–5.2 contingencies (if Exp 8/10 partially succeed)
            OR §6.1 β-VAE sweep fallback (if Exp 8 fails decisively)
```

### Decision gates

**After Exp 8 (direction codebook feasibility):**

- If K ≤ 1024 gives <30° error, good utilization, acceptable audio → proceed to Exp 10
- If K = 4096 gives 30–45° error, marginal audio → try §5.1a (product quantization) or §5.1c (lower-dim β-VAE) before proceeding
- If K = 4096 gives >45° error, degraded audio → direction space doesn't compress; fallback to §6.1 (β-VAE sweep) or §6.4 (chunk-wise generation)

**After Exp 10 (discrete direction AR model):**

- If rollout direction accuracy > 20% at k=4 AND audio better than continuous baseline → proceed to Exp 11 (pipeline integration). This is the paper.
- If rollout stays on-manifold but directional accuracy is low → partial win; investigate §5.3 (re-anchoring, beam search, hybrid)
- If rollout still collapses (categorical too uncertain) → the prediction problem is fundamental, not structural; reassess

---

## 9. Paper Narrative Update

Post-Stage 2, Discussion 11 identified Option C ("why audio is hard and what partially helps") as the strengthening narrative. The direction quantization experiments could elevate this to a stronger story:

### Option E: "Factored discrete-continuous latent dynamics for stable real-time audio generation" (if direction VQ succeeds)

Story: Continuous AR audio generation suffers from a fundamental rollout instability: predicted continuous vectors drift off the data manifold within 2 steps, regardless of representation, training procedure, or inference-time correction. We trace this to the direction of change being the perceptually critical variable and show that quantizing directions to a single learned codebook (no iterative unrolling) while keeping magnitudes continuous provides the structural stability of discrete tokens for the variable that matters most, without sacrificing the latency advantages of continuous latent generation. The factored approach extends the autonomous rollout horizon from 2 to N steps and reduces or eliminates the need for CALM's compensating mechanisms (noise injection, dual-context architecture).

**Contributions:**
1. Diagnostic framework establishing the 2-step continuous rollout ceiling across 10+ conditions
2. Direction/magnitude decomposition and perceptual validation (direction >> magnitude)
3. Direction codebook learning and analysis of directional structure in audio latent dynamics
4. Factored discrete-direction / continuous-magnitude AR model with stable multi-step rollout
5. Integration with real-time pipeline demonstrating latency-equivalent or better performance

### Option C remains the fallback

If direction quantization produces only partial improvement, the paper becomes: "We comprehensively characterized why continuous AR audio generation is hard, discovered that direction dominates perception, and showed that direction quantization provides a structural advantage that extends — but does not fully solve — the rollout horizon."

---

## 10. Relation to Engram / Memory

The direction codebook *is* the direction archetype concept from Discussion 5, realized as a VQ codebook rather than a soft retrieval mechanism. If Exp 8 succeeds in finding concentrated directional structure, the original Engram hypothesis is partially vindicated in a different form: not "memory-augmented dynamics" but "structured direction prediction via learned archetypes."

The memory-as-basis concept (Discussion 5 §X) maps directly onto the codebook: each archetype is a basis direction, and the model selects which basis direction to move along. The continuous magnitude controls how far along that direction to move. This is a cleaner formulation than soft retrieval over a memory bank, and it's trainable end-to-end.

**Updated status:** The Engram concept has evolved from lookup-based memory → residual memory → soft retrieval over archetypes → direction codebook. Each iteration is simpler and more directly motivated by experimental findings. The codebook version is the most concrete and testable.

---

## Appendix A: Technical Notes on Spherical K-Means

Standard k-means minimizes Euclidean distance. For unit vectors on S^{D-1}, spherical k-means minimizes angular distance (equivalently, maximizes cosine similarity):

1. Initialize K centroids on S^{D-1} (e.g., random unit vectors, or k-means++ with cosine distance)
2. **Assign:** each d_t to the centroid with highest cosine similarity
3. **Update:** each centroid to the L2-normalized mean of its assigned directions: c_k ← mean({d_t : assign(d_t) = k}) / ‖mean(...)‖
4. Repeat until convergence

This is equivalent to k-means on normalized vectors with cosine distance. Libraries: scikit-learn with normalized data, or custom implementation.

**Angular distance conversion:** cos(θ) = d_t · c_k, so θ = arccos(d_t · c_k) in radians, × 180/π for degrees.

## Appendix B: Latency Analysis for Direction VQ vs CALM Baseline

**CALM baseline (per frame):**
- Transformer backbone forward pass: predicts continuous μ, σ for consistency model input
- Consistency model: 1 function evaluation → audio frame
- Total: T_backbone + T_consistency

**Direction VQ (per frame):**
- Transformer backbone forward pass: predicts K-way logits + magnitude parameters
- Softmax + sample: negligible
- Codebook lookup (gather): negligible
- Scale: z_{t+1} = z_t + m · c_k: negligible
- Consistency model: 1 function evaluation → audio frame (unchanged)
- Total: T_backbone + T_consistency + ε

The backbone forward pass is the same computational cost (same architecture, different output head — softmax over K vs continuous parameters). The consistency model is unchanged. The only addition is the codebook lookup and scaling, which are O(D) operations (negligible compared to transformer forward pass).

**Net latency impact:** ≈ 0. The approach is latency-neutral relative to CALM.

## Appendix C: Cross-Stage Experimental Summary

| Exp | Stage | What was tested | Key finding | Status |
|---|---|---|---|---|
| Phase 0–1 | 0 | Clustering, MDN prediction | Direction predictable (cos 0.61), magnitude random | Complete |
| Phase 3 | 0 | z_dyn projection | Learned projection improves cos (0.71 vs 0.61) | Complete |
| Phase 4.5 | 0 | Rollout training | Stability via mean collapse | Complete |
| Exp 1 | 1 | vMF direction/magnitude | cos 0.73 at k=1; degrades to 0.15 at k=4 | Complete |
| Exp 1B | 1 | vMF rollout fine-tuning | Catastrophic — teacher cos 0.73 → 0.12, rollout random | Complete |
| Exp 3 | 1 | Mimi vs EnCodec | Mimi better prediction, EnCodec more stable, neither survives rollout | Complete |
| Exp 3B | 1 | Perceptual direction/magnitude | **Direction >> magnitude perceptually** | Complete |
| Exp 4 | 1 | PCA linear readout | Higher cos, lower state_err, but 12× worse ΔNLL. Dimensionality helps dynamics, hurts prediction | Complete |
| Exp 5 | 2 | β-VAE ± AR losses | β-VAE 3× ΔNLL over PCA. AR losses marginal. Baseline only k=2 survivor | Complete |
| Exp 6 | 2 | Score-based correction | **Dead.** Zero improvement, harmful at larger steps | Complete |
| Exp 7 | 2 | Perceptual rollout | k=1 lossless, k=2 marginal, k=4+ degraded | Complete |
| **Exp 8** | **3** | **Direction codebook feasibility** | — | **Proposed** |
| **Exp 9** | **3** | **Near-zero magnitude handling** | — | **Proposed (embedded in Exp 8)** |
| **Exp 10** | **3** | **Discrete direction AR model** | — | **Proposed (gated on Exp 8)** |
| **Exp 11** | **3** | **CALM pipeline integration** | — | **Proposed (gated on Exp 10)** |