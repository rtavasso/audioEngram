# Research Discussion 6: Critical Analysis, vMF Factorization, and Experimental Priorities

**Date:** 2026-02-04
**Participants:** Claude (Opus 4.5) + Codex
**Context:** Deep dive following Phase 4/4.5 results on rollout stability and the teacher-forced vs rollout performance gap

---

## 1. Executive Summary

This discussion critically examines the proposed next steps from Discussion 5 (vMF direction modeling, memory-as-basis, direction archetype discovery) and develops refined experimental priorities. Key conclusions:

1. **The teacher-forced vs rollout gap (-35 vs -1 ΔNLL) is primarily an objective mismatch problem**, not an architectural one. The model learns "do nothing" because that minimizes cumulative loss under covariate shift.

2. **vMF factorization may help, but not for the reasons initially proposed.** The value isn't better likelihood—it's enabling "slow but correctly directed" trajectories that preserve directional information while hedging on magnitude.

3. **Diagnostic experiments should precede architectural changes.** Two cheap experiments can reveal whether the bottleneck is per-step error, cumulative drift, or objective mismatch.

4. **Memory-as-basis is not really Engram.** The continuous-key / soft-retrieval mechanism is fundamentally different from discrete lookup. This may be fine, but the framing should be honest.

5. **The most compelling paper framing is "continuous AR audio needs explicit uncertainty factorization"**—a general contribution rather than a memory-specific one.

---

## 2. Critical Assessment: Is vMF the Right Factorization?

### 2.1 The Standard Argument

Direction is predictable (cosine ~0.65-0.71), magnitude is not (R² < 0). Therefore, factorize Δz = magnitude × direction and model direction on the unit sphere with vMF.

### 2.2 Challenges to This Argument

**Challenge 1: Cosine ~0.65 sounds good but isn't that concentrated in practice.**

In D=512 dimensions, cosine 0.65 corresponds to ||d_true - d_pred||² = 0.70, which is substantial Euclidean distance. The estimated kappa ~850 puts the model near the limit of what single-mode vMF can express. If the true direction distribution is multimodal or heavy-tailed on the sphere, vMF will underfit.

**Counter-argument:** The relevant comparison isn't "is 0.65 good in absolute terms" but "is there structure relative to uniform." Two random unit vectors in D=512 have expected cosine ~0. Achieving 0.65 indicates real concentration. The question is whether vMF captures this better than isotropic Gaussian over full Δz.

**Challenge 2: Factorization doesn't solve the rollout problem.**

Errors in direction compound. Errors in magnitude compound. Separating them doesn't stop compounding.

**Counter-argument:** vMF sampling with high kappa produces directions much closer to the true direction than sampling from an isotropic Gaussian. If the model can be confident about direction while uncertain about magnitude, it produces "slow but correctly directed" trajectories rather than stationary ones. The key insight: current models conflate directional uncertainty with magnitude uncertainty into a single isotropic distribution.

**Challenge 3: The direction/magnitude independence assumption may be wrong.**

The proposal couples kappa to magnitude, but what if the relationship is more complex? Small-magnitude deltas might have multimodal directions (hesitation between phonemes); large-magnitude deltas might be unimodal but heavy-tailed.

**Verdict:** vMF is worth trying, but the experiment should be designed to be diagnostic. Track whether kappa varies with context, compare vMF to Gaussian likelihood on direction-only prediction, and verify that the direction/magnitude split is optimal (vs. learned subspace factorization).

### 2.3 Alternative Factorizations Considered

| Factorization | Description | Pros | Cons |
|--------------|-------------|------|------|
| Direction/magnitude (vMF) | Predict direction on sphere, magnitude separately | Clean geometric interpretation | Assumes independence |
| Subspace (PCA) | Factor into principal subspace + residual | Learned from data | May not align with dynamics |
| Modal (mixture) | Each mode = transition type | Captures discrete structure | Hard to interpret modes |
| Temporal | Low-pass smooth + high-frequency residual | Addresses "innovation-heavy" hypothesis | Information loss risk |

---

## 3. The Teacher-Forced vs Rollout Gap: Diagnosis and Hypotheses

### 3.1 The Numbers

- Before rollout fine-tuning: Teacher-forced ΔNLL ~-46, Rollout ΔNLL ~+761,868 (explosion)
- After rollout fine-tuning: Teacher-forced ΔNLL ~-39, Rollout ΔNLL ~-0.4

The fine-tuning converts catastrophic divergence into marginal improvement, at the cost of ~38 nats of teacher-forced performance.

### 3.2 Competing Hypotheses

**Hypothesis A: Objective Mismatch**

The K-step rollout loss with per-step supervision creates a perverse incentive. Under covariate shift (model sees p_model(z) not p_data(z)), the easiest way to minimize loss is to predict small deltas. "Do nothing" is stable regardless of where you are.

Evidence: After rollout training, mean predicted ||Δz|| drops from 10.76 to 0.024—a 450x reduction. This isn't hedging; it's functional collapse.

**Hypothesis B: Mode Averaging**

Under teacher forcing, the model sees states from p_data(z). Under rollout, it sees p_model(z). If the model's predictions are multimodal, and p_model(z) puts mass in regions where different modes overlap, the averaged prediction is small.

**Hypothesis C: Insufficient Rollout Diversity**

Deterministic mean rollouts during training expose the model to only one trajectory per ground truth sequence. Sample-based rollouts would expose it to many trajectories, potentially learning more robust policies.

### 3.3 Diagnostic to Distinguish Hypotheses

**Recommended diagnostic:** Compare the model's k=1 behavior during rollout training vs teacher-forcing training.

- Train Model A with teacher-forcing only
- Train Model B with rollout training
- Evaluate **both** models under teacher-forcing at k=1

If Model B predicts smaller Δz at k=1 even under teacher-forcing, it has learned a global conservative prior. If Model B's k=1 predictions match Model A's under teacher-forcing but shrink under rollout, it's learned a conditional policy.

---

## 4. The Teacher-Forcing Injection Experiment

### 4.1 Experimental Design

During evaluation (not training), run rollout but periodically inject ground-truth states:

| Mode | Description |
|------|-------------|
| A | Inject z_true at every step (= teacher forcing) |
| B | Inject z_true at k=4, 8, 12 (periodic correction) |
| C | Inject z_true only at k=1 (initial condition only) |
| D | Pure rollout (no injection) |

Measure ΔNLL at each step under each mode.

### 4.2 What Each Outcome Tells Us

| Outcome | Interpretation | Implication |
|---------|---------------|-------------|
| Advantage decays slowly (B ≈ A) | Model is robust to occasional drift; periodic correction sufficient | Scheduled sampling might help |
| Advantage drops immediately at k=2 with Mode C | One-step error is already catastrophic | Need to train on distribution of predicted states |
| Mode B shows "reset" effect (advantage high after injection, then decays) | Model can recover from drift if given correct conditioning | Distribution shift is recoverable |
| No mode shows advantage | Model's predictions are useless regardless of conditioning | Phase 1 "advantage" was artifact |

### 4.3 Additional Measurements

- **Trajectory divergence curves:** Plot ||z_pred_k - z_true_k|| vs k for each mode
- **Per-step ΔNLL conditioned on prior accuracy:** Is step-k prediction better when step-(k-1) was accurate?
- **Cosine similarity vs L2 error tracking:** During rollout, track both cos(z_pred, z_true) and ||z_pred - z_true||. If cosine stays high while L2 diverges, directional information is preserved.

---

## 5. Memory-as-Basis: Is This Still Engram?

### 5.1 Discrete Engram vs Continuous Soft-Retrieval

| Property | Discrete Engram | Memory-as-Basis |
|----------|----------------|-----------------|
| Key type | Discrete (token sequence) | Continuous (latent state) |
| Retrieval | Exact match → O(1) lookup | Similarity-based → soft attention |
| Value | Fixed embedding per key | Weighted combination of prototypes |
| Gradients | Flow to retrieved embedding only | Flow to all prototypes |
| Computational primitive | Table lookup | Basis expansion |

### 5.2 Is the Distinction Fundamental?

**Yes.** Lookup works by equality ("this key matches that key"). Continuous representations require similarity ("this key is close to that key"). Similarity-based retrieval is inherently fuzzy—you never retrieve exactly the right thing.

However, both mechanisms share the core principle: **offload reusable structure to an explicit store**. In language, the store contains token-sequence → continuation mappings. In audio, the store would contain direction prototypes that can be combined to approximate any transition.

**Verdict:** Memory-as-basis is not Engram, but it preserves Engram's spirit. The honest framing: "learned direction codebook for continuous dynamics" rather than "Engram for audio."

### 5.3 Alternative: More Engram-Like Design

If we want to preserve the Engram retrieval mechanism:

1. Store actual direction prototypes from training (k-means centroids on observed directions)
2. Retrieve by context similarity (soft top-k)
3. Use retrieved directions as mixture means in a movMF

This preserves "lookup actual stored patterns" while adapting to continuous representations.

---

## 6. The Modal Structure Hypothesis

### 6.1 Hypothesis

Audio dynamics are **piecewise continuous with discrete transition types**:

- Within a phoneme: dynamics are smooth (slow drift in formant space)
- At boundaries: dynamics jump (discrete transition type)
- The *type* of transition (vowel→stop, fricative→vowel) is discrete/categorical
- The *realization* of that type varies continuously (speaker, prosody, coarticulation)

This maps onto the direction/magnitude split:
- Direction ≈ transition type (categorical-ish, memorizable)
- Magnitude ≈ transition intensity (continuous, sample-able)

### 6.2 Evidence

- Direction is more predictable than magnitude → transition "type" is more stereotyped
- Phase 0 clustering failed → types aren't capturable by simple context clustering
- Phase 1 MDN succeeded → mixture model can capture multiple types per context

### 6.3 Diagnostic

Take the trained Phase 1 MDN. For each prediction:
1. Identify the most-weighted mixture component
2. Cluster component indices across all predictions
3. Check if frames with same dominant component share linguistic features (phoneme context, prosodic position)

**Caveat:** Mimi wasn't trained with phoneme-level supervision. Its dynamics might correlate with acoustic features (spectral flux, energy envelope) but not linguistic features. Memory-as-basis might capture acoustic transition types rather than phonetic ones.

---

## 7. Sample-Based Rollout Training: Implementation Details

### 7.1 The Problem with Mean Rollouts

Deterministic mean rollouts during training expose the model to only one trajectory per ground truth sequence. The model learns that "mean prediction = correct" and doesn't experience the diversity of outcomes that sampling produces.

### 7.2 Implementation Options

**Option A: REINFORCE**
```
L = E_{Δz ~ p(Δz|z)} [L_rollout(Δz)]
∇L ≈ L_rollout(Δz_sample) · ∇ log p(Δz_sample|z)
```
High variance; requires careful baseline design.

**Option B: Gumbel-Softmax for component selection**
- Sample component k using Gumbel-Softmax (differentiable)
- Sample Δz ~ N(μ_k, σ_k) using reparameterization
- Fully differentiable but biased at high temperature

**Option C: Marginalization over components (recommended)**
```
L_rollout = E_{k ~ Cat(π)} [L_k(rollout using μ_k, σ_k)]
         ≈ (1/M) Σ_{m=1}^M L_{k_m}  where k_m ~ Cat(π)
```
Samples different mixture components but uses reparameterized Gaussian for Δz given component. Lower variance than pure REINFORCE.

---

## 8. Recovering Magnitude After Conservative Training

### 8.1 Approach A: Two-Stage Generation

1. Generate direction trajectory: d_{1:K} via vMF sampling
2. Generate magnitudes conditioned on directions: m_{1:K} ~ p(m | d_{1:K}, z_0)

The second stage can be a separate small model, or even non-autoregressive (since directions are fixed). This decouples "where to go" from "how fast to go."

**Analogy:** Motion capture systems often factor skeletal pose from motion velocity.

### 8.2 Approach B: Magnitude as Explicit Uncertainty

Make the direction/magnitude split explicit in the distribution:
- Direction: high kappa vMF (confident)
- Magnitude: high variance log-normal (uncertain)

At inference, choose magnitude strategy:
- Sample from learned distribution
- Use a magnitude schedule (curriculum over generation)
- Condition on global signal (speaking rate, energy)

**Challenge:** "Start conservative, increase over time" assumes conservatism is monotonically wrong. Need calibration data.

---

## 9. Perceptual Loss Considerations

### 9.1 Multi-Horizon Reconstruction Loss

Instead of per-step NLL, compute trajectory-level perceptual quality:
```
L_traj = ||decode(ẑ_{1:K}) - decode(z_{1:K})||_mel
```

**Pros:** Rewards trajectories that produce similar audio, not point-by-point latent matching.

**Cons:**
- Decoder aliasing: if decoder is locally linear, reduces to weighted L2
- Trajectory-level loss doesn't encourage step-level accuracy (model might "save up" error)
- Expensive: requires decoding, mel computation, backprop through decoder

### 9.2 Frozen Encoder Perceptual Loss (Recommended)

Use WavLM or Hubert features instead of Mimi re-encoding:
```
L_percep = ||WavLM(decode(z_pred)) - WavLM(decode(z_true))||
```

**Advantages:**
- WavLM is trained for noisy/variable audio (more robust than Mimi encoder)
- Captures phonetic content explicitly (ASR-adjacent training)
- Avoids feedback loop from using same encoder for generation and evaluation

---

## 10. Paper Framing Options

### 10.1 Option A: "Engram transfers to audio via factorization"

Memory-based prediction helps continuous audio modeling, but requires factorizing dynamics into direction (discrete-ish, memorizable) and magnitude (continuous, sample-able).

**Strength:** Novel application of Engram principle.
**Weakness:** Contingent on memory working (lowest probability experiment).

### 10.2 Option B: "Rollout training is the bottleneck, not representation"

The Phase 0-4 journey shows that structure exists but is destroyed by training on the wrong objective. Contribution: diagnostic framework for objective mismatch vs representation limitations.

**Strength:** Valuable lessons learned.
**Weakness:** Feels like a negative-result paper.

### 10.3 Option C: "Continuous AR audio needs explicit uncertainty factorization" (Recommended)

Some aspects of audio dynamics are categorical (transition type), some are continuous (transition intensity). Conflating them hurts. The direction/magnitude split is the key insight; memory is secondary.

**Strengths:**
- General contribution (applies to any continuous AR model)
- Diagnostic framework is useful even if memory doesn't help
- Falsifiable claim about the structure of the problem

---

## 11. Probability Estimates and Disagreements

| Experiment | Claude Estimate | Codex Estimate | Notes |
|------------|-----------------|----------------|-------|
| Teacher-forcing injection diagnostic | 0.8 | 0.9 | Both agree this is almost certainly informative |
| vMF vs Gaussian factorization | 0.4 | 0.5 | Moderate confidence; unit normalization is clean inductive bias |
| vMF rollout training → larger magnitudes | 0.25 | 0.15 | **Key disagreement:** Codex more pessimistic; hedging may be fundamental |
| Direction archetype discovery | 0.35 | 0.45 | Codex more optimistic; Phase 1 MDN success is evidence |
| Memory-as-basis improves over pure parametric | 0.2 | 0.15 | Both skeptical; value-add of explicit memory is unclear |

**Codex's concern on rollout training:** Even with vMF factorization, the model faces credit assignment. If bold magnitude prediction causes rollout failure, how does the model know whether direction or magnitude was wrong? Factorization might not break the conservatism feedback loop.

---

## 12. Recommended Experimental Roadmap

### 12.1 Immediate Priority (Diagnostic Phase)

1. **Teacher-forcing injection experiment (Modes A-D)**
   Cost: <1 hour. Information value: high regardless of outcome.

2. **K=1 behavior comparison across training regimes**
   Compare teacher-forcing-trained vs rollout-trained models under teacher-forcing evaluation.

3. **Cosine vs L2 tracking during rollout**
   Does directional information persist while L2 diverges?

4. **MDN component analysis**
   Do dominant mixture components correlate with linguistic/acoustic features?

### 12.2 Secondary Priority (If diagnostics support)

5. **vMF factorization as drop-in for Phase 1**
   Test on single-step prediction before tackling rollout.

6. **Sample-based rollout training**
   Compare to deterministic mean rollouts.

### 12.3 Tertiary Priority (If earlier steps succeed)

7. **Direction archetype extraction**
   Spherical k-means on observed directions; evaluate cross-speaker transfer.

8. **Memory-as-basis**
   Only after archetypes are validated.

### 12.4 Fallback Plan

If direction archetypes don't emerge or memory doesn't help, Option B (diagnostic paper) is still publishable. Document the Phase 0-4 journey as a case study in objective mismatch.

---

## 13. Key Insights and Quotes

### On the Rollout Gap
> "The model isn't 'hedging'—it's correctly optimizing the loss you gave it. The K-step rollout loss penalizes deviations from ground truth at each step. The minimum of this loss is achieved when predicted states track ground truth states. But under rollout, the predicted state drifts from ground truth. The model's 'solution' is to minimize the impact of this drift by predicting small deltas."

### On vMF Factorization
> "The value of vMF isn't better likelihood—it's enabling 'slow but correctly directed' trajectories. Current models conflate directional uncertainty with magnitude uncertainty into a single isotropic distribution. Factorization lets us treat them differently."

### On Memory-as-Basis
> "The operational question is whether audio continuations have discrete modal structure (favoring hard lookup) or smooth manifold structure (favoring soft interpolation). My intuition is that speech has discrete structure (phonemes, prosody patterns) that might favor hard lookup, while music/ambient audio is more continuous."

### On Research Strategy
> "The highest-EV path: Diagnostic experiments first → vMF factorization → direction archetype extraction → only then consider memory. This lets you stop early if any link breaks while still producing publishable insights."

---

## 14. Open Questions for Future Discussions

1. **Does the MDN already capture transition types?** If mixture components are interpretable, we already have implicit memory.

2. **What is the "correct" magnitude recovery strategy?** Two-stage generation vs inference-time calibration vs conditioning on global signals.

3. **How does this relate to discrete audio codecs?** If EnCodec/SoundStream latents show better rollout stability, that supports the "VQ regularization creates AR-friendliness" hypothesis.

4. **Is perceptual quality actually correlated with latent-space metrics?** We should listen to outputs at different checkpoints.

5. **Can we close the loop on Engram?** What would a truly Engram-like memory for audio look like, and would it help?

---

## Appendix A: Implementation Notes for Teacher-Forcing Injection

```python
def evaluate_with_injection(model, z_true, injection_schedule):
    """
    injection_schedule: dict mapping step k to True (inject) or False (rollout)
    e.g., {0: True, 1: False, 2: False, 3: False, 4: True, ...}
    """
    z_current = z_true[0]  # Always start with ground truth
    results = []

    for k in range(1, len(z_true)):
        # Predict from current state
        delta_pred = model.predict(z_current)
        delta_true = z_true[k] - z_true[k-1]

        # Compute NLL of true delta under predicted distribution
        nll = model.nll(delta_true, z_current)
        results.append({'k': k, 'nll': nll, 'injected': injection_schedule.get(k, False)})

        # Update state: inject or rollout
        if injection_schedule.get(k, False):
            z_current = z_true[k]  # Teacher forcing
        else:
            z_current = z_current + delta_pred  # Rollout

    return results
```

---

## Appendix B: Direction Archetype Discovery Protocol

1. **Extract directions:** d_t = Δz_t / ||Δz_t|| for frames where ||Δz_t|| > threshold
2. **Dimensionality reduction:** PCA to 90% variance (curse of dimensionality mitigation)
3. **Spherical clustering:** k-means with cosine distance, sweep K ∈ {32, 64, 128, 256, 512}
4. **Cluster quality metrics:**
   - Within-cluster tightness (mean cosine to centroid)
   - Variance explained
   - Cluster size distribution entropy
5. **Membership predictability:** Train classifier context → cluster_id, evaluate on held-out speakers
6. **Linguistic/acoustic correlation:** Check if clusters correlate with phoneme identity, prosodic position, spectral features

**Decision criteria:**
- If K exists with membership accuracy >60% and tightness >0.75, archetypes are real → proceed to memory-as-basis
- If no such K exists, archetypes are not usable → stick with pure parametric vMF
