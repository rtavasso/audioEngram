# Engineer Onboarding: Audio Engram Research Summary

**Date:** 2026-02-04
**Purpose:** Bring engineers up to speed on research trajectory and next experiments

---

## 1. Project Context (30-second version)

We're testing whether **lookup-based memory** (successful in LLMs via Engram) can improve **autoregressive generation of continuous audio latents**. The core question:

> Do short-horizon audio latent dynamics have reusable local structure that can be captured by lookup, improving over pure parametric prediction?

We use **Mimi** (a neural audio codec) to encode audio into 512-dimensional continuous latents at 12.5 Hz, then try to predict the next latent given context.

---

## 2. What We've Learned (Phases 0-4)

### Phase 0: Structure Detection — FAILED
- Tried clustering Mimi latent contexts to predict next-step dynamics
- **Result:** Clusters explained ~0% of dynamics variance (ratios 0.997-0.988 vs random baseline 0.995-0.999)
- **Conclusion:** Naive Engram-style lookup doesn't work for Mimi latents

### Phase 1: Parametric Baseline — SUCCEEDED
- Trained MDN (Mixture Density Network) to predict Δz given context
- **Result:** Teacher-forced ΔNLL ~-46 nats/frame (35% entropy reduction)
- **Key finding:** Direction is predictable (cosine ~0.65-0.71), magnitude is not (R² < 0)

### Phase 4: Rollout Training — PARTIALLY SUCCEEDED
- Trained with K=16 step unrolled loss to prevent divergence during generation
- **Before:** Rollout ΔNLL ~+761,868 (catastrophic explosion)
- **After:** Rollout ΔNLL ~-0.4 (stable but conservative)
- **Problem:** Model learned to predict near-zero deltas (||Δz|| dropped 450x)

### The Core Problem
Teacher-forcing: ΔNLL = -35 nats (great)
Rollout: ΔNLL = -1 nat (barely better than baseline)

The model "hedges" during rollout by predicting "stay still" because any confident prediction compounds into disaster.

---

## 3. Key Insights from Recent Discussions

### Insight 1: The Direction/Magnitude Split
- **Direction** of Δz is predictable (cosine ~0.65-0.71)
- **Magnitude** of Δz is unpredictable (R² < 0)
- Current MDN conflates these, forcing the model to hedge on both

### Insight 2: vMF Factorization
Model direction and magnitude separately:
```
Δz = magnitude × direction
direction ~ vMF(μ_dir, κ)      # concentrated on unit sphere
magnitude ~ LogNormal(μ_m, σ_m) # uncertain
```

This enables **"slow but correctly directed"** trajectories instead of **"stationary"** ones.

**Hypothesis:** Perceptual quality correlates more with direction (spectral contours, phonetic trajectory) than magnitude (speed). Preserving direction while hedging on magnitude may sound better than predicting nothing.

### Insight 3: The Gap May Be Representation-Specific
Mimi latents are "innovation-heavy" by design (optimized for reconstruction, not predictability). Comparisons:
- Game world models achieve stable rollouts because observations are engineered to be low-entropy
- Audio at 12.5 Hz may alias phoneme boundaries (50-100ms typical)
- Other codecs (EnCodec) or higher frame rates might show dramatically different predictability

### Insight 4: Modal Structure is Acoustic, Not Phonetic
Mimi has no phoneme supervision. Direction "archetypes" would be:
- Formant movement patterns (F1 rising + F2 falling)
- Pitch contour types (rising, falling, level)
- Energy transitions (onset, offset, stress)

These are continuous but may have recurring patterns amenable to soft memory lookup.

### Insight 5: Memory-as-Basis (Not Engram)
Continuous latents can't do exact lookup. Instead, use memory as a **learned basis of direction prototypes**:
```python
α = softmax(cos_similarity(z_t, prototype_keys))
μ_dir = normalize(Σ_k α_k · prototype_directions_k)
```

This preserves Engram's spirit (offload reusable structure) with different mechanics.

---

## 4. Experiments to Implement

### Tier 1: Immediate Priority

#### Experiment 1: vMF Factorization (Drop-in for MDN)
**Goal:** Test whether factorizing direction/magnitude improves rollout behavior

**Implementation:**
```python
# Direction head
mu_dir = normalize(f_dir(z_t))  # D-dim unit vector
log_kappa = g_kappa(z_t)        # scalar concentration

# Magnitude head
mu_mag = h_mu(z_t)              # scalar
log_sigma_mag = h_sigma(z_t)    # scalar

# Loss
L = vMF_NLL(d_true | mu_dir, exp(log_kappa))
  + LogNormal_NLL(m_true | mu_mag, exp(log_sigma_mag))
```

**Sampling:** Use Wood's algorithm for high-dimensional vMF (O(1) iterations regardless of D=512)

**Metrics to track:**
- ΔNLL (teacher-forced and rollout)
- cos(Δz_pred, Δz_true) at each rollout step
- ||Δz_pred|| / ||Δz_true|| ratio at each rollout step
- **Listen to outputs** — is "slow but correct" perceptually better?

**Success criterion:** cos(Δz_pred, Δz_true) > 0.6 at K=16 rollout

#### Experiment 2: Teacher-Forcing Injection Diagnostic
**Goal:** Understand where the rollout gap comes from

**Implementation:** During evaluation, inject ground-truth states periodically:
| Mode | Description |
|------|-------------|
| A | Inject z_true every step (= teacher forcing) |
| B | Inject z_true at k=4, 8, 12 (periodic correction) |
| C | Inject z_true only at k=1 (initial condition only) |
| D | Pure rollout (no injection) |

**What to measure:**
- ΔNLL at each step under each mode
- ||z_pred_k - z_true_k|| trajectory divergence curves
- Does advantage "reset" after injection?

**Interpretation:**
- If B ≈ A: Model is robust to occasional drift → scheduled sampling might help
- If advantage drops immediately at k=2 with Mode C: One-step error already catastrophic
- If Mode B shows "reset" effect: Distribution shift is recoverable

#### Experiment 3: Representation Comparison
**Goal:** Determine if the gap is Mimi-specific or fundamental to audio

**Implementation:**
- Extract EnCodec latents for LibriSpeech subset
- Run Phase 1 predictor baseline on EnCodec
- Test Mimi at 50 Hz instead of 12.5 Hz

**Success criterion:** If other representation shows >2x ΔNLL improvement, switch representations

---

### Tier 2: Conditional on Tier 1 Success

#### Experiment 4: Direction Archetype Discovery
**Goal:** Find reusable direction prototypes for memory-as-basis

**Protocol:**
1. Extract directions: d_t = Δz_t / ||Δz_t|| for ||Δz_t|| > threshold
2. PCA to 90% variance
3. Spherical k-means for K ∈ {32, 64, 128, 256, 512}
4. Evaluate:
   - Within-cluster tightness (mean cosine to centroid)
   - Cross-speaker transfer accuracy
   - Correlation with acoustic features (spectral flux, pitch change, energy)

**Success criterion:** K exists with tightness >0.75 and cross-speaker accuracy >60%

#### Experiment 5: Sample-Based Rollout Training
**Goal:** Expose model to diverse trajectories during training

**Current problem:** Deterministic mean rollouts train model that "mean = correct"

**Implementation:** Marginalization over MDN components:
```python
L_rollout = E_{k ~ Cat(π)} [L_k(rollout using μ_k, σ_k)]
         ≈ (1/M) Σ_{m=1}^M L_{k_m}  where k_m ~ Cat(π)
```

Sample different mixture components but use reparameterized Gaussian for Δz given component.

---

### Tier 3: Longer-term

#### Experiment 6: Memory-as-Basis Implementation
Only after archetypes are validated in Experiment 4.

#### Experiment 7: Score-Based Correction
If vMF rollouts still drift off-manifold:
1. Train unconditional score model on Mimi latents via denoising
2. Post-hoc Langevin correction: z ← z + ε·s(z) + √(2ε)·ξ

**Probability estimate:** 0.35 that this helps without prohibitive cost

#### Experiment 8: Two-Stage Generation
If vMF succeeds but magnitude remains problematic:
1. Stage 1 (AR): Generate direction trajectory
2. Stage 2 (parallel): Sample magnitudes conditioned on global features (speaker, speaking rate)

---

## 5. Decision Tree

```
Implement vMF + magnitude factorization
           |
           v
Does rollout stability improve vs MDN?
          / \
        Yes  No
         |    |
         v    v
   Direction  Debug magnitude
   archetype  issues; try κ(m)
   discovery        |
         |          v
         v      Still no?
   Do archetypes    |
   exist?           v
      / \       Representation
    Yes  No     comparison
     |    |
     v    v
   Memory- Stick with
   as-basis  pure vMF
     |
     v
   Does memory help?
      / \
    Yes  No
     |    |
     v    v
   Success! Score-based
            correction
```

---

## 6. Probability Estimates (Team Consensus)

| Hypothesis | Probability | Notes |
|------------|-------------|-------|
| vMF preserves direction under rollout (cos >0.6 at K=16) | 0.60 | Most promising immediate intervention |
| Archetypes exist (K with tightness >0.75) | 0.50 | Acoustic (not phonetic) structure |
| Memory-as-basis helps (if archetypes exist) | 0.25 | Soft retrieval can handle continuous structure |
| Score-based correction improves stability | 0.30 | Complex; defer until simpler approaches exhausted |
| Representation change shows >2x improvement | 0.65 | Gap may be Mimi-specific |

---

## 7. Implementation Notes

### vMF Sampling (Wood's Algorithm)
```python
def sample_vmf(mu, kappa, D):
    """Sample from von Mises-Fisher distribution."""
    # Step 1: Sample w via rejection (1D, efficient)
    b = (-2*kappa + np.sqrt(4*kappa**2 + (D-1)**2)) / (D-1)
    a = (D-1 + 2*kappa + np.sqrt(4*kappa**2 + (D-1)**2)) / 4
    d = 4*a*b / (1+b) - (D-1)*np.log(D-1)

    while True:
        eps = np.random.beta((D-1)/2, (D-1)/2)
        w = (1 - (1+b)*eps) / (1 - (1-b)*eps)
        t = 2*a*b / (1 - (1-b)*eps)
        u = np.random.uniform()
        if (D-1)*np.log(t) - t + d >= np.log(u):
            break

    # Step 2: Sample v uniformly orthogonal to mu
    v = np.random.randn(D)
    v = v - (v @ mu) * mu
    v = v / np.linalg.norm(v)

    # Step 3: Combine
    x = w*mu + np.sqrt(1 - w**2)*v
    return x
```

### vMF Log-Likelihood
```python
def vmf_log_prob(x, mu, kappa, D):
    """Log probability under vMF."""
    # Log normalization constant
    log_C = (D/2 - 1) * np.log(kappa) - (D/2) * np.log(2*np.pi) \
            - np.log(scipy.special.iv(D/2 - 1, kappa))
    return log_C + kappa * (mu @ x)
```

### Cosine to Kappa Conversion
For D=512, using E[μᵀd] ≈ 1 - (D-1)/(2κ):
- Observed cosine ~0.7 → κ ≈ 850

---

## 8. Key Files to Read

| File | Purpose |
|------|---------|
| `TEAM_BRIEF.md` | Original project motivation and Phase 0 specification |
| `RESEARCH_DISCUSSION5.md` | vMF formulation, memory-as-basis concept |
| `RESEARCH_DISCUSSION6.md` | Critical analysis, diagnostic experiments |
| `RESEARCH_DISCUSSION7.md` | Pushback analysis, updated probability estimates |

---

## 9. Success Criteria Summary

**Experiment 1 (vMF):** cos(Δz_pred, Δz_true) > 0.6 at K=16, perceptual quality > baseline
**Experiment 3 (Representation):** Other representation shows >2x ΔNLL improvement
**Experiment 4 (Archetypes):** K exists with tightness >0.75, cross-speaker accuracy >60%

---

## 10. Paper Framing (If Successful)

**Recommended framing:** "Continuous AR audio needs explicit uncertainty factorization"

- Some aspects of audio dynamics are categorical (transition type/direction)
- Some are continuous (transition intensity/magnitude)
- Conflating them hurts; the direction/magnitude split is the key insight
- Memory is secondary to this structural contribution

This framing is general (applies to any continuous AR model) and falsifiable.
