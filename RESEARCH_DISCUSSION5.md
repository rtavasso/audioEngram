# Research Discussion 5: vMF Direction Modeling, Memory-as-Basis, and Experimental Priorities

**Date:** 2026-02-04
**Participants:** Claude (Opus 4.5) + Codex
**Context:** Deep dive following Phase 4/4.5 results on rollout stability and the teacher-forced vs rollout performance gap

---

## 1. Executive Summary

This discussion revisits the Engram hypothesis in light of Phases 0-4.5 findings. Key conclusions:

1. **The original Engram formulation (lookup-based memory with discrete keys) is fundamentally limited for continuous audio latents** because there are no exact matches—only approximate similarity.

2. **A reformulated approach may succeed**: factorize dynamics into direction (predictable) and magnitude (unpredictable), model direction with von Mises-Fisher distributions on the unit sphere, and use memory as a "learned basis" of direction prototypes rather than as a residual corrector.

3. **The teacher-forced vs rollout gap** (-35 vs -1 nats ΔNLL) is likely due to mean shrinkage: the model learns to hedge during rollout because confident predictions compound into catastrophe.

4. **A concrete experimental roadmap** is proposed: (1) minimal vMF factorization, (2) direction archetype discovery, (3) vMF rollout training, (4) memory-as-basis if warranted.

---

## 2. The Central Puzzle: Why Does Rollout Training Lose Teacher-Forced Advantage?

### 2.1 Observed Gap

After Phase 4.5 rollout fine-tuning:
- Teacher-forced ΔNLL: ~-35 nats/frame (substantial improvement over baseline)
- Rollout ΔNLL: ~-1 nat/frame (barely better than baseline)

The model becomes stable during rollout but loses most of its predictive advantage.

### 2.2 Hypothesis: Mean Shrinkage

When training with K-step unrolled loss, the model optimizes:
```
L_rollout = sum_{k=1}^{K} NLL(Δz_k | ẑ_{k-1})
```

The gradient pushes toward predictions that do not compound into disaster. The simplest way to satisfy this is to **shrink the mean toward zero** and **inflate sigma**.

Evidence: Post-finetune, `mean ||Δẑ||` becomes "extremely small." The model is hedging—predicting "approximately stay where you are" is safer than committing to a direction that might be wrong after accumulated drift.

### 2.3 The Distribution Shift Problem

Under teacher forcing, the model sees states from `p_data(z)`. Under rollout, it sees states from its own induced distribution `p_model(z)`. These diverge by construction unless the model is perfect.

The gap may reflect: on-distribution states contain strong predictive signal; off-distribution states (even slightly off) do not. This is a fundamental limitation—you cannot make `p_model(z)` match `p_data(z)` by optimizing likelihood alone.

### 2.4 Reframing: Is the Gap Actually Bad?

The rollout ΔNLL of -1 means the model retains *some* predictive advantage even after 16 steps of accumulation. Most AR models in high-dimensional continuous spaces explode entirely. Ours merely becomes conservative.

The question is not "why did we lose 34 nats" but "can we recover some of those nats while maintaining stability?"

---

## 3. vMF Direction Modeling: A Better Factorization

### 3.1 The Direction/Magnitude Split

Phase 1 established:
- Direction (unit vector of Δz) is predictable: cosine ~0.61-0.71
- Magnitude (||Δz||) is not predictable: R² < 0 at all horizons

This suggests factoring:
```
Δz = magnitude × direction
```

where direction lives on the unit sphere and magnitude is a positive scalar.

### 3.2 Why vMF for Direction?

The current MDN predicts full Δz as a mixture of isotropic Gaussians. Direction is implicit—extracted post-hoc for evaluation.

A von Mises-Fisher (vMF) distribution directly models directions on the unit sphere:
```
p(d | μ, κ) = C_D(κ) · exp(κ · μᵀd)
```

where:
- μ is the mean direction (unit vector)
- κ is the concentration parameter (higher = more peaked)
- C_D(κ) is a normalization constant

### 3.3 Translating Observed Cosine to Kappa

For D=512 dimensions, using the approximation:
```
E[μᵀd] ≈ 1 - (D-1)/(2κ)
```

With observed cosine ~0.7:
```
κ ≈ 511/(2 × 0.3) ≈ 850
```

This is a moderate-to-high concentration regime—not a point mass, but far from uniform.

### 3.4 Magnitude-Dependent Kappa

The direction of a small-magnitude delta is dominated by noise. The model should be uncertain about direction when magnitude is small.

Proposed coupling:
```
μ_dir, log_κ_base, μ_mag, σ_mag = network(z_t)
magnitude ~ LogNormal(μ_mag, σ_mag)
κ = exp(log_κ_base) × sigmoid(magnitude / scale)
direction ~ vMF(μ_dir, κ)
Δz = magnitude × direction
```

This naturally downweights direction prediction for small deltas.

### 3.5 Sampling from High-Dimensional vMF

Standard rejection sampling has low acceptance rates in D=512. **Wood's algorithm (1994)** is the efficient alternative:

1. Sample scalar w from a 1D marginal (efficient rejection sampling)
2. Sample direction v uniformly on the (D-1)-sphere orthogonal to μ
3. Combine: x = w·μ + √(1-w²)·v

This has O(1) expected iterations regardless of D.

### 3.6 Magnitude Distribution

Magnitude may have heavy tails (silence vs transitions). Recommended: **mixture of log-normals** with state-dependent weights:
```
p(m) = Σ_k π_k · LogNormal(m | μ_k, σ_k)
```

Start with K=2 components (small/large magnitudes) and increase if needed.

---

## 4. Memory Revisited: From Lookup to Learned Basis

### 4.1 Why Lookup Fails for Continuous Representations

In language, Engram works because:
- Tokens are discrete: "cat" = "cat"
- Exact matches exist
- Lookup retrieves exactly the right information

In continuous latents:
- States are continuous: z₁ ≈ z₂ but z₁ ≠ z₂
- No exact matches
- Lookup retrieves approximately relevant information

This asymmetry is fundamental. Lookup works by equality; continuous representations require similarity-based retrieval, which is inherently fuzzy.

### 4.2 Memory-as-Basis: A Reformulation

Instead of "lookup and copy," think of memory as a "learned basis" for directions:
```
prototypes = {d_1, ..., d_K}  # K direction prototypes on unit sphere
α = softmax(attention(z_t, prototypes))  # soft selection
μ_mem = normalize(Σ_k α_k · d_k)  # weighted direction
```

The model's job is not to find a similar key and copy its value, but to express the dynamics as a combination of learned prototypes.

This sidesteps the key-matching problem. The question becomes: can direction dynamics be well-approximated by a sparse linear combination of a fixed set of basis vectors?

### 4.3 Do Direction Archetypes Exist?

This is the key empirical question. Testable via:

1. **Extract direction vectors**: d_t = Δz_t / ||Δz_t|| for frames where ||Δz_t|| > threshold
2. **Dimensionality reduction**: PCA to 90% variance (curse of dimensionality mitigation)
3. **Spherical clustering**: k-means with cosine distance, sweep K ∈ {32, 64, 128, 256, 512}
4. **Cluster quality**: within-cluster tightness (mean cosine to centroid), variance explained
5. **Membership predictability**: train classifier context → cluster_id, evaluate on held-out speakers

**Decision criteria:**
- If K exists with membership accuracy >60% and tightness >0.75, archetypes are real → pursue memory-as-basis
- If no such K exists, archetypes are not usable → stick with pure parametric vMF

### 4.4 Other Memory Roles Considered

| Role | Assessment |
|------|------------|
| Memory for kappa prediction | Skeptical—kappa is a scalar, parametric network can learn this directly |
| Memory for magnitude modes | If modes exist, mixture model already captures them |
| Memory for rollout anchoring | May help stability but could pull toward wrong part of manifold; worth trying with care |

---

## 5. Rollout Training: What We Learned and What to Try Next

### 5.1 Current State

K=16 rollout training converts catastrophic divergence into stable rollouts. But:
- The model becomes conservative (small mean predictions)
- Teacher-forced advantage is largely lost during rollout

### 5.2 Sample-Based Rollout Training

Instead of mean rollouts (deterministic), train with sampled rollouts:
```
ẑ_{t+1} = ẑ_t + sample(p(Δz | ẑ_t))
```

This exposes the model to a wider range of trajectories and may prevent conservative collapse.

**Challenge**: High variance gradients if using REINFORCE.

**Solutions**:
- Rao-Blackwellization: compute expected gradient over mixture component selection analytically
- Reparameterization: Gumbel-Softmax for component selection (biased at high temperature)

### 5.3 Distributional Matching

Likelihood-based training encourages hedging. Alternative: train a discriminator to distinguish rolled-out trajectories from ground truth. The dynamics model is trained to fool the discriminator.

This directly optimizes "produce plausible trajectories" rather than "match this specific trajectory."

Risks: mode collapse, training instability. Could use discriminator as auxiliary loss alongside NLL.

### 5.4 Why Increasing K Alone May Not Help

If the model has already learned "play it safe" at K=16, longer K will reinforce this strategy. The intervention needed is not longer horizons but different objectives or architectural constraints.

---

## 6. The Mimi Representation: Is It AR-Hostile?

### 6.1 Potential Issues

Mimi was trained for:
- Reconstruction quality
- Quantization (RVQ)
- Semantic preservation (WavLM distillation)

It was **not** trained for:
- Predictable dynamics
- Smooth trajectories
- Low conditional entropy given context

### 6.2 Why Quantization May Create Innovation-Heavy Latents

RVQ encourages latents to be close to codebook centroids, not smooth over time. The latent trajectory may "jump" between centroids to minimize quantization error at each frame, creating artificial discontinuities.

### 6.3 The Causal Encoder Limitation

For streaming, Mimi's encoder is causal. It must commit to a latent before knowing future audio. This creates "catch-up" dynamics where the latent jumps to accommodate new information.

### 6.4 Options If Representation Is the Bottleneck

1. **Fine-tune encoder**: Add temporal smoothness loss (acceleration penalty) during continued training
2. **Post-hoc transformation**: Train lightweight network to smooth Mimi latents
3. **Accept and adapt**: Treat innovation-heavy representation as given, design model to handle it

Fine-tuning risks: decoder expects original distribution; changing encoder may cause artifacts.

---

## 7. Connection Between Smoothing and vMF

### 7.1 The Hypothesis

Smoothing removes high-frequency variation. In the smoothed space:
```
Δx_smooth = Δx_raw - Δnoise
```

The direction of Δx_smooth is more consistent because jitter is removed. This could increase kappa (tighter directional distribution) and improve predictability.

### 7.2 The Tradeoff

- **Gain**: More predictable directions, higher kappa, better AR modeling
- **Cost**: Information about high-frequency structure may be lost

The key question: Is the high-frequency structure (a) perceptually important and (b) predictable?

If unimportant: discard it, model only smooth part.
If important but unpredictable: model as noise, sample from unconditional distribution.
If important and predictable: do not smooth—model it properly.

### 7.3 Proposed Hybrid

Smooth aggressively for direction modeling (high kappa vMF on smoothed representation). Model magnitude in raw space (captures energy of transitions, which is perceptually important).

---

## 8. Updated Thesis

**One-paragraph summary:**

Autoregressive modeling of continuous audio latents from Mimi faces a fundamental challenge: the dynamics contain both predictable structure (direction of change) and unpredictable variation (magnitude of change). The original Engram hypothesis—that lookup-based memory can offload local pattern recognition as in language models—fails in its naive form because continuous latents lack exact matches. However, a reformulated version may succeed: if the space of latent *directions* is low-dimensional and populated by reusable archetypes (common phonetic transitions, prosodic contours), then memory can store these archetypes as a learned basis. The parametric model's role shifts from predicting directions from scratch to selecting and refining archetypes. Magnitude, being fundamentally unpredictable, should be modeled separately with an appropriate heavy-tailed distribution and magnitude-dependent directional uncertainty. Rollout stability requires training on unrolled sequences; memory's role during rollout is not to correct predictions but to anchor states toward the training manifold when drift occurs. The viability of this approach hinges on whether direction archetypes exist—a testable hypothesis via spherical clustering and cross-speaker generalization analysis.

---

## 9. Experimental Roadmap

### 9.1 Priority Order

| Priority | Experiment | Effort | Key Question |
|----------|------------|--------|--------------|
| 1 | vMF + magnitude factorization (minimal) | Low | Does factorization help rollout stability? |
| 2 | Direction archetype discovery | Medium | Do reusable direction prototypes exist? |
| 3 | vMF rollout training (K=16, K=32) | Low | How does vMF behave under rollout? |
| 4 | Memory-as-basis | Medium-High | Does explicit memory close the gap? |
| 5 | Temporal smoothing | Medium | Does representation-level intervention help? |

### 9.2 Minimum Viable vMF Experiment

Swap MDN head for vMF + log-normal heads:
```python
# Direction head
mu_dir = normalize(f_dir(z_t))  # D-dimensional unit vector
log_kappa = g_kappa(z_t)        # scalar

# Magnitude head
mu_mag = h_mu(z_t)              # scalar
log_sigma_mag = h_sigma(z_t)    # scalar

# Loss
L = vMF_NLL(d_true | mu_dir, exp(log_kappa))
  + LogNormal_NLL(m_true | mu_mag, exp(log_sigma_mag))
```

Do not need full magnitude mixture or magnitude-dependent kappa initially. Test the factorization first.

### 9.3 Decision Tree

```
Implement minimal vMF + magnitude factorization
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
      / \       Revisit
    Yes  No     assumptions
     |    |
     v    v
   vMF   Stick with
   rollout  pure vMF
   training
     |
     v
   Does gap persist?
      / \
    Yes  No
     |    |
     v    v
   Memory- Success!
   as-basis
     |
     v
   Does memory help?
      / \
    Yes  No
     |    |
     v    v
   Success! Temporal
            smoothing
```

### 9.4 Direction Archetype Discovery Protocol

1. Extract directions: d_t = Δz_t / ||Δz_t|| for ||Δz_t|| > threshold
2. PCA to 90% variance
3. Spherical k-means for K ∈ {32, 64, 128, 256, 512, 1024}
4. Metrics per K:
   - Within-cluster tightness (mean cosine to centroid)
   - Variance explained
   - Cluster size distribution (entropy)
5. Membership predictability: classifier context → cluster_id, held-out speakers
6. Decision: sweet spot with accuracy >60% and tightness >0.75 → proceed to memory-as-basis

---

## 10. Key Insights and Quotes

### On Mean Shrinkage
> "Under rollout training, predicting 'approximately stay where you are' is much safer than predicting 'move confidently in the right direction.' A small mean with large variance gets moderate NLL on any outcome, while a confident directional prediction gets catastrophic NLL if the direction is wrong after a few steps of drift."

### On Memory's Fundamental Limitation
> "Lookup works by equality (this key matches that key). Continuous representations require similarity (this key is close to that key). Similarity-based retrieval is inherently fuzzy."

### On the Reformulated Thesis
> "The spirit is similar—offload reusable structure to an explicit memory mechanism—but the implementation is substantially different. This is appropriate: the original Engram was designed for discrete tokens, and we are adapting to continuous dynamics."

### On Research Strategy
> "Stay disciplined. One intervention at a time, with clear metrics and decision criteria at each step. If you add everything at once and it works, you will not know why. If it fails, you will not know what to fix."

---

## 11. Open Questions for Future Discussions

1. **Is magnitude truly unpredictable, or predictable from different features?** Test: condition magnitude on utterance-level features (speaker, energy, duration). If R² improves, consider multi-rate architecture.

2. **Does smoothing help direction predictability?** Quick test: apply 3-tap moving average, re-measure direction cosine.

3. **What is the right rollout objective?** Likelihood encourages hedging. Consider adversarial/distributional objectives as auxiliary losses.

4. **At what timescale does predictable structure exist?** 12.5 Hz may be aliasing phoneme boundaries. Test at higher frame rates (50-100 Hz).

5. **Is this still "Engram"?** The final system may look quite different—soft retrieval, direction prototypes, learned basis rather than hash lookup. The contribution becomes "the principle of separating reusable structure from context-specific computation extends to continuous domains, but requires different mechanisms."

---

## Appendix A: vMF Implementation Notes

### Wood's Algorithm for Sampling

```python
def sample_vmf(mu, kappa, D):
    """Sample from von Mises-Fisher distribution.

    Args:
        mu: mean direction, shape (D,), unit norm
        kappa: concentration parameter, scalar > 0
        D: dimensionality

    Returns:
        x: sample, shape (D,), unit norm
    """
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
    v = v - (v @ mu) * mu  # project out mu component
    v = v / np.linalg.norm(v)  # normalize

    # Step 3: Combine
    x = w*mu + np.sqrt(1 - w**2)*v
    return x
```

### vMF Log-Likelihood

```python
def vmf_log_prob(x, mu, kappa, D):
    """Compute log probability under vMF.

    The normalization constant involves Bessel functions.
    For numerical stability, use log-space computation.
    """
    # Log normalization constant
    # C_D(kappa) = kappa^{D/2-1} / ((2*pi)^{D/2} * I_{D/2-1}(kappa))
    log_C = (D/2 - 1) * np.log(kappa) - (D/2) * np.log(2*np.pi) \
            - np.log(scipy.special.iv(D/2 - 1, kappa))

    # Log probability
    log_p = log_C + kappa * (mu @ x)
    return log_p
```

---

## Appendix B: Literature References

- **Engram (original)**: Conditional memory for local pattern retrieval in LLMs
- **RSSM / Dreamer**: World models with state/innovation factorization
- **vMF**: Directional statistics on unit spheres (Mardia & Jupp, 2000)
- **Wood's algorithm**: Efficient vMF sampling (Wood, 1994)
- **Scheduled sampling**: Curriculum from teacher-forcing to free-running (Bengio et al., 2015)
- **CALM**: Continuous audio latent modeling with consistency heads
- **Mimi**: Streaming neural audio codec with RVQ and semantic distillation
