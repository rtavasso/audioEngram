# Research Brief: Latent Space Design for Autoregressive Audio Modeling

## Executive Summary

We ran a carefully designed experiment to test whether lookup-based memory mechanisms (successful in large language models) could improve autoregressive generation of continuous audio latents. The experiment failed—but the failure points toward a more fundamental question: **Are standard audio encoders producing latent representations that are optimal for autoregressive modeling, or are they optimized for the wrong objective?**

This brief summarizes the background, presents the negative result, and proposes a research direction focused on designing latent spaces specifically for AR modeling.

---

## Part I: Background

### The Two Papers

**CALM (Continuous Audio Language Models):** Demonstrates that audio can be generated autoregressively in the continuous latent space of a VAE, avoiding the lossy quantization of discrete codec approaches (like RVQ). CALM achieves strong results but requires several stabilization techniques: injecting noise during training, using a short-context transformer for local detail, and replacing diffusion with consistency modeling. These work, but they're engineering patches for a fundamental difficulty—continuous latents are hard to model autoregressively.

**Engram (Conditional Memory for LLMs):** Shows that large language models benefit from explicit lookup-based memory for local patterns. Instead of reconstructing common phrases like "Alexander the Great" from scratch at every occurrence, the model retrieves a pre-computed embedding via O(1) lookup. This frees up network depth for higher-level reasoning. The key finding: allocating ~20-25% of sparse parameters to memory (instead of more MoE experts) improves performance across diverse tasks—not just knowledge retrieval, but reasoning, code, and math.

### The Hypothesis We Tested

We asked: **Does audio have the same kind of reusable local structure that makes Engram work in language?**

In language, knowing the recent tokens strongly constrains what comes next. "Alexander the" almost certainly precedes "Great." This is *Type 2 structure*—the same prediction rule applies across many contexts.

In audio, we hypothesized that knowing the recent latent trajectory might similarly constrain the next-step dynamics. Phoneme transitions, coarticulation effects, and prosodic patterns might create recurring "dynamical motifs" that could be stored and retrieved rather than recomputed.

### The Experiment (Phase 0)

Before building any models, we tested whether the requisite structure exists:

1. Extract continuous latents from a pretrained Mimi encoder (the architecture underlying CALM) on LibriSpeech
2. For each frame, record the recent context and the next-step velocity (Δx = x_t - x_{t-1})
3. Cluster contexts using deliberately coarse representations (mean-pooling, PCA + VQ with small codebooks)
4. Measure how much clustering reduces variance in the velocity

**Decision criterion:** Proceed only if within-cluster variance ratio falls below 0.6 (clusters explain >40% of dynamics variance).

### The Result

| Condition | Variance Ratio | Random Baseline |
|-----------|---------------|-----------------|
| Mean pool → VQ(64) | 0.997 | 0.999 |
| PCA(8) → VQ(256) | 0.988 | 0.995 |

**The clustering explains essentially nothing.** Coarse context does not predict next-step dynamics. The conditional distribution p(Δx | coarse_context) is indistinguishable from the marginal p(Δx).

**Conclusion:** Engram-style lookup is not viable for Mimi's latent representation. There is no reusable local structure to exploit.

---

## Part II: Why Did It Fail?

The failure is specific to *this representation*, not necessarily to audio in general.

Mimi's encoder was trained to:
- Reconstruct audio accurately
- Quantize well (for RVQ)
- Preserve semantic content (via distillation from WavLM)

It was **not** trained to produce latents with predictable dynamics or clusterable transitions. The encoder's objective says nothing about whether the resulting latent space is easy to model autoregressively.

This is the standard pipeline for audio generation:
1. Train encoder to minimize reconstruction loss
2. Freeze encoder
3. Train AR model on frozen latents

The AR model is asked to predict sequences in a space designed for a completely different purpose.

---

## Part III: What Latent Structure Would Make Engram Work?

For Engram to work, the latent space must satisfy:

**The conditional distribution p(x_t | coarse(context)) must have significantly lower entropy than the marginal p(x_t).**

This requires two geometric properties:

### 1. Locally low-dimensional dynamics

At any given context, the distribution of likely next steps should concentrate in a small region—not a diffuse cloud over the full latent space. The conditional covariance should have low effective rank.

### 2. Globally discrete modes

The *types* of local dynamics should be finite and recurring. There should be K "dynamical modes" that repeat across speakers and utterances. The space of conditional distributions should be well-approximated by a finite mixture.

Together, these imply a **mixture of low-dimensional manifolds** structure:

```
x_t ≈ μ_mode(context) + ε_t
```

Where mode is discrete (K values) and ε is low-dimensional residual noise.

Standard VAEs have no pressure toward this structure. KL regularization actively discourages it—pushing toward smooth Gaussians rather than discrete mixtures.

---

## Part IV: Research Questions

This leads to the core questions I want to explore:

### Question 1: Can we design an encoder/decoder such that the latent structure works well with Engram?

**Sub-question 1a:** What training objectives would encourage Engram-friendly structure?

Candidates:
- **Predictability loss:** Reward encoders that produce latents where a simple model (lookup table, linear map) can predict x_t from coarse context
- **Conditional concentration:** Penalize high variance of x_t within context clusters
- **Mode factorization:** Explicitly decompose latents into discrete mode + continuous residual

**Sub-question 1b:** Is such an encoder necessarily a VAE?

The VAE framework (encoder → latent → decoder with KL regularization) may not be the right structure. Alternatives:
- VQ-VAE with dynamics-aware codebook learning
- Autoencoder with explicit mode-residual factorization
- Contrastive predictive coding adapted for generation
- Flow-based models with structured base distributions

The question is whether reconstruction-based training can ever produce AR-optimal latents, or whether we need a fundamentally different objective.

### Question 2: What is the structure of latent representations that makes them optimal for AR modeling?

This is the theoretical question underlying everything.

**Hypothesis:** AR-optimal latents have:
- Smooth, predictable local dynamics (low Lipschitz constant on trajectories)
- Disentangled factors (content, speaker, prosody in orthogonal subspaces)
- Discrete global modes with continuous local variation
- Information rate matched to what the AR model can handle (not preserving unpredictable noise)

**Key tension:** Reconstruction-optimal and AR-optimal may conflict. A latent space that preserves everything (excellent reconstruction) might be harder to model than one that smooths out unpredictable variation.

**Empirical question:** Can we characterize existing latent spaces (Mimi, EnCodec, SoundStream, DAC) along these axes? Which properties correlate with AR modeling difficulty?

### Question 3: Do two complementary latent representations benefit AR modeling more than either alone?

**The idea:** Instead of a single latent space trying to serve both reconstruction and AR modeling, use two:

1. **Reconstruction latent (z_rec):** Optimized for faithful reconstruction, like standard Mimi. May be high-dimensional, noisy, hard to predict.

2. **Dynamics latent (z_dyn):** Optimized for predictable local structure, Engram-friendly. May lose some fine detail but captures the "skeleton" of the trajectory.

The AR model operates primarily on z_dyn (which is easy to predict), while z_rec provides the detail needed for high-fidelity output.

**Architectural options:**
- Hierarchical: z_dyn is a coarse summary of z_rec
- Parallel: Two separate encoders, decoder takes both
- Residual: z_dyn captures predictable component, z_rec captures residual

**Hypothesis:** The combination outperforms either alone because:
- z_dyn alone: Easy to model but loses detail
- z_rec alone: Full detail but hard to model (current situation)
- Both: Easy modeling of structure + full detail for reconstruction

This is analogous to coarse-to-fine hierarchies in image generation, but motivated by AR modeling difficulty rather than spatial resolution.

---

## Part V: Concrete Next Steps

### Immediate (analysis, no training)

1. **Characterize existing latent spaces:** Run Phase 0 analysis on EnCodec, SoundStream, DAC, and any other available pretrained audio encoders. Do any of them show more structure than Mimi?

2. **Analyze Mimi's latent geometry:** What do the principal components of Δx capture? Is variance concentrated in a few dimensions? Is there structure in Δx itself (even if context doesn't predict it)?

3. **Literature review:** What work exists on AR-aware representation learning? Predictive coding, world models, dynamics-aware autoencoders in other domains (video, robotics)?

### Short-term (small-scale training)

4. **Predictability-regularized VAE:** Train Mimi-like encoder with added loss term rewarding predictability from coarse context. Sweep regularization strength. Run Phase 0 on resulting latents.

5. **Mode-factorized encoder:** Explicitly decompose latent into discrete mode (VQ) + continuous residual. Does this expose structure that vanilla VAE hides?

### Medium-term (if early results are promising)

6. **Dual-latent architecture:** Implement and test the two-representation idea. Does z_dyn + z_rec outperform either alone?

7. **Full AR modeling:** Train CALM-style models on the new representations. Do they require less stabilization (noise injection, etc.)? Do they achieve better quality at matched compute?

---

## Part VI: Open Questions for Discussion

1. **Is the discrete mode structure real or imposed?** Does audio actually have recurring dynamical motifs, or would forcing this structure harm generation quality? How would we know?

2. **What's the right level of coarseness?** Our Phase 0 used very aggressive compression (8-dimensional PCA, 64-256 clusters). Is there a sweet spot where structure emerges?

3. **Does the encoder need to see the future?** Contrastive predictive coding uses future prediction as a training signal. Could a bidirectional encoder during training (but causal during inference) help?

4. **How does this relate to semantic vs acoustic structure?** Mimi uses WavLM distillation for semantic content. Is the lack of dynamical structure because Mimi prioritizes semantic over acoustic regularity?

5. **Is there relevant work in other domains?** Video prediction, robotic control, and music all involve AR modeling of continuous trajectories. Have those fields solved this problem?

---

## Summary

We tested whether lookup-based memory could improve AR audio generation. It can't—at least not with current latent representations. But the failure reveals a deeper issue: standard audio encoders aren't designed to produce latents that are easy to model autoregressively.

The research direction is to design latent spaces specifically for AR modeling—either through modified training objectives, explicit structural factorization, or complementary dual representations.

The core bet: **There's significant headroom in AR audio generation that's currently being left on the table because we're using representations optimized for the wrong objective.**