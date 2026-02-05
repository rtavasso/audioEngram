# Research Discussion: Latent Space Design for AR Audio Modeling

**Date:** 2026-02-02
**Participants:** Claude (Opus 4.5) + Codex
**Context:** Follow-up discussion on BRIEF.md findings

---

## 1. Revisiting the Phase 0 Failure

### Original Result
- Clustering Mimi latent contexts (mean-pool → VQ or PCA → VQ) explained ~0% of next-step dynamics variance
- Variance ratios of 0.997 and 0.988 against random baselines of 0.999 and 0.995
- Conclusion drawn: Engram-style lookup not viable for Mimi's representation

### Challenges to This Interpretation

**Challenge 1: The metric may be dominated by "latent junk"**
- Some latent dimensions may have high variance but weak effect on decoded audio
- Within-cluster variance of Δx in raw latent space can stay ~1.0 even if *audio-relevant* dynamics are predictable
- **Fix:** Measure variance in a decoder-aware metric (weight Δx by decoder Jacobian / perceptual sensitivity, or measure predictability of decoded mel features)

**Challenge 2: The coarse key destroys predictive equivalence**
- Mean-pooling destroys temporal structure within the context window
- Engram in text benefits from keys that preserve predictive equivalence classes
- If the compression discards speaker/pitch/tempo normalization or phase-like variables, it destroys clusterability even if a different key would work
- **Fix:** Learn a key k(context) explicitly optimized so nearest neighbors have similar next-step distributions

**Challenge 3: MSE/R² can be near-zero even when conditional entropy drops**
- If p(Δx|c) is multimodal and symmetric (two likely directions), the conditional mean is a bad predictor → low R²
- But the distribution is still more structured than marginal
- **Fix:** Use probabilistic metrics (NLL under mixture/flow) or contrastive classification of futures

**Challenge 4: Cluster count may underfit**
- 64-256 clusters vs potentially ~1000+ distinct motifs
- Counter-argument: even with underfitting, we'd expect *some* variance reduction; getting ~0 suggests the dependency is genuinely weak under that key

### Diagnostic Sanity Check
Train a predictor f(context_full) → Δx using linear/MLP/small Transformer:
- If R² is still ~0: representation is genuinely "dynamics-hostile" at this frame rate
- If R² > 0 but clustering fails: the issue is "wrong key/compression" not "no structure exists"

---

## 2. Why Standard Audio Encoders May Not Be AR-Optimal

### The Training Objective Mismatch
Mimi's encoder was trained to:
- Reconstruct audio accurately
- Quantize well (for RVQ)
- Preserve semantic content (via distillation from WavLM)

It was **not** trained to produce latents with:
- Predictable dynamics
- Clusterable transitions
- Low conditional entropy given coarse context

### The Rate/Distortion Hypothesis
Quantization pressure in discrete codecs (EnCodec, DAC) might inadvertently create more predictable dynamics:

1. **Bottlenecking** (limited codebook/bitrate) forces encoder to drop high-entropy, hard-to-predict micro-variation
2. **Clustering/piecewise constancy**: VQ pressure makes latents live near finite prototypes, making local transitions more repeatable
3. **Task simplification**: AR over discrete tokens is classification; continuous AR is regression where small errors compound

**Testable prediction:** Run Phase 0 + predictor baselines on EnCodec/SoundStream/DAC latents at matched temporal resolution

---

## 3. Theoretical Framework: RSSM-Style Factorization

### Core Insight from World Models
In RSSM-style world models (PlaNet/Dreamer lineage), the split between predictable state and stochastic innovation is enforced both architecturally and via loss:

**Architectural:**
- Deterministic recurrent state: h_t = f(h_{t-1}, a_{t-1}, z_{t-1})
- Per-step stochastic latent z_t
- Prior p(z_t | h_t) = what's predictable
- Posterior q(z_t | h_t, o_t) = what the observation forces you to add

**Loss-based:**
- ELBO with reconstruction + KL term: KL(q(z_t|...) || p(z_t|...))
- The KL is the "innovation budget": if something can be carried in h_t (predictable), it's cheaper than paying KL each step

**Practical tricks:**
- KL balancing / free bits / KL schedules to prevent collapse or runaway residual
- These are essentially *rate constraints* on the innovation channel

### Translation to Audio
- z_dyn = predictable state (analogous to h_t)
- z_rec = innovation/residual (analogous to z_t)
- The KL budget on z_rec forces information into z_dyn unless it's truly unpredictable

---

## 4. Proposed Architecture: Dual-Latent Factorization

### Option I: Transform Mimi Latents (Freeze Decoder)
Fastest iteration path. Keep Mimi encoder/decoder frozen, learn a new latent state-space on top.

**Notation:**
- x_t ∈ R^256 = Mimi latents at 12.5 Hz
- z_dyn,t = predictable state (slow, AR-friendly)
- z_rec,t = innovation/residual (detail, mostly local)
- h_t = deterministic recurrent summary (optional)

**Components:**

1. **State encoder (causal):**
   ```
   z_dyn,t = e_dyn(x_≤t)  or  e_dyn(x_{t-L:t})
   ```

2. **Innovation posterior (local):**
   ```
   q_φ(z_rec,t | x_t, z_dyn,t) = N(μ_φ(·), diag(σ²_φ(·)))
   ```
   Critical: don't feed long context into this. Architecturally enforce "innovation is local."

3. **Innovation prior (predictable from state):**
   ```
   p_ψ(z_rec,t | z_dyn,t) = N(μ_ψ(z_dyn,t), diag(σ²_ψ(z_dyn,t)))
   ```
   This is the innovation budget hook.

4. **Latent reconstructor:**
   ```
   p_θ(x_t | z_dyn,t, z_rec,t) = N(g_θ(z_dyn,t, z_rec,t), σ²_x I)
   ```

5. **Dynamics model (AR target):**
   ```
   p_ω(z_dyn,t | z_dyn,<t)  — Transformer/RNN
   ```

**Training Loss (ELBO + distillation):**
```
L = E_q[-log p_θ(x_t | z_dyn,t, z_rec,t)]           # reconstruct Mimi latents
  + β · KL(q_φ(z_rec,t|x_t,z_dyn,t) || p_ψ(z_rec,t|z_dyn,t))  # innovation budget
  + λ · (-log p_ω(z_dyn,t | z_dyn,<t))              # make z_dyn AR-predictable
```

**Inference-time generation:**
1. Generate z_dyn,1:T ~ p_ω(·) autoregressively
2. Sample z_rec,t ~ p_ψ(z_rec,t | z_dyn,t) (local, parallel)
3. Produce x̂_t = g_θ(z_dyn,t, z_rec,t)
4. Decode x̂_1:T with frozen Mimi decoder to waveform

### Option II: End-to-End Dual-Latent Codec
Same structure, but reconstruct waveform/features directly instead of Mimi latents. More likely to yield actual audio gains, but slower to iterate.

### Getting z_rec at Inference Time

| Option | Description | Tradeoffs |
|--------|-------------|-----------|
| A) Deterministic | z_rec = g(z_dyn) | Not really an innovation channel; works if decoder is already stochastic |
| B) Conditional diffusion/flow | p(z_rec \| z_dyn) via diffusion | Most common high-fidelity pattern; adds cost |
| C) Learned conditional noise | Gaussian/mixture/flow conditioned on z_dyn | Good middle ground |
| D) Unconditional prior | Sample from learned marginal | Loses instance-specific detail |
| E) Multi-rate (recommended) | z_dyn slow, z_rec fast but local | Keeps innovation generator cheap |

**Multi-rate detail:** Make z_dyn slow (12.5 Hz or slower) and z_rec strictly local (short-context AR or lightweight diffusion over a few high-rate frames conditioned on nearby z_dyn).

---

## 5. Preventing Failure Modes

### Failure Mode: Decoder Ignores z_dyn
If z_rec is too powerful, the decoder learns to ignore z_dyn entirely.

**Mitigations (combine all):**
1. **Capacity control on z_rec:** KL/bitrate limit, dimensionality, dropout at channel level
2. **Train with prior-sampled z_rec:** A fraction of the time, decode from z_dyn + samples from p(z_rec|z_dyn), not posterior
3. **KL penalty:** Explicitly penalize KL(q(z_rec|x,z_dyn) || p(z_rec|z_dyn))
4. **Occasional masking:** Sometimes set z_rec = sample from prior (prefer this over z_rec = 0)

### Failure Mode: Co-Adaptation / Bad Equilibria
The λ term creates "make whatever you emit easy to predict" pressure. Encoder + dynamics model can collude to make z_dyn trivially predictable while z_rec carries everything.

**Mitigations:**
1. **Tight innovation budget:** Makes routing around z_dyn expensive
2. **Utility objective tied to long-range coherence:** Replace/augment λ with multi-horizon prediction of future Mimi latents or audio features from z_dyn alone
3. **Rollout reconstruction loss (strongest):**
   - Roll out z_dyn forward using dynamics model for K steps (no teacher forcing)
   - At each step, sample z_rec ~ p(z_rec|z_dyn)
   - Decode to x̂_{t:t+K} and penalize reconstruction of ground-truth x_{t:t+K}
   - Use curriculum on K (start at 1-2 steps, grow to 8-16)
   - Mix teacher-forced and free-running rollouts

### Failure Mode: Direction vs Magnitude Confound
Even if R² is low, direction of Δx might be predictable while magnitude is not.

**Diagnostic decomposition:**
- Predict unit direction u_t = Δx / ||Δx|| (evaluate cosine similarity, model with vMF)
- Predict log-magnitude m_t = log(||Δx|| + ε)
- Report explained variance along principal subspace of Δx

If direction is predictable but magnitude isn't: "the *type* of transition is structured, the *intensity* is stochastic/controlled by slow factors" — exactly where mode/state split helps.

---

## 6. Multi-Rate Architecture Details

If z_dyn needs to be slower than 12.5 Hz (e.g., 3 Hz for ~333ms resolution):

### Option A: Piecewise-Constant State (Recommended First)
- z_dyn,k lives on slow grid
- For Mimi frame t, map to k = floor(t / r)
- Pros: Simplest, very "world model"-ish
- Cons: Boundary artifacts if state changes within segment

### Option B: Learned Upsampler
- Small network produces per-frame conditioning c_t = U(z_dyn, t)
- Transposed conv / interpolation + MLP
- Pros: Smooth transitions
- Cons: Upsampler can become "the real model" if too strong

### Option C: Two-Stream with Cross-Attention
- Model z_dyn on slow stream AR
- Model x_t/z_rec,t on fast stream with cross-attention to slow tokens
- Pros: Expressive, standard in multirate transformers
- Cons: Heavier; confounds representation with model capacity

---

## 7. Literature Connections

### Directly Relevant Prior Art

| Work | Connection |
|------|------------|
| **FHVAE** (speech factorization) | Sequence-level vs segment-level latents = "slow state + local residual" |
| **Dreamer/RSSM** | Exactly the innovation-budget framing we're adapting |
| **VQ-VAE-2 / Jukebox** | Multi-scale discrete codes with coarse-to-fine = z_dyn/z_rec with VQ |
| **CPC / wav2vec** | Optimizes discriminative predictability, not generative simplicity |
| **Koopman / E2C** | "Learn embedding where dynamics become simple (linear-ish)" |
| **VRNN / SRNN / Deep Kalman Filter** | "Learn latent Markov state; pay KL for innovations" |

### Key Insight: CPC → Generative Gap
CPC/InfoNCE optimizes *discriminative* predictability, not "make conditional distribution simple":
- A representation can be excellent at ranking true futures vs negatives while being hard to model with AR Gaussian/mixture head
- The field bridges this by: self-supervised reps → discretize/cluster → model tokens generatively
- Consistent with "rate/quantization pressure creates AR-friendliness" hypothesis

### Disentanglement Connection
Disentanglement helps AR *if* it aligns with dynamical timescales and causal structure:
- Slow identity/prosody separated from fast content/residual → simpler dynamics per factor
- Vanilla β-VAE disentanglement doesn't guarantee this alignment
- Sequential disentanglement literature (speech factorization) is most relevant

### Theory of AR-Optimal Representations
The ideal state is the **minimal sufficient statistic of the past for predicting the future** (causal states in computational mechanics, PSRs in control).

Our practical goal: representation that makes the conditional **easy for a chosen model class** (AR transformer + Gaussian/mixture head + memory). This is an information-bottleneck / rate-distortion trade with an explicit dynamics model in the loop.

---

## 8. Experimental Roadmap

### Phase 1: Predictor Baseline (CRITICAL - Do First)

**Goal:** Determine if Mimi latents have any predictable structure at 12.5 Hz

**Method:**
1. Train mixture-density network or lightweight normalizing flow for p(Δx | context)
2. Report ΔNLL = NLL[p(Δx | full context)] - NLL[p(Δx)] at multiple horizons (k=1,2,4,8)
3. Report teacher-forced vs free-running gap
4. Decompose into direction vs magnitude predictability

**Interpretation:**
- ΔNLL ≈ 0: Representation is genuinely innovation-heavy at 12.5 Hz (major update against the whole direction)
- ΔNLL >> 0: Proceed to Phase 2

### Phase 2: Cross-Encoder Comparison

**Goal:** Test rate/distortion hypothesis

**Method:**
- Run Phase 1 diagnostics on EnCodec, SoundStream, DAC latents at matched temporal resolution
- Compare conditional vs marginal NLL across representations

**Interpretation:**
- If discrete/quantized latents show much better predictability: supports "rate/distortion pressure creates AR-friendly structure"

### Phase 3: RSSM Factorization (Option I)

**Goal:** Learn z_dyn + z_rec factorization on top of frozen Mimi

**Ablation priorities:**
1. dim(z_dyn) × KL_target (sweep jointly on coarse grid first)
2. Context length for z_dyn encoder
3. Rollout horizon K for reconstruction loss (curriculum)

**Lower priority initially:**
- p(z_rec|z_dyn) complexity (start Gaussian)
- Dynamics model architecture (start simple)

**Key metrics:**
- Is z_dyn easier to model AR than raw Mimi latents? (conditional NLL)
- What is the innovation rate (KL budget used by z_rec)?
- Does reconstruction quality degrade?

### Phase 4: Engram Integration

**Goal:** Test if Engram-style lookup helps for z_dyn

**Prerequisite:** Phase 3 shows z_dyn is AR-predictable

**Method:**
- Add memory mechanism to dynamics model for z_dyn
- Compare with/without memory at matched parameter count

### Phase 5: End-to-End Evaluation

**Goal:** Compare against CALM on generation quality/efficiency

**Metrics:**
- Total generative cost (FLOPs/latency) at fixed perceptual quality
- Training stability (does it require less noise injection, etc.?)
- Bits/innovation rate: how much KL per second is in z_rec?

---

## 9. Probability Estimates (Codex's Bets)

| Outcome | Probability | Reasoning |
|---------|-------------|-----------|
| Predictor baseline shows meaningful predictability (ΔNLL >> 0) | 0.55 | Phase 0 flatness is concerning, but "wrong key/metric" loophole is real |
| RSSM factorization yields z_dyn easier for AR than raw Mimi | 0.65 | KL innovation budget + prior-sampled training is strong inductive bias |
| If factorization works, Engram helps for z_dyn | 0.45 | Even if AR-friendly, z_dyn might be "smooth-but-nonrepeating" not memorizable |
| Full pipeline beats CALM on quality or efficiency | 0.25 | High bar; frame success as "simpler training / fewer tricks" initially |

**Most likely failure point:** (1) or (3) — either representation is genuinely innovation-heavy, or learned state doesn't have reusable motifs even if predictable.

---

## 10. Single Most Important Next Experiment

**Train the best-effort probabilistic predictor on current Mimi latents.**

Report:
1. ΔNLL = NLL[p(Δx | full context)] - NLL[p(Δx)] (teacher-forced)
2. Same ΔNLL at multiple horizons (k=1,2,4,8)
3. One free-running rollout metric to see teacher-forced vs rollout gap
4. Direction vs magnitude decomposition

**Why this is decisive:**
- If ΔNLL ≈ 0 even with strong predictor: huge update against "representation is hiding structure at 12.5 Hz"
- If ΔNLL >> 0: strongly supports moving to Option I and tells us what horizons/timescales to target

---

## 11. Key Quotes and Insights

> "The λ term is not 'make z_dyn useful', it's 'make z_dyn compressible by ω'. That is necessary for AR-friendliness, but not sufficient for carrying the *right* information."

> "Define 'structure' operationally as 'what you need to predict the future over long horizons under your inference-time constraints.'"

> "The non-handwavy claim you want to be able to make is: 'we factorized state vs innovation so the long-range model only carries low-entropy structure; the remaining entropy is local and cheap to sample.'"

> "If you truly made the process more predictable, the innovation channel should shrink (or become more local), and long-horizon modeling should get easier."

> "CPC can yield 'directional / categorical' predictability without yielding low MSE."

---

## 12. Open Questions

1. **Is the discrete mode structure real or imposed?** Does audio actually have recurring dynamical motifs, or would forcing this structure harm generation quality?

2. **What's the right level of coarseness?** Is there a sweet spot where structure emerges?

3. **Does the encoder need to see the future?** Could a bidirectional encoder during training (but causal during inference) help?

4. **How does this relate to semantic vs acoustic structure?** Is the lack of dynamical structure because Mimi prioritizes semantic over acoustic regularity?

5. **Representation bottleneck vs model bottleneck:** How do we distinguish "the representation is the problem" from "we just need better/bigger AR models"?

---

## Appendix: Stability Training Recipe

**For Option I (RSSM factorization):**

1. **KL warmup:** β from 0 → target over N steps
2. **Free-bits / target-KL:** Per frame for z_rec to prevent collapse or runaway
3. **Stop-gradient tricks:** Feed sg(z_dyn) into prior network early if z_dyn collapses
4. **Prior-sampled training:** Decode with z_rec ~ p(z_rec|z_dyn) a good fraction of time
5. **Rollout curriculum:** Start K=1-2, grow to 8-16; mix teacher-forced and free-running
6. **Feature losses for long rollouts:** Don't rely only on MSE; use mel/low-rank projections
