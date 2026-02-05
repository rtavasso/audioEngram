# Research Discussion 7: Critical Pushback Analysis

**Date:** 2026-02-04
**Participants:** Claude (Opus 4.5) + Codex
**Context:** Deep critical engagement with outstanding questions following Phase 0-4 results

---

## Executive Summary

This discussion responds to five fundamental pushbacks on the research direction. Key findings:

1. **The gap may not be "fundamental" but rather representation-specific.** Audio at 12.5 Hz Mimi latents may be innovation-heavy by design, not by necessity. Game observations in world models have lower entropy *by construction* (discrete action spaces, engineered state representations).

2. **vMF's value is not about preventing error compounding—it's about changing the *form* of compounding.** "Slow but correctly directed" trajectories may preserve perceptual structure better than "stationary" or "fast but directionless" ones. The hypothesis: perceptual quality correlates more with direction than magnitude.

3. **The "modal structure" in Mimi latents is likely acoustic, not phonetic.** Directions may correspond to spectral envelope changes, pitch contours, and energy dynamics—all continuous but with recurring patterns. This changes archetype discovery from "find phoneme transitions" to "find acoustic motifs."

4. **Score function approaches and diffusion-style correction deserve serious consideration.** The score ∇log p(z) provides "which direction moves toward high-density regions"—exactly what we need for rollout stability. This connects to recent work on score-based trajectory correction.

5. **The reservoir computing analogy is surprisingly apt but underspecified.** Mimi encoder as fixed reservoir, z_dyn as readout state. The key insight: reservoirs exploit high-dimensional transient dynamics; we're trying to compress those dynamics into low-dimensional AR-friendly state. This is backwards from standard reservoir computing.

---

## 1. ON THE "FUNDAMENTAL" NATURE OF THE GAP: World Models vs Audio

### 1.1 Your Challenge

> "World models (Dreamer, IRIS, DIAMOND) achieve stable long-horizon rollouts in continuous observation spaces. What's different about audio?"

This is the right question. If world models can handle continuous 64-128 dimensional observations at 4-16 Hz for thousands of steps, why can't we handle 512-dimensional audio latents at 12.5 Hz?

### 1.2 Key Differences: Game Observations vs Audio Latents

| Property | Game Observations (Atari, DMC) | Mimi Audio Latents |
|----------|--------------------------------|--------------------|
| **Entropy rate** | Low: most pixels static between frames | High: speech is inherently high-bandwidth signal |
| **Action conditioning** | Yes: dynamics caused by discrete actions | No: generative (unconditional on actions) |
| **Markov order** | Low: state + last action suffices | Unknown: may require long context |
| **Engineering** | Observations designed for control (diagnostic) | Latents optimized for reconstruction, not AR |
| **Stochasticity** | Game state is deterministic given actions | Audio has irreducible aleatoric uncertainty |
| **Training regime** | Model-based RL: rollout errors penalized via reward | Generative: no reward signal to correct drift |

### 1.3 The Entropy Density Hypothesis

You're right to suspect audio has higher entropy density. Let's quantify:

**Atari at 64x64x3 RGB, 4 Hz:**
- Most pixels unchanged frame-to-frame (background static)
- Effective degrees of freedom: object positions, scores, ~10-100 dims
- Conditional entropy given last state + action: very low

**Mimi latents at 512-dim, 12.5 Hz:**
- Every dimension changes every frame (no "background")
- Conditional entropy: our Phase 1 results show ΔNLL ~-46 nats teacher-forced
- Baseline entropy: -130 nats
- **Entropy reduction: 35%** (46/130)

For comparison, language models achieve 60-80% entropy reduction (predictable given context). World models likely achieve >80% for game observations.

**Conclusion:** Audio at this representation is inherently more innovation-heavy than game observations.

### 1.4 But Is This Fundamental to Audio or Specific to Mimi?

**Critical distinction:** Mimi latents were optimized for:
1. Reconstruction quality
2. RVQ quantization
3. Semantic content (WavLM distillation)

They were **not** optimized for predictability. In contrast, world model observations are often engineered to be Markovian and low-entropy (e.g., stacking frames, extracting features).

**Testable hypotheses:**
1. **Frame rate:** Audio at 12.5 Hz may alias phoneme boundaries (typical phones are 50-100ms). Increasing to 50 Hz might reveal smoother dynamics.
2. **Representation:** A VAE trained with temporal smoothness losses might have lower entropy dynamics.
3. **Discrete codecs:** EnCodec/SoundStream with VQ might force lower entropy through quantization pressure (your "VQ creates AR-friendliness" hypothesis from Discussion 1).

**Action item:** Run Phase 0-1 diagnostics on EnCodec latents at matched frame rate. If predictability dramatically improves, the gap is representation-specific, not fundamental to audio.

### 1.5 Where You're Right

Audio probably does have higher *intrinsic* entropy than vision:
- Speech has ~40 phonemes with coarticulation effects
- Prosody adds continuous variation (pitch, duration, intensity)
- Speaker characteristics add another layer of variation

But "higher" doesn't mean "impossible." It means we need better inductive biases.

### 1.6 Where I Was Wrong

In Discussion 6, I treated the rollout gap as primarily an objective mismatch problem. Your pushback highlights that **game observations are a different regime entirely**. World models succeed not just because of better training, but because the task is easier:
- Observations are low-entropy by design
- Actions provide strong conditioning signal
- Reward provides supervision beyond likelihood

We're tackling a harder problem. The gap may be partially fundamental, partially representation-specific, and partially objective-related. Disentangling these requires comparative experiments.

---

## 2. ON vMF NOT SOLVING THE PROBLEM: The Form of Compounding

### 2.1 Your Challenge

> "If the model predicts direction with cos(d_hat, d_true) = 0.7, and magnitude m_hat = 0, errors still compound. Is there a regime where 'slow but correctly directed' is perceptually better than 'stationary' or 'fast but wrong direction'?"

This is a profound reframing. Let me think through it rigorously.

### 2.2 The Space of Rollout Failures

Consider three failure modes:

| Mode | Magnitude | Direction | Trajectory | Perceptual Impact |
|------|-----------|-----------|------------|-------------------|
| **A: Stationary** | m=0 | n/a | Stays put | Monotone/robotic; prosody lost |
| **B: Correct direction, slow** | m << m_true | cos ~0.7 | Right path, lagging | Natural contours, reduced dynamics |
| **C: Wrong direction, fast** | m = m_true | cos ~0 | Random walk | Incoherent; phonetic structure lost |

Current models (post-rollout training) are in **Mode A**: mean predicted ||Δz|| = 0.024 vs true 10.76.

Your hypothesis: **Mode B might be perceptually superior to A**, and possibly even preferable to attempting Mode C and failing.

### 2.3 Why This Might Be True

**Argument 1: Perceptual quality correlates more with direction than magnitude**

Direction in latent space likely corresponds to:
- **Spectral tilt** (which formants are moving)
- **Phonetic trajectory** (vowel → consonant transitions)
- **Prosodic contour** (rising vs falling pitch)

Magnitude corresponds to:
- **Speed** of transition
- **Energy** of utterance

If you preserve formant trajectories but at 50% speed, you get slower speech. If you scramble formant trajectories, you get unintelligible noise.

**Argument 2: Temporal compression is less catastrophic than spectral scrambling**

Human listeners are remarkably tolerant of time-scale modification (we can understand speech at 0.5x-2x speed). We are much less tolerant of spectral distortion.

**Argument 3: Magnitude uncertainty is irreducible, direction uncertainty is not**

Phase 1 showed:
- Direction cosine: 0.69 (concentrated)
- Magnitude R²: <0 (unpredictable)

If magnitude is fundamentally aleatoric (depends on speaking rate, emotion, individual variation), then hedging on magnitude while committing to direction may be the Bayes-optimal strategy.

### 2.4 The vMF Factorization Hypothesis (Refined)

**Claim:** vMF factorization enables Mode B (correct direction, uncertain magnitude) rather than forcing Mode A (stationary).

**Mechanism:**
```
Current MDN: p(Δz) = Σ π_k N(μ_k, σ_k I)
→ Hedging on Δz conflates directional and magnitude uncertainty
→ Rollout training shrinks ||μ_k|| to minimize risk

vMF + magnitude: p(Δz) = p(direction) × p(magnitude)
→ p(direction) = vMF(μ_dir, κ)  [concentrated]
→ p(magnitude) = LogNormal(μ_m, σ_m)  [uncertain]
→ Rollout training can shrink magnitude without destroying direction
```

**Key insight:** By factorizing, we allow the model to express "I'm confident about WHERE we're going (direction), uncertain about HOW FAST (magnitude)."

### 2.5 Testable Prediction

If this hypothesis is correct, vMF-trained models under rollout should exhibit:
1. **Preserved directional alignment** even after K steps
2. **Reduced magnitude** relative to ground truth
3. **Better perceptual quality** than MDN-trained models at same rollout horizon

**Diagnostic:**
- Plot cos(Δz_pred, Δz_true) vs ||Δz_pred||/||Δz_true|| at each rollout step
- For vMF: expect cosine stays high (~0.6-0.7) while magnitude ratio drops
- For MDN: expect both cosine and magnitude ratio drop together

### 2.6 The Post-Hoc Magnitude Upscaling Idea

You suggest:
> "Maybe the RIGHT thing to do is preserve direction and let magnitude be sampled/upscaled post-hoc."

**Two-stage generation:**
1. **Stage 1 (AR):** Generate direction trajectory d_{1:T} ~ p(d_t | d_{<t}, z_0)
2. **Stage 2 (parallel):** Sample magnitudes m_{1:T} ~ p(m | context, style)
   - Condition on global features: speaker, speaking rate, utterance length
   - Or use a separate magnitude model (non-AR or short-context AR)

**Advantages:**
- Decouples "what to say" (direction/content) from "how to say it" (magnitude/prosody)
- Magnitude model can be simpler (maybe even non-autoregressive)
- Rollout training focuses on preserving content, not matching exact dynamics

**Challenges:**
- Requires identifying global conditioning variables for magnitude
- Risk: if magnitude is uncoupled from direction, phonetic structure may be lost (e.g., stop consonants require rapid transitions)

**Verdict:** Worth exploring. Start with simplest version: generate directions AR, sample magnitudes from unconditional empirical distribution. Evaluate perceptual quality.

### 2.7 Where I Agree

vMF doesn't "solve" compounding in the sense of eliminating it. But it **changes the failure mode** from "collapse to stationary" to "slow but correct." If perceptual quality is more sensitive to directional errors than magnitude errors, this is a meaningful improvement.

### 2.8 Where I'm Uncertain

The hypothesis depends on:
1. Perceptual quality correlating more with direction than magnitude (needs perceptual evaluation)
2. vMF actually preserving direction under rollout (needs empirical test)
3. Magnitude being safely sample-able post-hoc (unclear; may destroy phonetic timing)

**Probability estimate:** 0.55 that vMF rollout exhibits Mode B behavior (correct direction, reduced magnitude) rather than collapsing to Mode A (stationary). This is the single most important empirical question right now.

---

## 3. ON THE MODAL STRUCTURE HYPOTHESIS: Acoustic vs Phonetic

### 3.1 Your Challenge

> "Mimi has no phonetic supervision. What if the 'directions' are NOT phoneme transitions but spectral envelope changes, pitch contours, energy envelopes? These are continuous, not discrete. Does this change the archetype hypothesis?"

Absolutely, and this is a critical reframing.

### 3.2 Phonetic vs Acoustic Structure

**Original hypothesis (implicit):**
- Directions correspond to phoneme transitions (vowel→stop, fricative→vowel)
- These are discrete categories (finite phoneme inventory)
- Therefore, directions cluster into discrete archetypes

**Your reframing:**
- Directions correspond to acoustic features (formant movements, F0 contours, energy envelopes)
- These are **continuous** but may have **recurring patterns**
- Archetypes are not discrete categories but regions of high density in direction space

### 3.3 What Mimi Actually Encodes

Mimi was trained with:
1. **Reconstruction loss:** Minimize ||x - decode(encode(x))||
2. **VQ loss:** Latents close to RVQ codebook centroids
3. **Semantic distillation:** Match WavLM representations

WavLM is trained on masked prediction (SSL) and captures:
- Phonetic content (can be fine-tuned for ASR)
- Speaker identity (can be fine-tuned for verification)
- Prosody (pitch, stress patterns)

But WavLM is **frame-level**, not transition-level. Mimi latents likely encode:
- **Spectral envelope** (formant structure at each frame)
- **Pitch** (if preserved by WavLM)
- **Energy**

The **dynamics** (Δz) would then reflect:
- **Formant movements:** (e.g., F1 rising + F2 falling = back vowel → front vowel)
- **Pitch contours:** Rising, falling, level
- **Energy changes:** Onset, offset, stress

### 3.4 Continuous Patterns with Recurring Structure

Your point: these are continuous, not discrete. True. But **continuous doesn't mean unstructured**.

**Analogy:** Human motion capture data
- Joint angles are continuous
- But motion primitives exist: "reaching," "walking," "sitting"
- These primitives are not discrete categories but attractors in a continuous space
- Clustering works because similar motions (different people walking) follow similar trajectories

**Applied to audio:**
- Formant movements are continuous
- But phonetic categories impose constraints (e.g., /i/ → /a/ always involves F1 rising)
- Even without phonetic labels, these constraints create recurring patterns
- "Archetypes" are not discrete bins but regions of high density

### 3.5 Does This Change the Archetype Discovery Protocol?

**Original protocol (from Discussion 5):**
1. Extract directions d_t = Δz_t / ||Δz_t||
2. PCA to 90% variance
3. Spherical k-means for K ∈ {32, 64, 128, 256, 512}
4. Evaluate cluster quality and membership predictability

**Updated protocol (same mechanics, different interpretation):**
1. **Extract directions** (unchanged)
2. **PCA** (unchanged)
3. **Spherical clustering** (unchanged)
4. **Evaluate:**
   - **Cluster tightness:** Do we find concentrated regions?
   - **Cross-speaker transfer:** Are clusters speaker-independent?
   - **Acoustic correlation:** Do clusters correlate with acoustic features (spectral flux, pitch change, energy derivative)?
   - **NOT phonetic correlation** (Mimi has no phonetic labels)

**Key difference:** We're looking for **acoustic motifs**, not phonetic categories. Success looks like:
- K=256-512 clusters with tightness >0.75
- Clusters that generalize across speakers
- Clusters that correlate with interpretable acoustic events (formant movement types, pitch contour types)

### 3.6 Does Continuous Structure Undermine Memory-as-Basis?

**Your concern:** If archetypes are continuous, discrete memory lookup won't work.

**Response:** "Memory-as-basis" was always soft retrieval, not hard lookup:
```
α = softmax(cos_similarity(z_t, prototype_keys))
μ_dir = normalize(Σ_k α_k · prototype_directions_k)
```

This works fine for continuous structure. The prototypes span the space of common directions; soft weighting interpolates smoothly.

**Analogy:** Basis functions in function approximation (Fourier basis, RBF basis). Prototypes are like basis functions—you express any direction as a weighted sum.

### 3.7 Revised Hypothesis Statement

**Original:** "Discrete phoneme transitions create discrete direction archetypes."

**Revised:** "Acoustic dynamics (formant movements, pitch contours, energy envelopes) create recurring patterns in direction space. These patterns are continuous but structured, enabling soft memory-based retrieval."

### 3.8 Where You're Right

The structure is acoustic, not phonetic. This means:
- Archetypes may be more speaker-dependent (vocal tract differences)
- Archetypes may be more context-dependent (prosody, speaking rate)
- The number of archetypes may be larger (continuous space needs more prototypes)

But none of this invalidates the approach. It just changes the scale and interpretation.

---

## 4. SCORE FUNCTION APPROACH: Diffusion Meets Autoregression

### 4.1 Your Proposal

> "Instead of modeling p(Δz | z) directly, model the score function ∇_z log p(z). The score tells you 'which direction moves toward high-density regions.' During rollout, use the score to keep trajectories on-manifold."

This is a fascinating idea. Let me think through it carefully.

### 4.2 Score Matching Basics

In diffusion models and score-based generative models:
- **Score function:** s(z) = ∇_z log p(z)
- Intuitively: points "uphill" toward regions of high probability density
- Training: minimize ||s_θ(z) - ∇_z log p(z)||² via score matching (denoising)

### 4.3 Application to Temporal Sequences

**Standard AR:** Model p(z_t | z_{<t}) and sample z_t ~ p(·|z_{<t})

**Score-based correction:**
1. Predict next state: z_pred ~ p(z_t | z_{<t})
2. Compute score: s(z_pred) = ∇_z log p(z)
3. Correct toward manifold: z_corrected = z_pred + ε · s(z_pred)

**Why this might help:**
- AR predictions drift off-manifold due to compounding errors
- Score provides "local gradient toward training distribution"
- Correction pulls predictions back on-manifold

### 4.4 Has This Been Tried for Temporal Sequences?

**Yes, in several forms:**

| Work | Approach | Domain |
|------|----------|--------|
| **Score-based diffusion for time series** | Model p(x_{1:T}) with score, sample via Langevin dynamics | Finance, climate |
| **Diffusion-based video generation** | Diffusion over latent trajectories | Video (MCVD, RaMViD) |
| **DYffusion (2023)** | Diffusion models for dynamical systems | Physics simulations |
| **Score-based trajectory optimization** | Model score of state distribution, use for planning | Robotics |

**Key pattern:** Most successful applications use diffusion **over entire trajectories**, not step-by-step correction. Single-step score correction tends to be unstable.

### 4.5 Challenges for Audio Latent AR

**Challenge 1: Score of what distribution?**

The score s(z_t) = ∇_z log p(z_t) requires marginal p(z_t), but we want conditional p(z_t | z_{<t}).

Options:
- **Marginal score:** Model p(z) over all frames (ignores temporal conditioning)
- **Conditional score:** Model ∇_z log p(z_t | z_{<t}) (not clear how to use this; it's a gradient wrt z_t, not z_{t-1})
- **Trajectory score:** Model ∇_{z_{1:T}} log p(z_{1:T}) and optimize entire trajectory (expensive)

**Challenge 2: Score doesn't tell you where to go next**

The score ∇_z log p(z) tells you "from position z, move in this direction to increase density." But for generation, you're at z_t and need to predict z_{t+1}. The score at z_t doesn't directly give you z_{t+1}.

**Possible resolution:** Use score to constrain AR predictions:
```
Objective: max p(z_t | z_{<t})  subject to  ||z_t - z_t^0|| < ε
where z_t^0 = z_{t-1} + Δz_pred
and constraint enforces staying near manifold via score-based projection
```

This is a hybrid: AR for dynamics, score for regularization.

**Challenge 3: Training the score model**

Score matching requires:
- Large dataset of z samples
- Denoising: add noise, train model to predict score
- Computational cost: additional model parallel to AR model

### 4.6 Practical Instantiation: Denoising AR States

**Concrete proposal:**

1. **Train score model:** s_φ(z) ≈ ∇_z log p(z) via denoising score matching on Mimi latents
2. **AR with denoising step:**
   ```
   Δz_pred = AR_model(z_t)
   z_raw = z_t + Δz_pred
   z_corrected = denoise(z_raw, s_φ, n_steps=1-5)
   ```
3. **Denoising:** Langevin dynamics: z ← z + ε·s_φ(z) + √(2ε)·ξ

**Advantages:**
- Score model is separate (can train offline on Mimi latents)
- Correction is post-hoc (doesn't require changing AR training)
- Explicit manifold constraint

**Disadvantages:**
- Two models (AR + score)
- Denoising adds latency (though 1-5 steps is cheap)
- Score model may not know about temporal coherence (trained on marginal)

### 4.7 Alternative: Incorporate Score Loss into AR Training

**Training objective:**
```
L = NLL(z_t | z_{<t}) + λ · ||s_φ(z_pred) - s_φ(z_true)||²
```

This encourages AR predictions to land in regions where score is similar to true states (i.e., on-manifold).

**Advantages:**
- Single model
- Score provides implicit regularization during training

**Disadvantages:**
- Requires pre-trained score model
- Gradient flow may be difficult (score model should be frozen or carefully co-trained)

### 4.8 My Assessment

**Pros:**
- Score-based methods have strong theoretical grounding
- Addresses the "off-manifold drift" problem directly
- Could complement AR (not replace it)

**Cons:**
- Significant engineering complexity (two models, denoising loop)
- Score of marginal p(z) may not be the right object (we need conditional)
- Unclear whether single-step corrections suffice (trajectory-level diffusion is expensive)

**Probability estimate:** 0.35 that score-based correction improves rollout stability without prohibitive cost.

**Recommendation:**
- **Defer** until we've tested vMF factorization
- If vMF rollout still drifts off-manifold, score-based denoising is a logical next step
- Practical first step: train unconditional score model on Mimi latents (via denoising), apply post-hoc to rollouts, measure whether it helps

---

## 5. RESERVOIR COMPUTING ANALOGY: Fixed Dynamics, Learned Readout

### 5.1 Your Proposal

> "Echo State Networks separate fixed nonlinear reservoir (captures dynamics) from learned linear readout (makes predictions). Could Mimi encoder BE the reservoir and z_dyn the readout?"

This is a surprisingly deep connection. Let me unpack it carefully.

### 5.2 Reservoir Computing Recap

**Standard Echo State Network (ESN):**
1. **Input:** x_t (e.g., sensor data)
2. **Reservoir:** Large fixed recurrent network: h_t = f(W_in · x_t + W_res · h_{t-1})
   - W_in, W_res are **fixed** (random or pre-designed)
   - High-dimensional (100s-1000s of units)
   - Captures nonlinear dynamics
3. **Readout:** Linear map: y_t = W_out · h_t
   - W_out is **learned**
   - Trained via simple regression (least squares)

**Key principle:** Reservoir provides rich nonlinear expansion; readout extracts task-relevant features.

**Why it works:**
- High-dimensional reservoirs have long memory via transient dynamics
- Fixed weights avoid expensive recurrent training
- Linear readout is efficient and stable to train

### 5.3 Mapping to Audio Generative Modeling

**Your proposal:**
- **Reservoir = Mimi encoder (frozen)**
- **Readout state = z_dyn (learned)**

Let's formalize:
1. **Input:** Audio x_t
2. **Reservoir (Mimi encoder):** z_t = Encode(x_t)
   - High-dimensional (512)
   - Captures nonlinear dynamics
   - **Fixed** (pre-trained, frozen)
3. **Readout state (z_dyn):** Low-dimensional state extracted from z_t
4. **Dynamics model:** p(z_dyn,t | z_dyn,<t) (AR on readout state)

### 5.4 Where the Analogy Works

**Similarity 1: Separation of concerns**
- Reservoir/Mimi: Captures complex nonlinear transformations
- Readout/z_dyn: Extracts predictable low-dimensional structure

**Similarity 2: Avoiding expensive training**
- Reservoir: Fixed weights (no backprop through time)
- Mimi: Pre-trained, frozen (no need to retrain encoder)

**Similarity 3: High-dimensional expansion**
- Reservoir: Projects input to high-dimensional space with rich dynamics
- Mimi: Projects audio to 512-dimensional latent space

### 5.5 Where the Analogy Breaks Down

**Critical difference 1: Purpose of high-dimensional representation**

**ESN reservoir:**
- Purpose: Create **transient dynamics** that maintain long memory
- Mechanism: Recurrent connections with carefully tuned spectral radius
- High dimensionality enables rich dynamics (different units respond to different timescales)

**Mimi encoder:**
- Purpose: Create **compact representation** for reconstruction
- Mechanism: Encoder-decoder architecture, no recurrence
- High dimensionality is for fidelity, not dynamics

**Critical difference 2: Information flow**

**ESN:**
- Reservoir receives input at every timestep: h_t = f(x_t, h_{t-1})
- Reservoir dynamics are driven by input stream
- Readout is purely feedforward from reservoir state

**Our setup:**
- Mimi encoder is frame-independent: z_t = Encode(x_t) (no temporal conditioning in encoder)
- z_dyn must extract dynamics from **sequence** of z_t (not single frame)
- z_dyn is recurrent/autoregressive, not a feedforward readout

**Critical difference 3: Training objective**

**ESN:**
- Readout optimized for **prediction** of target y_t
- Simple supervised learning (regression)

**Our setup:**
- z_dyn optimized for **generation** (autoregressive likelihood)
- Also must reconstruct z_t (or x_t) via decoder
- Multi-objective, adversarial (if using rollout training)

### 5.6 The Backwards Direction

You note:
> "Reservoirs exploit high-dimensional transient dynamics; we're trying to compress those dynamics into low-dimensional AR-friendly state. This is backwards."

**Exactly.** Standard reservoir computing:
- Low-dimensional input → High-dimensional reservoir → Low-dimensional readout
- Expansion in the middle enables rich dynamics

**Our task:**
- High-dimensional Mimi latents → Low-dimensional z_dyn → High-dimensional reconstruction
- Compression in the middle for AR tractability

This is **inverse reservoir computing**: finding low-dimensional coordinates that explain high-dimensional observations.

### 5.7 What We Can Borrow from Reservoir Computing

**Insight 1: Separation of representation and prediction**

Reservoir computing succeeds because:
1. Reservoir provides good features (nonlinear expansion, memory)
2. Readout is simple (linear, easy to train)

Applied to our task:
1. Mimi provides good features (semantic content, spectral structure)
2. z_dyn should extract simple, predictable features (low entropy dynamics)

**Insight 2: Fixed representation can be sufficient**

Reservoirs work with **random fixed weights**. You don't need to train the representation for the task—you just need it to be rich enough.

Applied to our task:
- Mimi is pre-trained for reconstruction, not generation
- But if Mimi latents are rich enough, we can extract z_dyn that IS good for generation
- We don't need to fine-tune Mimi encoder

**Insight 3: Dynamics live in a low-dimensional subspace**

Even though reservoir state is high-dimensional, the task-relevant dynamics often live in a low-dimensional subspace (hence linear readout works).

Applied to our task:
- Mimi latents are 512-dimensional
- But predictable dynamics may live in a ~32-64 dimensional subspace
- z_dyn should capture this subspace

### 5.8 Concrete Application: Dimensionality Reduction as Readout

**Reservoir-inspired approach:**

1. **Extract Mimi latent sequence:** z_{1:T} from audio x
2. **Learn projection:** z_dyn,t = W · [z_{t-L:t}]
   - W is a learned linear projection (like ESN readout)
   - Captures predictable subspace
3. **AR dynamics on z_dyn:** p(z_dyn,t | z_dyn,<t)
4. **Reconstruct:** z_t = Decode(z_dyn,t) + residual

This is **linear dynamical system (LDS)** in the latent space—a classic reservoir computing readout.

**Advantages:**
- Simplicity (linear projection)
- Interpretability (can analyze W to see which Mimi dimensions are predictable)
- Stability (linear maps are well-behaved)

**Disadvantages:**
- Linear projection may be too weak (dynamics may be nonlinear)
- Requires residual to reconstruct full z_t (adds complexity)

### 5.9 My Assessment

The reservoir computing analogy is **insightful but incomplete**:

**What it gets right:**
- Separation of complex representation (Mimi) from simple dynamics (z_dyn)
- Fixed representation can be sufficient (don't need to fine-tune encoder)
- Predictable structure lives in a low-dimensional subspace

**What it misses:**
- We're doing **compression** (inverse reservoir), not expansion
- Mimi encoder is not recurrent (no temporal dynamics)
- Our task is generative, not predictive (different optimization landscape)

**Practical takeaway:** The reservoir computing perspective suggests:
1. **Keep Mimi frozen** (like fixed reservoir)
2. **Learn simple readout** (e.g., linear or shallow network z_t → z_dyn,t)
3. **Test whether low-dimensional dynamics suffice** (like ESN readout dimension)

**Probability estimate:** 0.4 that a reservoir-inspired linear readout (z_dyn = W · [z_{t-L:t}]) improves AR predictability vs full nonlinear encoder.

---

## 6. Synthesis: Revised Research Priorities

### 6.1 What I Now Believe

1. **The gap is not fundamental to audio, but may be representation-specific.** Mimi latents are innovation-heavy by design. Better representations (higher frame rate, temporal smoothness, discrete quantization) might close the gap.

2. **vMF factorization may enable "slow but correct" generation**, which could be perceptually preferable to stationary collapse. This is the most promising immediate intervention.

3. **"Modal structure" is acoustic, not phonetic**, and continuous, not discrete. Archetypes are regions of high density in direction space, not discrete categories.

4. **Score-based correction is promising but complex.** Defer until vMF is tested. If vMF drifts off-manifold, score-based denoising is next.

5. **Reservoir computing analogy suggests keeping Mimi frozen and learning simple readout.** Test linear projections before nonlinear z_dyn encoders.

### 6.2 Updated Experimental Roadmap

**Tier 1 (Immediate):**
1. **vMF factorization + rollout training**
   - Test whether direction is preserved while magnitude shrinks
   - Measure perceptual quality (listen to outputs)
   - Decision: Proceed if cos(Δz_pred, Δz_true) > 0.6 at K=16

2. **Representation comparison**
   - Run Phase 0-1 on EnCodec, SoundStream latents
   - Test frame rate hypothesis (Mimi at 50 Hz instead of 12.5 Hz)
   - Decision: If other representations show dramatically better predictability, switch

**Tier 2 (Conditional on Tier 1 success):**
3. **Acoustic archetype discovery**
   - Spherical k-means on directions
   - Correlate clusters with acoustic features (spectral flux, pitch change, energy)
   - Decision: Proceed to memory-as-basis if K exists with tightness >0.75 and cross-speaker transfer

4. **Reservoir-inspired linear readout**
   - z_dyn = W · [z_{t-L:t}] (linear projection)
   - Compare AR predictability: linear vs nonlinear z_dyn encoder
   - Decision: Use linear if comparable performance (Occam's razor)

**Tier 3 (Longer-term):**
5. **Score-based correction**
   - Train unconditional score model on Mimi latents
   - Post-hoc denoising of AR rollouts
   - Decision: Deploy if vMF rollouts drift off-manifold

6. **Two-stage generation (direction + magnitude)**
   - Stage 1: AR on directions
   - Stage 2: Sample magnitudes conditioned on global features
   - Decision: Final architecture if it improves perceptual quality

### 6.3 Decision Criteria

**"Slow but correct" hypothesis (vMF):**
- **Success:** cos(Δz_pred, Δz_true) > 0.6 at K=16, perceptual quality > baseline
- **Failure:** cos drops below 0.3, or perceptual quality worse than baseline
- **Decision:** Success → proceed to archetypes; Failure → try score-based or representation change

**Archetype discovery:**
- **Success:** K ∈ {256-512} with mean cluster cosine >0.75, cross-speaker accuracy >60%
- **Failure:** No K with tightness >0.6 or speaker-specific clusters only
- **Decision:** Success → memory-as-basis; Failure → pure parametric vMF

**Representation comparison:**
- **Success:** Other representation (EnCodec / higher frame rate) shows ΔNLL >2x improvement
- **Failure:** All representations show similar predictability
- **Decision:** Success → switch representation; Failure → problem is objective/architecture, not representation

---

## 7. Where You Changed My Mind

### 7.1 Game Observations Are Easier Than I Realized

I underestimated how engineered game observations are. They're designed to be Markovian, low-entropy, and diagnostic. Audio is none of these. The gap may be partially fundamental.

### 7.2 "Slow but Correct" Is a Coherent Strategy

I initially dismissed vMF as "just delaying the compounding problem." You're right that the **form** of compounding matters. If perceptual quality correlates more with direction than magnitude, vMF could genuinely help.

### 7.3 Acoustic vs Phonetic Structure

I was implicitly assuming phonetic structure (because that's how humans think about speech). But Mimi doesn't know about phonemes—it knows about acoustic features. Archetypes should be acoustic motifs, not phonetic categories.

### 7.4 The Reservoir Computing Connection

This analogy is deeper than I initially appreciated. The "inverse reservoir" framing (compressing high-dimensional dynamics to low-dimensional AR-friendly state) clarifies our task and suggests concrete approaches (linear readouts).

---

## 8. Where I'm Pushing Back

### 8.1 Score Functions Are Not a Silver Bullet

The score ∇_z log p(z) provides useful information, but:
- It's the score of marginal p(z), not conditional p(z_t | z_{<t})
- Single-step corrections may be unstable
- Trajectory-level diffusion is expensive

Score-based methods are powerful, but they're a research direction unto themselves. We should exhaust simpler approaches (vMF, better objectives) before diving into score matching.

### 8.2 Reservoir Analogy Is Imperfect

Mimi encoder is not a reservoir:
- No recurrence (frame-independent)
- Not designed for temporal dynamics
- Purpose is compression, not expansion

The useful insight is "separate representation from dynamics," but we shouldn't over-index on the reservoir framing.

### 8.3 Two-Stage Generation Is Risky

Decoupling direction from magnitude post-hoc sounds appealing, but:
- Phonetic timing may be lost (stop consonants need rapid transitions)
- Magnitude is not fully independent of direction (formant movement speed matters)
- Requires identifying global conditioning variables (non-trivial)

I'd test this only after confirming that magnitude is truly unconditionable on local context.

---

## 9. Open Questions

1. **What is the entropy rate of speech in different representations?**
   - Compare Mimi 12.5 Hz vs EnCodec vs 50 Hz Mimi
   - Quantify conditional entropy reduction (ΔNLL / baseline NLL)
   - Decision: Is this a representation problem or a fundamental audio problem?

2. **Does perceptual quality correlate more with direction or magnitude?**
   - Synthetic experiment: Generate trajectories with varied cos(d_pred, d_true) and ||m_pred - m_true||
   - Human evaluation: Which matters more?
   - Decision: If direction matters more, vMF is correct factorization

3. **Can we visualize learned mixture components?**
   - For each MDN component k, collect frames where it's dominant
   - Analyze acoustic properties (spectral flux, pitch, energy)
   - Decode to audio and listen
   - Decision: If components correspond to interpretable acoustic events, mode structure is real

4. **What does "on-manifold" mean for Mimi latents?**
   - Train density model (flow, VAE) on Mimi latents
   - Measure likelihood of ground truth vs rolled-out states
   - Decision: If rolled-out states have much lower likelihood, score-based correction could help

---

## 10. Probability Estimates (Updated)

| Hypothesis | Previous | Updated | Reasoning |
|------------|----------|---------|-----------|
| vMF preserves direction under rollout (cos >0.6 at K=16) | 0.55 | 0.60 | Your "slow but correct" argument is compelling |
| Archetypes exist (K with tightness >0.75) | 0.45 | 0.50 | Acoustic (not phonetic) structure is more plausible |
| Memory-as-basis helps (if archetypes exist) | 0.15 | 0.25 | Soft retrieval can handle continuous structure |
| Score-based correction improves stability | 0.35 | 0.30 | More skeptical after thinking through mechanics |
| Representation change (EnCodec / higher Hz) shows >2x ΔNLL improvement | 0.55 | 0.65 | More convinced gap is representation-specific |
| Linear readout (reservoir-style) comparable to nonlinear z_dyn | 0.40 | 0.45 | Simple hypotheses deserve more credit |

---

## 11. Recommended Immediate Actions

**This week:**
1. **Implement minimal vMF + magnitude factorization**
   - Drop-in replacement for MDN head
   - Train on teacher-forcing first (baseline)
   - Report: ΔNLL, direction cosine, magnitude R²

2. **vMF rollout training (K=16)**
   - Same rollout protocol as Phase 4
   - Track cos(Δz_pred, Δz_true) and ||Δz_pred||/||Δz_true|| separately
   - Decision: Proceed if directional preservation observed

3. **Listen to outputs**
   - Generate audio from rolled-out latents (vMF vs MDN)
   - Qualitative evaluation: Is "slow but correct" perceptually better?
   - Record informal notes on quality differences

**Next week:**
4. **Direction archetype discovery (if vMF succeeds)**
   - Spherical k-means on directions
   - Evaluate cross-speaker transfer
   - Correlate with acoustic features

5. **Representation comparison**
   - Extract EnCodec latents for LibriSpeech subset
   - Run Phase 1 predictor baseline
   - Compare ΔNLL: Mimi vs EnCodec

---

## 12. Literature to Read

**On world models and video:**
- DIAMOND (2024): "Diffusion for World Modeling"
- IRIS (2024): "Recurrent world models with implicit differentiation"
- DreamerV3 (2023): Latest RSSM architecture

**On score-based temporal modeling:**
- DYffusion (2023): "Dynamics-informed diffusion"
- Score-based trajectory optimization (Chen et al., 2022)
- CSDI (2021): "Conditional score-based diffusion for time series imputation"

**On vMF distributions:**
- Guu et al. (2018): "Generating sentences by editing prototypes" (uses vMF for text)
- Xu & Durrett (2018): "Spherical latent spaces for stable variational autoencoders"
- Davidson et al. (2018): "Hyperspherical VAE"

**On reservoir computing:**
- Jaeger (2001): Original ESN paper
- Lukoševičius & Jaeger (2009): "Reservoir computing approaches to recurrent neural network training"
- Pathak et al. (2018): "Model-free prediction of large spatiotemporally chaotic systems from data: A reservoir computing approach"

---

## 13. Final Thoughts

Your pushbacks have sharpened my thinking considerably. The key insights:

1. **The gap is representation-specific, not fundamental.** We should test other representations before concluding audio is inherently intractable.

2. **Factorization changes the failure mode.** "Slow but correct" may be better than "stationary," and much better than "fast but wrong." vMF enables this.

3. **Structure is acoustic, not phonetic.** This changes how we interpret archetypes (regions of high density, not discrete categories) but doesn't invalidate the approach.

4. **Score functions are powerful but complex.** Defer until simpler approaches (vMF, better objectives) are exhausted.

5. **Reservoir computing suggests simple readouts.** Test linear projections before complex encoders.

The highest-EV path: **vMF rollout → representation comparison → archetypes (if warranted) → score-based correction (if needed)**.

This lets us stop early if vMF fails, pivot to better representations if Mimi is the bottleneck, and scale to more complex methods only if justified.

I'm excited to see the vMF rollout results. If directional preservation holds, this could be the key insight. If not, representation comparison becomes critical.

Let's build the vMF head and find out.
