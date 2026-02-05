# Research Discussion 9: Pivot to Representation Design — Why the Encoder Is the Bottleneck

**Date:** 2026-02-05  
**Participants:** Riley + Claude (Opus 4.6)  
**Context:** Post-Tier 1 results analysis, CALM paper review, and project reorientation  
**Status:** Pivot proposal for team review

---

## Executive Summary

Eight discussions and five experimental phases have converged on a single conclusion: **the dynamics model is not the bottleneck; the representation is.** This document proposes a pivot from "better dynamics on frozen Mimi latents" to "design latent representations that are inherently AR-friendly," informed by everything we've learned and by a close reading of the CALM paper (Rouard et al., 2025) that motivated this line of work.

**The core argument in three steps:**

1. **Mimi's latents weren't designed for AR generation.** They were optimized for reconstruction + RVQ quantization + semantic distillation. Our experiments show their dynamics are catastrophically sensitive to perturbation — a single step of predicted context causes immediate divergence (Exp2 injection diagnostic: state error ~5e13 by step 2).

2. **CALM's authors already know this.** Their solution is to inject noise during training (making the backbone robust to corrupted context) and add a short-context transformer on clean latents (restoring local detail). These are workarounds for a representation that isn't AR-friendly, not solutions to the underlying problem. Their own ablation shows removing these tricks destroys quality (FAD: 0.93 → 8.38).

3. **A representation designed for AR-friendliness from the start should not need these workarounds.** If the latent space is smooth, low-entropy in its dynamics, and has bounded sensitivity to perturbation, then simple dynamics models should produce stable rollouts without noise injection, short-context bypasses, or consistency-model sampling heads.

**Proposed contribution:** Design and evaluate VAE training objectives that produce AR-friendly continuous audio latents, using our diagnostic framework (Phases 0–4.5 + Tier 1) to measure what makes a representation "AR-friendly" and to compare against Mimi.

---

## 1. What We Learned (Phases 0–4.5 + Tier 1)

### 1.1 The findings that matter

| Phase | Finding | Implication |
|-------|---------|-------------|
| Phase 0 | Clustering explains ~0% of dynamics variance | Coarse keys don't capture structure (but structure may still exist) |
| Phase 1 | Learned predictor: ΔNLL = -185 at k=1 | Predictable structure exists; Phase 0's method was blind to it |
| Phase 1 | Direction cos ~0.61, magnitude R² < 0 | *Type* of transition is predictable; *intensity* is stochastic |
| Phase 3 | z_dyn more predictable than raw Mimi (cos 0.71) | Learned factorization can extract a better state |
| Phase 3 | z_rec unused (recon_prior ≈ recon_post) | Innovation channel never became meaningful |
| Phase 4 | Memory helps teacher-forced, not rollout | Memory patches one-step errors, not the real bottleneck |
| Phase 4.5 | Rollout training: +761K → -0.99 ΔNLL | Stability is achievable — but only via mean collapse (‖Δẑ‖: 10.76 → 0.024) |
| Tier 1 Exp1 | vMF: good direction cos (0.73) but ΔNLL positive | LogNormal magnitude blow-up poisons the joint likelihood |
| **Tier 1 Exp2** | **Injection diagnostic: divergence at step 2** | **The dynamics function is undefined off-manifold. One-step error is catastrophic.** |

### 1.2 The diagnosis

Every dynamics-side intervention we tried (memory, vMF factorization, rollout training, state matching) either failed or achieved stability only through collapse. The injection diagnostic (Exp2) explains why: **the function mapping z_t → Δz_t is catastrophically sensitive to perturbation of z_t.** This is not a prediction-quality problem — it's a **representation smoothness** problem.

A dynamics function trained on data from p_data(z) has no reason to be well-behaved at z + ε where ε is small but moves z off the data manifold. If the data manifold is thin (high-dimensional latents with complex, folded geometry), even tiny perturbations leave the manifold, and the dynamics function produces garbage.

### 1.3 Why Mimi latents are particularly bad for AR

Mimi was trained with three objectives, none of which encourage AR-friendly dynamics:

1. **Reconstruction** — encourages high-fidelity encoding, which means the latent space packs information densely. Dense packing = thin manifold = high sensitivity to perturbation.

2. **RVQ quantization** — encourages the latent space to be amenable to vector quantization. VQ pressure creates codebook structure, not smooth dynamics. (Ironically, the *discrete* tokens from RVQ might have better AR properties than the continuous pre-quantization latents, because VQ creates piecewise-constant regions.)

3. **Semantic distillation (WavLM)** — encourages phonetic discriminability, not temporal smoothness. Two latent frames that sound similar but correspond to different phonemes are pushed apart, potentially creating sharp boundaries in latent space that make dynamics discontinuous.

None of these objectives say anything about: temporal smoothness of trajectories, predictability of dynamics, robustness to perturbation of context, or low conditional entropy of next-step distributions.

---

## 2. What CALM Tells Us

### 2.1 CALM's architecture is a symptom, not a solution

CALM (Rouard et al., 2025) achieves strong results on speech and music generation with continuous latents. But a careful reading reveals that much of the architecture exists to compensate for representation-level problems:

**Noise injection (§4.1):** During training, each latent frame is corrupted: x̃_s = k_s ε_s + √(1-k_s) x_s. This makes the backbone robust to corrupted context at inference time (when it sees its own predictions). The authors state explicitly: "Noise injection prevents error accumulation during inference, but [...] is insufficient alone for high-quality music generation."

**Short-context transformer (§4.2):** A separate clean-context pathway over the last K=10 frames. This exists because noise injection destroys fine-grained information in the backbone's conditioning. The short-context transformer restores local detail from clean latents. The ablation (Table 5) shows this is the single most important component: removing it degrades FAD from 0.93 to 4.03.

**Consistency head (§4.2):** Rather than predicting Δx directly (as our experiments do), CALM uses a consistency model to sample from the full conditional distribution p(x_s | z_s). This implicitly handles multimodality, magnitude uncertainty, and distributional complexity — all things our MDN/vMF heads struggle with.

### 2.2 CALM's VAE is already partially AR-friendly

CALM trains its own VAE (§4.1, §5.1), not Mimi's continuous latents:

- **32-dimensional** (vs our 512-dimensional Mimi latents) — dramatically lower dimensional, meaning the data manifold is lower-dimensional and less sensitive to perturbation.
- **KL regularization** — enforces a Gaussian prior on the latent space, which smooths it and prevents the kind of thin-manifold, high-sensitivity structure that Mimi's unregularized latents exhibit.
- **Semantic distillation** — same as Mimi (WavLM), but applied to the full latent rather than just the first codebook.

The 32-dim KL-regularized VAE is already a step toward AR-friendliness. But CALM still needs noise injection and short-context transformers, suggesting the VAE alone isn't sufficient — it wasn't explicitly optimized for temporal dynamics.

### 2.3 The gap CALM leaves open

CALM's solution is effective but has costs:

1. **Information destruction in long-range context.** Noise injection forces the backbone to operate on corrupted inputs. This limits the backbone's ability to encode fine-grained long-range dependencies.

2. **Architectural complexity.** The dual-transformer (long noised + short clean) architecture exists entirely to compensate for the noise injection. Without the representation problem, this wouldn't be needed.

3. **Hyperparameter sensitivity.** The noise level, short-context window K, and the interaction between backbone and consistency head create a fragile balance. Too little noise → instability. Too much → quality loss.

4. **Longer-horizon degradation.** The authors note that music generation degrades after 10–15 seconds without the full architecture, suggesting the workarounds have limited temporal reach.

**The opportunity:** A representation that is inherently AR-friendly should not need noise injection or dual-context architecture. Simple dynamics models should work. This simplifies the architecture, reduces hyperparameter sensitivity, and may extend the stable generation horizon.

---

## 3. The Pivot: Representation Design for AR-Friendliness

### 3.1 What "AR-friendly" means (operational definition)

A latent representation is AR-friendly to the degree that:

1. **Predictability:** The conditional entropy H(x_t | x_{<t}) is low relative to the marginal entropy H(x_t). Measurable via Phase 1 ΔNLL diagnostic.

2. **Smoothness:** The dynamics function f: x_t → p(Δx_t | x_t) is Lipschitz-continuous — small perturbations in x_t produce small changes in the predicted distribution. Measurable via the injection diagnostic (Exp2): how many steps before divergence?

3. **Rollout stability:** Multi-step autoregressive generation stays on or near the data manifold. Measurable via rollout ΔNLL and trajectory divergence curves.

4. **Reconstruction quality:** The representation still allows high-fidelity audio reconstruction. Measurable via standard codec metrics (PESQ, STOI, MOSNet, mel distance).

The first three properties are about dynamics; the fourth is about information. The core tension is that optimizing for reconstruction tends to create dense, thin-manifold latent spaces that are hostile to AR modeling. The research question is: **can we train a VAE that achieves good reconstruction while explicitly optimizing for AR-friendly dynamics?**

### 3.2 Proposed VAE training objectives

We propose training VAEs on LibriSpeech train-clean-100 with combinations of the following losses:

**Standard losses (always present):**
- `L_recon`: Reconstruction loss (multi-scale spectral + time-domain)
- `L_KL`: KL regularization against Gaussian prior (β-VAE style)

**AR-friendliness losses (the experimental variables):**

**A. Temporal smoothness loss:**
```
L_smooth = E_t[ ||x_t - x_{t-1}||² ]
```
Penalizes large frame-to-frame jumps. This directly reduces the magnitude of Δx, making dynamics lower-entropy. Risk: may blur transients (stop consonants, onsets).

**B. Dynamics predictability loss:**
```
L_pred = E_t[ -log p_φ(Δx_t | x_{t-W:t-1}) ]
```
Train a lightweight predictor φ jointly with the encoder, and penalize the encoder for producing latents whose dynamics are hard to predict. This is the most direct attack on AR-friendliness — it literally optimizes "make the future predictable from the past."

**C. Latent Jacobian regularization:**
```
L_jacobian = E_t[ ||J_enc(audio_t) - J_enc(audio_{t-1})||_F ]
```
Penalizes rapid changes in the encoder's Jacobian between adjacent frames. This encourages the encoder to map nearby audio frames to nearby latent points with consistent local geometry, reducing the "folding" that creates thin manifolds.

**D. Noise robustness loss (CALM-inspired):**
```
L_noise = E_t,ε[ ||dec(x_t + σε) - audio_t||² ]
```
Train the decoder to be robust to small perturbations of the latent. If the decoder is smooth, the dynamics function doesn't need to be perfect — small errors in predicted latents still decode to reasonable audio.

**E. Rollout reconstruction loss (curriculum):**
```
L_rollout = E[ Σ_{k=1}^{K} ||dec(x̂_k) - audio_k||² ]
where x̂_{k+1} = x̂_k + f_φ(x̂_k)  (rolled out with lightweight dynamics)
```
Directly optimizes: "latents that are autoregressively rolled out still decode to good audio." This is the most expensive but most directly aligned with the downstream task. Requires curriculum on K.

### 3.3 Experimental design

#### Phase R1: VAE Training Sweep

Train a grid of VAEs on LibriSpeech train-clean-100, varying:

| Dimension | Values | Rationale |
|-----------|--------|-----------|
| Latent dim | {32, 64, 128} | CALM uses 32; Mimi uses 512. Lower dim = smoother manifold but less capacity |
| β (KL weight) | {0.01, 0.1, 1.0} | Controls manifold thickness. Higher = smoother but lossy |
| AR loss | {none, L_smooth, L_pred, L_smooth + L_pred} | The experimental variable |
| Frame rate | 12.5 Hz | Match Mimi for comparability |

Architecture: Mimi-style encoder/decoder (causal convolutions + transformer layers), but with VAE bottleneck instead of RVQ. Semantic distillation (WavLM) optional — include for one condition to test interaction.

**Total conditions:** 3 × 3 × 4 = 36 VAEs. Each is small (LibriSpeech-100 is ~100 hours). Training time estimate: ~2–4 GPU-hours per VAE on a single A100. Full sweep: ~72–144 GPU-hours.

#### Phase R2: AR-Friendliness Diagnostic Battery

For each trained VAE, run the diagnostic battery we've built:

1. **Phase 1 predictor baseline** (ΔNLL at k=1,2,4,8) — measures predictability
2. **Injection diagnostic** (Exp2 modes A–D) — measures perturbation sensitivity
3. **Rollout stability** (train dynamics model, measure rollout ΔNLL and divergence horizon) — measures long-horizon viability
4. **Reconstruction quality** (PESQ, STOI, mel distance on held-out speakers) — measures information preservation

Report as a multi-objective Pareto front: reconstruction quality vs AR-friendliness metrics.

#### Phase R3: Best-Representation Dynamics Evaluation

Take the top 3–5 representations from the Pareto front and run the full dynamics pipeline:

1. Phase 3 (RSSM factorization) — does z_dyn / z_rec split work better on AR-friendly latents?
2. Phase 4.5 (rollout training) — does rollout training achieve stability *without* mean collapse?
3. Phase 1-on-Phase 3 (z_dyn predictability) — is z_dyn more predictable than the best raw latents?

This tests whether the downstream dynamics modeling improves when the representation is better.

#### Phase R4: End-to-End Generation and Listening

For the best representation(s):

1. Generate audio from rolled-out latents
2. Decode through VAE decoder
3. Compare perceptually: AR-friendly VAE vs Mimi vs CALM-style (noise injection + short context)
4. Report: informal listening panel + MOS if warranted

### 3.4 Decision criteria

| Outcome | Interpretation | Next step |
|---------|---------------|-----------|
| AR loss VAEs show >2× ΔNLL improvement AND divergence horizon >10 steps AND reconstruction within 10% of Mimi | **AR-friendly representations are learnable and dramatically help** | Full paper on representation design |
| AR loss helps predictability but not stability (still diverges fast) | Representation helps but isn't sufficient; dynamics-side work still needed | Combine AR-friendly VAE with rollout training |
| AR loss degrades reconstruction without helping stability | Tension between reconstruction and AR-friendliness is too steep | Investigate whether the tradeoff is fundamental |
| No AR loss variant meaningfully improves over β-VAE alone | KL regularization + low dim is sufficient; explicit AR losses are unnecessary | Simpler story: "just use a KL-regularized VAE" |

---

## 4. What We Keep From the Previous Work

The pivot doesn't discard the Phase 0–4.5 work. It reframes it as **the diagnostic framework** that motivates and evaluates the representation design:

### 4.1 Diagnostic contributions (already complete)

- **Phase 0 → Phase 1 reconciliation:** Demonstrates that variance-ratio metrics miss structure that probabilistic predictors find. This is a methodological contribution independent of the representation.
- **Direction/magnitude decomposition:** Characterizes *what* is predictable (direction) vs stochastic (magnitude) in audio dynamics. This generalizes across representations.
- **Injection diagnostic (Exp2):** A cheap, decisive test for rollout viability. One-step perturbation → measure divergence horizon. This becomes a standard tool in the diagnostic battery.
- **Phase 4.5 rollout training:** Demonstrates that stability is achievable but current representations force mean collapse. This motivates asking "what representation wouldn't collapse?"

### 4.2 Paper narrative (updated)

**Title direction:** "AR-Friendly Audio Representations: Why the Encoder Matters More Than the Dynamics Model"

**Story arc:**

1. **Motivation:** Continuous AR audio generation suffers from catastrophic rollout instability. CALM solves this with architectural workarounds (noise injection, dual-context transformers). Can we solve it at the representation level instead?

2. **Diagnostic framework (Phases 0–1, Exp2):** We develop a battery of diagnostics that measure AR-friendliness: predictability (ΔNLL), perturbation sensitivity (injection diagnostic), and rollout stability. Applied to Mimi, these reveal that the representation is the binding constraint — divergence happens within 1–2 steps regardless of dynamics model quality.

3. **Representation design (Phase R1–R2):** We train VAEs with explicit AR-friendliness objectives and show that [results here: which losses help, what the Pareto tradeoff looks like].

4. **Downstream impact (Phase R3–R4):** AR-friendly representations enable [stable rollouts / simpler dynamics models / better generation quality] without the architectural workarounds CALM requires.

5. **Conclusion:** For continuous AR audio generation, representation design is the primary lever. Investing in AR-friendly encoders pays larger dividends than sophisticated dynamics models or architectural patches.

---

## 5. Relation to Engram / Memory

The original project hypothesis — that Engram-style memory could help AR audio generation — has been thoroughly tested and found wanting:

- **Phase 0:** Clustering-based lookup explains nothing.
- **Phase 4:** Residual memory helps teacher-forced prediction but adds nothing after rollout training.
- **Discussion 5–7:** The reformulated "memory-as-basis" / direction archetype idea remains untested but the probability estimates are low (15–25%).

The honest conclusion: **discrete lookup memory does not transfer from language to continuous audio latents.** The reasons are structural (no exact key matches, continuous similarity instead of discrete equality, high innovation rate in audio). This is a valid negative finding.

However, the *principle* behind Engram — that reusable local structure should be exploited rather than recomputed — may still apply at the representation level. An AR-friendly VAE that produces latents with reusable dynamical motifs would be the right substrate for memory-like mechanisms. We should revisit memory only after establishing that such a representation exists.

---

## 6. Practical Considerations

### 6.1 Compute budget

| Phase | Compute estimate | Timeline |
|-------|-----------------|----------|
| R1: VAE sweep (36 conditions) | ~100–150 A100-hours | 1–2 weeks with 4 GPUs |
| R2: Diagnostic battery (36 × 4 diagnostics) | ~50 A100-hours | 1 week (parallelizable) |
| R3: Full dynamics pipeline (top 5) | ~50 A100-hours | 1 week |
| R4: Generation + listening | ~10 A100-hours + human time | 3–5 days |
| **Total** | **~250 A100-hours** | **~4–5 weeks** |

This is comparable to the compute already spent on Phases 0–4.5 and represents a reasonable investment for a potential paper.

### 6.2 Implementation reuse

Most of the existing codebase transfers directly:

- **Phase 1 predictor baseline:** Drop-in on any latent store (just change `data.latents_dir`)
- **Injection diagnostic (Exp2):** Same — operates on any latent representation
- **Phase 3 RSSM factorizer:** Same architecture, new latent input
- **Phase 4.5 rollout training:** Same

New code needed:
- VAE training script (adapt from Mimi architecture, replace RVQ with VAE bottleneck)
- AR-friendliness loss implementations (L_smooth, L_pred, L_jacobian)
- Orchestration script for the sweep + diagnostic battery

### 6.3 Risk mitigation

**Risk 1: AR-friendliness losses destroy reconstruction quality.**  
Mitigation: The sweep includes a "no AR loss" baseline (pure β-VAE). If explicit AR losses don't help beyond what KL regularization provides, that's still a finding — "just use a low-dim KL-regularized VAE" is a useful result.

**Risk 2: LibriSpeech-100 is too small / narrow.**  
Mitigation: We're testing the *principle*, not building a production system. If the principle works on LibriSpeech-100, scaling to larger datasets is straightforward. If it doesn't work on clean read speech, it won't work on harder data either.

**Risk 3: VAE architecture matters more than the loss.**  
Mitigation: Fix the architecture across conditions (Mimi-style encoder/decoder). Only vary the training objective. This isolates the effect of the loss from architectural confounds.

**Risk 4: Low-dim latents can't reconstruct speech well enough.**  
Mitigation: CALM demonstrates that 32-dim VAE latents at 12.5 Hz achieve competitive reconstruction (Table 1 in their paper: PESQ 2.42, STOI 0.90, MOSNet 3.15). Our sweep includes 32, 64, and 128 dims to map the quality frontier.

---

## 7. Immediate Next Steps (Before the Full Pivot)

Before committing to the full R1–R4 plan, two cheap experiments can de-risk the pivot:

### 7.1 Quick validation: β-VAE baseline (2–3 days)

Train a single 32-dim β-VAE on LibriSpeech-100 with Mimi's encoder/decoder architecture (replace RVQ with Gaussian bottleneck, add KL loss). Run the Phase 1 predictor baseline and injection diagnostic on its latents.

**If ΔNLL improves substantially over Mimi AND divergence horizon extends beyond 2 steps:** The pivot is strongly motivated. Proceed to full sweep.

**If results are comparable to Mimi:** The pivot may not work. Reconsider whether the issue is dimensionality, KL regularization, or something else.

### 7.2 Complete Exp3 representation comparison (1–2 days)

Fix the EnCodec environment issue and run Phase 1 on EnCodec latents. This tells us whether the predictability gap is Mimi-specific.

---

## 8. Summary of the Proposed Research Direction

**From:** "Can Engram-style memory improve AR audio generation on frozen Mimi latents?"

**To:** "What properties make a continuous audio representation AR-friendly, and can we train VAEs that have these properties?"

**The diagnostic framework we built (Phases 0–4.5) becomes the evaluation tool, not the contribution.** The contribution is the finding that representation design is the primary lever for continuous AR audio stability, supported by:

1. A diagnostic battery for AR-friendliness (predictability, perturbation sensitivity, rollout stability)
2. A systematic comparison of VAE training objectives on these diagnostics
3. Evidence that AR-friendly representations reduce or eliminate the need for CALM's architectural workarounds
4. (If applicable) A demonstration that simple dynamics models produce stable, perceptually good audio on AR-friendly latents

This positions the work as a general contribution to continuous AR generation — applicable to audio, but potentially to video, robotics, and any domain where continuous latents are generated autoregressively.

---

## Appendix A: CALM Paper — Key Numbers for Comparison

| Metric | CALM VAE (32-dim) | Mimi VQ-VAE (8 RVQ) |
|--------|-------------------|----------------------|
| MOSNet | 3.15 | 3.11 |
| ABX (phonetic) | 8.1% | 9.4% |
| PESQ | 2.42 | 2.13 |
| STOI | 0.90 | 0.87 |
| Latent dim | 32 | 8 codebooks × vocab |
| Frame rate | 12.5 Hz | 12.5 Hz |
| KL regularization | Yes | No (RVQ) |

CALM's 32-dim VAE matches or exceeds Mimi on all reconstruction metrics despite 16× lower dimensionality. This is strong prior evidence that low-dim KL-regularized VAEs are viable for speech.

## Appendix B: VAE Architecture Specification (Draft)

Based on Mimi's encoder/decoder with modifications:

- **Encoder:** Causal convolution stack (stride pattern matching 12.5 Hz) + causal transformer layers
- **Bottleneck:** Gaussian VAE (μ, log σ → sample z = μ + σε) replacing Mimi's RVQ
- **Decoder:** Mirror of encoder (transposed convolutions + transformer)
- **Discriminator:** Multi-scale STFT discriminator (standard for audio VAE-GANs)
- **Semantic distillation:** Optional WavLM cosine loss on latent (test with/without)

Hyperparameters to fix across conditions:
- Frame rate: 12.5 Hz
- Encoder/decoder architecture: identical across all conditions
- Training: AdamW, 200K steps, batch size 32 (tune as needed on LibriSpeech-100)
- Discriminator: same architecture and loss across all conditions

Hyperparameters to sweep: latent dim, β, AR-friendliness loss (as specified in §3.3).

## Appendix C: Diagnostic Battery Specification

For each VAE, run:

1. **Extract latents:** Encode LibriSpeech-100 train and eval splits → zarr store
2. **Phase 1 predictor:** MDN on Δx, horizons k ∈ {1, 2, 4, 8}, report ΔNLL, direction cos, logmag R²
3. **Injection diagnostic:** Train dynamics MLP on z_t → Δz_t, run Modes A–D, report divergence horizon (step at which state_err > 100 × median ‖Δz‖)
4. **Rollout stability:** Train dynamics MLP with K=16 rollout, report rollout ΔNLL and mean ‖Δẑ‖ / mean ‖Δz_true‖ (magnitude ratio — 1.0 is ideal, 0.0 is collapse)
5. **Reconstruction quality:** PESQ, STOI, mel-spectrogram distance on eval speakers

All scripts exist and require only `data.latents_dir` to be pointed at the new zarr store.