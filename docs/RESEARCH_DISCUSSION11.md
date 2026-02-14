# Research Discussion 11: Stage 2 Results — Representation Redesign Meets Reality

**Date:** 2026-02-06
**Participants:** Riley + Claude (Opus 4.6)
**Context:** Stage 2 experiments (Exp 5, 6, 7) from Discussion 10 §6 are complete. This document reports findings, evaluates against the §6 decision gates, and charts the next direction.
**Status:** Results analysis and updated recommendations

---

## Executive Summary

Stage 2 is done. Three experiments tested the two main hypotheses from Discussion 10: (1) learned low-dimensional representations with AR-friendliness objectives produce more stable rollout dynamics, and (2) score-based manifold correction at inference time can recover from rollout drift. The findings:

1. **Score-based correction is dead.** Langevin correction on Mimi latents produced zero improvement. The best corrected configuration was slightly *worse* than uncorrected rollout (state_err 11720 vs 11643). Larger correction steps cause divergence to infinity. The manifold geometry problem cannot be patched at inference time with a learned score field.

2. **The β-VAE produces high-quality reconstructions with dramatically better prediction than PCA.** All four VAE variants achieved mel distance ~2.47 and L1 ~0.036, with ΔNLL of -41 to -45 — far exceeding PCA-32's -14.7. The learned encoder recovers the information PCA discards while maintaining a 32D bottleneck. This confirms the core representation hypothesis.

3. **AR-friendliness objectives show mixed effects depending on the metric.** In the injection diagnostic (16-step free-running), smooth+pred is the clear winner (ΔNLL -4.18 vs baseline +0.29). But in per-horizon rollout evaluation, the baseline β-VAE actually outperforms all AR-loss variants at k=1 and k=2. The losses reshape *how* things fail without preventing failure.

4. **Rollout still collapses universally at k≥2.** Every variant — baseline, smooth, pred, combined — produces inf/NaN rollout NLL by k=2. The fundamental 2-step rollout ceiling from Stage 1 persists despite the representation change. The problem is not purely about representation dimensionality or manifold smoothness.

5. **Perceptual rollout WAVs are available for listening.** 50 WAV files at 48kHz across 10 utterances and 5 rollout lengths, decoded through Mimi from vMF dynamics model output.

**Against Discussion 10 §6 decision gates:** The β-VAE beats PCA-32 on ΔNLL (✓) and produces excellent reconstructions (✓). But the AR-loss variants do not *substantially* outperform the baseline β-VAE — the effect is modest, inconsistent across metrics, and does not extend the rollout horizon. We are between "AR-friendliness objectives have marginal value" and "just use a low-dim KL-regularized VAE."

---

## 1. Stage 2 Experimental Results

### 1.1 Exp 5: β-VAE with AR-Friendliness Objectives

Trained four ARFriendlyVAE variants wrapping Mimi encoder (project=False) with Conv1d bottleneck to 32D, each for 10,000 steps on LibriSpeech train-clean-100 (5000 utterances). Then extracted latents and ran the full Phase 1 diagnostic battery.

#### VAE Training

| Variant | β | λ_smooth | λ_pred | Final loss | Recon | KL | Smooth | Pred |
|---------|---|----------|--------|------------|-------|-----|--------|------|
| baseline_betavae | 0.01 | 0 | 0 | 7.40 | 6.60 | 68 | — | — |
| betavae_smooth | 0.01 | 0.1 | 0 | 7.50 | 6.60 | 68 | 1.8 | — |
| betavae_pred | 0.01 | 0 | 0.01 | 7.24 | 6.59 | 63 | — | 0.97 |
| betavae_smooth_pred | 0.01 | 0.1 | 0.01 | 7.42 | 6.59 | 64 | 1.83 | 0.94 |

**Observations:** Reconstruction loss is virtually identical across variants (~6.59–6.60). KL is slightly lower for pred variants (63–64 vs 68), suggesting the predictability loss encourages a more structured posterior. The pred loss converges to ~0.95, meaning the lightweight MLP predictor achieves reasonable (but not excellent) next-step prediction on the latent deltas during VAE training.

#### Reconstruction Quality

| Variant | Recon L1 | Recon Mel |
|---------|----------|-----------|
| baseline_betavae | 0.0363 | 2.479 |
| betavae_smooth | 0.0362 | 2.462 |
| betavae_pred | 0.0362 | 2.471 |
| betavae_smooth_pred | 0.0362 | 2.468 |

All variants are effectively identical on reconstruction. The AR-friendliness objectives do not degrade audio quality. This is a necessary condition — the representation changes are free in reconstruction terms.

#### Phase 1 Diagnostic Battery (Per-Horizon Rollout)

| Variant | k | Teacher NLL | Rollout NLL | Gap |
|---------|---|-------------|-------------|-----|
| **baseline** | 1 | -55.67 | **-0.92** | 54.75 |
| | 2 | -47.15 | **-38.85** | **8.30** |
| | 4 | -41.35 | inf | — |
| | 8 | -43.16 | 1.16e+19 | — |
| **smooth** | 1 | -56.59 | -1.36 | 55.23 |
| | 2 | -48.38 | 1.36e+35 | — |
| | 4 | -42.48 | NaN | — |
| | 8 | -44.09 | 1.60e+16 | — |
| **pred** | 1 | -55.97 | +10.23 | 66.20 |
| | 2 | -49.39 | NaN | — |
| | 4 | -41.67 | inf | — |
| | 8 | -44.27 | 1.39e+35 | — |
| **smooth_pred** | 1 | -56.30 | +2.42 | 58.72 |
| | 2 | -48.62 | inf | — |
| | 4 | -42.92 | inf | — |
| | 8 | -44.15 | 5.08e+16 | — |

**Key finding: the baseline β-VAE outperforms all AR-loss variants on per-horizon rollout.** At k=1, the baseline achieves rollout_nll=-0.92 (negative = still informative) while pred variants are positive (+10.23, +2.42 = worse than unconditional). At k=2, the baseline is the *only* variant that doesn't collapse (rollout_nll=-38.85, gap=8.30). This is unexpected and important.

**Interpretation:** The predictability loss (L_pred) constrains the latent dynamics to be more predictable by the co-trained MLP predictor, but this doesn't translate to better rollout by the separately-trained Phase 1 MDN. The predictor sees gradients through reparameterization during VAE training, creating a latent space optimized for *its own* predictions — a different model (the Phase 1 MDN) doesn't necessarily benefit.

#### Injection Diagnostic (16-Step Free-Running)

| Variant | Mode | cos | state_err | ΔNLL |
|---------|------|-----|-----------|------|
| **baseline** | A_teacher | 0.580 | 0.000 | -41.97 |
| | D_rollout | 0.060 | 0.952 | **+0.29** |
| **smooth** | A_teacher | 0.561 | 0.000 | -45.56 |
| | D_rollout | 0.057 | 0.956 | -2.61 |
| **pred** | A_teacher | 0.555 | 0.000 | -45.64 |
| | D_rollout | **0.086** | 0.924 | +2.40 |
| **smooth_pred** | A_teacher | 0.542 | 0.000 | -45.47 |
| | D_rollout | 0.067 | **0.918** | **-4.18** |

**Here the story reverses.** In the injection diagnostic (16-step free-running from a k=1 MDN), smooth_pred wins on state_err (0.918, 3.6% lower than baseline) and ΔNLL (-4.18 vs +0.29). The pred variant achieves the highest rollout direction cos (0.086, 43% above baseline's 0.060).

**The contradiction:** Baseline wins per-k rollout, smooth_pred wins injection diagnostic. These are measuring different things:
- Per-k rollout: separate MDN models trained at each horizon, rollout evaluated at that specific horizon
- Injection diagnostic: single k=1 MDN model running 16-step free-running forecast

The injection diagnostic is the more realistic test (a single model doing extended rollout), suggesting smooth_pred's advantages manifest at longer horizons even if the individual-k models don't show it. But the effect size is small — all variants' D_rollout cos (0.057–0.086) is effectively random in 32D space (expected random |cos| ≈ 0.18).

#### Summary: β-VAE vs Stage 1 Representations

| Representation | Dim | Eval ΔNLL | D_rollout state_err | D_rollout cos |
|----------------|-----|-----------|---------------------|---------------|
| Mimi (raw) | 512 | -154.3 | 79.1 | 0.068 |
| EnCodec encoder | 128 | -79.5 | 13.9 | 0.093 |
| PCA-32 | 32 | -13.6 | 47.9 | 0.099 |
| **β-VAE baseline** | **32** | **-41.4** | **0.952*** | **0.060** |
| **β-VAE smooth_pred** | **32** | **-44.7** | **0.918*** | **0.067** |

*State_err values are not directly comparable across representations because L2 norms scale with dimensionality and representation magnitude. The β-VAE 32D latents have different scales than Mimi 512D or PCA 32D latents.

**Clear win:** β-VAE ΔNLL (-41 to -45) is 3× better than PCA-32 (-13.6) at the same bottleneck dimension. The learned encoder captures far more predictable structure than linear projection while maintaining a 32D manifold. The representation hypothesis from Discussion 10 §3.2 is confirmed — a learned encoder can operate on a fundamentally different point on the dimensionality–quality tradeoff.

### 1.2 Exp 6: Score-Based Manifold Correction

Trained ScoreNetwork (1024-dim, 4-layer MLP) via denoising score matching on Mimi 512D latent frames for 20,000 steps. Applied Langevin correction during vMF rollout with swept hyperparameters.

| n_steps | step_size | D_rollout cos | D_corrected cos | D_rollout state_err | D_corrected state_err |
|---------|-----------|---------------|-----------------|---------------------|----------------------|
| 1 | 0.001 | 0.075 | 0.074 | 11643 | 11720 |
| 1 | 0.01 | 0.075 | 0.075 | 11643 | 12702 |
| 1 | 0.1 | 0.075 | 0.080 | 11643 | 2,507,858 |
| 3 | 0.001 | 0.075 | 0.074 | 11643 | 11966 |
| 3 | 0.01 | 0.075 | 0.075 | 11643 | 17169 |
| 3 | 0.1 | 0.075 | 0.076 | 11643 | **inf** |
| 5 | 0.001 | 0.075 | 0.074 | 11643 | 12059 |
| 5 | 0.01 | 0.075 | 0.077 | 11643 | 362,678 |
| 5 | 0.1 | 0.075 | NaN | 11643 | NaN |

**Total failure.** No correction configuration improves state_err. The smallest step sizes produce negligible changes. Medium step sizes worsen state_err by 1–30×. Large step sizes cause divergence to infinity or NaN.

**Why it fails:** The score model learns ∇_z log p(z) — the gradient pointing toward high-density regions of the *unconditional* latent distribution. But rollout drift doesn't move states to low-density regions of the *spatial* distribution — it moves them to regions that are *temporally* inconsistent. A state z can be perfectly plausible as an individual frame while being catastrophically wrong as the continuation of a trajectory. The score field corrects spatial plausibility, not temporal coherence.

This is a fundamental mismatch. To make score correction work, you'd need a *conditional* score model s(z_t | z_{t-1}, ..., z_{t-W}) — essentially learning the full dynamics model's correction field, which is the same problem we're already trying to solve.

### 1.3 Exp 7: Perceptual Rollout Evaluation

Generated 50 WAV files (10 utterances × 5 rollout lengths: 1, 2, 4, 8, 16 steps) plus 10 GT references, decoded through Mimi from vMF rollout trajectories. Output at 48kHz.

**Subjective listening evaluation** (performed by Riley):
- **k=1 (80ms): indistinguishable from GT.** Single-step prediction is perceptually lossless.
- **k=2 (160ms): barely perceptible.** A faint artifact at the rollout point, rest perfect.
- **k=4 (320ms): short distorted blip.** Clearly audible but brief (~1/3 second).
- **k=8 (640ms): ~0.5s distorted section.** Unmistakable degradation.
- **k=16 (1.28s): ~1s of obvious distortion.** Still localized to the rollout segment; GT prefix and suffix sound perfect.

The distortion is always at the same point (the rollout segment, starting ~1/3 through the file). Everything outside the rollout window is literal GT decoded through Mimi and sounds identical to the reference.

**Note:** These rollouts use the Stage 1 vMF model on *Mimi* latents, not on β-VAE latents. They represent the baseline perceptual quality of rollout in the original 512D space. The perceptual result establishes that individual step quality is not the problem — error accumulation is.

---

## 2. Decision Gate Evaluation

Evaluating against Discussion 10 §6 decision gates:

### Gate 1: Score Correction (§6.2)

> If correction extends divergence horizon from ~2 steps to >8 steps → viable

**Result: NO.** Correction extends nothing. Zero improvement at any hyperparameter setting. Score-based manifold correction is ruled out for this problem.

> If no improvement → the manifold geometry is too thin for score-based recovery; representation redesign is the only path

**This is the outcome.** The unconditional score field doesn't capture the information needed to fix temporal drift. We proceed without this tool.

### Gate 2: Baseline β-VAE (§6.1)

> If baseline β-VAE beats PCA-32 on ΔNLL and matches on stability → the learned encoder recovers information PCA loses. AR losses are worth testing.

**Result: YES on ΔNLL.** β-VAE ΔNLL = -41.4, PCA-32 ΔNLL = -13.6 (3× improvement). The learned encoder dramatically outperforms linear projection at preserving predictable structure.

**Stability comparison is complicated.** State_err values aren't directly comparable across representations due to different latent scales. But all β-VAE variants still collapse at k≥2 rollout, just like all Stage 1 representations.

### Gate 3: AR-Loss Variants (§6.1)

> If L_pred substantially outperforms baseline β-VAE on rollout stability → central thesis validated.

**Result: MIXED.** smooth_pred shows modest improvement in injection diagnostic metrics (ΔNLL -4.18 vs +0.29, state_err 0.918 vs 0.952). But baseline outperforms on per-k rollout (survives k=2 while others collapse). L_pred reshapes the failure pattern without extending the horizon.

> If neither helps → KL + low dim is the entire story; AR-specific objectives don't add value.

**Close to this outcome.** The AR objectives provide marginal, metric-dependent improvements. They don't change the fundamental picture. The dimensionality reduction + KL regularization from the baseline β-VAE is responsible for most of the improvement over raw Mimi.

---

## 3. Cross-Cutting Analysis

### 3.1 The Rollout Ceiling Persists

The central finding of Stage 2 is negative: **the 2-step rollout ceiling identified in Stage 1 survives the representation change.** Despite moving from 512D Mimi to a learned 32D manifold optimized with AR-friendliness objectives, rollout still collapses by k=2 in every variant tested.

This has implications for the project thesis. Discussion 10 §6 framed Stage 2 as testing whether the problem was about manifold geometry (fixable by representation redesign) vs something more fundamental. The evidence now weighs toward the latter:

| Stage | What was tested | Did rollout survive k=2? |
|-------|----------------|--------------------------|
| 1 | Mimi 512D (vMF) | No (cos→0 by step 2) |
| 1 | EnCodec 128D (MDN) | No (cos=0.09 by step 4) |
| 1 | PCA-32 (MDN) | No (cos=0.10 by step 4) |
| 1 | vMF rollout fine-tuning | No (cos=-0.004) |
| **2** | **β-VAE 32D baseline** | **Marginal** (k=2 rollout_nll=-38.85, but k=4 inf) |
| **2** | **β-VAE 32D + L_smooth** | **No** (k=2 = 1.36e+35) |
| **2** | **β-VAE 32D + L_pred** | **No** (k=2 = NaN) |
| **2** | **β-VAE 32D + L_smooth + L_pred** | **No** (k=2 = inf) |
| **2** | **Score correction on Mimi** | **No** (made things worse) |

Ten experimental conditions across two stages. One marginal success (baseline β-VAE survives k=2 with gap=8.3). The pattern is robust.

### 3.2 What the β-VAE Did Accomplish

Despite the persistent rollout ceiling, the β-VAE demonstrates real value:

1. **3× ΔNLL over PCA-32 at the same dimensionality.** The learned encoder captures far more predictable structure than linear projection. This validates the representation learning approach even if rollout is still hard.

2. **Excellent reconstruction.** Mel distance ~2.47, L1 ~0.036. The 32D bottleneck preserves nearly all audio-relevant information. This is competitive with codec-grade reconstruction.

3. **The baseline β-VAE is the first representation where k=2 rollout doesn't immediately explode.** rollout_nll=-38.85 at k=2 (gap=8.3 from teacher) is far from perfect but meaningfully finite. Every other tested representation produces inf/NaN at k=2. This suggests the learned manifold *is* smoother — the improvement just isn't enough to survive extended rollout.

### 3.3 Why AR-Loss Objectives Underperformed

The predictability loss (L_pred) was expected to be the key intervention. Instead, it showed mixed effects. Diagnosis:

1. **Predictor mismatch.** L_pred trains a lightweight MLP predictor jointly with the VAE. But the diagnostic battery uses a separately-trained MDN. The latent space is optimized for *the wrong predictor*. The co-trained MLP exploits features of the latent space that the MDN doesn't leverage, and vice versa.

2. **Gradient competition.** L_pred's gradient flows through z (via reparameterization) back to the encoder. This competes with the reconstruction gradient, which wants z to preserve audio information. The equilibrium may sacrifice some reconstruction-relevant structure for predictability, without enough gain to matter for a different downstream model.

3. **Scale of the effect.** λ_pred=0.01 may be too small. The pred loss (~0.95) contributes only ~0.01 to the total loss of ~7.4. The reconstruction loss dominates the encoder's gradients 600:1. The AR objectives may need to be much stronger to reshape the latent manifold.

4. **Smoothness hurts short-horizon rollout.** L_smooth penalizes latent jumps, which should help rollout. But the smooth variants all collapse at k=2 while the baseline survives. The smoothness penalty may be removing sharp transitions that the dynamics model needs as landmarks — smoothing the manifold also removes the features that make prediction tractable.

### 3.4 The Baseline β-VAE Anomaly

The most interesting result is the baseline β-VAE's k=2 survival. This variant has no AR objectives — just reconstruction + KL. Yet it's the only representation across both stages where k=2 rollout produces a finite, meaningful NLL (-38.85, gap=8.3).

Possible explanations:

1. **KL regularization provides the right inductive bias.** The KL penalty pushes the posterior toward N(0,I), creating a latent space with bounded, Gaussian-like geometry. This may be more important for rollout stability than explicit AR losses.

2. **Unconstrained optimization finds a natural sweet spot.** Without competing smoothness/predictability gradients, the encoder can freely optimize for reconstruction + KL, landing on a manifold that happens to have favorable dynamics.

3. **The AR losses over-constrain.** By adding smoothness and predictability objectives, we force the encoder into a compromise that's worse for rollout than the unconstrained optimum. The VAE already produces a smooth manifold via KL; adding L_smooth on top is redundant and distortionary.

This is reminiscent of regularization in deep learning: moderate regularization (KL only) helps, but excessive regularization (KL + smooth + pred) can hurt by constraining the model away from good solutions.

---

## 4. Updated Probability Estimates

| Hypothesis | Disc. 10 | Post-Stage 2 | Reasoning |
|------------|----------|--------------|-----------|
| β-VAE alone (KL + low dim) improves over Mimi for AR | 0.50 | **0.70** | Confirmed: 3× ΔNLL over PCA-32, k=2 survival. Real improvement, just not enough |
| Explicit AR losses (L_smooth, L_pred) help beyond β-VAE baseline | 0.40 | **0.15** | Marginal, metric-dependent. Baseline outperforms on the most important metric (k=2 rollout) |
| Score-based correction improves rollout stability | 0.40 | **<0.05** | Decisively refuted. Zero improvement, actively harmful at larger step sizes |
| Problem is fundamental to audio at 12.5 Hz | 0.30 | **0.50** | 10 experimental conditions, 2 stages, 1 marginal success. Pattern is robust |
| Stronger β-VAE training (more steps, higher β, larger bottleneck) could extend rollout | — | **0.35** | New. Current training is only 10k steps; higher β or larger bottleneck untested |
| Multi-step VAE rollout training (unroll through VAE during training) could help | — | **0.30** | New. Current L_pred only trains a surrogate predictor; actual rollout training might work better |
| Frame rate reduction (< 12.5 Hz) extends rollout horizon | — | **0.25** | Still untested but increasingly plausible given universal failure at current rate |
| Transformer dynamics model (vs MLP MDN) could extend rollout | — | **0.20** | Different model class might capture long-range patterns that MDN cannot |

---

## 5. What We Now Know

### Established facts (Stage 1 + Stage 2)

- **Learned low-dim representations beat linear projection.** β-VAE 32D ΔNLL is 3× PCA-32 at the same dimension. Information recovery is real.
- **Reconstruction is not the bottleneck.** All VAE variants produce mel ~2.47; audio quality is preserved. The problem is purely in dynamics, not encoding.
- **KL regularization helps rollout.** β-VAE baseline is the first and only representation to survive k=2 rollout with finite, meaningful NLL.
- **Explicit AR losses (L_smooth, L_pred) provide marginal benefit at best.** They reshape the failure mode without extending the horizon. The baseline β-VAE outperforms on the most critical metric.
- **Score correction is not viable.** Unconditional score fields don't capture temporal coherence information. Ruled out.
- **The 2-step rollout ceiling is robust.** 10 conditions, 2 stages, representations from 32D to 512D, linear to learned. Rollout collapses by k=2 in every case (baseline β-VAE marginal exception).
- **Direction dominates perception** (from Stage 1). Even modest rollout direction accuracy should be perceptually useful, but no model sustains it.

### Open questions

- **Is more aggressive β-VAE training the answer?** 10k steps, β=0.01, 32D are one point in a large space. Higher β, more steps, different bottleneck dimensions could shift the tradeoff.
- **Does end-to-end rollout training (unrolling through the VAE encoder during dynamics training) help?** Current setup trains VAE and dynamics model separately. Joint training with actual rollout gradients could produce representations that resist dynamics-model-specific drift.
- **Is 12.5 Hz fundamentally too fast?** Each step is 80ms of audio. Perhaps at 6.25 Hz (160ms steps) or 3.125 Hz (320ms), the dynamics become predictable enough for multi-step rollout.
- **Would a different dynamics model architecture change the picture?** The Phase 1 MDN is a simple MLP. A transformer with attention over the context window might capture longer-range dependencies.
- **~~What do the Exp 7 perceptual rollout WAVs sound like?~~** ANSWERED: k=1 sounds identical to GT. k=2 has barely perceptible artifacts. k=4+ clearly distorted. Practical target is k=2–4.

---

## 6. Recommendations for Stage 3

The fundamental question has narrowed: **why does rollout universally collapse at step 2, and what breaks the ceiling?**

We've eliminated several hypotheses:
- ~~Representation too high-dimensional~~ → β-VAE 32D still fails
- ~~Manifold geometry too thin~~ → Score correction doesn't help
- ~~Need smoother dynamics~~ → L_smooth makes things worse
- ~~Need more predictable dynamics~~ → L_pred doesn't transfer to new predictor

What remains untested:

### 6.1 β-VAE Hyperparameter Sweep (HIGH priority, ~1 week)

The current β-VAE experiment tested one configuration (β=0.01, 32D, 10k steps). The baseline's k=2 survival is encouraging but marginal. A targeted sweep could find the operating point where the β-KL tradeoff maximally extends rollout:

- **β sweep:** 0.001, 0.01, 0.1, 1.0. Higher β → stronger regularization → smoother manifold, at reconstruction cost.
- **Bottleneck dimension sweep:** 16, 32, 64, 128. The 32D choice was inherited from CALM; the optimal point may differ.
- **Training duration:** 50k steps (current: 10k). Longer training may be needed for the encoder to find the smoothest manifold.

**Decision criterion:** If any configuration achieves finite rollout NLL at k=4, the ceiling is breakable and we enter full optimization mode. If the sweep confirms k=2 as the hard ceiling regardless of β/dim/duration, the problem is deeper than representation.

### 6.2 End-to-End Rollout Training (HIGH priority, ~2 weeks)

The fundamental limitation of the current approach: **the VAE and dynamics model are trained separately.** The L_pred surrogate predictor is a weak proxy — its gradients don't match what the actual Phase 1 MDN would need.

Proposed: train the VAE with an actual multi-step rollout loss:
1. Encode audio → z sequence
2. Train MDN on z for a few hundred steps (inner loop)
3. Roll out MDN for K steps
4. Backpropagate rollout loss through z (via reparameterization) to the encoder
5. The encoder learns to produce representations where *this specific dynamics model* can sustain rollout

This is computationally expensive (inner loop + outer loop) but directly optimizes for the objective we care about. It's conceptually similar to Exp 1B's rollout fine-tuning, but applied to the *representation* rather than the dynamics model.

**Risk:** The same mean-collapse failure mode from Exp 1B could appear — the encoder might learn to produce constant latents (zero dynamics = zero rollout error). Mitigation: reconstruction loss anchors the encoder to preserve audio information; the KL term prevents posterior collapse.

### 6.3 Perceptual Listening Results (COMPLETED)

Listening evaluation of Exp 7 WAVs has been performed. The rollout script splices GT prefix → rollout → GT suffix, with rollout starting at ~1/3 through the utterance.

**Results:**

| Rollout | Duration | Perceptual quality |
|---------|----------|-------------------|
| k=1 | 80ms | **Indistinguishable from GT** |
| k=2 | 160ms | Barely perceptible distortion at rollout point |
| k=4 | 320ms | Short distorted blip, rest perfect |
| k=8 | 640ms | ~0.5s clearly distorted section |
| k=16 | 1.28s | ~1s of obvious distortion |

Everything outside the rollout segment sounds identical to GT in every file. The distortion is localized precisely to the rollout frames, confirming: (a) the Mimi decoder faithfully reproduces GT latents, and (b) the dynamics model's error accumulation is the sole source of quality degradation.

**Critical finding: k=1 is perceptually lossless.** The dynamics model's single-step prediction produces audio indistinguishable from ground truth. This means individual step prediction quality is not the bottleneck — the problem is purely error accumulation across steps.

**Implications for the practical bar:**
- For applications with teacher forcing every 80ms (k=1): **already solved.** The current system produces perfect audio.
- For 160ms gaps (k=2): marginal — barely audible artifacts. The baseline β-VAE's k=2 rollout survival (gap=8.3) may be perceptually sufficient.
- For 320ms+ gaps (k=4+): clear artifacts. This is the regime that needs improvement.
- The perceptual bar is "survive k=2–4," not k=8+. This is closer to achievable than feared.

### 6.4 Frame Rate Reduction (MEDIUM priority, ~2 weeks)

If the β-VAE sweep (§6.1) confirms the k=2 ceiling, frame rate becomes the leading hypothesis. At 12.5 Hz, each step is 80ms. At 6.25 Hz, each step is 160ms — the dynamics between frames are smoother, and the current k=1 rollout quality would cover twice as much audio.

This requires retraining the Mimi encoder/decoder at a different stride, which is expensive. But it's the only untested structural variable that could fundamentally change the dynamics landscape.

### 6.5 Transformer Dynamics Model (LOW priority, ~2 weeks)

The Phase 1 MDN uses a fixed-size context window (8 frames) flattened through an MLP. A transformer could attend to longer context and capture patterns the MDN misses. However:
- The rollout ceiling appears at step 2, well within the 8-frame window
- The problem is more likely accumulation error than context length
- A transformer would be expensive to train for marginal expected benefit

Defer unless §6.1 and §6.2 fail to move the needle.

### Priority Order (updated after §6.3 listening)

The perceptual results change the priority calculus. k=1 is already lossless, and the practical target is k=2–4 (160–320ms), not k=8+. This makes the problem more tractable.

```
Week 1:    §6.1 β-VAE hyperparameter sweep (β, dim, duration)
Week 2-3:  §6.2 End-to-end rollout training (if sweep fails to break k=4)
Week 3-4:  §6.4 Frame rate reduction (if rollout training also fails)
```

### Decision Gate: After §6.1

- If any β-VAE config achieves finite k=4 rollout → the ceiling is soft, optimize within this framework
- If all configs hit k=2 ceiling → proceed to §6.2 (end-to-end rollout training), which changes the training paradigm rather than the representation
- If §6.2 also fails → §6.4 (frame rate reduction) or reassess the core thesis

---

## 7. Paper Narrative Update

Discussion 10 §8 outlined four narrative options. Post-Stage 2:

**Option A** ("representation design is the primary lever") is *partially* supported. The β-VAE clearly outperforms raw Mimi and PCA for prediction, and the baseline β-VAE is the only representation with marginal k=2 rollout survival. But it doesn't break the rollout ceiling — "primary lever" oversells the result.

**Option C** ("why audio is hard and what partially helps") is strengthening. The systematic evidence across 10+ conditions that rollout collapses by step 2 — regardless of representation, model, or training procedure — is itself a significant finding. Combined with the perceptual validation (direction >> magnitude) and the β-VAE's partial improvement, this forms a coherent story:

> Continuous AR audio generation fails not because existing representations are bad, but because the dynamics of audio latent spaces are inherently challenging for autoregressive prediction. Learned representations (β-VAE) partially mitigate this through dimensionality reduction and KL regularization, extending the rollout horizon from ~1 to ~2 steps. However, explicit AR-friendliness objectives (smoothness, predictability) provide marginal additional benefit, and inference-time correction (score matching) is ineffective. The perceptual importance of directional accuracy over magnitude accuracy suggests that even modest improvements in rollout stability could yield disproportionate audio quality gains.

This is a publishable story. Stage 3 experiments (§6.1–6.4) could strengthen it toward Option A if the hyperparameter sweep or end-to-end training breaks the ceiling.

---

## 8. Relation to Engram / Memory

The memory-as-basis hypothesis remains gated on rollout stability (as noted in Discussion 10 §9). The β-VAE's 32D manifold is actually a better substrate for archetype discovery than raw Mimi — lower-dimensional, KL-regularized, with predictable structure. If the hyperparameter sweep (§6.1) or rollout training (§6.2) extends the horizon to k≥4, the 32D β-VAE latent space becomes the natural target for direction archetype extraction.

**Updated status:** Still gated on rollout stability. The β-VAE provides the right representation; we need the dynamics to be stable enough to observe recurring patterns.

---

## Appendix A: Complete Experimental Configuration

### Exp 5: β-VAE with AR-Friendliness

- **Architecture:** Mimi encoder (project=False, frozen) → Conv1d(512,32) × 2 (mu, logvar) → reparameterize → Conv1d(32,512) → Mimi decoder (unfrozen)
- **Training:** 10k steps, batch_size=4, lr=1e-4, fp16 AMP, segment_sec=4.0
- **Data:** LibriSpeech train-clean-100, 5000 utterances
- **Losses:** Multi-scale STFT (windows 256/512/1024/2048) + time-domain L1 + β·KL + λ_smooth·L_smooth + λ_pred·L_pred
- **Predictor:** 2-layer MLP (32×8 → 256 → 32), lr=1e-3, window_size=8
- **Diagnostic:** Phase 1 MDN (8-component, 1024-hidden, 2-layer) for k∈{1,2,4,8}, injection diagnostic (16-step, 4 modes)
- **Duration:** 3h 29m on TITAN X Pascal
- **Run ID:** 20260206_174326

### Exp 6: Score-Based Manifold Correction

- **Score model:** 4-layer MLP (512→1024→512), sinusoidal sigma embedding, denoising score matching
- **Training:** 20k steps, batch_size=256, lr=1e-3, σ∈[0.01, 1.0]
- **Correction sweep:** n_steps∈{1,3,5} × step_size∈{0.001,0.01,0.1} × σ=0.1 (9 configs)
- **Baseline dynamics:** vMF k=1 model from Exp 1
- **Duration:** 4m 39s on TITAN X Pascal
- **Run ID:** 20260206_172109

### Exp 7: Perceptual Rollout Evaluation

- **Dynamics model:** vMF k=1 from Exp 1
- **Decoder:** Mimi autoencoder (24kHz → 48kHz resample)
- **Utterances:** 10 eval utterances, rollout lengths {1,2,4,8,16}
- **Output:** 50 WAV files + 10 GT references at 48kHz
- **Duration:** 5s on TITAN X Pascal
- **Run ID:** 20260206_172050

### Compute Summary

| Experiment | GPU Time | Key Output |
|------------|----------|------------|
| Exp 5 (β-VAE, 4 variants) | 3h 29m | summary.csv, 4× metrics.json, 4× injection_diag.json, 4× VAE checkpoints |
| Exp 6 (score correction) | 4m 39s | summary.csv, score checkpoint |
| Exp 7 (perceptual rollout) | 5s | manifest.json, 60 WAV files |

Total Stage 2 compute: approximately 3.6 GPU-hours on TITAN X Pascal.

---

## Appendix B: Cross-Stage Rollout Comparison

All per-horizon rollout results across both stages (k=1 MDN models, rollout NLL):

| Representation | Dim | k=1 rollout_nll | k=2 rollout_nll | k=4 rollout_nll |
|----------------|-----|-----------------|-----------------|-----------------|
| Mimi (raw) | 512 | — | — | — |
| EnCodec encoder | 128 | — | — | — |
| PCA-32 | 32 | — | — | — |
| β-VAE baseline | 32 | -0.92 | **-38.85** | inf |
| β-VAE smooth | 32 | -1.36 | 1.36e+35 | NaN |
| β-VAE pred | 32 | +10.23 | NaN | inf |
| β-VAE smooth_pred | 32 | +2.42 | inf | inf |

*Stage 1 per-horizon rollout NLLs not directly available in same format (different evaluation code paths); injection diagnostic data used for Stage 1 comparisons in main text.*

The baseline β-VAE's k=2 survival (rollout_nll = -38.85, gap = 8.30) is unique across all conditions tested.
