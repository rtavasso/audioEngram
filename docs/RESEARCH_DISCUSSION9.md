# Research Discussion 9 (Revised): Representation Design as a Hypothesis — What We Know, What We Don't, and What to Test Next

**Date:** 2026-02-05  
**Participants:** Riley + Claude (Opus 4.6)  
**Context:** Post-Tier 1 results analysis, CALM paper review, and consideration of project direction  
**Status:** Proposal for team discussion — not a decision document

---

## Executive Summary

Eight discussions and five experimental phases have produced a rich diagnostic picture but have not yet resolved the central question: **is the bottleneck the dynamics model, the representation, or both?** This document argues that the representation deserves serious investigation as a potential primary lever, but acknowledges that several cheaper experiments from earlier discussions remain unfinished and should be completed before committing to a multi-week pivot.

**The argument in outline:**

1. Tier 1 results show that vMF teacher-forced direction cosine (0.73) is our best yet, but a LogNormal magnitude implementation bug prevented any rollout evaluation. The injection diagnostic (Exp2) confirms catastrophic off-manifold sensitivity. The representation comparison (Exp3/EnCodec) remains incomplete.

2. A close reading of CALM suggests that much of its architecture compensates for representation-level fragility, though reasonable people could interpret those components as principled design choices rather than workarounds.

3. Training a VAE with explicit AR-friendliness objectives is a promising direction — but before committing to a 36-condition sweep, we should complete the cheap experiments Discussion 7 identified as decision-relevant: fix the magnitude sampling, run EnCodec, and listen to "slow but correct" outputs.

**This document proposes a two-stage plan:** finish the deferred diagnostics first (1–2 weeks), then decide whether the representation pivot is warranted based on what those diagnostics reveal.

---

## 1. What We Actually Know (Phases 0–4.5 + Tier 1)

### 1.1 Findings with strong evidence

| Phase | Finding | Strength of evidence |
|-------|---------|---------------------|
| Phase 0 → 1 | Clustering misses structure that learned predictors find | Strong: ΔNLL = -185 at k=1 vs variance ratio ≈ 1.0 |
| Phase 1 | Direction is predictable (cos ~0.61), magnitude is not (R² < 0) | Strong: consistent across horizons and splits |
| Phase 3 | z_dyn is more predictable than raw Mimi (cos 0.71 vs 0.61) | Moderate: single configuration, but clean improvement |
| Phase 4.5 | Rollout training converts catastrophic divergence to stability | Strong: +761K → -0.99 ΔNLL, reproducible across sweep points |
| Phase 4.5 | Stability comes at the cost of mean collapse (‖Δẑ‖: 10.76 → 0.024) | Strong: consistent across all sweep points (tw=0, 0.1; sw=0, 0.05, 0.2) |
| Tier 1 Exp2 | One step of predicted context causes immediate divergence | Strong: state_err ~5e13 by step 2 across modes B/C/D |

### 1.2 Findings that are suggestive but incomplete

| Finding | What's missing |
|---------|---------------|
| vMF direction cosine (0.73) is our best | **No rollout evaluation** — LogNormal E[m] blow-up prevented testing the "slow but correct" hypothesis. Discussion 8 diagnosed this as a sampling bug (using E[m] instead of median), not a refutation of factorization. |
| Exp2 shows off-manifold catastrophe | **Single representation only** — we don't know if EnCodec or other representations show the same sensitivity. This was identified in Discussion 7 as a first-order question. |
| Residual memory adds little after rollout training | **Only tested with conservative mean-rollout dynamics** — memory might matter more if the dynamics model makes non-trivial predictions (i.e., after fixing mean collapse). |
| z_rec is unused in Phase 3 | **Limited hyperparameter exploration** — may reflect the specific capacity/KL regime tested, not an inherent limitation. |

### 1.3 What we cannot claim yet

We should be careful about two tempting narratives:

**"Every dynamics-side intervention failed."** This overstates the case. vMF never got a rollout test. Rollout training achieved stability (a major result) — the problem is mean collapse, which is an objective design issue, not evidence that dynamics modeling is hopeless. The Phase 4.5 sweep explored only one axis (teacher_weight, state_weight); we haven't tried sample-based rollout training, adversarial objectives, or the vMF rollout protocol Discussion 7 specified.

**"The representation is the bottleneck."** This is a plausible hypothesis, but it's currently supported more by reasoning from first principles (Mimi wasn't optimized for AR) than by comparative evidence (we haven't shown that a different representation does better). The strongest evidence would be an EnCodec or β-VAE comparison — neither exists yet.

---

## 2. The Representation Hypothesis: Why It's Plausible

Despite the caveats above, there are good reasons to take the representation hypothesis seriously.

### 2.1 Mimi's training objectives don't encourage AR-friendly dynamics

Mimi was trained for reconstruction + RVQ quantization + WavLM semantic distillation. None of these objectives encourage temporal smoothness, predictability of dynamics, or robustness to context perturbation. This is the argument from Discussion 1 (§2), Discussion 7 (§1.4), and it remains compelling.

The specific concerns:

- **Reconstruction pressure** → dense information packing → thin data manifold → high sensitivity to perturbation. This is consistent with Exp2's immediate divergence.
- **RVQ quantization** → codebook structure, not smooth dynamics. (Ironically, discrete tokens from RVQ may have better AR properties than pre-quantization continuous latents — a hypothesis from Discussion 1 that remains untested.)
- **WavLM distillation** → phonetic discriminability, not temporal smoothness. May create sharp boundaries in latent space.

### 2.2 CALM's architecture is informative (but interpretable multiple ways)

CALM (Rouard et al., 2025) achieves strong results with continuous latents, but several architectural choices suggest the authors are compensating for representation-level fragility:

- **Noise injection (§4.1):** Corrupts latent frames during training to make the backbone robust to prediction errors at inference. This is explicitly described as preventing error accumulation.
- **Short-context transformer (§4.2):** A clean-context pathway over the last K=10 frames. Ablation shows removing it degrades FAD from 0.93 to 4.03 — the single most important component.
- **Consistency head (§4.2):** Samples from the full conditional rather than predicting Δx directly, implicitly handling multimodality and magnitude uncertainty.

**The "workaround" interpretation:** These components exist because the representation isn't AR-friendly. A better representation would make them unnecessary.

**The "principled design" interpretation:** Noise injection is data augmentation for robustness (a standard technique). The short-context transformer is multi-scale conditioning (common in sequence models). The consistency head handles distributional complexity (a natural choice for continuous targets).

Both interpretations are defensible. The empirical test is whether a representation designed for AR-friendliness reduces the need for these components. But we should be honest that "workaround" is our reading, not an established fact.

### 2.3 CALM's VAE is already partially AR-friendly

CALM trains its own 32-dimensional KL-regularized VAE, not Mimi's continuous latents. This is 16× lower dimensional and includes KL regularization that smooths the manifold. Yet CALM still needs noise injection and dual-context architecture, suggesting the VAE alone isn't sufficient — at least without explicit temporal objectives. This is the gap we'd be targeting.

### 2.4 The observation that should temper our confidence

Discussion 7 (§1.3) estimated that Mimi achieves ~35% entropy reduction (ΔNLL/baseline NLL) vs >60% for language models and likely >80% for game observations. If audio is fundamentally more innovation-heavy, then *no* representation will make dynamics as easy as in domains where AR is already solved. The question is whether we can move from 35% to, say, 50-60% — enough to make simple dynamics models viable. We don't know the answer.

---

## 3. What Discussion 7 Said We Should Do (and What We Haven't Done)

Discussion 7 laid out a tiered experimental plan with clear decision criteria. Before proposing new work, we should assess what's been completed:

### 3.1 Tier 1 items from Discussion 7

| Item | Status | Decision criterion |
|------|--------|-------------------|
| vMF + rollout training (K=16) | ❌ **Not done** — teacher-forced only; rollout blocked by LogNormal bug | cos(Δz_pred, Δz_true) > 0.6 at K=16 |
| Representation comparison (EnCodec) | ❌ **Not done** — environment issue | >2× ΔNLL improvement → switch representation |
| Listen to outputs | ❌ **Not done** | Qualitative: is "slow but correct" perceptually better? |

### 3.2 Discussion 7 contingency logic

Discussion 7 proposed: **vMF rollout → representation comparison → archetypes (if warranted) → score-based correction (if needed)**.

The decision tree was:
- If vMF rollout preserves direction (cos > 0.6 at K=16): proceed to archetypes
- If vMF fails AND representation comparison shows >2× improvement: switch representation
- If both fail: consider score-based correction

**We are currently at the first node of this tree, having not completed any of the three Tier 1 experiments to their decision criteria.** Proposing a major pivot from this position skips the decision framework we agreed on.

### 3.3 Other deferred items

| Item | Source | Status |
|------|--------|--------|
| Score-based manifold correction | Discussion 7 §4 (p=0.35) | Not explored. Exp2's off-manifold catastrophe is exactly the scenario where this was recommended. |
| Linear readout baseline (reservoir-inspired) | Discussion 7 §5 (p=0.40) | Not explored. Would test whether simple dimensionality reduction of Mimi latents improves AR-friendliness without training new encoders. |
| Perceptual evaluation of direction vs magnitude | Discussion 7 §9 Q2 | Not explored. Would directly validate or invalidate the motivation for the representation pivot. |

---

## 4. Proposed Plan: Finish Diagnostics, Then Decide

### 4.1 Stage 1: Complete deferred experiments (1–2 weeks)

These are all cheap and directly decision-relevant:

**Experiment A: Fix vMF magnitude and run rollout evaluation** (2–3 days)

Fix the LogNormal sampling bug (use median m = exp(μ) instead of E[m] = exp(μ + ½σ²), or clamp log_sigma tightly). Then:

1. Re-run Exp1 with fixed magnitude → report ΔNLL under both teacher forcing and rollout
2. Run vMF rollout training (K=16) per Discussion 7 protocol
3. Track cos(Δz_pred, Δz_true) and ‖Δz_pred‖/‖Δz_true‖ separately at each rollout step
4. **Decision:** If cos > 0.6 at K=16 and ‖Δz_pred‖/‖Δz_true‖ > 0.1 (not collapsed), the vMF factorization is working and the representation pivot is less urgent.

**Experiment B: Complete EnCodec representation comparison** (1–2 days)

Fix the environment issue. Run Phase 1 predictor baseline on EnCodec latents at matched frame rate.

1. Report ΔNLL, direction cosine, logmag R² for k ∈ {1, 2, 4, 8}
2. Run injection diagnostic (Exp2 modes A–D) on EnCodec latents
3. **Decision:** If EnCodec shows substantially better predictability or longer divergence horizon, the gap is representation-specific and the pivot is strongly motivated. If results are comparable, the gap may be more fundamental to audio.

**Experiment C: Perceptual evaluation of "slow but correct"** (1–2 days)

Generate audio from rolled-out latents under different regimes:

1. Mode A: Stationary (post-rollout-training mean collapse, ‖Δẑ‖ ≈ 0.024)
2. Mode B: Ground-truth directions, reduced magnitude (simulate "slow but correct")
3. Mode C: Random-direction rollout at ground-truth magnitude
4. Baseline: Teacher-forced (ground truth)

Decode through Mimi decoder. Listen. Record informal quality notes.

**Decision:** If Mode B sounds substantially better than Mode A, directional preservation matters more than exact magnitude — validating vMF and potentially reducing the urgency of the representation pivot. If Mode B sounds terrible (because the decoder is sensitive to magnitude), the representation pivot gains urgency.

**Experiment D: Linear readout baseline** (1 day)

Learn a linear projection z_dyn = W · [z_{t-L:t}] of Mimi latents to a 32–64 dimensional subspace. Run Phase 1 predictor and injection diagnostic on the projected latents.

**Decision:** If linear projection substantially improves AR-friendliness metrics, dimensionality reduction alone may be sufficient — supporting the representation hypothesis but suggesting a simpler intervention than training new VAEs.

### 4.2 Stage 1 decision matrix

| Stage 1 outcome | Interpretation | Stage 2 action |
|-----------------|---------------|----------------|
| vMF rollout works (cos > 0.6, no collapse) | Dynamics-side factorization is viable | Pursue vMF direction + archetypes + perceptual eval |
| vMF fails, but EnCodec dramatically better | Gap is Mimi-specific | Consider representation switch or VAE pivot |
| vMF fails, EnCodec similar, linear projection helps | Dimensionality is the issue, not the objective | Train low-dim VAE (simpler than full pivot) |
| vMF fails, EnCodec similar, linear projection doesn't help | May be fundamental to audio at this frame rate | Consider frame rate experiments, score-based correction, or "negative but informative" paper framing |
| "Slow but correct" sounds good perceptually | Direction > magnitude for perception | Validates vMF approach regardless of NLL metrics |

### 4.3 Stage 2: Representation pivot (if warranted) (~4 weeks)

If Stage 1 evidence supports the representation hypothesis — particularly if EnCodec or linear projection shows substantial improvement, or if all dynamics-side interventions including fixed vMF fail — then the VAE training sweep becomes well-motivated.

The design follows the original Discussion 9 proposal, with adjustments:

#### Phase R1: VAE Training Sweep

Train VAEs on LibriSpeech train-clean-100, varying:

| Dimension | Values | Rationale |
|-----------|--------|-----------|
| Latent dim | {32, 64, 128} | CALM uses 32; Mimi uses 512. Lower dim = smoother manifold |
| β (KL weight) | {0.01, 0.1, 1.0} | Controls manifold thickness |
| AR loss | {none, L_smooth, L_pred, L_smooth + L_pred} | The experimental variable |
| Frame rate | 12.5 Hz | Match Mimi for comparability |

Architecture: Mimi-style encoder/decoder with VAE bottleneck instead of RVQ. Total: 36 conditions, ~100–150 A100-hours.

**AR-friendliness losses:**

- **L_smooth** = E_t[ ‖x_t - x_{t-1}‖² ] — penalizes large frame-to-frame jumps
- **L_pred** = E_t[ -log p_φ(Δx_t | x_{t-W:t-1}) ] — jointly trained lightweight predictor penalizes unpredictable dynamics
- **L_jacobian** (optional) = penalizes rapid changes in encoder Jacobian between adjacent frames
- **L_noise** (CALM-inspired, optional) = decoder robustness to small latent perturbations
- **L_rollout** (expensive, optional) = decode rolled-out latents and penalize reconstruction error

#### Phase R2: Diagnostic Battery

For each VAE, run the existing diagnostic framework:

1. Phase 1 predictor baseline (ΔNLL at k=1,2,4,8)
2. Injection diagnostic (Exp2 modes A–D)
3. Rollout stability (dynamics model with K=16 rollout)
4. Reconstruction quality (PESQ, STOI, mel distance)

#### Phase R3: Best-Representation Dynamics Evaluation

Top 3–5 representations from Pareto front → full dynamics pipeline (Phase 3 RSSM, Phase 4.5 rollout training).

#### Phase R4: End-to-End Generation and Listening

Generate audio, decode, compare perceptually.

### 4.4 Decision criteria for the representation pivot

| Outcome | Interpretation | Next step |
|---------|---------------|-----------|
| AR-loss VAEs show >2× ΔNLL improvement AND divergence horizon >10 steps AND reconstruction within 10% of Mimi | AR-friendly representations are learnable and dramatically help | Full paper on representation design |
| AR loss helps predictability but not stability | Representation helps but isn't sufficient alone | Combine AR-friendly VAE with rollout training |
| AR loss degrades reconstruction without helping stability | Tension is too steep | Investigate whether the tradeoff is fundamental |
| No AR loss variant meaningfully improves over β-VAE alone | KL regularization + low dim is sufficient | Simpler story: "just use a KL-regularized VAE" |

---

## 5. What We Keep From Previous Work

The Phase 0–4.5 diagnostic framework is valuable regardless of which direction we take:

- **Phase 0 → Phase 1 reconciliation:** Methodological contribution (variance-ratio metrics miss structure that probabilistic predictors find)
- **Direction/magnitude decomposition:** Characterizes the geometry of predictability in audio dynamics
- **Injection diagnostic (Exp2):** Cheap, decisive test for rollout viability — reusable for any representation
- **Phase 4.5 rollout training:** Demonstrates that stability is achievable but current representations force mean collapse — the motivating finding

---

## 6. Relation to Engram / Memory

The original Engram hypothesis has been partially tested:

- **Phase 0:** Clustering-based lookup explains nothing. (Strong negative)
- **Phase 4:** Residual memory helps teacher-forced prediction (+3.4 nats ΔNLL) but adds nothing after rollout training. (Moderate negative — but rollout-trained models make nearly zero-magnitude predictions, limiting what any corrector can add.)
- **Discussion 5–7 reformulation** (memory-as-basis / soft retrieval over direction archetypes): **Untested.** Probability estimates from Discussion 7: 0.25 for memory-as-basis helping, 0.50 for archetypes existing.

The honest summary: naive discrete memory doesn't transfer from language to continuous audio latents. The refined continuous-memory hypothesis remains untested and is lower priority than the dynamics/representation questions, but shouldn't be declared dead without testing. If vMF rollout succeeds and direction archetypes are found (Discussion 5's protocol), memory-as-basis becomes a natural follow-up.

---

## 7. Relation to Score-Based Correction

Discussion 7 (§4) proposed score-based manifold correction as a contingency: if vMF rollout drifts off-manifold, train an unconditional score model on Mimi latents and apply post-hoc Langevin correction. Probability estimate: 0.35.

The Tier 1 Exp2 results (immediate off-manifold catastrophe) are precisely the scenario where this was recommended. A score model ∇_z log p(z) could provide "which direction moves toward high-density regions" — exactly what's needed when the dynamics function is undefined off-manifold.

This remains a viable alternative or complement to the representation pivot. It's architecturally simpler (no new encoder training; score model trains on existing latents) and could be tested in parallel. The tradeoff: it treats the symptom (off-manifold drift) rather than the cause (fragile manifold geometry), but treating symptoms quickly is sometimes the right research strategy.

---

## 8. Risks and Honest Uncertainties

### 8.1 Risk: We're rationalizing a pivot based on incomplete evidence

The strongest version of this critique: we haven't finished the Discussion 7 plan, vMF never got a fair rollout test, EnCodec comparison doesn't exist yet, and we're proposing a 4-5 week detour based on reasoning-from-principles plus one experiment (Exp2) that could be addressed by simpler interventions (score correction, better rollout objectives).

**Mitigation:** Stage 1 addresses this directly. We commit to finishing the cheap experiments before committing compute to VAE training. The pivot is a hypothesis to be tested, not a conclusion to be acted on.

### 8.2 Risk: AR-friendliness losses destroy reconstruction quality

If the Pareto front is steep (small gains in predictability require large reconstruction losses), the approach may not be practical.

**Mitigation:** The sweep includes a "no AR loss" baseline (pure β-VAE). CALM demonstrates that 32-dim KL-regularized VAEs achieve competitive reconstruction (PESQ 2.42, STOI 0.90). The question is whether explicit AR losses help beyond what KL alone provides.

### 8.3 Risk: The problem is fundamental to audio, not Mimi-specific

If speech at 12.5 Hz is genuinely high-entropy regardless of representation, no amount of encoder redesign will help.

**Mitigation:** Stage 1 Experiment B (EnCodec) and D (linear readout) provide evidence on this. Discussion 7's entropy analysis suggests audio is harder than games/language but not impossible — 35% entropy reduction is meaningful, and better representations might push it higher.

### 8.4 Uncertainty: Is CALM's architecture actually suboptimal?

We're framing CALM's noise injection and short-context transformer as costs to be eliminated. But CALM works well. Multi-scale conditioning and robustness training are established techniques. Even if we produce a "cleaner" representation, the end-to-end system might still benefit from these components.

The strongest claim we can make is not "CALM's architecture is wrong" but "a representation designed for AR-friendliness should require *less* of these compensating mechanisms." The degree to which they become unnecessary is an empirical question.

---

## 9. Updated Probability Estimates

| Hypothesis | Discussion 7 | Updated | Reasoning |
|------------|-------------|---------|-----------|
| vMF preserves direction under rollout (cos > 0.6 at K=16) | 0.60 | 0.50 | Tier 1 teacher-forced results are encouraging (cos 0.73) but magnitude bug prevented testing; mild downward update for implementation difficulty |
| Representation change shows >2× ΔNLL improvement | 0.65 | 0.60 | Still plausible but EnCodec comparison would be the decisive test |
| β-VAE alone (no AR loss) substantially improves over Mimi | — | 0.55 | CALM's 32-dim VAE is already partially AR-friendly; KL + low dim may be the main lever |
| Explicit AR losses help beyond β-VAE baseline | — | 0.35 | Unclear whether L_pred and L_smooth add value beyond what KL regularization provides |
| Score-based correction improves rollout stability | 0.30 | 0.35 | Exp2 results are exactly the scenario where score correction should help |
| Linear readout competitive with nonlinear z_dyn | 0.45 | 0.45 | Unchanged; still untested |
| Memory-as-basis helps (if archetypes exist) | 0.25 | 0.20 | Slight downward update; continuous structure (Discussion 7 §3) makes hard archetypes less likely |

---

## 10. Immediate Next Steps (Recommended Priority Order)

**This week (Stage 1 experiments):**

1. **Fix vMF LogNormal sampling** (use median magnitude) and re-run Exp1 + rollout training per Discussion 7 protocol. Track direction cosine and magnitude ratio separately at each rollout step. *This is the single highest-value experiment because it directly tests whether the most promising dynamics-side intervention works.*

2. **Fix EnCodec environment issue** and run Phase 1 + Exp2 on EnCodec latents. *Cheapest way to test whether the gap is Mimi-specific.*

3. **Generate and listen to "slow but correct" audio** (Mode B synthetic experiment from Discussion 7 §2.3). *Cheap perceptual validation of the core assumption.*

4. **Linear readout baseline** (z_dyn = W · z, 32-64 dims). *Tests whether dimensionality alone is the issue.*

**Next week (Stage 1 → Stage 2 decision):**

5. Review Stage 1 results against decision matrix (§4.2).
6. If representation pivot is warranted: begin β-VAE baseline (single 32-dim model, 2-3 days).
7. If vMF works: proceed to direction archetype discovery and vMF perceptual evaluation.

**Weeks 3–6 (conditional on Stage 1):**

8. Full VAE sweep (if pivot), or vMF + archetypes + memory-as-basis (if dynamics path), or score-based correction (if both fail).

---

## 11. Paper Narrative Options

The narrative depends on what Stage 1 reveals:

### Option A: "Representation design is the primary lever" (if pivot succeeds)

Story: Continuous AR audio suffers from rollout instability because standard codec representations aren't AR-friendly. We develop a diagnostic battery, show that representation properties (predictability, smoothness, rollout stability) dominate dynamics model choice, and train VAEs with AR-friendliness objectives that improve stability without CALM's architectural workarounds.

### Option B: "vMF factorization enables stable directional generation" (if vMF works)

Story: The direction/magnitude decomposition reveals that audio dynamics have concentrated directional structure. vMF factorization preserves this structure under rollout while allowing magnitude to be handled separately. "Slow but correctly directed" generation is perceptually superior to mean collapse.

### Option C: "Diagnostics for continuous AR: why audio is hard and what helps" (if everything partially works)

Story: We develop a diagnostic framework for AR-friendliness of continuous representations, characterize why audio latents are harder than game observations, and show that a combination of representation design and dynamics factorization is needed. This is a methodological contribution with partial positive results.

### Option D: "Negative results and diagnostics" (if nothing works well)

Story: We systematically tested whether Engram-style memory, vMF factorization, rollout training, and representation redesign can solve the continuous AR audio rollout problem. Each intervention produced informative failures. The diagnostic framework and failure analysis are the contribution.

All four options are publishable. The key is to let the experiments determine which one we write, rather than choosing the narrative first and selecting experiments to support it.

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

CALM's 32-dim VAE matches or exceeds Mimi on reconstruction metrics despite 16× lower dimensionality. This is prior evidence that low-dim KL-regularized VAEs are viable for speech, though it doesn't tell us whether the resulting dynamics are AR-friendly without CALM's noise injection and dual-context architecture.

## Appendix B: Diagnostic Battery Specification

For any new representation, run:

1. **Extract latents:** Encode LibriSpeech-100 train and eval splits → zarr store
2. **Phase 1 predictor:** MDN on Δx, horizons k ∈ {1, 2, 4, 8}, report ΔNLL, direction cos, logmag R²
3. **Injection diagnostic:** Train dynamics MLP on z_t → Δz_t, run Modes A–D, report divergence horizon (step at which state_err > 100 × median ‖Δz‖)
4. **Rollout stability:** Train dynamics MLP with K=16 rollout, report rollout ΔNLL and mean ‖Δẑ‖ / mean ‖Δz_true‖ (magnitude ratio — 1.0 is ideal, 0.0 is collapse)
5. **Reconstruction quality:** PESQ, STOI, mel-spectrogram distance on eval speakers

All scripts exist and require only `data.latents_dir` to be pointed at the new zarr store.

## Appendix C: VAE Architecture Specification (Draft, for Stage 2)

Based on Mimi's encoder/decoder with modifications:

- **Encoder:** Causal convolution stack (stride pattern matching 12.5 Hz) + causal transformer layers
- **Bottleneck:** Gaussian VAE (μ, log σ → sample z = μ + σε) replacing Mimi's RVQ
- **Decoder:** Mirror of encoder (transposed convolutions + transformer)
- **Discriminator:** Multi-scale STFT discriminator (standard for audio VAE-GANs)
- **Semantic distillation:** Optional WavLM cosine loss on latent (test with/without)

Hyperparameters fixed across conditions: frame rate (12.5 Hz), encoder/decoder architecture, training schedule (AdamW, 200K steps, batch 32), discriminator.

Hyperparameters to sweep: latent dim, β, AR-friendliness loss (as specified in §4.3).