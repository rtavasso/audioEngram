# Phase 1 Results: Predictor Baseline on Mimi Latents

**Date:** 2026-02-02
**Status:** Experiment complete, results analyzed

---

## 1. Motivation

Prior work (Phase 0) tested whether Engram-style lookup memory could improve autoregressive modeling of continuous audio latents. The test clustered Mimi encoder contexts using coarse representations (mean-pooling, PCA + VQ) and measured whether clustering reduced variance in next-step dynamics (Δx = x_t - x_{t-1}).

**Phase 0 Result:** Variance ratios of 0.997 and 0.988 against random baselines—clustering explained essentially 0% of dynamics variance.

**Phase 0 Conclusion:** No reusable local structure exists in Mimi's latent representation.

**Challenge to Phase 0:** The negative result could reflect:
1. The representation genuinely lacks predictable structure, OR
2. The coarse key/metric was wrong (mean-pooling destroys trajectory shape; MSE/variance-ratio misses multimodal structure)

Phase 1 tests this directly: train a learned predictor and measure probabilistic predictability (NLL) rather than relying on clustering with hand-designed keys.

---

## 2. Experimental Design

### 2.1 Research Question

**Primary question:** Does context provide predictive information about next-step Mimi latent dynamics (Δx), as measured by a learned probabilistic model?

**Secondary questions:**
- How does predictability decay with context staleness (k)?
- Is the predictable structure in the *direction* of change, the *magnitude* of change, or both?
- How do errors compound in autoregressive rollout?

### 2.2 Data

**Encoder:** Mimi (the VAE underlying CALM)
**Source audio:** LibriSpeech
**Latent properties:**
- Frame rate: 12.5 Hz (80ms per frame)
- Dimensionality: 512 (continuous Mimi latents in this repo)

**Train/eval split:** By speaker (inherited from Phase 0 splits).

**Note on sample counts:** The validity constraint depends on k (`t >= (W-1)+k`), so the effective number of frames varies slightly across horizons. Ideally report `n` per horizon and split.

### 2.3 Task Formulation

**Lagged-context prediction (primary):** For each horizon k ∈ {1, 2, 4, 8}, train a model to predict the next-step delta at frame t from a stale context window:

```
p(Δx_t | x_{t-k-W+1 : t-k})
```

where:
- x_t ∈ ℝ^512 is the Mimi latent at frame t
- Δx_t = x_t - x_{t-1}
- W is the context window length (W=8 in the run config)

**Horizon interpretation:**
| k | Context staleness |
|---|---------------------|
| 1 | context ends 80ms before target |
| 2 | context ends 160ms before target |
| 4 | context ends 320ms before target |
| 8 | context ends 640ms before target |

This is **not** “predict k steps into the future.” The target is always the same one-step delta Δx_t; k increases how stale the conditioning window is. This directly probes how predictable the dynamics are, and (secondarily) how robust prediction is when recent context is unreliable.

### 2.4 Model Architecture

Mixture Density Network (MDN) with diagonal Gaussian mixture:
- Input: flattened context window, dimension W×D (W=8, D=512 → 4096)
- Backbone: 2-layer MLP, width 1024, GELU
- Output: K=8 mixture components with (π_k, μ_k, σ_k) over Δx_t ∈ ℝ^512
- Numerical stability: log-σ clamp

### 2.5 Metrics

**Primary metric: ΔNLL (nats saved by conditioning)**

```
ΔNLL(k) = NLL[p(Δx_t | x_{t-k-W+1 : t-k})] - NLL[p(Δx_t)]
```

- NLL is reported in **nats per frame**. Divide by log(2) to convert to bits.
- ΔNLL < 0 means context helps; ΔNLL ≈ 0 means context provides no information

**Secondary metrics:**

1. **Direction cosine similarity:** Decompose prediction into direction and magnitude:
   - Ground truth direction: u_t = Δx / ||Δx||
   - Predicted direction: û_t = Δx̂ / ||Δx̂||
   - Metric: cos(u_t, û_t) averaged over evaluation set
   - Interpretation: 1.0 = perfect direction prediction, 0.0 = random

2. **Log-magnitude R²:**
   - Ground truth: m_t = log(||Δx|| + ε)
   - Predicted: m̂_t
   - Metric: R² = 1 - Var(m_t - m̂_t) / Var(m_t)
   - Interpretation: 1.0 = perfect magnitude prediction, 0.0 = predicting mean, <0 = worse than mean

3. **Rollout gap:** Compare teacher-forced NLL to autoregressive rollout NLL:
   - Teacher-forced: model sees ground-truth context at each step
   - Rollout: model sees its own predictions as context
   - Gap measures context-corruption sensitivity / error compounding

### 2.6 Baseline

The marginal baseline models p(Δx_t) without any context as a diagonal Gaussian fit on training deltas. This is the "null hypothesis" that context provides no information.

---

## 3. Results

### 3.1 Primary Results: Conditional vs Marginal NLL

| Horizon k | Eval NLL (conditional) | Eval NLL (baseline) | ΔNLL | Interpretation |
|-----------|------------------------|---------------------|------|----------------|
| 1 | 1075.5 | 1260.9 | **-185.4** | Context helps substantially |
| 2 | 1129.8 | 1260.9 | **-131.1** | Context helps |
| 4 | 1176.0 | 1260.9 | **-84.8** | Context helps |
| 8 | 1191.9 | 1258.7 | **-66.8** | Context helps |

**Key finding:** ΔNLL is negative across all horizons, indicating that the full (continuous) context window provides meaningful predictive information about next-step dynamics. The effect decays with context staleness but remains present even when the context ends ~640ms before the target.

**Important nuance (vs Phase 0):** Phase 1 demonstrates structure under full-context conditioning; it does not show that this structure is compressible into a small discrete key suitable for O(1) lookup. So “Phase 0 failed” and “Phase 1 ΔNLL << 0” can both be true.

To put scale on this (eval split):
- k=1: ΔNLL = -185.44 nats/frame ≈ -0.362 nats/dim ≈ -0.523 bits/dim, which is ~267 bits/frame (~3.34 kbps at 12.5 Hz) of uncertainty removed vs the unconditional baseline.
- k=8: ΔNLL = -66.79 nats/frame ≈ -0.130 nats/dim ≈ -0.187 bits/dim, which is ~96 bits/frame (~1.20 kbps at 12.5 Hz).

### 3.2 Direction vs Magnitude Decomposition

| Horizon k | Direction cos | Log-mag R² |
|-----------|---------------|------------|
| 1 | 0.611 | -0.595 |
| 2 | 0.392 | -4.814 |
| 4 | 0.107 | -33.84 |
| 8 | 0.038 | -56.26 |

**Key finding:** Direction is predictable (cos > 0) at short horizons, decaying toward random (cos → 0) by k=4-8. Magnitude is *not* predictable—R² is negative at all horizons, indicating the model performs worse than simply predicting the mean log-magnitude.

**Interpretation detail:** These direction/magnitude metrics are computed from the model’s conditional mean (mixture-averaged μ). If magnitude is broad/multimodal, the mean can have systematically wrong magnitude even when the NLL improves.

**Interpretation:** The *type* of transition (which direction in latent space) is structured and predictable from context. The *intensity* of transition (how far to move) is stochastic and unpredictable.

**Caveat:** Both “direction cos” and “log-mag R²” are computed from the model’s *conditional mean* Δx̂ (mixture-averaged μ). If p(Δx|context) is multimodal or sign-symmetric, the conditional mean can collapse toward 0 (hurting cosine and magnitude metrics) even while the conditional density improves substantially (ΔNLL << 0). In other words, these mean-based decompositions may understate the amount of *distributional* predictability present.

### 3.3 Rollout Stability

| Horizon k | Rollout gap (NLL) |
|-----------|-------------------|
| 1 | — |
| 2 | — |
| 4 | ∞ |
| 8 | 5.69 × 10^10 |

**Key finding:** Rollout-context likelihood becomes numerically extreme at larger k. This is consistent with “off-manifold context” failure: once x̂ drifts, the conditional head can become overconfident in the wrong region (large ||μ|| relative to σ), producing enormous NLL. In practice, “∞” here is likely float overflow rather than a meaningful information-theoretic infinity.

**Diagnostic detail:** This rollout metric still scores the *true* Δx_t under p(Δx_t | context), but the context is built from generated x̂. It measures context corruption sensitivity (teacher-forced vs rollout-context), not “k-step forecasting accuracy.”

**Missing results for k=1,2:** In the raw table for this run, rollout values are blank for k=1 and k=2. That typically means the rollout diagnostic did not run or returned no usable utterances for those horizons; this should be re-run/verified before making strong claims about rollout stability at short k.

### 3.4 Train/Eval Discrepancy (Anomaly)

| Horizon k | Train ΔNLL | Eval ΔNLL |
|-----------|------------|-----------|
| 1 | +2028.2 | -185.4 |
| 2 | +11856.8 | -131.1 |
| 4 | -83.3 | -84.8 |
| 8 | +74.1 | -66.8 |

For k=1 and k=2, training ΔNLL is large and *positive* (worse than baseline) while evaluation ΔNLL is negative (better than baseline). This is atypical—normally training performance is equal to or better than evaluation.

**Possible explanations:**
1. Instrumentation/aggregation bug: `train_*` may not be computed via the same evaluation path as `eval_*` (e.g., mixing “training loss” and “eval NLL”, or mismatched (context, Δx) pairs).
2. Data mismatch in the train-eval iterator (row filtering/ordering) leading to misalignment on the train pass only.
3. Numerical instability that triggers only on the train evaluation subset.

Until this is resolved, treat the eval numbers as the reliable signal for “structure exists across held-out speakers”, and treat `train_dnll` as suspect.

One additional observation: the unconditional baseline NLL differs between train and eval (e.g., 1260.34 vs 1260.89 at k=1). This is expected because it is the *same* fitted baseline p(Δx) being evaluated on two different distributions; it does not imply the baseline was refit on eval.

---

## 4. Interpretation

### 4.1 Phase 0 vs Phase 1 Reconciliation

| Approach | Result | Conclusion |
|----------|--------|------------|
| Phase 0: Clustering with coarse keys | Variance ratio ≈ 1.0 | No structure found |
| Phase 1: Learned probabilistic predictor | ΔNLL = -185 to -67 | Structure exists |

The discrepancy is explained by the limitations of Phase 0's methodology:

1. **Key design:** Mean-pooling and PCA-VQ are information-destroying compressions that may not preserve predictive equivalence classes. A learned predictor can discover the relevant features.

2. **Metric choice:** Variance ratio (equivalent to R² on the mean prediction) misses multimodal structure. If the true conditional p(Δx_t | context) is multimodal, the conditional mean can be a poor predictor even when the distribution is highly structured. NLL captures distributional structure that variance-based metrics miss.

3. **Model capacity:** Clustering with 64-256 codes may underfit. A neural network has more capacity to capture complex conditional dependencies.

But importantly: Phase 1 shows structure exists with full-context conditioning; Phase 0 specifically tested whether structure survives aggressive coarsening into a small key (a requirement for lookup). Phase 1 does not “resurrect lookup” unless we also learn an AR-equivalence key.

### 4.2 Implications for AR Audio Modeling

**Finding 1: Predictable structure exists but is missed by naive methods.**

The Mimi latent space contains meaningful predictive structure at the 80ms frame rate. This structure is:
- Learnable by a probabilistic model
- Not captured by simple clustering/quantization of context
- Present across multiple prediction horizons (80ms to 640ms)

**Finding 2: Direction is predictable, magnitude is not.**

This decomposition suggests a natural factorization:
- **Predictable component (direction):** What *type* of transition will occur—e.g., phoneme boundary, pitch change, silence onset
- **Unpredictable component (magnitude):** How *intense* the transition is—e.g., speaking rate, emphasis, exact timing

This aligns with the hypothesis that audio dynamics have discrete "modes" (types of transitions) with continuous variation (intensity/timing).

**Finding 3: Rollout instability is catastrophic.**

Even with meaningful single-step predictability, autoregressive generation fails. This explains why CALM requires multiple stabilization techniques (noise injection, short-context transformers, consistency modeling). The representation is not *AR-friendly*—small errors compound into divergence.

### 4.3 Support for Proposed Research Direction

These results support the RSSM-style dual-latent factorization proposed in RESEARCH_DISCUSSION.md:

| Proposed component | Supported by |
|--------------------|--------------|
| z_dyn (predictable state) | Direction is predictable (cos = 0.61 at k=1) |
| z_rec (innovation/residual) | Magnitude is unpredictable (R² < 0) |
| KL budget on z_rec | Magnitude variance should be "paid for" as innovation |
| Rollout reconstruction loss | Rollout instability confirms need for explicit long-horizon training |

The direction/magnitude decomposition maps directly onto the state/innovation split:
- z_dyn should capture *which direction* the trajectory will move (predictable from context)
- z_rec should capture *how far* it moves (sampled from a learned prior)

---

## 5. Limitations and Caveats

### 5.1 Single Encoder

Results are specific to Mimi. Other audio encoders (EnCodec, SoundStream, DAC) may show different patterns. Cross-encoder comparison is needed to determine whether these findings generalize.

### 5.2 Single Dataset

Results are on LibriSpeech (read English speech). Structure may differ for:
- Spontaneous speech
- Non-English languages
- Music
- General audio

### 5.3 Train/Eval Discrepancy

The anomalous train ΔNLL values (positive when eval is negative) require investigation before these results can be considered fully validated.

### 5.4 Predictor Architecture

Results depend on the specific model architecture used. A different architecture might find more or less structure.

One practical next tweak is to improve robustness of the likelihood head under off-manifold contexts (heavier-tailed components, stronger σ regularization, or explicit context normalization), which may also reduce rollout blow-ups.

---

## 6. Next Steps

Based on these results, the recommended next steps are:

### 6.1 Immediate (Validation)

1. **Investigate train/eval discrepancy** for k=1,2
2. **Cross-encoder comparison:** Run same experiment on EnCodec/DAC latents

### 6.2 Short-term (Architecture Development)

3. **Implement Option I:** RSSM-style factorization of Mimi latents into z_dyn + z_rec
4. **Ablation grid:** Sweep dim(z_dyn) × KL_budget for z_rec
5. **Evaluate:** Is z_dyn easier to model AR than raw Mimi latents?

### 6.3 Medium-term (Integration)

6. **Engram integration:** Test if lookup memory helps for z_dyn
7. **End-to-end evaluation:** Compare against CALM baselines

---

## 7. Summary

**Research question:** Does Mimi's latent space contain predictable structure that Phase 0 clustering missed?

**Answer:** Yes. A learned probabilistic predictor achieves ΔNLL of -185 to -67 across prediction horizons of 80ms to 640ms. The predictable structure is primarily in the *direction* of latent transitions (cos similarity up to 0.61), not the *magnitude* (R² < 0 at all horizons).

**Implication:** The Phase 0 failure reflects limitations of the clustering methodology, not absence of structure. However, rollout instability remains catastrophic, motivating the proposed dual-latent factorization to separate predictable state from unpredictable innovation.

---

## Appendix A: Raw Results

```
horizon_k,slice,train_nll,train_nll_baseline,train_dnll,eval_nll,eval_nll_baseline,eval_dnll,eval_direction_cos,eval_logmag_r2,rollout_gap_nll,rollout_gap_dnll
1,all,3288.57,1260.34,2028.23,1075.45,1260.89,-185.44,0.6114,-0.5949,,
2,all,13117.18,1260.34,11856.85,1129.82,1260.89,-131.07,0.3924,-4.8140,,
4,all,1177.05,1260.34,-83.28,1176.04,1260.89,-84.85,0.1068,-33.8358,inf,inf
8,all,1332.29,1258.14,74.15,1191.87,1258.67,-66.79,0.0384,-56.2588,56921627075.45,56921627075.45
```

## Appendix B: Experimental Configuration

These values correspond to the `configs/phase1.yaml` defaults unless overridden.

- **Encoder:** Mimi (kyutai/mimi)
- **Frame rate:** 12.5 Hz
- **Latent dim:** 512
- **Context window:** W=8 frames, flattened (4096-dim)
- **Horizons:** k ∈ {1, 2, 4, 8} (context staleness)
- **Conditional model:** MDN, diagonal Gaussian mixture, K=8
- **Backbone:** 2-layer MLP, width 1024, GELU
- **Optimizer:** AdamW
- **Training:** 25,000 steps, batch size 256
- **LR / WD:** 1e-3 / 1e-4
- **Grad clip:** 1.0
