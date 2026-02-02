# Background

# Conditional Memory for Continuous Audio Latent Dynamics: Technical Synopsis

## Executive Summary

This document provides the conceptual foundation for a research project investigating whether lookup-based memory mechanisms—successful in large language models—can improve autoregressive generation of continuous audio latents. The project tests a specific structural hypothesis about audio representations and is designed to fail fast if that hypothesis is false.

---

## Part I: The Problem We Are Trying to Solve

### The State of Audio Generation

Modern audio generation systems face a fundamental tension between two representation strategies.

**Discrete token models** represent audio as sequences of codes from a learned codebook, typically using Residual Vector Quantization (RVQ). This makes audio "look like text" and allows direct application of language modeling techniques. The approach works well and powers systems like MusicGen, AudioLM, and Mimi. However, quantization is inherently lossy. Achieving high fidelity requires deep codebook hierarchies (8-32 levels), which increases computational cost and introduces complex dependencies between codebook levels. Modeling these dependencies requires architectural compromises like delay patterns or auxiliary RQ-Transformer heads.

**Continuous latent models** represent audio in the latent space of a variational autoencoder, avoiding quantization entirely. The CALM paper demonstrates this approach can match or exceed discrete models in quality while being more computationally efficient. However, continuous latents are harder to model autoregressively. The model must predict a continuous vector at each step, and small errors accumulate over time, causing generated audio to drift off the training manifold. CALM addresses this through several stabilization techniques: injecting noise into the context during training, using a short-context transformer to preserve local detail, and replacing diffusion sampling with consistency models.

These stabilization techniques work, but they are fundamentally *workarounds* for an architectural limitation. The transformer backbone must learn everything from scratch at every step: both the global structure of the audio and the local dynamics of what typically happens next. This is computationally expensive and potentially unstable.

### What Language Models Teach Us

Recent work on large language models reveals an interesting structural insight. A significant fraction of modeling effort is spent reconstructing *static, local, stereotyped patterns*—things like named entities ("Alexander the Great"), formulaic phrases ("by the way"), and common collocations. These patterns are highly predictable from local context and appear repeatedly across different documents and contexts.

The Engram paper introduces *conditional memory* as a solution: a lookup table indexed by recent context that retrieves pre-computed embeddings for common patterns. This is implemented as a hash-based N-gram embedding that maps discrete token sequences to vectors in O(1) time. The key finding is that adding this memory module to a Mixture-of-Experts transformer improves performance across diverse benchmarks—not just knowledge retrieval tasks where memory is obviously useful, but also reasoning, code generation, and mathematics.

The mechanism appears to be that memory *offloads local pattern reconstruction* from the transformer backbone, freeing up depth and capacity for higher-level reasoning. Empirical analysis shows that Engram-augmented models reach "prediction-ready" representations earlier in the network, as if the model has gained effective depth.

### The Core Question

This project asks whether the same principle applies to continuous audio latents:

> **Do short-horizon audio latent dynamics exhibit reusable local structure that can be captured by a lookup mechanism, and if so, does exploiting this structure improve autoregressive modeling?**

This is not a question about whether we can beat benchmarks by adding parameters. It is a question about the *nature* of audio representations—specifically, whether there exist recurring patterns in how audio latents evolve over short time horizons that are stable enough across speakers and contexts to be worth storing and retrieving rather than recomputing.

---

## Part II: Why This Might Work (and Why It Might Not)

### The Optimistic Case

Audio, particularly speech, has strong local regularities. Phoneme transitions follow articulatory constraints. Coarticulation effects create predictable dynamics between adjacent sounds. Prosodic patterns impose structure at slightly longer timescales. These regularities are the reason N-gram models work at all for speech recognition, and why phone-level language models can generate intelligible (if robotic) speech.

In the latent space of a well-trained VAE, these regularities should manifest as *clusters of similar trajectories*. When the recent context looks like "mid-vowel transitioning toward a stop consonant," the distribution of likely next-frame dynamics should be relatively concentrated, regardless of which specific vowel or which specific speaker. If this clustering is strong enough, a lookup table indexed by coarsened context could retrieve a useful prior on next-step dynamics.

This prior would not replace the continuous prediction—it would *augment* it. The transformer backbone would still model the full conditional distribution, but it would receive a hint about the likely region of latent space. This could reduce the entropy the backbone must model, improve stability during autoregressive rollout, and potentially allow the backbone to focus more capacity on longer-range structure.

### The Pessimistic Case

The analogy to language may not hold. Language has several properties that make lookup particularly effective:

**Discrete, symbolic structure.** Words and phrases are categorical objects. The same phrase "Alexander the Great" is identical every time it appears, making it trivially indexable. Audio latents are continuous vectors with no natural discretization.

**Reusable patterns.** In language, the same syntactic constructions and collocations appear across millions of documents by millions of authors. In audio, speaker-specific characteristics (vocal tract shape, speaking rate, accent) might dominate, making "patterns" highly individual rather than universal.

**Stable keys.** In language, the context used to index memory (recent tokens) is discrete and doesn't drift. In audio generation, the context is *generated* latents, which may drift from the training distribution. A corrupted key retrieves an irrelevant or harmful prior.

If audio latent dynamics are dominated by speaker-specific variation, or if the "patterns" are too continuous and context-dependent to discretize usefully, lookup will fail. This is not a theoretical concern—it is the *most likely outcome*.

### What We Must Demonstrate

For the project to proceed, we must establish that audio latents exhibit what we call *Type 2 structure*:

**Type 1 structure** means the future is predictable from the past. All successful autoregressive models exploit Type 1 structure; it is necessary but not sufficient for our purposes.

**Type 2 structure** means the *same prediction rule* applies across many different contexts. The mapping from "coarse description of recent dynamics" to "likely next dynamics" is stable across different speakers, different utterances, and different positions in an utterance.

Language has extreme Type 2 structure because syntax and vocabulary are shared across speakers. Audio might have much weaker Type 2 structure because so much variation is individual.

---

## Part III: The Experimental Design

### Phase 0: Testing for Structure (The Gatekeeper)

Before building any models, we must determine whether the requisite structure exists. This is a pure data analysis task that can be completed in a few days with minimal compute.

**The setup:** Take a trained CALM speech VAE and extract latent sequences for LibriSpeech. For each frame, record the recent context (previous W frames) and the next-step dynamics (Δx = x_t - x_{t-1}).

**The test:** Cluster the contexts using deliberately coarse representations (aggressive PCA, small VQ codebooks). Then measure how much the clustering reduces variance in the dynamics.

If coarse context clusters predict next-step dynamics significantly better than chance, Type 2 structure exists. If they don't, lookup is fundamentally unsuited to this domain.

**Critical constraints:**

1. **Coarse conditioning.** The context representation must be information-poor. If we condition on the full high-dimensional context, prediction will improve trivially due to continuity. We need to show that *coarsened* context still predicts dynamics.

2. **Cross-speaker transfer.** Clusters must be learned on one set of speakers and evaluated on held-out speakers. If structure only appears when clusters are speaker-specific, it is not reusable in the way Engram requires.

3. **Robustness checks.** Structure must persist when we exclude silence frames, utterance boundaries, and other potential confounds.

**Decision criteria:** We proceed to modeling only if:
- Within-cluster variance ratio falls below 0.6 (clusters explain >40% of dynamics variance)
- Cross-speaker degradation is less than 15-20%
- Structure persists across confound checks

If these criteria are not met, the project terminates with a negative structural result. This is a valid and publishable finding about the nature of audio latent spaces.

### Phase 1: Minimal Engram Modeling

If Phase 0 passes, we implement a minimal version of the Engram mechanism for audio:

1. Coarsely quantize the recent context to obtain a discrete key
2. Look up a learned embedding vector
3. Apply a scalar gate conditioned on the backbone's hidden state
4. Add the gated embedding to the short-context transformer's output

This is much simpler than the full Engram architecture (no multi-head hashing, no convolution, no multi-branch integration). If this minimal version shows no benefit, additional complexity will not save the approach.

**Comparisons must be iso-compute.** We cannot simply add Engram on top of CALM and declare victory if metrics improve. We must show that at fixed computational cost, a smaller backbone plus memory outperforms a larger backbone without memory. Otherwise we have merely demonstrated that more parameters help.

**Metrics span multiple concerns:**
- Teacher-forced likelihood (does memory help modeling?)
- Rollout stability (does memory reduce drift during generation?)
- Downstream quality (WER, speaker similarity on generated speech)

### Phase 2: Noise Injection Ablation

CALM injects noise into the long-context backbone during training to make it robust to inference-time error accumulation. This is effective but somewhat unprincipled—a training-time hack to compensate for an inference-time problem.

If Engram provides a structural solution to the same problem, it should reduce or eliminate the need for noise injection. We test this by training CALM with and without Engram at various noise levels. If CALM+Engram maintains stability without noise while CALM alone fails, that is strong evidence that memory provides genuine robustness rather than just additional capacity.

**Diagnostic: gate dynamics.** We track how the Engram gate magnitude evolves during autoregressive rollout. Two patterns are informative:
- If the gate decays over time, the model has learned to trust memory less as rollout proceeds (because keys become unreliable). This suggests memory acts as a bootstrap prior.
- If the gate stays high while quality degrades, memory is actively harmful—retrieving inappropriate priors based on corrupted keys.

---

## Part IV: Why Phase 0 Is Designed the Way It Is

The Phase 0 specification is unusually detailed because every choice has consequences for the validity of the conclusions.

### Context Representation

We test three conditioning schemes of increasing sophistication:

1. **Mean-pooled context → VQ(K=64):** Extremely coarse. Averages away all temporal structure within the window. If this works, structure is very robust.

2. **Flattened context → PCA(8) → VQ(K=256):** Preserves some temporal structure in the principal components. More informative keys, but still very low-dimensional.

3. **PCA(8) → quartile binning:** Axis-aligned discretization. Tests whether learned (VQ) vs simple (binning) discretization matters.

The progression tests how much information the key needs to be useful. If only the finest conditioning shows structure, keys must be high-fidelity, which undermines the efficiency argument for lookup.

### Target Variable

We predict Δx (velocity) rather than x (position) because position prediction is dominated by trivial continuity (x_t ≈ x_{t-1}). Velocity is the interesting signal—how the latent is *changing*, not where it currently is.

Velocity is normalized globally (not per-speaker) to ensure the analysis asks about absolute dynamics rather than speaker-relative dynamics.

### Temporal Scales

We test context windows ending at different lags before the target (1, 2, and 4 frames back). This simulates the situation at inference time, where the most recent context frames are generated rather than ground truth. If structure disappears with even a small lag, the approach is fragile to the inherent staleness of generated context.

### Sample Size and Statistical Validity

With ~4.5 million frames from LibriSpeech train-clean-100, we have ample data for stable variance estimation. Clusters with fewer than 100 samples are excluded, and we report what fraction of data falls in excluded clusters. If more than 10% of data is excluded, the clustering is too fine.

Crucially, we compute statistics *on the same sample set* for both numerator and denominator of the variance ratio. Filtering introduces subtle biases if not handled carefully.

### Controls

**Random cluster baseline:** We permute cluster assignments (preserving the size distribution) to verify that the variance ratio is actually measuring structure rather than filtering artifacts. This should yield a ratio near 1.0.

**Speaker-level statistics:** We report variance ratio per speaker and examine the distribution. A single pooled number can hide bimodal behavior where structure exists for some speakers but not others.

**Confound slices:** We compute metrics separately on high-energy frames, utterance-medial frames, and other subsets to ensure structure is not an artifact of silence or prosodic boundaries.

---

## Part V: Interpreting Outcomes

### If Phase 0 Fails

The most likely outcome is that Phase 0 fails: variance ratios above 0.65, or severe degradation on held-out speakers, or structure that disappears when confounds are controlled.

This would mean: **Audio latent dynamics do not exhibit reusable short-horizon structure.** The variation in how latents evolve is too continuous, too speaker-specific, or too context-dependent to compress into discrete keys.

This is a meaningful negative result. It tells us that the success of Engram in language relies on language-specific properties (discrete symbols, shared vocabulary, reusable syntax) that do not transfer to audio. Continuous attention, as in CALM's short-context transformer, is the appropriate inductive bias for this domain.

### If Phase 0 Passes But Phase 1 Fails

Structure might exist but not be *exploitable* by the minimal Engram architecture. Possible reasons:
- The backbone is already capturing this structure internally, making memory redundant
- The structure exists but isn't learnable through the simple gating mechanism
- Memory helps under teacher forcing but not during rollout (key corruption dominates)

Diagnosing this requires probing whether backbone representations correlate with cluster IDs, testing memory under teacher forcing only, and examining gate dynamics.

### If Both Phases Succeed

This would mean: **Conditional memory provides genuine benefits for continuous audio latent modeling**, either through improved efficiency (same quality at lower compute) or improved stability (better rollouts without noise injection hacks).

This would be a significant finding. It would suggest that the discrete/continuous divide in audio generation is partially artificial—that what matters is not discreteness per se, but whether the model has an explicit mechanism for reusing local structure. It would open a research direction exploring memory-augmented continuous models more broadly.

---

## Part VI: Relationship to Existing Work

### CALM

Our work builds directly on CALM. We use their VAE architecture, their consistency-based sampling head, and their insight that continuous latents can match discrete tokens in quality. Our contribution is asking whether an additional architectural primitive (memory) can improve on their approach.

The short-context transformer in CALM already addresses local conditioning. Our question is whether *lookup* provides benefits over *reconstruction*—that is, whether storing patterns is better than recomputing them. This is not obvious; attention is very flexible, and lookup is rigid.

### Engram

We adapt Engram's core insight—that conditional memory complements conditional computation—to a domain where it was not designed to work. The technical challenges are different (continuous keys, distribution shift during generation), but the conceptual framework transfers.

Our minimal Engram is much simpler than the full architecture. We deliberately strip away multi-head hashing (less relevant with smaller key spaces), convolution (we already have a context window), and multi-branch integration (we're testing the core idea, not optimizing throughput). If the minimal version fails, the full version would need strong justification.

### Broader Context

This project sits at the intersection of several research threads:
- Memory-augmented neural networks (going back to Neural Turing Machines)
- Sparse mixture models (MoE, product-of-experts)
- Continuous generative models (VAEs, diffusion, flow matching, consistency)
- Audio generation (codec-based and latent-space approaches)

The specific question—whether lookup-based memory helps continuous autoregressive modeling—has not been systematically tested. Audio is a good domain to test it because the structure is rich enough that memory might help but continuous enough that it might not.

---

## Part VII: What the Engineer Needs to Know

### Core Task

Implement Phase 0 exactly as specified. This is a data analysis pipeline, not a modeling task. The goal is to determine whether structure exists before committing to expensive training runs.

### Critical Invariants

1. **No data leakage.** All fitting (PCA, VQ, normalization statistics) uses only training speakers. Evaluation uses transforms only.

2. **Consistent sample sets.** Variance ratio numerator and denominator must be computed on exactly the same samples after filtering.

3. **Correct indexing.** Context windows must end at the right lag before the target. Off-by-one errors here invalidate everything. Write unit tests with hand-constructed data.

4. **VAE configuration.** Frame rate and sample rate must come from the actual checkpoint, not hardcoded assumptions. The entire analysis depends on correct temporal alignment.

### What Success Looks Like

A successful Phase 0 produces a clear table showing variance ratios across conditions, splits, and confound slices, along with visualizations of cluster structure. The decision to proceed or stop should be obvious from the numbers.

A successful implementation is one where:
- The random-cluster control yields ratio ~1.0
- Results are reproducible across runs with the same seed
- All filtering and exclusion is documented and accounted for
- The report makes the decision criteria and outcomes explicit

### Timeline

Phase 0 should take 3-5 days for an experienced engineer:
- Day 1: Data pipeline (LibriSpeech download, VAE inference, frame extraction)
- Day 2: Conditioning and clustering (PCA, VQ fitting, cluster assignment)
- Day 3: Metrics and controls (variance ratio, entropy, random baseline)
- Day 4: Confound analysis and visualization
- Day 5: Report generation and review

The compute requirements are minimal—a few hours of GPU time for VAE inference, then CPU-only analysis.

---

## Conclusion

This project tests whether a successful architectural idea from language models—conditional memory for local pattern retrieval—transfers to continuous audio generation. The hypothesis is plausible but far from certain. The experimental design is structured to fail fast and fail informatively.

If the hypothesis fails, we learn something about the structural differences between language and audio. If it succeeds, we open a new direction for audio generation research. Either outcome advances understanding.

The Phase 0 specification is the critical path. Implement it exactly, verify the invariants, and let the data decide.

# Implementation Plan: Phase 0 (LibriSpeech + CALM VAE Latents + Clustering)

## 0) Tech stack

* Python 3.11
* PyTorch (for VAE inference)
* torchaudio (resampling, loading)
* numpy, scipy
* scikit-learn (PCA, k-means)
* pandas (report tables)
* matplotlib (plots)
* argparse (config)
* zarr or parquet for storage (see below)

## 1) Repo structure

```
audio-engram-phase0/
  README.md
  pyproject.toml
  configs/
    phase0.yaml
  src/
    phase0/
      __init__.py
      data/
        librispeech.py
        splits.py
        io.py
      vae/
        calm_vae.py
        infer_latents.py
      features/
        context.py
        energy.py
        normalization.py
      clustering/
        vq.py
        pca.py
        baselines.py
      metrics/
        variance_ratio.py
        entropy_diag_gauss.py
        speaker_stats.py
      analysis/
        run_phase0.py
        report.py
        plots.py
      utils/
        seed.py
        logging.py
  scripts/
    00_download_librispeech.sh
    01_make_speaker_splits.py
    02_infer_latents.py
    03_build_phase0_dataset.py
    04_fit_conditioning.py
    05_eval_metrics.py
    06_make_report.py
  outputs/
    phase0/
      <run_id>/
        metrics.json
        tables.csv
        plots/
  tests/
    test_indexing.py
    test_variance_ratio.py
    test_reproducibility.py
```

---

## 2) Data artifacts (explicit formats)

### 2.1 Latents store

Store per-utterance latents (and metadata) after VAE inference.

Recommended: **zarr** (fast chunked arrays) or **parquet** if you prefer row-based.

**zarr layout:**

```
latents.zarr/
  <utterance_id>/
    x: float32 [T, D]
    energy: float32 [T]          # optional
    timestamps: float32 [T]      # seconds
    speaker_id: int32
```

Also write an index table:
`latents_index.parquet`

* utterance_id
* speaker_id
* n_frames
* duration_sec
* path_audio

### 2.2 Phase 0 sample table

Don’t store full context windows per frame (too large). Instead store:

* utterance_id, speaker_id, t (frame index)
  and compute contexts on the fly from stored x arrays.

Create:
`phase0_frames.parquet`

* utterance_id
* speaker_id
* t
* pos_frac (t / T)  (for utterance position confound)
* energy (optional)

This keeps Phase 0 memory-light.

---

## 3) Speaker split logic (reproducible)

Script: `01_make_speaker_splits.py`

* parse LibriSpeech metadata → speaker list
* deterministic shuffle with seed
* select 200 speakers train, 51 eval
* save:

  * `splits/train_speakers.txt`
  * `splits/eval_speakers.txt`
  * `splits/train_utt_ids.txt`
  * `splits/eval_utt_ids.txt`

Also store 10% utterance holdout *within train speakers* for kmeans validation:

* `splits/train_utt_ids_train.txt`
* `splits/train_utt_ids_val.txt`

---

## 4) VAE inference

Script: `02_infer_latents.py`
Inputs:

* VAE checkpoint path + expected sample rate + hop settings
* list of utterances
  Steps:

1. load waveform
2. resample to VAE sample rate (only if needed)
3. run encoder → x[T, D]
4. compute energy per frame:

   * simplest: frame RMS of waveform using the same hop/window alignment as latents
   * store energy[T]
5. save to `latents.zarr`

Acceptance checks:

* assert no NaNs
* record D, hop_sec, mean/std of x
* verify frame rate ≈ expected (derive from timestamps)

---

## 5) Build phase0 frame index

Script: `03_build_phase0_dataset.py`
Filter rules:

* duration >= 3 sec
* frames t must satisfy: t >= (W + max_context_lag) and t < T
  Where max_context_lag depends on your “Δt” conditioning-lag variant.
  If you keep your current “context ends at t-Δt” idea:
* for lag L ∈ {1,2,4}, need t- L >= 1 and t-W-L >= 0

Store:

* utterance_id, speaker_id, t, pos_frac, energy, split

Also precompute:

* “is_high_energy” flag using speaker-independent threshold:

  * compute global median energy over train speakers
  * mark frames above median

---

## 6) Conditioning feature computation (three conditions)

All conditioning features must be computed *only from training speakers* for fit.

### 6.1 Common: context extraction

Module: `features/context.py`

Given utterance latents x[T,D], time t, window W, lag L:

* define end = t - L
* context frames are x[end-W : end]  (length W)

Return:

* mean pooled: c_mean[D]
* flattened: c_flat[W*D]

Unit test (`test_indexing.py`):

* hand-constructed x where each frame equals its index → verify correct slices.

### 6.2 Condition 1: Mean-pool + VQ K=64

Script: `04_fit_conditioning.py --cond mean_vq64`

* compute c_mean for all frames from train speakers
* fit kmeans K=64 (kmeans++ init, max_iter=100, n_init=10)
* save centroids
* assign cluster IDs for train and eval frames (nearest centroid)
* record assignment distances (confidence proxy)

### 6.3 Condition 2: Flatten + PCA(8) + VQ K=256

Same script:

* compute c_flat for train frames
* fit PCA(n=8) on train frames
* project train + eval frames → c_pca[8]
* fit kmeans K=256 on train c_pca
* assign IDs + distances

### 6.4 Condition 3: PCA(8) + quantile binning

* compute PCA as above
* for each PC dimension, compute quartile edges on train set
* bin each coordinate to {0,1,2,3} and hash into integer bin_id
* discard bins with <100 samples (but compute excluded mass properly)

Also implement **Condition 0: random clusters**:

* sample cluster IDs with the *same empirical cluster size histogram* as the learned IDs, or simpler:

  * permute learned cluster IDs across samples (preserves histogram exactly)
    This is the best control.

---

## 7) Target construction + normalization

Module: `features/normalization.py`

For each frame:

* Δx = x[t] - x[t-1]
* compute μ_global, σ_global over Δx from training speakers only (over the same subset used in evaluation; easiest: compute on all eligible frames before cluster filtering)
* normalize Δx_norm = (Δx - μ) / σ

Important: handle σ≈0 dims:

* clamp σ_d = max(σ_d, 1e-6)

Store μ, σ as artifacts.

Optional ablations:

* without per-speaker mean centering of x
* with centering

---

## 8) Metric computation

### 8.1 Cluster filtering

Filter clusters with count >= 100 **within each split** (train/eval). But note:

* if you filter differently in train vs eval, comparisons are messy.
  Better:
* define “effective clusters” from train split only
* apply same set to eval (exclude eval samples whose assigned cluster is not effective)
  Report excluded mass in both.

### 8.2 Variance ratio

Module: `metrics/variance_ratio.py`

Given Δx_norm samples and cluster IDs:

* compute within-cluster SSE
* compute total SSE over same included samples S
* ratio = SSE_within / SSE_total

Also compute per-speaker ratios:

* for each speaker, compute ratio using that speaker’s included samples
* report mean/std and CI

### 8.3 Diagonal Gaussian entropy reduction

Module: `metrics/entropy_diag_gauss.py`

* unconditional variances per dim on S
* per-cluster variances per dim
* compute:
  $$
  H \propto \frac12 \sum_d \log \sigma_d^2
  $$

(constants cancel)

* compute relative reduction:
  $$
  (H - H_{\text{cond}}) / H
  $$

### 8.4 Confound slices

Compute all metrics on:

* all frames
* high-energy only
* utterance medial only (pos_frac in [0.17, 0.83]) and compare to initial/final

---

## 9) Diagnostics you requested (plus two that I think are mandatory)

### Required plots (your list + minimal extras)

* histogram of cluster sizes (train)
* per-cluster variance vs size
* 2D PCA scatter of μ_k (cluster mean Δx) with point size ~ cluster count

### Mandatory extra diagnostics

1. **Gatekeeping control:** random cluster baseline results on the same filtered set
2. **Distance/confidence analysis:**

   * plot variance ratio as a function of “keep top-q% most confident assignments”
   * if ratio improves sharply when keeping only confident assignments, you’ve found “structure but only with reliable keys”, which is relevant to later gating.

---

## 10) Run orchestrator + reporting

Script: `analysis/run_phase0.py`

* runs all conditions × lags
* saves a single `metrics.json` and `tables.csv` with rows:

  * condition, lag, split, slice (all/high_energy/medial), ratio, entropy_reduction, excluded_mass, n_eff_clusters, speaker_mean, speaker_std, etc.

Script: `06_make_report.py`

* dumps:

  * decision matrix table
  * the plots
  * a short “success/failure” paragraph using templated text and actual numbers

---

## 11) Acceptance tests (what “done” means)

Phase 0 implementation is correct if:

1. **Indexing test passes** (context slicing exactly as specified)
2. **Random-cluster baseline** yields ratio ≈ 1.0 (within noise) on all slices

   * if you see 0.9 systematically, you have a bug or filtering artifact
3. **Reproducibility:** same seed → identical kmeans centroids and metrics
4. **No leakage:** PCA/VQ fit uses only train speakers; eval uses transform + assign only
5. **Exclusion accounting:** excluded mass reported for both train and eval and uses identical “effective cluster set” derived from train