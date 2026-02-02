# Summary

Phase 0 is mostly structurally aligned with `SPEC.md` (speaker-based splits, correct context indexing with unit tests, required conditioning methods present), but there are several **spec-breaking gaps** that can invalidate conclusions: confound slices currently **refit clustering per slice**, key reporting fields (notably **excluded mass** and **per-speaker CI outputs**) are missing, the **decision criteria logic** is not faithful to the spec (train-only checks, random baseline only a warning, `abs()` on degradation), and parts of the metrics pipeline have subtle sample-set and stability issues (entropy reduction denominator/sign, potential feature/target misalignment risk for PCA conditions).

# What changed

- No files changed.

# Rationale

You asked for a rigorous audit of `src/phase0/` and `scripts/` against `SPEC.md`, specifically around invariants, conditioning, metrics, confounds, and decision criteria. This report calls out where the implementation matches the spec and where it deviates in ways that could cause false “pass/fail” outcomes or misleading variance-ratio/entropy results.

# Risks / edge cases

## 1) Critical invariants (SPEC “must-haves”)

### 1.1 No data leakage (PCA/VQ/norm stats fit on train speakers only)

**Mostly OK**, but with a spec-adjacent concern:

- Speaker splits are correctly constructed and used:
  - `scripts/01_make_speaker_splits.py` + `src/phase0/data/splits.py` create speaker-based train/eval splits.
  - `scripts/03_build_phase0_dataset.py` sets `split` by speaker ID.
  - `src/phase0/analysis/run_phase0.py` fits PCA/k-means/quartiles and normalization stats using only `frames[split=="train"]`.

**Deviation from SPEC intent (VAE config sourcing):**
- SPEC requires VAE sample rate / hop / frame rate / latent dim to come from the checkpoint, not from “assumptions”.
- `scripts/02_infer_latents.py` reads these from the Mimi model, but then **asserts they match** `configs/phase0.yaml` (which hardcodes 24kHz, 12.5Hz, 512). That’s a sanity check, but the pipeline is still *config-driven*, not “checkpoint-as-source-of-truth”.
  - Risk: swapping checkpoints requires manual config edits; if assertions are removed later, you could silently misalign temporal semantics.

### 1.2 Consistent sample sets (numerator/denominator computed on same filtered samples)

**Variance ratio (VR) is correct**:
- `src/phase0/metrics/variance_ratio.py` explicitly filters once (effective clusters) and computes `SSE_total` and `SSE_within` on the same filtered sample set.

**But there are two important consistency risks elsewhere:**

1) **Potential feature/target misalignment risk for PCA conditions (Condition 2/3)**  
   In `src/phase0/analysis/run_phase0.py`, features for mean and flat are collected via *separate passes*:
   - `train_feat_mean, train_deltas, ... = collect_features_and_deltas(..., mode="mean")`
   - `train_feat_flat, _, _, _ = collect_features_and_deltas(..., mode="flat")`
   Then PCA conditions use `train_feat_flat` but still use `train_deltas` from the *mean* pass.
   - If either pass drops/keeps frames differently (exception paths, missing utterances, etc.), cluster IDs won’t correspond to the deltas being scored.
   - This is spec-critical because it can silently corrupt metrics while still producing “reasonable” numbers.

2) **Entropy reduction can violate “same sample set” due to small per-cluster counts**
   - `src/phase0/metrics/entropy.py` drops clusters with `<2` samples inside `compute_per_cluster_entropy`, which changes the effective sample set for conditional entropy while unconditional entropy uses all filtered samples.
   - Effective clusters are defined by train counts ≥100, but eval can still have low counts for some “effective” clusters, so this can occur on eval.

### 1.3 Correct indexing (context windows end at `t-L`, length `W`)

**Good / matches SPEC exactly**:
- `src/phase0/features/context.py` implements:
  - `end = t - lag`
  - `context = x[end-W : end]`
- `tests/test_indexing.py` provides hand-constructed off-by-one coverage (explicitly requested by SPEC).

### 1.4 VAE alignment (frame rate = 12.5 Hz, latent dim = 512)

**Operationally aligned (Mimi is 12.5Hz/512)**, but with the “checkpoint-vs-config source-of-truth” caveat above:
- `mimi_autoencoder.py` documents and exposes the 12.5Hz/512 properties.
- `scripts/02_infer_latents.py` asserts these equal config values.

---

## 2) Conditioning schemes (3 required)

The three required schemes are implemented via `configs/phase0.yaml` and `src/phase0/analysis/run_phase0.py`:

- Condition 1: mean-pool → kmeans K=64 (`mean_pool_vq`, `k=64`)
- Condition 2: flatten → PCA(8) → kmeans K=256 (`pca_vq`, `pca_dim=8`, `k=256`)
- Condition 3: flatten → PCA(8) → quartile binning (`pca_quartile`, `pca_dim=8`)

**Deviations / missing aspects:**

- **Assignment distances (“confidence proxy”) are computed but not used or reported**:
  - `assign_clusters()` returns distances, but `run_phase0.py` discards them (no confidence curve, no “keep top-q%” analysis).
  - SPEC explicitly requests distance/confidence analysis as a “mandatory extra diagnostic”.

- **Quartile bin “discard <100 samples” is not implemented as an explicit artifact step**:
  - You *do* filter bins through `effective_clusters` for metrics, which is equivalent for VR, but you don’t report excluded mass or bin stats in the Phase 0 outputs.

- `scripts/04_fit_conditioning.py` does not match SPEC’s CLI shape:
  - SPEC expects `04_fit_conditioning.py --cond ...` to fit a single condition; current script fits all at once (not fatal, but it diverges from spec’d workflow).

---

## 3) Metrics implementation

### 3.1 Variance ratio

**Correct formula and sample-set handling:**
- `variance_ratio = SSE_within / SSE_total` computed on the same filtered set in `src/phase0/metrics/variance_ratio.py`.

**Missing spec-required reporting:**
- SPEC requires excluded-mass reporting and effective-cluster accounting.
- `src/phase0/clustering/vq.py` already has `compute_cluster_stats()` (including `excluded_mass`), but `src/phase0/analysis/run_phase0.py` does not record:
  - excluded mass (train/eval)
  - cluster size histograms (train)
  - fraction excluded threshold check (>10% warning/fail)

### 3.2 Diagonal Gaussian entropy reduction

**Implemented, but with two spec-relevant issues:**
- Formula denominator differs from SPEC:
  - SPEC: `(H - H_cond) / H`
  - Code: `(H - H_cond) / abs(H)` in `src/phase0/metrics/entropy.py`
  - This changes meaning when `H` is negative (possible after normalization/filtering) and is not what SPEC states.
- Potential sample-set mismatch for conditional entropy as noted in 1.2.

### 3.3 Per-speaker statistics with CI

**Partially implemented but not surfaced:**
- `src/phase0/metrics/speaker_stats.py` computes per-speaker metrics and a 95% CI for variance ratio in `aggregate_speaker_metrics()`.
- `src/phase0/analysis/run_phase0.py` **does not store CI bounds** (it only stores mean/std).
- `scripts/06_make_report.py` explicitly notes per-speaker detail isn’t available (“we don’t have per-speaker data here”).

### 3.4 Cross-speaker degradation

Implemented as relative degradation using **speaker-mean** VR:
- `compute_cross_speaker_degradation()` returns `(eval_mean - train_mean)/train_mean`.

**Decision logic bug risk (see below):**
- `src/phase0/analysis/report.py` uses `abs(degradation) > max` which can incorrectly fail if eval is *better* (negative degradation with large magnitude).

---

## 4) Confound checks

Required confounds from the prompt:

- High-energy frames only:
  - `scripts/03_build_phase0_dataset.py` computes global median energy on train speakers only and sets `is_high_energy`.
  - `src/phase0/analysis/run_phase0.py` slices by `is_high_energy`.

- Utterance-medial frames only (`pos_frac in [0.17, 0.83]`):
  - Implemented in `apply_slice_filter()`.

- Random cluster baseline (permuted IDs):
  - Implemented in `run_condition()` via `permute_cluster_ids()`.

**Major spec deviation: confound slices currently refit the clustering model per slice**
- In `src/phase0/analysis/run_phase0.py`, for each `slice_name`, you:
  1) filter frames to that slice
  2) collect features from that subset
  3) fit PCA/kmeans/bins on that subset

SPEC’s intent for confounds is “hold the conditioning/clustering fixed, then measure metrics on subsets” so you can say structure *persists* under confound removal rather than “structure exists when you refit clusters on that confound”. Refitting can hide failures (the clusterer adapts to the slice), undermining the gatekeeping purpose.

**Missing slice requested by SPEC’s implementation plan:**
- SPEC asks to compare medial to initial/final; implementation only has medial (no explicit initial/final slices).

**Random baseline computed only on train**
- `run_condition()` only computes `random_baseline_variance_ratio` for train IDs, not eval.

---

## 5) Decision criteria (gatekeeping)

Decision logic in `src/phase0/analysis/report.py` deviates in several important ways:

- **Uses train VR only** for the primary threshold check:
  - SPEC’s gate is about reusable structure and cross-speaker transfer; the natural “gate” should prioritize eval/held-out speakers (or require both train and eval < threshold).
- **Confound robustness checks use train-only VR**, not eval.
- **Random baseline ≈ 1.0 is only a warning**, not a fail condition:
  - SPEC acceptance tests say if you see ~0.9 systematically you likely have a bug/filter artifact; this should be treated as a hard failure or at least block “pass”.
- **Uses `abs(degradation)`**, which can fail if eval improves substantially (negative degradation with large magnitude).

---

## Additional implementation bugs / inconsistencies worth flagging

- Model artifact overwrites in `src/phase0/analysis/run_phase0.py`:
  - PCA/bin models are saved as `{condition_name}_pca.pkl` and `{condition_name}_bins.pkl` with **no lag/slice in filename**, and are overwritten repeatedly across the nested loops.
- Pipeline duplication / inconsistency between scripts:
  - `scripts/04_fit_conditioning.py` computes cluster stats (excluded mass, cluster sizes) but `scripts/05_eval_metrics.py` calls `run_full_analysis()` which refits models and does not consume `04`’s artifacts.
  - `scripts/06_make_report.py` plots cluster sizes from `04`’s stats JSON, but the metrics come from `05`/`run_phase0.py`, so plots and metrics can describe **different fitted models** (especially given the per-slice refitting issue).
- Missing “mandatory extra diagnostics” from SPEC:
  - No confidence-vs-variance-ratio curve is generated or saved.
  - No per-cluster variance vs size plot.
  - No 2D PCA scatter of cluster-mean Δx plot.

---

## Validation note

- All tests pass (`pytest`): indexing, reproducibility, variance ratio (31 tests).
- These tests do **not** cover the biggest spec risks above (slice refitting, excluded-mass reporting, decision criteria correctness, entropy formula/sample-set alignment, mean/flat collection alignment).

# Follow-ups

Highest-impact fixes to become SPEC-compliant (in order):

1) In `src/phase0/analysis/run_phase0.py`, fit conditioning (PCA/VQ/bins) **once per lag on all train frames**, assign cluster IDs for all frames, then compute metrics on `all/high_energy/medial` by **filtering the sample set**, not refitting clusters per slice.
2) Make feature/delta collection single-pass so Condition 2/3 features cannot misalign with deltas (store and reuse `(utterance_id, t)` frame keys).
3) Record and report excluded-mass (train/eval) and effective cluster counts per condition/lag, and fail/flag if excluded mass >10% as SPEC describes.
4) Surface per-speaker distributions and CI bounds in `metrics.json`/`tables.csv`, and generate the per-speaker plot from actual per-speaker values.
5) Fix entropy reduction to match SPEC formula exactly and ensure unconditional/conditional entropies are computed on identical sample sets.
6) Fix `make_decision()` to:
   - gate primarily on eval (held-out speakers) or require both train+eval < 0.6
   - treat random baseline deviations as failure (or at least block pass)
   - remove `abs()` on degradation (only penalize worse eval)
7) Add the SPEC-mandated plots and confidence diagnostics using `src/phase0/analysis/plots.py` (functions exist but aren’t wired into the report).