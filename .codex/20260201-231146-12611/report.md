# Summary

A 12GB Colab OOM is still plausible at 100K+ samples primarily due to (a) storing multiple large `[N, W*D]` feature matrices in RAM at once, (b) list→`np.array(...)` materialization peaks during feature collection, (c) PCA fit/transform creating full-size centered copies, and (d) the current `assign_clusters()` broadcast (`[batch, K, D]`) which is *especially* large for `mean_pool_vq` (D=512) even with batching.

Assumptions for estimates below (from `configs/phase0.yaml`): latent dim `D=512`, context window `W=8` so flattened dim `W*D=4096`, and float32 unless noted.

---

# What changed

- No files changed (analysis only).

---

# Rationale

## 1) Large array allocations (persistent)

### Flat context features are inherently large
- `scripts/04_fit_conditioning.py:148-151` collects `feat_flat` as a full dense array.
- `src/phase0/analysis/run_phase0.py:162-168` stores both `features_mean` and `features_flat` (and deltas) in RAM.

Memory for **N=100,000**:
- `features_flat` float32 `[N, 4096]`: **~1562.5 MiB (1.53 GiB)**.
- `features_mean` float32 `[N, 512]`: **~195.3 MiB**.
- `deltas` float32 `[N, 512]`: **~195.3 MiB**.

So per split (train *or* eval) in `run_phase0.collect_all_features_and_deltas()` you’re at roughly:
- **~1.53 GiB + 0.19 GiB + 0.19 GiB ≈ 1.91 GiB** (plus Python/pandas overhead).

### `run_full_analysis()` holds train and eval arrays simultaneously
- `src/phase0/analysis/run_phase0.py:379-388` collects and retains both `train_data` and `eval_data` concurrently.

If train and eval are each ~100K samples, just these persistent arrays are roughly:
- **~1.91 GiB (train) + ~1.91 GiB (eval) ≈ 3.82 GiB**, before normalization, PCA, clustering, slicing copies, etc.

### Normalization produces additional full-size arrays
- `src/phase0/analysis/run_phase0.py:395-396` calls `normalize_delta()` for both splits.
- `src/phase0/features/normalization.py:102` does `(delta - mu) / sigma` which typically allocates at least one full intermediate plus the output.

For **each** split:
- `train_deltas_norm` float32 `[N, 512]`: **~195.3 MiB** (same for eval).
So add **~390.6 MiB** persistent across both splits (and often more transient during the operation).

## 2) Data collection loops that accumulate in memory (and spike at materialization)

### Lists of per-sample arrays + final stacked arrays => peak ~2× (or worse)
- `scripts/04_fit_conditioning.py:40-85` accumulates `features_list`, `deltas_list`, then converts to `np.array(...)`.
- `src/phase0/analysis/run_phase0.py:119-168` does the same but for **both** mean and flat features, plus deltas.

Key peak behavior:
- Each sample’s `feat_flat` is created as a standalone numpy array (`flatten()`), stored in a Python list, and then copied again into a contiguous `[N, 4096]` array at `np.array(...)`.
- At the moment of conversion, you can transiently hold:
  - the **list of N arrays** (already ~1.53 GiB of raw float32 data for flat),
  - plus the new contiguous **output array** (~1.53 GiB),
  - plus overhead for 100K Python objects.
- That makes **~3.0 GiB+ peak** *just for flat features*, inside `collect_all_features_and_deltas()`.

### `get_context_mean()` returns float64 by default (extra memory in the lists)
- `src/phase0/features/context.py:43-44` uses `context.mean(axis=0)`; numpy’s default mean accumulator/output for float32 input is typically float64.
- That means the per-sample arrays appended at:
  - `scripts/04_fit_conditioning.py:68` and
  - `src/phase0/analysis/run_phase0.py:146`
  are likely float64 until the final `dtype=np.float32` conversion.

For **N=100K**, mean features in the list can be roughly:
- float64 `[N, 512]`: **~390.6 MiB** in list element payloads alone (before stacking to float32).

### `get_context_flat()` always copies per sample
- `src/phase0/features/context.py:76-77` uses `context.flatten()`, which **always** returns a copy.
- This guarantees per-sample allocations of size `4096 * 4 = 16 KiB`.
- For 100K samples, that is ~1.53 GiB of allocations just to populate the list, *before* the final stacked array is built.

## 3) PCA fitting on high-dimensional data

### PCA fit on `[N, 4096]` can be multi-gigabyte even before SVD work buffers
- `src/phase0/clustering/pca.py:53-55` calls `sklearn.decomposition.PCA.fit(features)`.

Risk factors:
- sklearn PCA centers data; centering frequently creates a full-size copy of X (size `N*D`).
- sklearn often operates in float64 internally depending on validation paths and solver decisions, which can double memory.

For **N=100K, D=4096**:
- X float32: **~1.53 GiB**
- X float64: **~3.05 GiB**
- A centered copy of X is another **~1.53–3.05 GiB**.
So PCA fit can easily push transient usage into **~3–6+ GiB** territory *just for the PCA input and centering*, excluding other arrays already resident (train/eval features/deltas).

### PCA projection (`project_pca`) creates a full centered intermediate
- `src/phase0/clustering/pca.py:80-82`:
  - `centered = features - model.mean` allocates a full `[N, D]` array.
  - `np.dot(centered, model.components.T)` then computes the low-d output.

If `features` is `features_flat` `[100K, 4096]` float32:
- `centered` is another **~1.53 GiB** transient allocation per call.

In `run_full_analysis`, PCA-based conditions call `project_pca()` for:
- train (`src/phase0/analysis/run_phase0.py:251-253` via `assign_condition_clusters()`)
- eval (same path)
So you get **repeated 1.53 GiB transient spikes** during cluster assignment for PCA-based conditions.

## 4) Broadcast operations creating huge intermediate arrays

### The biggest immediate OOM risk: `assign_clusters()` diff tensor
- `src/phase0/clustering/vq.py:96-103`:
  - `diff = batch[:, None, :] - model.centroids[None, :, :]` creates `[batch, K, D]`.
  - `sq_dists = np.sum(diff**2, axis=2)` also materializes `diff**2` (another `[batch, K, D]` temporary).

For the **mean_pool_vq** condition (`K=64`, `D=512`) with default `batch_size=10000`:
- `diff` size = `10000*64*512` float32 = **~1250 MiB (1.22 GiB)**.
- `diff**2` can be another **~1.22 GiB** transient.
- Peak inside the loop can be **~2.4+ GiB** *just for distance computation*, on top of already-resident train/eval feature matrices.

This is a direct OOM trigger when combined with:
- resident `features_flat` (~1.53 GiB per split),
- resident `features_mean`/`deltas`/normalized deltas,
- PCA centering copies (for PCA conditions),
- pandas frames and Python object overhead.

### Other broadcast/copy sites that add meaningful pressure
- `src/phase0/features/normalization.py:102` `(delta - mu) / sigma` (full-size intermediates for `[N, 512]`).
- `src/phase0/metrics/variance_ratio.py:28-31` `diff = deltas - global_mean` and `diff**2` for `[N, 512]` (extra hundreds of MiB transient for large slices).

### Slice filtering creates full copies (not views)
- `src/phase0/analysis/run_phase0.py:285-287` boolean indexing:
  - `deltas_slice = deltas_norm[slice_mask]` is a copy.
- For the `"all"` slice, that copy is essentially the whole array:
  - another **~195 MiB** for train + **~195 MiB** for eval per condition per lag (transient, but repeated).

### `apply_slice_mask()` builds a large dict repeatedly
- `src/phase0/analysis/run_phase0.py:73-75` builds `key_to_idx = {k: i ...}` over all `frame_keys`.
- Called at `src/phase0/analysis/run_phase0.py:452-453` for each slice, and inside the condition loop.
- For 100K keys, this is a non-trivial extra memory/time cost (tens of MB depending on Python’s dict overhead), but typically secondary compared to the multi-GB numpy allocations above.

---

# Risks / edge cases

- If “100K+ samples” means **per split per lag** (not total), memory scales linearly and the flat feature matrix alone can exceed Colab capacity when combined with PCA and clustering.
- Longer utterances increase per-utterance `x = latent_store.get_latents(utt_id)` size (`src/phase0/analysis/run_phase0.py:130` / `scripts/04_fit_conditioning.py:51`), which adds additional transient pressure.
- sklearn implementation details (PCA/KMeans dtype casting, solver choice, copies during centering) can change memory peaks substantially; worst-case peaks are plausibly several GiB higher than the “raw array size” math.
- The **mean_vq64** path is uniquely risky because it combines:
  - high D=512, and
  - a broadcast distance computation (`[batch, K, D]`) that is huge at `batch_size=10000`.

---

# Follow-ups

- Reduce/replace the `[batch, K, D]` broadcast in `src/phase0/clustering/vq.py:101-103`:
  - either lower `batch_size` specifically for high-D conditions (e.g., 500–2000), or
  - compute squared distances via `||x||^2 + ||c||^2 - 2 x·c` to avoid 3D intermediates entirely.
- Avoid list-of-arrays accumulation for flat features (`scripts/04_fit_conditioning.py:40-85`, `src/phase0/analysis/run_phase0.py:119-168`):
  - preallocate if N is known, or
  - write features incrementally to a memmap/zarr/npz and stream into PCA/KMeans.
- Make `get_context_mean()` stay float32 to avoid float64 per-sample arrays (`src/phase0/features/context.py:44`), e.g. by specifying an explicit dtype for the reduction.
- Replace `flatten()` with a view when possible (`src/phase0/features/context.py:77`) to reduce per-sample allocations (or build flat features into a preallocated output buffer).
- Use incremental / streaming dimensionality reduction for PCA-based conditions (`src/phase0/clustering/pca.py:53-55`), e.g. `IncrementalPCA` with chunking, to avoid holding/centering the full `[N, 4096]` matrix.
- In `run_full_analysis`, consider not holding both train and eval full feature matrices simultaneously (`src/phase0/analysis/run_phase0.py:379-388`): collect/transform/assign in phases to lower the resident set size.
- Reduce slice-copy overhead by computing metrics without materializing full slice copies where feasible (`src/phase0/analysis/run_phase0.py:285-287`), and avoid rebuilding `key_to_idx` for every slice (`src/phase0/analysis/run_phase0.py:73-75`).