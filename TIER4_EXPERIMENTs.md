# State-Conditioned Directional Structure in VAE Latent Dynamics

## Research Question

Does the temporal evolution of continuous VAE latents exhibit directional structure that is *state-dependent* — i.e., do latents in specific regions of the space evolve along a small number of preferred directions, even though no such structure exists globally?

## Motivation

Prior experiment: VQ over global deltas (z_{t+1} - z_t) showed ~50° average cosine angle to nearest centroid in 32d, approximately random. This refutes the hypothesis of *global* directional preferences. However, the global analysis may wash out *local* structure — different regions of latent space may have distinct, low-rank transition dynamics that cancel when aggregated.

If state-conditioned structure exists with small codebooks, it provides a tractable inductive bias for AR models: rather than predicting unconstrained continuous vectors, the model selects among a small set of plausible directions per state and predicts a residual. The value of this approach is directly proportional to how small the per-state codebook can be while capturing the dynamics.

## Dataset

LibriSpeech train-clean-100 (~100 hours). Encode all utterances through the VAE encoder to obtain latent sequences. Each utterance produces a sequence of latent vectors z_1, z_2, ..., z_T where z_t ∈ ℝ^32.

Record the total number of latent frames N across the dataset (report this).

## Preprocessing

### Delta Computation

For each consecutive pair in a latent sequence:

```
δ_t = z_{t+1} - z_t
```

Discard deltas that span utterance boundaries.

### Normalization

Compute two versions of the delta set:

1. **Raw deltas**: δ_t as-is (preserves magnitude information)
2. **Unit deltas**: δ̂_t = δ_t / ||δ_t|| (isolates directional information)

All directional analyses (cosine similarity, angular measurements) use unit deltas. Report the distribution of ||δ_t|| separately — if magnitude is highly variable, raw deltas will conflate magnitude clustering with directional clustering.

Filter out near-zero deltas (||δ_t|| < ε, suggest ε = 1e-4 times median norm) as their directions are noise-dominated.

---

## Experiment 1: Establish the Global Baseline

**Purpose**: Reproduce and properly characterize the global (unconditional) directional structure. This is the null model against which all subsequent experiments are compared.

### Procedure

1. Collect all unit deltas δ̂_t from the full dataset.
2. Run k-means on unit deltas for K_global ∈ {16, 32, 64, 128, 256, 512, 1024}.
3. For each K_global, compute:
   - **Mean angular distance** to nearest centroid: θ̄ = mean(arccos(|δ̂_t · c_nearest|))
   - **Median angular distance**
   - **Standard deviation of angular distance**
   - Full histogram of angular distances (bin width = 5°)

### Null Model

For each K_global, repeat with random unit vectors sampled uniformly on S^31 (use the standard method: sample from N(0,I) and normalize). Use the same number of samples as the real dataset. Compute the same statistics. This gives the expected angular distance under no structure.

### Deliverable

Table: K_global | θ̄_data | θ̄_random | θ̄_data - θ̄_random | median_data | median_random

Plot: Histogram overlay (data vs random) for each K_global.

**Interpretation**: If the gap θ̄_random - θ̄_data is small (< 5°) even at K_global = 1024, global directional structure is genuinely absent. If the gap is moderate (5-15°), there is weak global structure that the prior experiment may have underestimated.

---

## Experiment 2: State Quantization (C1)

**Purpose**: Partition latent space into regions for conditional analysis.

### Procedure

1. Collect all latent vectors z_t (not deltas) from the full dataset.
2. Run k-means for K_1 ∈ {64, 128, 256, 512, 1024, 2048}.
3. Assign each z_t to its nearest centroid: s_t = argmin_j ||z_t - c_j||.
4. For each K_1, report:
   - Cluster size distribution (histogram + min/max/mean/median counts per cluster)
   - Quantization MSE: mean(||z_t - c_{s_t}||²)
   - Percentage of clusters with fewer than 50 assigned deltas (these will have unreliable downstream statistics)

### Important

Do NOT use unit-normalized latents for state clustering — use the raw latent vectors. The states partition the actual latent space. Only deltas get normalized for directional analysis.

---

## Experiment 3: State-Conditioned Directional Analysis (Core Experiment)

**Purpose**: Test whether directional structure in deltas is state-dependent.

### Procedure

For each K_1 from Experiment 2:

1. For each state index j ∈ {0, ..., K_1 - 1}:
   a. Collect all unit deltas δ̂_t where s_t = j (i.e., z_t was assigned to state j).
   b. If the cluster has fewer than 50 deltas, skip it.
   c. Run k-means on these unit deltas for K_2 ∈ {4, 8, 16, 32, 64}.
   d. Compute mean angular distance θ̄_j to nearest delta centroid.

2. Aggregate across states:
   - **Weighted mean**: θ̄_cond = Σ_j (n_j / N) · θ̄_j where n_j is the count for state j
   - **Unweighted mean**: mean(θ̄_j) across states
   - Distribution of θ̄_j across states (some states may be highly structured, others not)

### Null Model (Critical)

For each (K_1, K_2) pair, construct the null by:

1. **Shuffled null**: Take all unit deltas, randomly shuffle their state assignments, then run the same per-state VQ and compute the same statistics. This preserves the global delta distribution but breaks the state-delta association. Repeat 5 times and average.

2. **Random null**: For each state cluster of size n_j, generate n_j random unit vectors on S^31, run k-means with K_2 centroids, compute angular distances. This gives the pure geometric baseline.

### Deliverable

Primary result table:

| K_1 | K_2 | θ̄_cond (data) | θ̄_cond (shuffled) | θ̄_cond (random) | Δ_structure = θ̄_shuffled - θ̄_data |
|-----|-----|----------------|--------------------|--------------------|--------------------------------------|

The key metric is **Δ_structure**: the improvement from state conditioning over unconditional. If Δ_structure is large (> 10°) at small K_2 (4-16), there is actionable state-dependent directional structure. If Δ_structure is small at all K_2, the dynamics are not state-dependent in a low-rank directional sense.

Secondary plot: Heatmap of θ̄_cond(K_1, K_2) with the shuffled null subtracted, showing where structure concentrates.

Histogram: Distribution of per-state θ̄_j for the best-performing (K_1, K_2) pair, annotated with the shuffled null mean. This reveals whether structure is uniform across states or concentrated in a few.

---

## Experiment 4: Temporal Context — Bigram Conditioning

**Purpose**: Test whether transition structure depends on recent trajectory, not just instantaneous state.

### Procedure

1. Using the best K_1 from Experiment 3, form bigram state indices: b_t = (s_{t-1}, s_t).
2. For each bigram with sufficient data (≥ 50 samples), run k-means on the associated unit deltas with K_2 from the best configuration in Experiment 3.
3. Compute the same angular distance statistics.
4. Compare θ̄_bigram against θ̄_unigram (Experiment 3) and the shuffled null.

### Deliverable

Table: θ̄_bigram | θ̄_unigram | θ̄_shuffled | θ̄_random

Report what fraction of bigrams have sufficient data. If most bigrams are too sparse, this approach doesn't scale and unigram conditioning is the practical ceiling.

**Interpretation**: If θ̄_bigram << θ̄_unigram, trajectory context matters and an AR model should condition on recent discrete history (directly analogous to Engram's N-gram structure). If θ̄_bigram ≈ θ̄_unigram, the instantaneous state captures the relevant conditioning.

---

## Experiment 5: Perceptual Validation

**Purpose**: Ground the quantitative analysis in perceptual quality. Angular distances are meaningless if they don't predict audible differences.

### Procedure

1. Take the best (K_1, K_2) configuration from Experiment 3.
2. For each latent z_t in a held-out set of ~20 utterances:
   a. Compute the ground-truth delta δ_t.
   b. Compute the state assignment s_t and find the nearest delta centroid ĉ for that state.
   c. Construct a "quantized trajectory" by replacing each delta with its nearest centroid (scaled by the original magnitude): z̃_{t+1} = z_t + ||δ_t|| · ĉ_{nearest}
3. Decode both the original and quantized trajectories through the VAE decoder.
4. Compute: PESQ, STOI, mel spectrogram L1, and (if available) a perceptual metric between original and reconstructed audio.
5. Also decode a "random direction" trajectory: z̃_{t+1} = z_t + ||δ_t|| · r̂ where r̂ is a random unit vector. This is the lower bound.

### Deliverable

Table: Condition | PESQ | STOI | Mel-L1

- Ground truth (encode → decode, no delta replacement)
- State-conditioned VQ deltas (nearest centroid direction, original magnitude)
- Global VQ deltas (nearest centroid from Experiment 1)
- Random direction (original magnitude)

**Interpretation**: If state-conditioned VQ deltas produce near-ground-truth audio, the discrete approximation is perceptually sufficient and the AR model only needs to select among centroids. If quality degrades substantially, K_2 is too small or the structure isn't tight enough.

---

## Experiment 6: Scaling Diagnostic

**Purpose**: Determine whether the state-conditioned structure is genuine low-rank dynamics or just a consequence of fine-grained partitioning.

### Procedure

Plot θ̄_cond as a function of K_1 × K_2 (total number of (state, delta) entries), with the shuffled null overlaid.

### Interpretation

- **If the data curve separates from the null early and flattens**: Genuine structure. A small total codebook captures most of the predictable dynamics.
- **If the data curve tracks the null but shifted down, improving only as total entries grow**: No special structure — you're just getting better nearest-neighbor coverage with more entries. This is the "memorization without generalization" regime.
- **Crossover point**: The K_1 × K_2 value where adding more entries gives diminishing returns tells you the effective dimensionality of the transition dynamics.

---

## Implementation Notes

### k-means on Unit Vectors

Standard k-means with Euclidean distance on unit vectors is equivalent to minimizing angular distance (since ||â - b̂||² = 2 - 2cos(θ)). Use sklearn's KMeans or faiss. No need for spherical k-means — Euclidean k-means on normalized vectors gives the same assignments. However, after fitting, re-normalize the centroids to unit vectors before computing angular distances.

### Computational Budget

The most expensive step is Experiment 3: K_1 values × K_1 clusters × K_2 values k-means runs. With K_1 = 2048 and 5 values of K_2, that's ~10,000 k-means runs on small subsets. Each should be fast (small n per cluster), but parallelize across clusters.

### Random Seed

Use 3 random seeds for all k-means runs. Report mean ± std of angular distances across seeds. If variance across seeds is comparable to Δ_structure, the result is not robust.

### Storage

For K_1 = 2048, you need to store per-cluster delta sets. With ~2M total frames and 32d, this is manageable in memory. Pre-compute all state assignments for each K_1, then iterate.

---

## Decision Criteria

After running all experiments, the path forward depends on the results:

**Strong positive** (Δ_structure > 10° at K_2 ≤ 16, perceptual validation passes): State-conditioned directional lookup is viable as an AR inductive bias. Design a model that predicts discrete (state, delta_direction) pairs and continuous magnitude/residual.

**Weak positive** (Δ_structure = 5-10° at K_2 ≤ 16, or strong only at K_2 ≥ 64): Some structure exists but may not be sufficient to meaningfully constrain the AR model. Consider whether the marginal benefit over unconstrained prediction justifies the architectural complexity.

**Negative** (Δ_structure < 5° at all scales, or only appears with large K_1 × K_2): State-dependent directional structure is insufficient. The latent dynamics are either too high-entropy locally or the relevant structure lives at longer temporal scales. Pivot to: (a) diffusion/flow-based next-step prediction, (b) longer-context conditioning (Experiment 4 results guide this), or (c) re-examining the VAE itself — the bottleneck may be that the learned manifold doesn't have the transition structure you need.

**Surprising finding — bigram >> unigram** (Experiment 4): The dynamics are trajectory-dependent, not state-dependent. This directly motivates the Engram-style N-gram lookup over quantized state sequences, and the research direction shifts to learning transition tables conditioned on short discrete histories.