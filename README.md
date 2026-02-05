# Phase 0: Audio Latent Structure Analysis

This project implements Phase 0 of the Conditional Memory for Audio Latents research (see `SPEC.md`). It determines whether audio latents from the Mimi VAE exhibit reusable local structure suitable for conditional memory mechanisms.

## Overview

Phase 0 is a **gatekeeper experiment**. Before building any models, we test whether the requisite structure exists in audio latent dynamics. If coarse context clusters predict next-step dynamics significantly better than chance, we proceed to modeling. If not, the project terminates with a valid negative result.

## Requirements

- Python 3.10+
- CUDA-capable GPU (for VAE inference)
- ~50GB disk space (for LibriSpeech + latents)

## Installation

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/YOUR_USERNAME/engramAudio.git
cd engramAudio

# Or if already cloned, initialize submodules
git submodule update --init --recursive

# Create virtual environment and install dependencies
# PyTorch with CUDA 12.4 support is installed automatically
uv sync

# Install moshi in editable mode
uv pip install -e ./moshi/moshi --python .venv/bin/python

# Verify GPU support
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Running the Experiments

Execute the scripts in order. Each script depends on outputs from previous steps.

### Step 1: Download LibriSpeech

Download the train-clean-100 subset (~6GB):

```bash
./scripts/00_download_librispeech.sh ./data
```

This creates `./data/LibriSpeech/train-clean-100/` with 251 speakers.

### Step 2: Create Speaker Splits

Split speakers into train (200) and eval (51) sets:

```bash
uv run python scripts/01_make_speaker_splits.py
```

**Output:** `outputs/phase0/splits/`
- `train_speakers.txt`, `eval_speakers.txt`
- `train_utt_ids.txt`, `eval_utt_ids.txt`

### Step 3: Extract Latents

Run Mimi VAE encoder on all utterances (GPU required, ~2-3 hours):

```bash
uv run python scripts/02_infer_latents.py --device cuda
```

**Output:**
- `outputs/phase0/latents.zarr/` - Continuous latents (512-dim @ 12.5 Hz)
- `outputs/phase0/latents_index.parquet` - Utterance metadata

### Step 4: Build Frame Index

Create the Phase 0 frame dataset with energy and position features:

```bash
uv run python scripts/03_build_phase0_dataset.py
```

**Output:** `outputs/phase0/phase0_frames.parquet`

### Step 5: Fit Conditioning Models

Fit PCA, k-means, and quartile bins on training data:

```bash
uv run python scripts/04_fit_conditioning.py
```

**Output:** `outputs/phase0/conditioning/`
- Fitted models (`.pkl` files)
- Normalization stats
- Cluster statistics

### Step 6: Evaluate Metrics

Compute variance ratio, entropy reduction, and cross-speaker metrics:

```bash
uv run python scripts/05_eval_metrics.py
```

**Output:**
- `outputs/phase0/metrics.json` - Full results
- `outputs/phase0/tables.csv` - Summary table

### Step 7: Generate Report

Create the final decision report:

```bash
uv run python scripts/06_make_report.py
```

**Output:**
- `outputs/phase0/report.txt` - Human-readable report with PASS/FAIL decision
- `outputs/phase0/decision.json` - Machine-readable decision
- `outputs/phase0/plots/` - Diagnostic visualizations

## Quick Start (All Steps)

Run the complete pipeline:

```bash
# Download data
./scripts/00_download_librispeech.sh ./data

# Run analysis pipeline
uv run python scripts/01_make_speaker_splits.py
uv run python scripts/02_infer_latents.py --device cuda
uv run python scripts/03_build_phase0_dataset.py
uv run python scripts/04_fit_conditioning.py
uv run python scripts/05_eval_metrics.py
uv run python scripts/06_make_report.py

# View results
cat outputs/phase0/report.txt
```

## Decision Criteria

From `SPEC.md`, Phase 0 **passes** if ALL conditions are met:

| Criterion | Threshold | Description |
|-----------|-----------|-------------|
| Variance Ratio | < 0.6 | Clusters explain >40% of dynamics variance |
| Cross-Speaker Degradation | < 20% | Structure generalizes to held-out speakers |
| Confound Robustness | Structure persists | High-energy and utterance-medial slices |
| Random Baseline | ≈ 1.0 | Confirms no filtering artifacts |

Phase 0 **fails** if ANY criterion is violated.

## Conditioning Schemes Tested

1. **mean_vq64**: Mean-pool context → VQ with K=64 clusters
2. **pca8_vq256**: Flatten context → PCA(8) → VQ with K=256 clusters
3. **pca8_quartile**: Flatten context → PCA(8) → Quartile binning

## Configuration

Edit `configs/phase0.yaml` to modify:

- Data paths and subset
- Context window size and lags
- Clustering parameters
- Decision thresholds

## Running Tests

```bash
uv run python -m pytest tests/ -v
```

Tests cover:
- Context window indexing correctness
- Variance ratio computation
- Reproducibility with fixed seeds

## Project Structure

```
engramAudio/
├── configs/
│   └── phase0.yaml          # Configuration
├── scripts/
│   ├── 00_download_librispeech.sh
│   ├── 01_make_speaker_splits.py
│   ├── 02_infer_latents.py
│   ├── 03_build_phase0_dataset.py
│   ├── 04_fit_conditioning.py
│   ├── 05_eval_metrics.py
│   └── 06_make_report.py
├── src/phase0/
│   ├── data/                 # Data loading, splits, I/O
│   ├── vae/                  # Latent extraction
│   ├── features/             # Context, normalization, energy
│   ├── clustering/           # VQ, PCA, quartiles, baselines
│   ├── metrics/              # Variance ratio, entropy, speaker stats
│   ├── analysis/             # Orchestration, reporting, plots
│   └── utils/                # Seeding, logging
├── tests/                    # Unit tests
├── outputs/phase0/           # Results (created by scripts)
├── mimi_autoencoder.py       # Mimi VAE wrapper
├── SPEC.md                   # Full specification
└── README.md                 # This file
```

## Interpreting Results

### If Phase 0 Passes

Audio latents exhibit reusable local structure. Proceed to Phase 1: Minimal Engram Modeling.

### If Phase 0 Fails

Audio latent dynamics do not exhibit sufficient reusable structure. This is a valid negative result indicating that:
- The variation in how latents evolve is too continuous, speaker-specific, or context-dependent
- Lookup-based memory (as in Engram) is unsuitable for this domain

## Tier 1 Experiments (Post-Phase0)

These implement the immediate-priority experiments described in `ENGINEER_ONBOARDING.md`.

```bash
# Exp 1: vMF direction + LogNormal magnitude (drop-in model)
uv run python scripts/tier1_exp1_vmf.py --config configs/tier1_exp1_vmf.yaml

# Exp 2: injection diagnostic (point this at a checkpoint produced by Exp 1)
uv run python scripts/tier1_exp2_injection.py --config configs/tier1_exp2_injection.yaml --checkpoint <path/to/*.pt>

# Exp 3: representation comparison (Mimi 12.5 vs EnCodec if available)
uv run python scripts/tier1_exp3_rep_compare.py --config configs/tier1_exp3_rep_compare.yaml
```

Speed tips:
- Tier1 scripts log `Device: ...` — if it says `cpu`, Exp1/Exp3 can take a very long time.
- For quicker iterations, cut `train.max_steps`, `context.horizons_k`, and `train.max_eval_samples`, and set `rollout.enabled: false`.
- You can also try increasing `train.num_workers` to overlap data prep (the Phase 1 IterableDataset is sharded across workers).
- Optional training knobs (in `train:`) include `compile`, `compile_mode`, `amp`, and `amp_dtype`.
- For Exp2, `diag.max_train_samples` caps baseline-fitting cost; reducing `diag.n_eval_utterances` / `diag.segments_per_utt` also helps.
- Continuous attention mechanisms (as in CALM) remain the appropriate approach

## References

- **SPEC.md**: Full technical specification
- **Engram Paper**: Conditional memory for language models
- **CALM Paper**: Continuous audio latent modeling
- **Mimi**: Neural audio codec from Kyutai
