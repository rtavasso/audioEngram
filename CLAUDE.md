# Project Configuration

## Running Python Files

Always run Python files using uv:

```bash
uv run python <script.py>
```

This ensures the correct virtual environment (.venv) and dependencies are used.

## Dependencies

Main dependencies are managed through pyproject.toml. PyTorch with CUDA 12.4 support is installed automatically via uv.

### Moshi Submodule

Moshi (the Mimi VAE) is included as a git submodule. After cloning, initialize it:

```bash
git submodule update --init --recursive
```

Then install in editable mode:

```bash
uv pip install -e ./moshi/moshi --python .venv/bin/python
```

## GPU Support

The project is configured to install PyTorch with CUDA 12.4 support. Verify with:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## Experiment Tracking

All experiments use a lightweight tracking system (`src/experiment.py`). Every run automatically records:

- **`<out_dir>/config.yaml`** — snapshot of the config used
- **`<out_dir>/run_info.json`** — provenance (git commit, Python/torch version, CUDA, timestamps, duration, key metrics)
- **`<out_dir>/run.log`** — full execution log
- **`outputs/manifest.jsonl`** — global append-only log (two lines per run: started + completed; crashed runs only have started)

### Viewing Results

```bash
uv run python scripts/collect_results.py                          # all completed runs
uv run python scripts/collect_results.py --latest                 # most recent per experiment
uv run python scripts/collect_results.py --experiment exp1_vmf    # filter by experiment
uv run python scripts/collect_results.py --no-save                # print only, skip file output
```

Outputs `outputs/results_table.csv` and `outputs/results_table.md` (markdown for paper drafts). Crashed runs are reported separately.

## Running Experiments

### Phase 0: Data Preparation (run first)

All Tier 1 experiments require Phase 0 data. Run these in order:

```bash
bash scripts/00_download_librispeech.sh ./data
uv run python scripts/01_make_speaker_splits.py --config configs/phase0.yaml
uv run python scripts/02_infer_latents.py --config configs/phase0.yaml
uv run python scripts/03_build_phase0_dataset.py --config configs/phase0.yaml
```

### Tier 1: Run All (convenience wrapper)

Runs Exp 1 → Exp 2 → Exp 3 sequentially with a shared run-id:

```bash
uv run python scripts/tier1_run_all.py --run-id my_run
uv run python scripts/tier1_run_all.py --run-id my_run --resume   # skip already-completed experiments
```

Options: `--exp1-config`, `--exp2-config`, `--exp3-config`, `--exp2-horizon-k` (default 1).

### Tier 1: Individual Experiments

**Exp 1 — vMF+LogNormal factorization** (baseline dynamics model):

```bash
uv run python scripts/tier1_exp1_vmf.py --config configs/tier1_exp1_vmf.yaml --run-id my_exp1
```

Options: `--slice {all,high_energy,utterance_medial}`. Trains models for each horizon k in config. Outputs checkpoints, `metrics.json`, `tables.csv`.

**Exp 1B — vMF rollout fine-tuning** (requires Exp 1 checkpoint):

```bash
uv run python scripts/tier1_exp1b_vmf_rollout_train.py \
    --config configs/tier1_exp1b_vmf_rollout.yaml \
    --checkpoint outputs/tier1/exp1_vmf/<RUN_ID>/checkpoints/vmf_k1_final.pt
```

Options: `--k`, `--max-steps`, `--lr` for quick overrides. Outputs `vmf_rollout_final.pt`, `injection_diag.json`, `summary.json`.

**Exp 2 — Injection diagnostic** (requires Exp 1 or 1B checkpoint):

```bash
uv run python scripts/tier1_exp2_injection.py \
    --config configs/tier1_exp2_injection.yaml \
    --checkpoint outputs/tier1/exp1_vmf/<RUN_ID>/checkpoints/vmf_k1_final.pt
```

Tests 4 modes: A_teacher (full teacher forcing), B_periodic, C_one_shot, D_rollout (pure free-running). Outputs `metrics.json`, `per_step.csv`, plots.

**Exp 3 — Representation comparison** (Mimi vs EnCodec):

```bash
uv run python scripts/tier1_exp3_rep_compare.py --config configs/tier1_exp3_rep_compare.yaml
```

Options: `--force` to recompute latents. Extracts latents for each representation, trains Phase 1, runs 4-mode injection diagnostic (A_teacher, B_periodic, C_one_shot, D_rollout) on each. Outputs `summary.csv`, `injection_diag.json`.

**Exp 3B — Synthetic trajectory audio** (perceptual validation):

```bash
uv run python scripts/tier1_exp3b_synthetic_audio.py --config configs/tier1_exp3b_synthetic.yaml
```

Generates WAV files for modes: GT, A_stationary, B_slow (scaled magnitude), C_wrong_dir (random direction). For listening tests.

**Exp 4 — PCA linear readout** (dimensionality reduction baseline):

```bash
uv run python scripts/tier1_exp4_linear_readout.py --config configs/tier1_exp4_linear_readout.yaml
```

Fits IncrementalPCA, projects latents to lower dims, trains Phase 1 on projected vs raw, runs 4-mode injection diagnostic (A_teacher, B_periodic, C_one_shot, D_rollout) on each. Outputs `summary.csv`, `injection_diag.json`.

### Stage 2: Individual Experiments

**Exp 5 — beta-VAE with AR-friendliness** (primary Stage 2 experiment):

```bash
uv run python scripts/tier2_exp5_betavae.py --config configs/tier2_exp5_betavae.yaml
uv run python scripts/tier2_exp5_betavae.py --config configs/tier2_exp5_betavae.yaml --variant baseline_betavae
uv run python scripts/tier2_exp5_betavae.py --config configs/tier2_exp5_betavae.yaml --max-steps 100  # quick test
```

Trains 4 VAE variants (baseline, +smooth, +pred, +both), extracts latents, runs Phase 1 diagnostic battery + injection diagnostic on each. Options: `--variant` to run single variant, `--max-steps` / `--phase1-max-steps` for quick runs. Outputs per-variant `phase1/`, `recon_metrics.json`, `summary.csv`.

**Exp 6 — Score-based manifold correction** (requires Phase 1 checkpoint):

```bash
uv run python scripts/tier2_exp6_score_correction.py \
    --config configs/tier2_exp6_score_correction.yaml \
    --checkpoint outputs/tier1/exp1_vmf/<RUN_ID>/checkpoints/vmf_k1_final.pt
```

Trains score model on Mimi latents, sweeps Langevin correction hyperparams (n_steps x step_size x sigma), runs modified injection diagnostic comparing D_rollout vs D_corrected. Outputs `sweep_results.json`, `summary.csv`.

**Exp 7 — Perceptual rollout evaluation** (requires Phase 1 checkpoint):

```bash
uv run python scripts/tier2_exp7_perceptual_rollout.py \
    --config configs/tier2_exp7_perceptual_rollout.yaml \
    --checkpoint outputs/tier1/exp1_vmf/<RUN_ID>/checkpoints/vmf_k1_final.pt
```

Decodes actual MDN rollout trajectories through Mimi decoder at various lengths (1,2,4,8,16 steps). Splices GT prefix → rollout → GT suffix → 48kHz WAV. Outputs WAVs + `manifest.json`.

**Exp 9 — Pocket-TTS Mimi VAE-GAN** (needs Phase 0 audio + splits):

```bash
uv run python scripts/tier2_exp9_pocket_mimi_vae.py --config configs/tier2_exp9_pocket_mimi_vae.yaml
uv run python scripts/tier2_exp9_pocket_mimi_vae.py --config configs/tier2_exp9_pocket_mimi_vae.yaml --max-steps 10  # quick test
```

Trains pocket-tts Mimi with 32D VAE bottleneck using full VAE-GAN loss (reconstruction + KL + adversarial + feature matching + WavLM distillation). Uses pretrained pocket-tts encoder (frozen) + decoder (fine-tuned), MS-STFT discriminator. After training, extracts 32D latents, runs Phase 1 diagnostic battery + injection diagnostic + reconstruction metrics. Options: `--max-steps`, `--phase1-max-steps`. Outputs `summary.csv`, `recon_metrics.json`, phase1 diagnostics.

### Stage 3: Individual Experiments

**Exp 11 — Consistency model for rollout stability** (requires Exp 5 latents + Exp 8 codebook + Exp 10 checkpoint):

```bash
uv run python scripts/tier3_exp11_consistency_rollout.py \
    --config configs/tier3_exp11_consistency_rollout.yaml
```

Trains a one-step Tweedie denoiser on 32D β-VAE latents, then sweeps sigma values across 6 rollout conditions (Direction AR argmax/categorical ± Tweedie, MDN ± Tweedie) at k=1,2,4,8,16. Options: `--score-checkpoint` to skip score training, `--score-max-steps` for quick runs, `--skip-audio`, `--skip-mdn`, `--sigma` for single sigma. Outputs `rollout_metrics.json`, `summary.csv`, `sigma_sweep.json`, `best_sigmas.json`, `rollout_audio/` WAVs.

### Experiment Dependency Graph

```
Phase 0 (00 → 01 → 02 → 03)
  ├── Exp 1 (vmf baseline)
  │     ├── Exp 1B (rollout fine-tuning, needs Exp 1 checkpoint)
  │     ├── Exp 2 (injection diagnostic, needs Exp 1 or 1B checkpoint)
  │     ├── Exp 6 (score correction, needs Exp 1 checkpoint + Phase 0 latents)
  │     └── Exp 7 (perceptual rollout, needs Exp 1 checkpoint + Mimi decoder)
  ├── Exp 3 (representation comparison, needs splits from 01)
  ├── Exp 3B (synthetic audio, needs latents from 02)
  ├── Exp 4 (PCA readout)
  └── Exp 5 (beta-VAE, needs Phase 0 audio + splits)
        ├── Exp 8 (direction codebook, needs Exp 5 latents)
        │     └── Exp 10 (direction AR, needs Exp 8 codebook + Exp 5 latents)
        │           └── Exp 11 (consistency rollout, needs Exp 10 + Exp 5 MDN + Exp 8 codebook)
        └── Exp 9 (pocket Mimi VAE)
```
