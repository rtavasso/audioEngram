#!/usr/bin/env python3
"""
Tier 1 - Experiment 4: PCA linear readout baseline.

Tests if dimensionality reduction alone captures predictable structure by:
1. Fitting IncrementalPCA on training Mimi latents for d_out in {32, 64}
2. Projecting all latents to new zarr stores
3. Building frames indices for projected latents
4. Running Phase 1 MDN baseline on projected latents
5. Comparing DNLL with raw 512-dim baseline

Usage:
  uv run python scripts/tier1_exp4_linear_readout.py --config configs/tier1_exp4_linear_readout.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import yaml
import zarr
from sklearn.decomposition import IncrementalPCA

from phase0.data.io import (
    LatentStore,
    save_latents_zarr,
    save_latents_index,
    load_latents_index,
    load_latents_zarr,
)
from phase0.data.splits import load_splits
from phase0.features.context import get_valid_frame_range
from phase0.features.energy import compute_median_energy
from experiment import register_run, finalize_run
from phase0.utils.logging import setup_logging, get_logger
from phase0.utils.seed import set_seed

from phase1.train_eval import _device_from_config, train_and_eval_for_k, write_results, fit_unconditional_baseline


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_injection_diagnostic(
    *,
    phase1_dir: Path,
    frames_index_path: str | Path,
    latents_dir: str | Path,
    latents_index_path: str | Path,
    splits_dir: str | Path,
    window_size: int,
    inj_cfg: dict,
    device,
    logger,
) -> dict | None:
    """Run 4-mode injection diagnostic on the k=1 MDN checkpoint."""
    ckpt_path = phase1_dir / "checkpoints" / "mdn_k1_final.pt"
    if not ckpt_path.exists():
        logger.warning(f"[exp4] No k=1 checkpoint at {ckpt_path}; skipping injection diagnostic")
        return None

    from phase1.checkpoints import load_phase1_checkpoint
    from phase1.injection_diag import run_injection_diagnostic

    model, _ckpt = load_phase1_checkpoint(ckpt_path, device=device)
    baseline = fit_unconditional_baseline(
        frames_index_path=frames_index_path,
        latents_dir=latents_dir,
        window_size=window_size,
        horizon_k=1,
        slice_name="all",
        max_samples=inj_cfg.get("max_train_samples"),
    )
    result = run_injection_diagnostic(
        model=model,
        baseline=baseline,
        latents_dir=latents_dir,
        latents_index_path=latents_index_path,
        splits_dir=splits_dir,
        horizon_k=1,
        window_size=window_size,
        k_steps=int(inj_cfg.get("k_steps", 16)),
        n_eval_utterances=int(inj_cfg.get("n_eval_utterances", 16)),
        segments_per_utt=int(inj_cfg.get("segments_per_utt", 8)),
        max_frames_per_utt=int(inj_cfg.get("max_frames_per_utterance", 2000)),
        seed=42,
        device=device,
        mode_inject_after_steps={
            "A_teacher": None,
            "B_periodic": [int(x) for x in inj_cfg.get("inject_after_steps_periodic", [4, 8, 12])],
            "C_one_shot": [int(x) for x in inj_cfg.get("inject_after_steps_one_shot", [1])],
            "D_rollout": [],
        },
        sample_from_model=False,
    )
    inj_path = phase1_dir / "injection_diag.json"
    with open(inj_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"[exp4] Wrote injection diagnostic: {inj_path}")
    return result


def _fit_pca(
    *,
    store: LatentStore,
    train_utt_ids: list[str],
    n_components: int,
    batch_size: int = 4096,
    logger,
) -> IncrementalPCA:
    """Fit IncrementalPCA on training latents."""
    pca = IncrementalPCA(n_components=n_components)
    buf = []
    n_frames = 0

    for utt_id in train_utt_ids:
        if utt_id not in store:
            continue
        x = store.get_latents(utt_id).astype(np.float32, copy=False)
        buf.append(x)
        n_frames += x.shape[0]

        # Flush when buffer is large enough
        total_rows = sum(b.shape[0] for b in buf)
        if total_rows >= batch_size:
            block = np.concatenate(buf, axis=0)
            pca.partial_fit(block)
            buf = []

    # Flush remainder
    if buf:
        block = np.concatenate(buf, axis=0)
        if block.shape[0] >= n_components:
            pca.partial_fit(block)

    ev = pca.explained_variance_ratio_
    logger.info(
        f"[exp4] PCA d={n_components}: fitted on {n_frames} frames, "
        f"explained_var_ratio={ev.sum():.4f} (top-5: {ev[:5]})"
    )
    return pca


def _project_and_save(
    *,
    store: LatentStore,
    latents_index: pd.DataFrame,
    pca: IncrementalPCA,
    out_zarr_path: Path,
    out_index_path: Path,
    logger,
) -> None:
    """Project all latents through PCA and save to new zarr store."""
    entries = []
    for _, row in latents_index.iterrows():
        utt_id = str(row["utterance_id"])
        if utt_id not in store:
            continue

        x, energy, timestamps, speaker_id = load_latents_zarr(
            utt_id, store.zarr_path
        )
        x_proj = pca.transform(x.astype(np.float32, copy=False)).astype(np.float32)

        save_latents_zarr(
            latents=x_proj,
            energy=energy,
            timestamps=timestamps,
            speaker_id=int(speaker_id),
            utterance_id=utt_id,
            zarr_path=out_zarr_path,
        )
        entries.append({
            "utterance_id": utt_id,
            "speaker_id": int(row["speaker_id"]),
            "n_frames": int(x_proj.shape[0]),
            "duration_sec": float(row["duration_sec"]),
            "audio_path": str(row.get("audio_path", "")),
        })

    save_latents_index(entries, out_index_path)
    logger.info(f"[exp4] Projected {len(entries)} utterances -> {out_zarr_path}")


def _build_frames_index(
    *,
    splits_dir: str | Path,
    latents_index_path: str | Path,
    latents_zarr_path: str | Path,
    window_size: int,
    max_lag: int,
    min_duration_sec: float,
    out_frames_index_path: str | Path,
    logger,
) -> None:
    """Build frames index for projected latents."""
    splits = load_splits(splits_dir)
    train_speaker_set = set(splits.train_speakers)
    eval_speaker_set = set(splits.eval_speakers)

    latents_index = load_latents_index(latents_index_path)
    frames_list = []
    train_energies = []

    for _, row in latents_index.iterrows():
        utt_id = str(row["utterance_id"])
        speaker_id = int(row["speaker_id"])
        n_frames = int(row["n_frames"])
        duration = float(row["duration_sec"])
        if duration < float(min_duration_sec):
            continue

        if speaker_id in train_speaker_set:
            split = "train"
        elif speaker_id in eval_speaker_set:
            split = "eval"
        else:
            continue

        try:
            _x, energy, _ts, _spk = load_latents_zarr(utt_id, latents_zarr_path)
        except Exception as e:
            logger.warning(f"[exp4] Could not load {utt_id}: {e}")
            continue

        if split == "train":
            train_energies.append(energy)

        first_valid, last_valid = get_valid_frame_range(n_frames, int(window_size), int(max_lag))
        for t in range(int(first_valid), int(last_valid)):
            frames_list.append({
                "utterance_id": utt_id,
                "speaker_id": speaker_id,
                "t": int(t),
                "pos_frac": float(t) / float(n_frames),
                "energy": float(energy[t]),
                "split": split,
            })

    frames = pd.DataFrame(frames_list)
    if frames.empty:
        raise RuntimeError("No frames found when building frames index.")

    median_energy = compute_median_energy(train_energies)
    frames["is_high_energy"] = frames["energy"] > float(median_energy)

    from phase0.data.io import save_frames_index
    save_frames_index(frames, out_frames_index_path)


def main() -> int:
    p = argparse.ArgumentParser(description="Tier1 Exp4: PCA linear readout")
    p.add_argument("--config", type=str, default="configs/tier1_exp4_linear_readout.yaml")
    p.add_argument("--run-id", type=str, default=None)
    args = p.parse_args()

    logger = setup_logging(name="tier1-exp4-linear-readout")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_id = args.run_id or _default_run_id()
    out_root = Path(cfg["output"]["out_dir"])
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    run = register_run(
        experiment="exp4_linear_readout", run_id=run_id, config_path=args.config,
        config=cfg, cli_args=sys.argv[1:], out_dir=out_dir, log_name="tier1-exp4-linear-readout",
    )

    set_seed(int(cfg.get("seed", 42)))
    device = _device_from_config(cfg["train"]["device"])

    data_cfg = cfg["data"]
    latents_dir = data_cfg["latents_dir"]
    latents_index_path = data_cfg["latents_index"]
    if latents_index_path is None:
        latents_index_path = str(Path(latents_dir).parent / "latents_index.parquet")
    frames_index_path = data_cfg["frames_index"]
    splits_dir = data_cfg["splits_dir"]
    min_duration_sec = float(data_cfg.get("min_duration_sec", 3.0))

    window_size = int(cfg["context"]["window_size"])
    horizons_k = [int(k) for k in cfg["context"]["horizons_k"]]
    max_lag = int(max(horizons_k))
    pca_dims = [int(d) for d in cfg["pca_dims"]]
    slice_name = str(cfg.get("slice", "all"))

    store = LatentStore(latents_dir)
    latents_index = load_latents_index(latents_index_path)

    # Identify train utterances for PCA fitting
    splits = load_splits(splits_dir)
    train_speaker_set = set(splits.train_speakers)
    train_df = latents_index[latents_index["speaker_id"].isin(train_speaker_set)]
    train_df = train_df[train_df["duration_sec"] >= min_duration_sec]
    train_utt_ids = train_df["utterance_id"].astype(str).tolist()

    logger.info(f"Run id: {run_id}")
    logger.info(f"Device: {device}")
    logger.info(f"PCA dims: {pca_dims}")
    logger.info(f"Train utterances for PCA: {len(train_utt_ids)}")

    # Also run raw 512-dim baseline for comparison
    all_rows = []

    # Raw baseline
    logger.info("[exp4] Running raw 512-dim Phase 1 baseline")
    raw_dir = out_dir / "raw_512"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_results = []
    for k in horizons_k:
        r = train_and_eval_for_k(
            frames_index_path=frames_index_path,
            latents_dir=latents_dir,
            splits_dir=splits_dir,
            latents_index_path=latents_index_path,
            out_dir=raw_dir,
            horizon_k=k,
            window_size=window_size,
            slice_name=slice_name,
            seed=int(cfg["train"]["seed"]),
            device=device,
            n_components=int(cfg["model"]["n_components"]),
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            n_hidden_layers=int(cfg["model"]["n_hidden_layers"]),
            dropout=float(cfg["model"]["dropout"]),
            min_log_sigma=float(cfg["model"]["min_log_sigma"]),
            max_log_sigma=float(cfg["model"]["max_log_sigma"]),
            batch_size=int(cfg["train"]["batch_size"]),
            num_workers=int(cfg["train"]["num_workers"]),
            max_steps=int(cfg["train"]["max_steps"]),
            lr=float(cfg["train"]["lr"]),
            weight_decay=float(cfg["train"]["weight_decay"]),
            grad_clip_norm=float(cfg["train"]["grad_clip_norm"]),
            log_every=int(cfg["train"]["log_every"]),
            eval_every=int(cfg["train"]["eval_every"]),
            save_every=int(cfg["train"]["save_every"]),
            shuffle_buffer=int(cfg["train"]["shuffle_buffer"]),
            max_train_samples=cfg["train"].get("max_train_samples"),
            max_eval_samples=cfg["train"].get("max_eval_samples"),
            rollout_enabled=bool(cfg["rollout"]["enabled"]),
            rollout_n_eval_utterances=int(cfg["rollout"]["n_eval_utterances"]),
            rollout_max_frames_per_utt=int(cfg["rollout"]["max_frames_per_utterance"]),
            rollout_sample_from_mixture=bool(cfg["rollout"]["sample_from_mixture"]),
            compile_model=bool(cfg["train"].get("compile", False)),
            compile_mode=str(cfg["train"].get("compile_mode", "default")),
            amp=bool(cfg["train"].get("amp", False)),
            amp_dtype=str(cfg["train"].get("amp_dtype", "bf16")),
        )
        raw_results.append(r)
        all_rows.append({
            "representation": "raw_512",
            "pca_dim": 512,
            "horizon_k": k,
            "eval_dnll": r.eval.get("dnll"),
            "eval_nll": r.eval.get("nll"),
            "eval_nll_baseline": r.eval.get("nll_baseline"),
            "rollout_gap_nll": r.rollout.get("gap_nll") if r.rollout else None,
        })
    write_results(raw_results, metrics_path=str(raw_dir / "metrics.json"), tables_path=str(raw_dir / "tables.csv"))

    # Injection diagnostic on raw baseline
    inj_cfg = cfg.get("injection", {})
    if cfg.get("run_injection", True):
        _run_injection_diagnostic(
            phase1_dir=raw_dir,
            frames_index_path=frames_index_path,
            latents_dir=latents_dir,
            latents_index_path=latents_index_path,
            splits_dir=splits_dir,
            window_size=window_size,
            inj_cfg=inj_cfg,
            device=device,
            logger=logger,
        )

    # PCA projections
    for pca_dim in pca_dims:
        logger.info(f"[exp4] === PCA d={pca_dim} ===")
        pca_dir = out_dir / f"pca_{pca_dim}"
        pca_dir.mkdir(parents=True, exist_ok=True)

        pca_zarr = pca_dir / "latents.zarr"
        pca_index = pca_dir / "latents_index.parquet"
        pca_frames = pca_dir / "phase0_frames.parquet"

        # 1. Fit PCA
        pca = _fit_pca(
            store=store,
            train_utt_ids=train_utt_ids,
            n_components=pca_dim,
            logger=logger,
        )

        # Save PCA model
        pca_model_path = pca_dir / "pca_model.npz"
        np.savez_compressed(
            str(pca_model_path),
            components=pca.components_,
            mean=pca.mean_,
            explained_variance_ratio=pca.explained_variance_ratio_,
            singular_values=pca.singular_values_,
        )

        # 2. Project and save
        _project_and_save(
            store=store,
            latents_index=latents_index,
            pca=pca,
            out_zarr_path=pca_zarr,
            out_index_path=pca_index,
            logger=logger,
        )

        # 3. Build frames index
        _build_frames_index(
            splits_dir=splits_dir,
            latents_index_path=pca_index,
            latents_zarr_path=pca_zarr,
            window_size=window_size,
            max_lag=max_lag,
            min_duration_sec=min_duration_sec,
            out_frames_index_path=pca_frames,
            logger=logger,
        )

        # 4. Run Phase 1 MDN
        phase1_dir = pca_dir / "phase1"
        phase1_dir.mkdir(parents=True, exist_ok=True)
        pca_results = []

        for k in horizons_k:
            r = train_and_eval_for_k(
                frames_index_path=pca_frames,
                latents_dir=pca_zarr,
                splits_dir=splits_dir,
                latents_index_path=pca_index,
                out_dir=phase1_dir,
                horizon_k=k,
                window_size=window_size,
                slice_name=slice_name,
                seed=int(cfg["train"]["seed"]),
                device=device,
                n_components=int(cfg["model"]["n_components"]),
                hidden_dim=int(cfg["model"]["hidden_dim"]),
                n_hidden_layers=int(cfg["model"]["n_hidden_layers"]),
                dropout=float(cfg["model"]["dropout"]),
                min_log_sigma=float(cfg["model"]["min_log_sigma"]),
                max_log_sigma=float(cfg["model"]["max_log_sigma"]),
                batch_size=int(cfg["train"]["batch_size"]),
                num_workers=int(cfg["train"]["num_workers"]),
                max_steps=int(cfg["train"]["max_steps"]),
                lr=float(cfg["train"]["lr"]),
                weight_decay=float(cfg["train"]["weight_decay"]),
                grad_clip_norm=float(cfg["train"]["grad_clip_norm"]),
                log_every=int(cfg["train"]["log_every"]),
                eval_every=int(cfg["train"]["eval_every"]),
                save_every=int(cfg["train"]["save_every"]),
                shuffle_buffer=int(cfg["train"]["shuffle_buffer"]),
                max_train_samples=cfg["train"].get("max_train_samples"),
                max_eval_samples=cfg["train"].get("max_eval_samples"),
                rollout_enabled=bool(cfg["rollout"]["enabled"]),
                rollout_n_eval_utterances=int(cfg["rollout"]["n_eval_utterances"]),
                rollout_max_frames_per_utt=int(cfg["rollout"]["max_frames_per_utterance"]),
                rollout_sample_from_mixture=bool(cfg["rollout"]["sample_from_mixture"]),
                compile_model=bool(cfg["train"].get("compile", False)),
                compile_mode=str(cfg["train"].get("compile_mode", "default")),
                amp=bool(cfg["train"].get("amp", False)),
                amp_dtype=str(cfg["train"].get("amp_dtype", "bf16")),
            )
            pca_results.append(r)
            all_rows.append({
                "representation": f"pca_{pca_dim}",
                "pca_dim": pca_dim,
                "horizon_k": k,
                "eval_dnll": r.eval.get("dnll"),
                "eval_nll": r.eval.get("nll"),
                "eval_nll_baseline": r.eval.get("nll_baseline"),
                "rollout_gap_nll": r.rollout.get("gap_nll") if r.rollout else None,
            })

        write_results(pca_results, metrics_path=str(phase1_dir / "metrics.json"), tables_path=str(phase1_dir / "tables.csv"))

        # Injection diagnostic on PCA-projected latents
        if cfg.get("run_injection", True):
            _run_injection_diagnostic(
                phase1_dir=phase1_dir,
                frames_index_path=pca_frames,
                latents_dir=pca_zarr,
                latents_index_path=pca_index,
                splits_dir=splits_dir,
                window_size=window_size,
                inj_cfg=inj_cfg,
                device=device,
                logger=logger,
            )

    # Summary comparison table
    summary = pd.DataFrame(all_rows)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(str(summary_path), index=False)
    logger.info(f"[exp4] Summary table:\n{summary.to_string()}")
    logger.info(f"[exp4] Wrote summary: {summary_path}")

    # Extract key metrics: k=1 DNLL for each representation
    km = {}
    for _, row in summary[summary["horizon_k"] == 1].iterrows():
        km[f"{row['representation']}_dnll_k1"] = row.get("eval_dnll")
    finalize_run(run, key_metrics=km)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
