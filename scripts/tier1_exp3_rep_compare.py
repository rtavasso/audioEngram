#!/usr/bin/env python3
"""
Tier 1 - Experiment 3: Representation comparison.

Compares Phase 1 predictability (teacher-forced + rollout gap diagnostic) across:
  - Mimi latent (12.5 Hz)
  - EnCodec encoder latents (if torchaudio EnCodec is available)

Single-command runner:
  uv run python scripts/tier1_exp3_rep_compare.py --config configs/tier1_exp3_rep_compare.yaml
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import yaml

from phase0.data.io import (
    save_latents_index,
    save_frames_index,
    load_latents_index,
    load_latents_zarr,
)
from phase0.data.librispeech import get_utterances
from phase0.data.splits import load_splits
from phase0.features.context import get_valid_frame_range
from phase0.features.energy import compute_median_energy
from phase0.utils.logging import setup_logging
from phase0.utils.seed import set_seed
from phase1.train_eval import _device_from_config, train_and_eval_for_k, write_results


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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
            logger.warning(f"[exp3] Could not load {utt_id}: {e}")
            continue

        if split == "train":
            train_energies.append(energy)

        first_valid, last_valid = get_valid_frame_range(n_frames, int(window_size), int(max_lag))
        for t in range(int(first_valid), int(last_valid)):
            frames_list.append(
                {
                    "utterance_id": utt_id,
                    "speaker_id": speaker_id,
                    "t": int(t),
                    "pos_frac": float(t) / float(n_frames),
                    "energy": float(energy[t]),
                    "split": split,
                }
            )

    frames = pd.DataFrame(frames_list)
    if frames.empty:
        raise RuntimeError("No frames found when building frames index.")

    median_energy = compute_median_energy(train_energies)
    frames["is_high_energy"] = frames["energy"] > float(median_energy)
    save_frames_index(frames, out_frames_index_path)


def _extract_mimi_latents(
    *,
    utterances,
    zarr_path: str | Path,
    index_path: str | Path,
    device: str,
    checkpoint: str | None,
    logger,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    amp: bool,
    amp_dtype: str,
) -> None:
    from mimi_autoencoder import load_mimi_autoencoder
    from phase0.vae.infer_latents import batch_infer_latents

    # Mimi's pretrained checkpoint exports a fixed latent frame rate (typically 12.5 Hz).
    # There is no supported "50 Hz" export mode without changing model internals/checkpoint.
    logger.info("[exp3] Loading Mimi autoencoder (fixed latent rate)...")
    autoencoder = load_mimi_autoencoder(checkpoint_path=checkpoint, device=device)
    autoencoder.eval()

    entries = batch_infer_latents(
        utterances=utterances,
        autoencoder=autoencoder,
        zarr_path=zarr_path,
        device=device,
        show_progress=True,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        prefetch_factor=int(prefetch_factor),
        amp=bool(amp),
        amp_dtype=str(amp_dtype),
    )
    save_latents_index(entries, index_path)


def _extract_encodec_latents(
    *,
    utterances,
    zarr_path: str | Path,
    index_path: str | Path,
    device: str,
    logger,
) -> None:
    try:
        import torch
        import torchaudio
    except Exception as e:
        raise RuntimeError("EnCodec extraction requires torch + torchaudio installed.") from e

    from phase0.data.io import save_latents_zarr
    from phase0.data.librispeech import load_audio
    from phase0.vae.infer_latents import compute_frame_energy

    # Try to load EnCodec model from torchaudio.
    model = None
    sample_rate = 24000
    if hasattr(torchaudio, "pipelines") and hasattr(torchaudio.pipelines, "ENCODEC_24KHZ"):
        bundle = torchaudio.pipelines.ENCODEC_24KHZ
        sample_rate = int(getattr(bundle, "sample_rate", 24000))
        model = bundle.get_model().to(device)
    elif hasattr(torchaudio, "models") and hasattr(torchaudio.models, "encodec_model_24khz"):
        model = torchaudio.models.encodec_model_24khz().to(device)
        sample_rate = 24000
    else:
        raise RuntimeError("Could not find an EnCodec model in torchaudio (expected pipelines.ENCODEC_24KHZ).")

    model.eval()
    logger.info(f"[exp3] Loaded EnCodec model (sr={sample_rate})")

    entries = []
    for utt in utterances:
        try:
            wav, _sr = load_audio(utt.audio_path, target_sr=sample_rate)
            wav = wav.unsqueeze(0).to(device)  # [1,1,T]
            with torch.inference_mode():
                if hasattr(model, "encoder"):
                    emb = model.encoder(wav)  # [B,C,T']
                else:
                    raise RuntimeError("EnCodec model missing .encoder; cannot extract continuous latents.")
            emb = emb.detach().float().cpu()
            lat = emb.squeeze(0).permute(1, 0).contiguous().numpy()  # [T',C]
            n_frames = int(lat.shape[0])

            # Energy aligned by uniform partition into n_frames slices.
            frame_size = int((int(wav.shape[-1]) + n_frames - 1) // max(n_frames, 1))
            energy = compute_frame_energy(wav.squeeze(0).cpu(), frame_size, n_frames)

            # Timestamps from effective frame rate (approx).
            duration_sec = float(utt.duration_sec)
            eff_frame_rate = float(n_frames) / max(duration_sec, 1e-6)
            timestamps = (np.arange(n_frames, dtype=np.float32) / eff_frame_rate).astype(np.float32, copy=False)

            save_latents_zarr(
                latents=lat,
                energy=energy,
                timestamps=timestamps,
                speaker_id=int(utt.speaker_id),
                utterance_id=str(utt.utterance_id),
                zarr_path=zarr_path,
            )
            entries.append(
                {
                    "utterance_id": str(utt.utterance_id),
                    "speaker_id": int(utt.speaker_id),
                    "n_frames": int(n_frames),
                    "duration_sec": float(utt.duration_sec),
                    "audio_path": str(utt.audio_path),
                }
            )
        except Exception as e:
            logger.warning(f"[exp3] EnCodec failed for {utt.utterance_id}: {e}")
            continue

    save_latents_index(entries, index_path)


def main() -> int:
    p = argparse.ArgumentParser(description="Tier1 Exp3: representation comparison")
    p.add_argument("--config", type=str, default="configs/tier1_exp3_rep_compare.yaml")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--force", action="store_true", help="Recompute latents/frames even if outputs exist")
    args = p.parse_args()

    logger = setup_logging(name="tier1-exp3-rep-compare")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_id = args.run_id or _default_run_id()
    out_root = Path(cfg["output"]["out_dir"])
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg.get("seed", 42)))

    device = _device_from_config(cfg["train"]["device"])
    window_size = int(cfg["context"]["window_size"])
    horizons_k = [int(k) for k in cfg["context"]["horizons_k"]]
    max_lag = int(max(horizons_k))

    librispeech_path = cfg["data"]["librispeech_path"]
    subset = cfg["data"]["subset"]
    min_duration_sec = float(cfg["data"]["min_duration_sec"])
    splits_dir = cfg["data"]["splits_dir"]

    splits = load_splits(splits_dir)
    all_speakers = splits.train_speakers + splits.eval_speakers
    utterances = get_utterances(librispeech_path, all_speakers, subset)
    utterances = [u for u in utterances if float(u.duration_sec) >= min_duration_sec]

    # Optionally downsample utterances for speed
    max_utts = cfg["data"].get("max_utterances")
    if max_utts is not None:
        utterances = utterances[: int(max_utts)]

    logger.info(f"Run id: {run_id}")
    logger.info(f"Device: {device}")
    logger.info(f"Utterances: {len(utterances)} (subset={subset})")

    rep_rows = []
    for rep in cfg["representations"]:
        name = str(rep["name"])
        rep_dir = out_dir / name
        rep_dir.mkdir(parents=True, exist_ok=True)

        latents_zarr = rep_dir / "latents.zarr"
        latents_index = rep_dir / "latents_index.parquet"
        frames_index = rep_dir / "phase0_frames.parquet"

        if args.force:
            # Best-effort cleanup (zarr is a directory)
            if latents_zarr.exists():
                import shutil

                shutil.rmtree(latents_zarr)
            for pth in (latents_index, frames_index):
                if pth.exists():
                    pth.unlink()

        if not latents_index.exists() or not latents_zarr.exists():
            rtype = str(rep["type"]).lower().strip()
            logger.info(f"[exp3] Extracting latents for {name} ({rtype}) -> {rep_dir}")
            if rtype == "mimi":
                _extract_mimi_latents(
                    utterances=utterances,
                    zarr_path=latents_zarr,
                    index_path=latents_index,
                    device=str(device),
                    checkpoint=rep.get("checkpoint"),
                    logger=logger,
                    batch_size=int(rep.get("batch_size", 1)),
                    num_workers=int(rep.get("num_workers", 0)),
                    prefetch_factor=int(rep.get("prefetch_factor", 2)),
                    amp=bool(rep.get("amp", True)),
                    amp_dtype=str(rep.get("amp_dtype", "bf16")),
                )
            elif rtype == "encodec":
                try:
                    _extract_encodec_latents(
                        utterances=utterances,
                        zarr_path=latents_zarr,
                        index_path=latents_index,
                        device=str(device),
                        logger=logger,
                    )
                except Exception as e:
                    logger.warning(f"[exp3] Skipping {name}: EnCodec extraction unavailable ({e})")
                    continue
            else:
                raise ValueError(f"Unknown representation type: {rtype}")

        if not frames_index.exists():
            logger.info(f"[exp3] Building frames index for {name}")
            _build_frames_index(
                splits_dir=splits_dir,
                latents_index_path=latents_index,
                latents_zarr_path=latents_zarr,
                window_size=window_size,
                max_lag=max_lag,
                min_duration_sec=min_duration_sec,
                out_frames_index_path=frames_index,
                logger=logger,
            )

        # Phase 1 baseline (MDN) on this representation
        logger.info(f"[exp3] Phase1 baseline (MDN) for {name}")
        rep_phase1_dir = rep_dir / "phase1"
        rep_phase1_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for k in horizons_k:
            r = train_and_eval_for_k(
                frames_index_path=frames_index,
                latents_dir=latents_zarr,
                splits_dir=splits_dir,
                latents_index_path=latents_index,
                out_dir=rep_phase1_dir,
                horizon_k=k,
                window_size=window_size,
                slice_name=str(cfg.get("slice", "all")),
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
                model_type="mdn",
                compile_model=bool(cfg["train"].get("compile", False)),
                compile_mode=str(cfg["train"].get("compile_mode", "default")),
                amp=bool(cfg["train"].get("amp", False)),
                amp_dtype=str(cfg["train"].get("amp_dtype", "bf16")),
            )
            results.append(r)

        write_results(results, metrics_path=str(rep_phase1_dir / "metrics.json"), tables_path=str(rep_phase1_dir / "tables.csv"))

        # Summary row: best eval ΔNLL across horizons (more negative is better)
        best = min((float(r.eval.get("dnll")) for r in results if r.eval.get("dnll") is not None), default=float("nan"))
        rep_rows.append({"representation": name, "best_eval_dnll": best, "phase1_dir": str(rep_phase1_dir)})

    summary = pd.DataFrame(rep_rows).sort_values("best_eval_dnll")
    summary_path = out_dir / "summary.csv"
    summary.to_csv(str(summary_path), index=False)
    logger.info(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
