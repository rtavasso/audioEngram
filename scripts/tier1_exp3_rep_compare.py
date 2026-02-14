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
from experiment import register_run, finalize_run
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
    model_id: str = "facebook/encodec_24khz",
    representation: str = "codes",
) -> None:
    from phase0.data.io import save_latents_zarr
    from phase0.data.librispeech import load_audio
    from phase0.vae.infer_latents import compute_frame_energy

    try:
        import torch
        from transformers import EncodecModel, AutoProcessor
    except Exception as e:
        raise RuntimeError(
            "EnCodec extraction requires `transformers` + `datasets` deps installed "
            "(see pyproject.toml)."
        ) from e

    representation = str(representation).lower().strip()
    if representation not in {"codes", "encoder"}:
        raise ValueError(f"Unknown EnCodec representation: {representation} (expected codes|encoder)")

    processor = AutoProcessor.from_pretrained(str(model_id))
    sample_rate = int(getattr(processor, "sampling_rate", 24000))
    model = EncodecModel.from_pretrained(str(model_id)).to(device)
    model.eval()
    logger.info(f"[exp3] Loaded EnCodec via transformers: {model_id} (sr={sample_rate}, rep={representation})")

    entries = []
    for utt in utterances:
        try:
            wav, _sr = load_audio(utt.audio_path, target_sr=sample_rate)
            audio = wav.squeeze(0).detach().cpu().numpy()

            inputs = processor(raw_audio=audio, sampling_rate=sample_rate, return_tensors="pt")
            input_values = inputs["input_values"].to(device)
            padding_mask = inputs.get("padding_mask")
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)

            with torch.inference_mode():
                if representation == "encoder":
                    if not hasattr(model, "encoder"):
                        raise RuntimeError("EncodecModel missing .encoder; cannot extract encoder features.")
                    enc_out = model.encoder(input_values)
                    if hasattr(enc_out, "last_hidden_state"):
                        enc = enc_out.last_hidden_state
                    elif torch.is_tensor(enc_out):
                        enc = enc_out
                    else:
                        raise RuntimeError("Unexpected encoder output type; expected Tensor or object with .last_hidden_state")

                    # Try to coerce to [1, C, T']
                    if enc.ndim == 3 and enc.shape[0] == 1:
                        pass
                    elif enc.ndim == 2 and enc.shape[0] == 1:
                        enc = enc.unsqueeze(1)
                    else:
                        raise RuntimeError(f"Unexpected encoder tensor shape: {tuple(enc.shape)}")

                    emb = enc.detach().float().cpu()
                    lat = emb.squeeze(0).permute(1, 0).contiguous().numpy()  # [T',C]
                else:
                    out = model(input_values, padding_mask)
                    if not hasattr(out, "audio_codes") or out.audio_codes is None:
                        raise RuntimeError("EncodecModel output missing audio_codes; cannot extract codes.")
                    codes = out.audio_codes.detach().cpu()
                    # Expected: [B, Q, T'] where Q=#codebooks.
                    if codes.ndim != 3 or codes.shape[0] != 1:
                        raise RuntimeError(f"Unexpected audio_codes shape: {tuple(codes.shape)}")
                    lat = codes.squeeze(0).permute(1, 0).contiguous().to(torch.float32).numpy()  # [T',Q]

            n_frames = int(lat.shape[0])

            # Energy aligned by uniform partition into n_frames slices.
            frame_size = int((int(wav.shape[-1]) + n_frames - 1) // max(n_frames, 1))
            energy = compute_frame_energy(wav, frame_size, n_frames)

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

    run = register_run(
        experiment="exp3_rep_compare", run_id=run_id, config_path=args.config,
        config=cfg, cli_args=sys.argv[1:], out_dir=out_dir, log_name="tier1-exp3-rep-compare",
    )

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
                        model_id=str(rep.get("model_id", "facebook/encodec_24khz")),
                        representation=str(rep.get("representation", "codes")),
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

        # Run injection diagnostic on k=1 checkpoint if available
        if cfg.get("run_injection", True):
            ckpt_path = rep_phase1_dir / "checkpoints" / "mdn_k1_final.pt"
            if ckpt_path.exists():
                from phase1.checkpoints import load_phase1_checkpoint
                from phase1.injection_diag import run_injection_diagnostic
                from phase1.train_eval import fit_unconditional_baseline
                import json as _json

                inj_model, _ckpt = load_phase1_checkpoint(ckpt_path, device=device)
                inj_baseline = fit_unconditional_baseline(
                    frames_index_path=frames_index,
                    latents_dir=latents_zarr,
                    window_size=window_size,
                    horizon_k=1,
                    slice_name=str(cfg.get("slice", "all")),
                    max_samples=cfg["train"].get("max_train_samples"),
                )
                inj_cfg = cfg.get("injection", {})
                inj_result = run_injection_diagnostic(
                    model=inj_model,
                    baseline=inj_baseline,
                    latents_dir=latents_zarr,
                    latents_index_path=latents_index,
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
                inj_path = rep_phase1_dir / "injection_diag.json"
                with open(inj_path, "w") as _f:
                    _json.dump(inj_result, _f, indent=2)
                logger.info(f"[exp3] Wrote injection diagnostic: {inj_path}")

        # Summary row: best eval ΔNLL across horizons (more negative is better)
        best = min((float(r.eval.get("dnll")) for r in results if r.eval.get("dnll") is not None), default=float("nan"))
        rep_rows.append({"representation": name, "best_eval_dnll": best, "phase1_dir": str(rep_phase1_dir)})

    summary = pd.DataFrame(rep_rows).sort_values("best_eval_dnll")
    summary_path = out_dir / "summary.csv"
    summary.to_csv(str(summary_path), index=False)
    logger.info(f"Wrote summary: {summary_path}")

    km = {}
    for row in rep_rows:
        km[f"{row['representation']}_best_dnll"] = row["best_eval_dnll"]
    finalize_run(run, key_metrics=km)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
