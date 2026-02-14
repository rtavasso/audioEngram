#!/usr/bin/env python3
"""
Stage 2 - Experiment 5: beta-VAE with AR-friendliness objectives.

For each VAE variant:
  1. Train VAE (audio -> encode -> losses -> backward)
  2. Extract latents for all utterances -> zarr
  3. Build frames index
  4. Run Phase 1 diagnostic battery (train_and_eval_for_k for k in horizons)
  5. Run injection diagnostic (4 modes)
  6. Compute reconstruction quality (mel distance on eval utterances)
  7. Write summary CSV + per-variant results

Usage:
  uv run python scripts/tier2_exp5_betavae.py --config configs/tier2_exp5_betavae.yaml
  uv run python scripts/tier2_exp5_betavae.py --config configs/tier2_exp5_betavae.yaml --variant baseline_betavae
  uv run python scripts/tier2_exp5_betavae.py --config configs/tier2_exp5_betavae.yaml --max-steps 100  # quick test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import yaml

from experiment import register_run, finalize_run
from phase0.data.librispeech import get_utterances
from phase0.data.splits import load_splits
from phase0.utils.logging import setup_logging
from phase0.utils.seed import set_seed
from phase1.train_eval import _device_from_config, train_and_eval_for_k, write_results


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_vae(cfg: dict, device: torch.device):
    """Build ARFriendlyVAE from config."""
    from mimi_autoencoder import load_mimi_autoencoder
    from stage2.vae import ARFriendlyVAE

    vae_cfg = cfg["vae"]
    autoencoder = load_mimi_autoencoder(
        checkpoint_path=vae_cfg.get("mimi_checkpoint"),
        device=str(device),
    )
    vae = ARFriendlyVAE(
        mimi_encoder=autoencoder.encoder,
        mimi_decoder=autoencoder.decoder,
        latent_dim=int(vae_cfg["latent_dim"]),
        encoder_dim=int(vae_cfg["encoder_dim"]),
        dec_hidden_dim=int(vae_cfg.get("dec_hidden_dim", 256)),
        freeze_encoder=bool(vae_cfg.get("freeze_encoder", True)),
        freeze_decoder=bool(vae_cfg.get("freeze_decoder", False)),
    ).to(device)
    return vae, autoencoder


def _compute_recon_metrics(
    vae,
    utterances: list,
    device: torch.device,
    n_utterances: int = 50,
    max_duration_sec: float = 10.0,
    sample_rate: int = 24000,
    logger=None,
) -> dict:
    """Compute reconstruction quality metrics on eval utterances."""
    from phase0.data.librispeech import load_audio

    vae.eval()
    l1_errors = []
    mel_errors = []

    selected = [u for u in utterances if u.duration_sec <= max_duration_sec][:n_utterances]

    for utt in selected:
        try:
            wav, sr = load_audio(utt.audio_path, target_sr=sample_rate)
            audio = wav.unsqueeze(0).to(device)  # [1, 1, T]
            length = audio.shape[-1]

            with torch.inference_mode():
                out = vae(audio)
                audio_hat = out["audio_hat"][..., :length]

            # Time-domain L1
            l1 = float(torch.nn.functional.l1_loss(audio_hat, audio).item())
            l1_errors.append(l1)

            # Mel spectrogram distance
            mel_err = _mel_distance(audio.squeeze(), audio_hat.squeeze(), sample_rate)
            mel_errors.append(mel_err)
        except Exception as e:
            if logger:
                logger.warning(f"[exp5] Recon eval failed for {utt.utterance_id}: {e}")
            continue

    return {
        "n_utterances": len(l1_errors),
        "l1_mean": float(np.mean(l1_errors)) if l1_errors else float("nan"),
        "l1_std": float(np.std(l1_errors)) if l1_errors else float("nan"),
        "mel_distance_mean": float(np.mean(mel_errors)) if mel_errors else float("nan"),
        "mel_distance_std": float(np.std(mel_errors)) if mel_errors else float("nan"),
    }


def _mel_distance(audio: torch.Tensor, audio_hat: torch.Tensor, sr: int, n_mels: int = 80) -> float:
    """Compute mel spectrogram L1 distance."""
    import torchaudio.transforms as T

    mel_transform = T.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=256, n_mels=n_mels,
    ).to(audio.device)

    mel = mel_transform(audio.squeeze())
    mel_hat = mel_transform(audio_hat.squeeze())
    log_mel = torch.log(mel.clamp_min(1e-5))
    log_mel_hat = torch.log(mel_hat.clamp_min(1e-5))
    return float(torch.nn.functional.l1_loss(log_mel_hat, log_mel).item())


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 2 Exp5: beta-VAE")
    p.add_argument("--config", type=str, default="configs/tier2_exp5_betavae.yaml")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--variant", type=str, default=None, help="Run only this variant")
    p.add_argument("--max-steps", type=int, default=None, help="Override VAE max_steps")
    p.add_argument("--phase1-max-steps", type=int, default=None, help="Override Phase 1 max_steps")
    args = p.parse_args()

    # Disable torch.compile on old GPUs
    if os.environ.get("NO_TORCH_COMPILE"):
        os.environ["TORCH_COMPILE_DISABLE"] = "1"

    logger = setup_logging(name="phase0")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_id = args.run_id or _default_run_id()
    out_root = Path(cfg["output"]["out_dir"])
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    run = register_run(
        experiment="exp5_betavae", run_id=run_id, config_path=args.config,
        config=cfg, cli_args=sys.argv[1:], out_dir=out_dir, log_name="phase0",
    )

    set_seed(int(cfg.get("seed", 42)))
    device = _device_from_config(cfg["train"]["device"])

    # Load utterances
    data_cfg = cfg["data"]
    splits = load_splits(data_cfg["splits_dir"])
    all_speakers = splits.train_speakers + splits.eval_speakers
    utterances = get_utterances(data_cfg["librispeech_path"], all_speakers, data_cfg["subset"])
    utterances = [u for u in utterances if u.duration_sec >= float(data_cfg["min_duration_sec"])]
    max_utts = data_cfg.get("max_utterances")
    if max_utts is not None:
        utterances = utterances[:int(max_utts)]

    audio_paths = [str(u.audio_path) for u in utterances]
    logger.info(f"[exp5] Run id: {run_id}, device: {device}, utterances: {len(utterances)}")

    # Separate eval utterances for reconstruction metrics
    eval_speaker_set = set(splits.eval_speakers)
    eval_utterances = [u for u in utterances if u.speaker_id in eval_speaker_set]

    # Filter variants
    variants = cfg["variants"]
    if args.variant:
        variants = [v for v in variants if v["name"] == args.variant]
        if not variants:
            logger.error(f"Variant '{args.variant}' not found in config")
            return 1

    summary_rows = []

    for var in variants:
        var_name = var["name"]
        var_dir = out_dir / var_name
        var_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[exp5] === Variant: {var_name} ===")

        # 1. Train VAE
        logger.info(f"[exp5] Training VAE for {var_name}...")
        vae, autoencoder = _build_vae(cfg, device)

        from stage2.vae_train import train_vae

        vae_train_cfg = cfg["vae_train"]
        vae_max_steps = args.max_steps or int(vae_train_cfg["max_steps"])

        train_result = train_vae(
            audio_paths=audio_paths,
            vae=vae,
            out_dir=var_dir,
            beta=float(var["beta"]),
            lambda_smooth=float(var.get("lambda_smooth", 0.0)),
            lambda_pred=float(var.get("lambda_pred", 0.0)),
            pred_window_size=int(vae_train_cfg.get("pred_window_size", 8)),
            pred_hidden_dim=int(vae_train_cfg.get("pred_hidden_dim", 256)),
            segment_sec=float(vae_train_cfg["segment_sec"]),
            sample_rate=int(vae_train_cfg["sample_rate"]),
            batch_size=int(vae_train_cfg["batch_size"]),
            num_workers=int(vae_train_cfg["num_workers"]),
            max_steps=vae_max_steps,
            lr=float(vae_train_cfg["lr"]),
            lr_predictor=float(vae_train_cfg.get("lr_predictor", 1e-3)),
            weight_decay=float(vae_train_cfg["weight_decay"]),
            grad_clip_norm=float(vae_train_cfg["grad_clip_norm"]),
            log_every=int(vae_train_cfg["log_every"]),
            save_every=int(vae_train_cfg["save_every"]),
            sample_every=int(vae_train_cfg.get("sample_every", 5000)),
            sample_output_sr=int(vae_train_cfg.get("sample_output_sr", 48000)),
            seed=int(cfg.get("seed", 42)),
            device=device,
            amp=bool(vae_train_cfg.get("amp", True)),
            amp_dtype=str(vae_train_cfg.get("amp_dtype", "fp16")),
        )
        logger.info(f"[exp5] VAE training done: {train_result['final_checkpoint']}")

        # 2. Extract latents
        logger.info(f"[exp5] Extracting VAE latents for {var_name}...")
        from stage2.vae_extract import extract_vae_latents, build_frames_index

        latents_zarr = var_dir / "latents.zarr"
        latents_index = var_dir / "latents_index.parquet"

        extract_vae_latents(
            vae=vae,
            utterances=utterances,
            zarr_path=latents_zarr,
            index_path=latents_index,
            device=device,
        )

        # 3. Build frames index
        window_size = int(cfg["context"]["window_size"])
        horizons_k = [int(k) for k in cfg["context"]["horizons_k"]]
        max_lag = max(horizons_k)

        frames_index = var_dir / "phase0_frames.parquet"
        build_frames_index(
            splits_dir=data_cfg["splits_dir"],
            latents_index_path=latents_index,
            latents_zarr_path=latents_zarr,
            window_size=window_size,
            max_lag=max_lag,
            min_duration_sec=float(data_cfg["min_duration_sec"]),
            out_frames_index_path=frames_index,
        )

        # 4. Phase 1 diagnostic battery
        logger.info(f"[exp5] Phase 1 diagnostic battery for {var_name}...")
        phase1_dir = var_dir / "phase1"
        phase1_dir.mkdir(parents=True, exist_ok=True)

        phase1_max_steps = args.phase1_max_steps or int(cfg["train"]["max_steps"])

        results = []
        for k in horizons_k:
            r = train_and_eval_for_k(
                frames_index_path=frames_index,
                latents_dir=latents_zarr,
                splits_dir=data_cfg["splits_dir"],
                latents_index_path=latents_index,
                out_dir=phase1_dir,
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
                max_steps=phase1_max_steps,
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

        write_results(
            results,
            metrics_path=str(phase1_dir / "metrics.json"),
            tables_path=str(phase1_dir / "tables.csv"),
        )

        # 5. Injection diagnostic
        ckpt_path = phase1_dir / "checkpoints" / "mdn_k1_final.pt"
        inj_result = None
        if ckpt_path.exists():
            from phase1.checkpoints import load_phase1_checkpoint
            from phase1.injection_diag import run_injection_diagnostic
            from phase1.train_eval import fit_unconditional_baseline

            inj_model, _ = load_phase1_checkpoint(ckpt_path, device=device)
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
                splits_dir=data_cfg["splits_dir"],
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
            with open(phase1_dir / "injection_diag.json", "w") as f:
                json.dump(inj_result, f, indent=2)
            logger.info(f"[exp5] Injection diagnostic written for {var_name}")

        # 6. Reconstruction quality
        recon_cfg = cfg.get("recon_eval", {})
        recon_metrics = _compute_recon_metrics(
            vae, eval_utterances, device,
            n_utterances=int(recon_cfg.get("n_utterances", 50)),
            max_duration_sec=float(recon_cfg.get("max_duration_sec", 10.0)),
            logger=logger,
        )
        with open(var_dir / "recon_metrics.json", "w") as f:
            json.dump(recon_metrics, f, indent=2)

        # Summary row
        best_dnll = min(
            (float(r.eval.get("dnll")) for r in results if r.eval.get("dnll") is not None),
            default=float("nan"),
        )
        row = {
            "variant": var_name,
            "beta": var["beta"],
            "lambda_smooth": var.get("lambda_smooth", 0.0),
            "lambda_pred": var.get("lambda_pred", 0.0),
            "best_eval_dnll": best_dnll,
            "recon_l1": recon_metrics.get("l1_mean", float("nan")),
            "recon_mel": recon_metrics.get("mel_distance_mean", float("nan")),
        }
        if inj_result and "modes" in inj_result:
            d_mode = inj_result["modes"].get("D_rollout", {})
            row["rollout_state_err"] = d_mode.get("state_err", float("nan"))
            row["rollout_cos"] = d_mode.get("cos", float("nan"))
        summary_rows.append(row)

        # Free GPU memory before next variant
        del vae, autoencoder
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Write summary
    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(str(summary_path), index=False)
    logger.info(f"[exp5] Summary written: {summary_path}")
    logger.info(f"\n{summary.to_string(index=False)}")

    km = {}
    for row in summary_rows:
        km[f"{row['variant']}_best_dnll"] = row["best_eval_dnll"]
        km[f"{row['variant']}_recon_mel"] = row.get("recon_mel", float("nan"))
    finalize_run(run, key_metrics=km)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
