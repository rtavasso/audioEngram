#!/usr/bin/env python3
"""
Phase 3: Export z_dyn sequences for each utterance from a trained checkpoint.

Writes a zarr store with per-utterance z_dyn arrays and an index parquet.

Usage:
  uv run python scripts/21_phase3_export_zdyn.py --config configs/phase3.yaml --checkpoint outputs/phase3/checkpoints/phase3_final.pt
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import torch
import yaml

from phase0.data.io import save_latents_zarr, save_latents_index, LatentStore
from phase0.utils.logging import setup_logging
from phase3.data import Phase3UtteranceDataset
from phase3.models import Factorizer
from phase3.normalization import NormStats, normalize_x
from phase3.train_eval import _device_from_config, compute_or_load_norm_stats


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description="Export z_dyn from Phase 3 checkpoint")
    parser.add_argument("--config", type=str, default="configs/phase3.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max-utts", type=int, default=None)
    parser.add_argument("--split", type=str, default="eval", help="train|eval|all")
    args = parser.parse_args()

    setup_logging(name="phase3-export")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = _device_from_config(cfg["train"]["device"])
    ckpt = torch.load(args.checkpoint, map_location=device)

    x_dim = int(cfg["model"]["x_dim"])
    z_dyn_dim = int(cfg["model"]["z_dyn_dim"])
    z_rec_dim = int(cfg["model"]["z_rec_dim"])

    norm_stats = None
    if cfg.get("normalization", {}).get("enabled", True):
        norm_stats = compute_or_load_norm_stats(
            latents_dir=cfg["data"]["latents_dir"],
            latents_index=cfg["data"]["latents_index"],
            splits_dir=cfg["data"]["splits_dir"],
            stats_file=cfg["normalization"]["stats_file"],
            x_dim=x_dim,
            min_duration_sec=float(cfg["data"]["min_duration_sec"]),
            max_train_utterances=cfg["normalization"].get("max_train_utterances"),
        )

    m = cfg["model"]
    model = Factorizer(
        x_dim=x_dim,
        z_dyn_dim=z_dyn_dim,
        z_rec_dim=z_rec_dim,
        dyn_encoder_hidden=int(m["dyn_encoder"]["hidden_dim"]),
        dyn_encoder_layers=int(m["dyn_encoder"]["num_layers"]),
        dyn_encoder_dropout=float(m["dyn_encoder"]["dropout"]),
        dyn_model_hidden=int(m["dyn_model"]["hidden_dim"]),
        dyn_model_layers=int(m["dyn_model"]["num_layers"]),
        dyn_model_dropout=float(m["dyn_model"]["dropout"]),
        dyn_model_min_log_sigma=float(m["dyn_model"]["min_log_sigma"]),
        dyn_model_max_log_sigma=float(m["dyn_model"]["max_log_sigma"]),
        posterior_hidden=int(m["posterior"]["hidden_dim"]),
        posterior_layers=int(m["posterior"]["num_layers"]),
        posterior_dropout=float(m["posterior"]["dropout"]),
        posterior_min_log_sigma=float(m["posterior"]["min_log_sigma"]),
        posterior_max_log_sigma=float(m["posterior"]["max_log_sigma"]),
        prior_hidden=int(m["prior"]["hidden_dim"]),
        prior_layers=int(m["prior"]["num_layers"]),
        prior_dropout=float(m["prior"]["dropout"]),
        prior_min_log_sigma=float(m["prior"]["min_log_sigma"]),
        prior_max_log_sigma=float(m["prior"]["max_log_sigma"]),
        recon_hidden=int(m["reconstructor"]["hidden_dim"]),
        recon_layers=int(m["reconstructor"]["num_layers"]),
        recon_dropout=float(m["reconstructor"]["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # We export from the raw latents store to avoid padding/collate.
    store = LatentStore(cfg["data"]["latents_dir"])

    # Determine which utterances to export
    import pandas as pd

    df = pd.read_parquet(cfg["data"]["latents_index"], columns=["utterance_id", "speaker_id", "n_frames", "duration_sec"])
    df = df[df["duration_sec"] >= float(cfg["data"]["min_duration_sec"])]
    if args.split != "all":
        from phase0.data.splits import load_splits

        splits = load_splits(cfg["data"]["splits_dir"])
        speakers = set(splits.train_speakers if args.split == "train" else splits.eval_speakers)
        df = df[df["speaker_id"].isin(speakers)]
    df = df.sort_values(["speaker_id", "utterance_id"])
    if args.max_utts is not None:
        df = df.iloc[: int(args.max_utts)]

    out_zarr = Path(cfg["output"]["zdyn_dir"])
    out_idx = Path(cfg["output"]["zdyn_index"])
    out_zarr.parent.mkdir(parents=True, exist_ok=True)

    index_entries = []
    for r in df.itertuples(index=False):
        utt_id = str(r.utterance_id)
        spk = int(r.speaker_id)
        x = store.get_latents(utt_id).astype(np.float32, copy=False)
        if norm_stats is not None:
            x = normalize_x(x, norm_stats).astype(np.float32, copy=False)
        xt = torch.from_numpy(x).unsqueeze(0).to(device)  # [1,T,D]
        z_dyn = model.e_dyn(xt)[0].detach().cpu().numpy().astype(np.float32, copy=False)  # [T,z]

        # Store as "x" in zarr for compatibility with LatentStore access pattern
        energy = np.zeros((z_dyn.shape[0],), dtype=np.float32)
        timestamps = np.arange(z_dyn.shape[0], dtype=np.float32) / 12.5
        save_latents_zarr(
            latents=z_dyn,
            energy=energy,
            timestamps=timestamps,
            speaker_id=spk,
            utterance_id=utt_id,
            zarr_path=out_zarr,
        )
        index_entries.append(
            {
                "utterance_id": utt_id,
                "speaker_id": spk,
                "n_frames": int(z_dyn.shape[0]),
                "duration_sec": float(r.duration_sec),
                "audio_path": "",
            }
        )

    save_latents_index(index_entries, out_idx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

