from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from sklearn.cluster import MiniBatchKMeans

from phase0.data.io import LatentStore, load_latents_index
from phase0.data.splits import load_splits
from phase0.data.librispeech import load_audio
from phase0.utils.seed import set_seed
from stage2.pocket_mimi_vae import build_pocket_mimi_vae
from phase1.train_eval import _device_from_config


StageName = Literal["exp1", "exp2", "exp3", "exp4", "exp5", "exp6"]


def parse_stages(spec: str) -> list[StageName]:
    stages: list[StageName] = []
    for s in (spec or "").split(","):
        s = s.strip().lower()
        if not s:
            continue
        if s not in {"exp1", "exp2", "exp3", "exp4", "exp5", "exp6"}:
            raise ValueError(f"Unknown stage '{s}'. Expected exp1..exp6.")
        stages.append(s)  # type: ignore[arg-type]
    if not stages:
        raise ValueError("No stages selected.")
    return stages


def _now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _unit_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def _random_unit_vectors(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(size=(n, d), dtype=np.float32)
    return _unit_rows(x)


def _angle_deg_from_dots(dots: np.ndarray) -> np.ndarray:
    dots = np.clip(np.abs(dots), -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def _chunked_mean_sq_dist(x: np.ndarray, centers: np.ndarray, labels: np.ndarray, chunk: int = 200_000) -> float:
    n = x.shape[0]
    acc = 0.0
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        diff = x[sl] - centers[labels[sl]]
        acc += float(np.sum(diff * diff))
    return acc / float(n)


def _hist_counts(values: np.ndarray, bin_deg: int) -> dict:
    if values.size == 0:
        return {"bin_deg": int(bin_deg), "bins": [], "counts": []}
    max_deg = 90.0
    bins = np.arange(0.0, max_deg + bin_deg, bin_deg)
    counts, edges = np.histogram(values, bins=bins)
    return {
        "bin_deg": int(bin_deg),
        "bins": [float(x) for x in edges.tolist()],
        "counts": [int(x) for x in counts.tolist()],
    }


@dataclass(frozen=True)
class LabelGroups:
    order: np.ndarray  # indices that sort labels ascending
    starts: np.ndarray  # [n_labels] start offsets into order
    ends: np.ndarray  # [n_labels] end offsets into order

    def indices(self, label: int) -> np.ndarray:
        return self.order[self.starts[label] : self.ends[label]]

    def count(self, label: int) -> int:
        return int(self.ends[label] - self.starts[label])


def _group_indices_by_label(labels: np.ndarray, n_labels: int) -> LabelGroups:
    labels_i = labels.astype(np.int32, copy=False)
    order = np.argsort(labels_i, kind="mergesort")
    sorted_labels = labels_i[order]
    all_labels = np.arange(int(n_labels), dtype=np.int32)
    starts = np.searchsorted(sorted_labels, all_labels, side="left").astype(np.int64, copy=False)
    ends = np.searchsorted(sorted_labels, all_labels, side="right").astype(np.int64, copy=False)
    return LabelGroups(order=order.astype(np.int32, copy=False), starts=starts, ends=ends)


@dataclass(frozen=True)
class CachePaths:
    cache_dir: Path
    z_path: Path
    delta_unit_path: Path
    delta_mag_path: Path
    delta_src_idx_path: Path
    delta_prev_idx_path: Path
    utt_table_path: Path
    stats_path: Path


def _cache_paths(out_dir: Path) -> CachePaths:
    cache_dir = _ensure_dir(out_dir / "cache")
    return CachePaths(
        cache_dir=cache_dir,
        z_path=cache_dir / "z.npy",
        delta_unit_path=cache_dir / "delta_unit.npy",
        delta_mag_path=cache_dir / "delta_mag.npy",
        delta_src_idx_path=cache_dir / "delta_src_idx.npy",
        delta_prev_idx_path=cache_dir / "delta_prev_idx.npy",
        utt_table_path=cache_dir / "utterances.parquet",
        stats_path=cache_dir / "dataset_stats.json",
    )


def build_or_load_cache(
    *,
    latents_dir: str | Path,
    latents_index: str | Path,
    splits_dir: str | Path,
    out_dir: Path,
    logger,
    eps_mult: float,
    max_utterances: int | None,
    max_deltas: int | None,
    overwrite: bool,
) -> dict:
    """
    Build (or load) a cached representation:
      - z: [M, D] float32 concatenated latent frames
      - delta_unit: [N, D] float32 unit deltas
      - delta_mag: [N] float32 magnitudes
      - delta_src_idx: [N] int32 source frame global index in z
      - delta_prev_idx: [N] int32 previous frame global index in z (or -1)
      - utterances table: utterance_id, speaker_id, audio_path, n_frames, z_offset

    Returns dict with memmapped arrays + metadata + preprocessing thresholds.
    """
    paths = _cache_paths(out_dir)
    have_all = (
        paths.z_path.exists()
        and paths.delta_unit_path.exists()
        and paths.delta_mag_path.exists()
        and paths.delta_src_idx_path.exists()
        and paths.delta_prev_idx_path.exists()
        and paths.utt_table_path.exists()
        and paths.stats_path.exists()
    )
    if have_all and not overwrite:
        try:
            logger.info(f"[exp12] Using existing cache at {paths.cache_dir}")
            utt = pd.read_parquet(paths.utt_table_path)
            with open(paths.stats_path) as f:
                stats = json.load(f)
            n_d = int(stats["n_deltas_total"])
            z = np.load(paths.z_path, mmap_mode="r")
            du = np.load(paths.delta_unit_path, mmap_mode="r")[:n_d]
            dm = np.load(paths.delta_mag_path, mmap_mode="r")[:n_d]
            src = np.load(paths.delta_src_idx_path, mmap_mode="r")[:n_d]
            prev = np.load(paths.delta_prev_idx_path, mmap_mode="r")[:n_d]
            eps = float(stats["near_zero_eps"])
            valid_mask = dm >= eps
            if max_deltas is not None and int(valid_mask.sum()) > int(max_deltas):
                logger.warning("[exp12] max_deltas override requested; will subsample valid deltas deterministically")
            return {
                "paths": paths,
                "utterances": utt,
                "stats": stats,
                "z": z,
                "delta_unit": du,
                "delta_mag": dm,
                "delta_src_idx": src,
                "delta_prev_idx": prev,
                "near_zero_eps": eps,
                "valid_delta_mask": valid_mask,
            }
        except Exception as e:
            logger.warning(f"[exp12] Cache load failed ({e}); rebuilding cache.")

    logger.info("[exp12] Building cache (this may take a while)...")
    store = LatentStore(latents_dir)
    index = load_latents_index(latents_index)
    splits = load_splits(splits_dir)
    split_train = set(splits.train_utterances)
    split_eval = set(splits.eval_utterances)

    # Keep only utterances present in the latents index and splits.
    index = index.copy()
    index["utterance_id"] = index["utterance_id"].astype(str)
    index = index[index["utterance_id"].isin(split_train.union(split_eval))]
    index = index.sort_values(["speaker_id", "utterance_id"]).reset_index(drop=True)
    if max_utterances is not None:
        index = index.iloc[: int(max_utterances)].reset_index(drop=True)

    utt_ids = index["utterance_id"].tolist()
    logger.info(f"[exp12] Cache input utterances: {len(utt_ids)}")
    if len(utt_ids) == 0:
        raise RuntimeError("No utterances found after filtering latents_index by splits.")

    # Preallocate arrays (sizes from index n_frames).
    n_frames_total = int(index["n_frames"].sum())
    d = int(store.get_latents(utt_ids[0]).shape[1])

    # Total deltas = sum(T-1).
    n_deltas_total = int((index["n_frames"] - 1).clip(lower=0).sum())
    if max_deltas is not None:
        n_deltas_total = min(n_deltas_total, int(max_deltas))

    logger.info(f"[exp12] Total frames M={n_frames_total}, deltas N≈{n_deltas_total}, D={d}")

    z = np.lib.format.open_memmap(paths.z_path, dtype=np.float32, mode="w+", shape=(n_frames_total, d))
    delta_unit = np.lib.format.open_memmap(paths.delta_unit_path, dtype=np.float32, mode="w+", shape=(n_deltas_total, d))
    delta_mag = np.lib.format.open_memmap(paths.delta_mag_path, dtype=np.float32, mode="w+", shape=(n_deltas_total,))
    delta_src_idx = np.lib.format.open_memmap(paths.delta_src_idx_path, dtype=np.int32, mode="w+", shape=(n_deltas_total,))
    delta_prev_idx = np.lib.format.open_memmap(paths.delta_prev_idx_path, dtype=np.int32, mode="w+", shape=(n_deltas_total,))

    # Build concatenated arrays.
    z_off = 0
    d_off = 0
    utt_rows = []
    rng = np.random.default_rng(0)  # deterministic sampling if max_deltas is used

    for i, row in index.iterrows():
        utt_id = str(row["utterance_id"])
        spk = int(row["speaker_id"])
        audio_path = str(row["audio_path"])
        x = store.get_latents(utt_id).astype(np.float32, copy=False)  # [T, D]
        t = int(x.shape[0])
        if t == 0:
            continue

        z[z_off : z_off + t] = x

        # Deltas for this utterance
        if t >= 2 and d_off < n_deltas_total:
            dx = x[1:] - x[:-1]  # [t-1, d]
            mags = np.linalg.norm(dx, axis=1).astype(np.float32, copy=False)
            dirs = dx / np.maximum(mags[:, None], 1e-8)

            # Optional global cap: subsample deltas uniformly across utterances.
            take = np.arange(t - 1, dtype=np.int64)
            if max_deltas is not None:
                remaining = n_deltas_total - d_off
                if remaining <= 0:
                    take = take[:0]
                elif take.size > remaining:
                    take = rng.choice(take, size=remaining, replace=False)
                    take.sort()

            n_take = int(take.size)
            if n_take > 0:
                delta_unit[d_off : d_off + n_take] = dirs[take]
                delta_mag[d_off : d_off + n_take] = mags[take]
                # Source frame global index for each delta is z_off + frame_idx
                src_idx = (z_off + take).astype(np.int32, copy=False)
                delta_src_idx[d_off : d_off + n_take] = src_idx
                # Previous frame global index (for bigram) is src-1 if within utterance, else -1
                prev_local = (take - 1)
                prev_global = (z_off + prev_local).astype(np.int32, copy=False)
                prev_global[prev_local < 0] = -1
                delta_prev_idx[d_off : d_off + n_take] = prev_global
                d_off += n_take

        utt_rows.append(
            {
                "utterance_id": utt_id,
                "speaker_id": spk,
                "audio_path": audio_path,
                "n_frames": t,
                "z_offset": int(z_off),
            }
        )
        z_off += t

        if (i + 1) % 200 == 0:
            logger.info(f"[exp12] Cached {i+1}/{len(index)} utterances")

        if max_deltas is not None and d_off >= n_deltas_total:
            # Still need to fill z fully for state clustering, so don't break.
            pass

    # Flush memmaps
    z.flush()
    delta_unit.flush()
    delta_mag.flush()
    delta_src_idx.flush()
    delta_prev_idx.flush()

    utt_df = pd.DataFrame(utt_rows)
    utt_df.to_parquet(paths.utt_table_path, index=False)

    # Compute near-zero threshold based on median ||δ||
    dm = np.array(delta_mag[:d_off], copy=True)
    median_mag = float(np.median(dm)) if dm.size else 0.0
    near_zero_eps = float(eps_mult * median_mag)

    stats = {
        "timestamp": _now_iso(),
        "latents_dir": str(latents_dir),
        "latents_index": str(latents_index),
        "splits_dir": str(splits_dir),
        "n_utterances": int(len(utt_df)),
        "latent_dim": int(d),
        "n_frames_total": int(n_frames_total),
        "n_deltas_total": int(d_off),
        "delta_mag_median": median_mag,
        "near_zero_eps_mult": float(eps_mult),
        "near_zero_eps": near_zero_eps,
        "delta_mag_percentiles": {
            "p5": float(np.percentile(dm, 5)) if dm.size else 0.0,
            "p25": float(np.percentile(dm, 25)) if dm.size else 0.0,
            "p50": float(np.percentile(dm, 50)) if dm.size else 0.0,
            "p75": float(np.percentile(dm, 75)) if dm.size else 0.0,
            "p95": float(np.percentile(dm, 95)) if dm.size else 0.0,
        },
    }
    with open(paths.stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Reload as mmap for uniform return type.
    z_r = np.load(paths.z_path, mmap_mode="r")
    du_r = np.load(paths.delta_unit_path, mmap_mode="r")
    dm_r = np.load(paths.delta_mag_path, mmap_mode="r")
    src_r = np.load(paths.delta_src_idx_path, mmap_mode="r")
    prev_r = np.load(paths.delta_prev_idx_path, mmap_mode="r")
    valid_mask = dm_r[:d_off] >= near_zero_eps

    logger.info(
        f"[exp12] Cache built: M={n_frames_total} frames, N={d_off} deltas, "
        f"near_zero_eps={near_zero_eps:.3e}, kept={int(valid_mask.sum())}"
    )

    return {
        "paths": paths,
        "utterances": utt_df,
        "stats": stats,
        "z": z_r,
        "delta_unit": du_r[:d_off],
        "delta_mag": dm_r[:d_off],
        "delta_src_idx": src_r[:d_off],
        "delta_prev_idx": prev_r[:d_off],
        "near_zero_eps": near_zero_eps,
        "valid_delta_mask": valid_mask,
    }


def _fit_kmeans_unit(
    x_unit: np.ndarray,
    k: int,
    seed: int,
    *,
    n_init: int,
    max_iter: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    km = MiniBatchKMeans(
        n_clusters=int(k),
        init="k-means++",
        n_init=int(n_init),
        max_iter=int(max_iter),
        batch_size=int(batch_size),
        random_state=int(seed),
    )
    km.fit(x_unit)
    centers = km.cluster_centers_.astype(np.float32, copy=False)
    centers = _unit_rows(centers)
    labels = km.labels_.astype(np.int32, copy=False)
    return centers, labels


def _mean_angle_unit_to_centroid(
    x_unit: np.ndarray,
    centers_unit: np.ndarray,
    labels: np.ndarray,
) -> float:
    dots = np.sum(x_unit * centers_unit[labels], axis=1)
    ang = _angle_deg_from_dots(dots)
    return float(np.mean(ang)) if ang.size else float("nan")


def _global_eval(
    *,
    x_unit: np.ndarray,
    k: int,
    seed: int,
    kmeans_cfg: dict,
    hist_bin_deg: int,
) -> dict:
    centers, labels = _fit_kmeans_unit(
        x_unit,
        k,
        seed,
        n_init=kmeans_cfg["n_init"],
        max_iter=kmeans_cfg["max_iter"],
        batch_size=kmeans_cfg["batch_size"],
    )
    dots = np.sum(x_unit * centers[labels], axis=1)
    ang = _angle_deg_from_dots(dots)
    return {
        "k": int(k),
        "seed": int(seed),
        "theta_mean_deg": float(np.mean(ang)),
        "theta_median_deg": float(np.median(ang)),
        "theta_std_deg": float(np.std(ang)),
        "hist": _hist_counts(ang, hist_bin_deg),
    }


def _write_json(path: Path, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _plot_hist_overlay(out_path: Path, hist_a: dict, hist_b: dict, title: str, label_a: str, label_b: str) -> None:
    import matplotlib.pyplot as plt

    bins = np.array(hist_a["bins"], dtype=np.float32)
    ca = np.array(hist_a["counts"], dtype=np.int64)
    cb = np.array(hist_b["counts"], dtype=np.int64)
    x = bins[:-1]
    width = float(hist_a["bin_deg"])
    plt.figure(figsize=(8, 4))
    plt.bar(x, ca, width=width, alpha=0.6, label=label_a, align="edge")
    plt.bar(x, cb, width=width, alpha=0.6, label=label_b, align="edge")
    plt.xlabel("Angular distance (deg)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_heatmap(out_path: Path, df: pd.DataFrame, value_col: str, title: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    pivot = df.pivot_table(index="k1", columns="k2", values=value_col, aggfunc="mean")
    plt.figure(figsize=(9, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis")
    plt.title(title)
    plt.xlabel("K2 (delta codebook per state)")
    plt.ylabel("K1 (state codebook)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _safe_pesq(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    try:
        from pesq import pesq
        mode = "wb" if sr == 16000 else "nb"
        return float(pesq(sr, ref, deg, mode))
    except Exception:
        return float("nan")


def _safe_stoi(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    try:
        from pystoi import stoi
        return float(stoi(ref, deg, sr, extended=False))
    except Exception:
        return float("nan")


def _mel_l1(ref_t: torch.Tensor, deg_t: torch.Tensor, sr: int, n_mels: int) -> float:
    mel = T.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=256, n_mels=int(n_mels)).to(ref_t.device)
    with torch.inference_mode():
        m0 = mel(ref_t)
        m1 = mel(deg_t)
        m0 = torch.log(m0.clamp_min(1e-5))
        m1 = torch.log(m1.clamp_min(1e-5))
        return float(torch.nn.functional.l1_loss(m1, m0).item())


def _resample_np(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    t = torch.from_numpy(x[None, :]).float()
    y = torchaudio.functional.resample(t, sr_in, sr_out)
    return y.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


class Tier4Exp12Runner:
    def __init__(self, *, cfg: dict, out_dir: Path, logger):
        self.cfg = cfg
        self.out_dir = out_dir
        self.logger = logger
        _ensure_dir(self.out_dir / "plots")
        _ensure_dir(self.out_dir / "tables")
        _ensure_dir(self.out_dir / "artifacts")

    def run(self, *, stages: list[StageName], overwrite_cache: bool) -> dict:
        key_metrics: dict = {}

        needs_cache = any(s in {"exp1", "exp2", "exp3", "exp4", "exp5"} for s in stages)
        cache = None
        du_all = None
        src_idx_all = None
        prev_idx_all = None
        if needs_cache:
            cache = build_or_load_cache(
                latents_dir=self.cfg["data"]["latents_dir"],
                latents_index=self.cfg["data"]["latents_index"],
                splits_dir=self.cfg["data"]["splits_dir"],
                out_dir=self.out_dir,
                logger=self.logger,
                eps_mult=float(self.cfg["preprocess"]["eps_mult"]),
                max_utterances=self.cfg["preprocess"].get("max_utterances"),
                max_deltas=self.cfg["preprocess"].get("max_deltas"),
                overwrite=overwrite_cache,
            )

            # Materialize filtered deltas for analysis.
            valid_mask = cache["valid_delta_mask"]
            du_all = np.array(cache["delta_unit"][valid_mask], dtype=np.float32, copy=True)
            dmag_all = np.array(cache["delta_mag"][valid_mask], dtype=np.float32, copy=True)
            src_idx_all = np.array(cache["delta_src_idx"][valid_mask], dtype=np.int32, copy=True)
            prev_idx_all = np.array(cache["delta_prev_idx"][valid_mask], dtype=np.int32, copy=True)

            # Optional cap (deterministic)
            max_deltas = self.cfg["preprocess"].get("max_deltas")
            if max_deltas is not None and du_all.shape[0] > int(max_deltas):
                self.logger.warning(f"[exp12] Subsampling valid deltas to max_deltas={max_deltas}")
                rng = np.random.default_rng(123)
                keep = rng.choice(np.arange(du_all.shape[0]), size=int(max_deltas), replace=False)
                keep.sort()
                du_all = du_all[keep]
                dmag_all = dmag_all[keep]
                src_idx_all = src_idx_all[keep]
                prev_idx_all = prev_idx_all[keep]

            cache_summary = {
                "n_deltas_valid": int(du_all.shape[0]),
                "near_zero_eps": float(cache["near_zero_eps"]),
                "delta_mag_mean": float(np.mean(dmag_all)) if dmag_all.size else 0.0,
                "delta_mag_median": float(np.median(dmag_all)) if dmag_all.size else 0.0,
            }
            _write_json(self.out_dir / "artifacts" / "cache_summary.json", cache_summary)

        if "exp1" in stages:
            assert du_all is not None
            res1 = self._run_exp1_global(du_all)
            key_metrics.update(res1.get("key_metrics", {}))
        state_models: dict = {}
        if any(s in {"exp2", "exp3", "exp4", "exp5"} for s in stages):
            assert cache is not None and src_idx_all is not None

            k1_list, seed_list = self._required_state_models(stages=stages)
            state_models = self._run_exp2_states(
                cache["z"],
                src_idx_all,
                stages=stages,
                k1_list=k1_list,
                seed_list=seed_list,
            )

        if "exp3" in stages:
            assert du_all is not None and src_idx_all is not None
            res3 = self._run_exp3_conditional(du_all, src_idx_all, state_models)
            key_metrics.update(res3.get("key_metrics", {}))
        else:
            res3 = {}

        if "exp4" in stages and bool(self.cfg.get("exp4_bigram", {}).get("enabled", True)):
            assert du_all is not None and src_idx_all is not None and prev_idx_all is not None
            self._run_exp4_bigram(du_all, src_idx_all, prev_idx_all, res3, state_models)

        if "exp5" in stages and bool(self.cfg.get("exp5_perceptual", {}).get("enabled", True)):
            assert cache is not None
            self._run_exp5_perceptual(res3, state_models, cache)

        if "exp6" in stages and bool(self.cfg.get("exp6_scaling", {}).get("enabled", True)):
            self._run_exp6_scaling()

        self._write_report()
        return {"key_metrics": key_metrics}

    def _required_state_models(self, *, stages: list[StageName]) -> tuple[list[int], list[int]]:
        """
        Decide which (K1, seed) state models need to be fit.

        - If exp3 is requested: we need all K1 values (for the grid).
        - If only exp4/exp5 are requested: use best K1 from artifacts if present,
          otherwise fall back to config values.
        """
        seeds = [int(s) for s in self.cfg.get("seeds", [int(self.cfg.get("seed", 42))])]
        k1_cfg = [int(k) for k in self.cfg.get("exp2_states", {}).get("k1", [])]
        if "exp3" in stages or "exp2" in stages:
            return k1_cfg, seeds

        cfg5 = self.cfg.get("exp5_perceptual", {}) if isinstance(self.cfg.get("exp5_perceptual"), dict) else {}
        state_seed = int(cfg5.get("state_seed") or seeds[0])

        # If this run only needs Exp5 (and maybe Exp6), fit just one seed.
        stages_set = set(stages)
        if "exp5" in stages_set and stages_set.issubset({"exp5", "exp6"}):
            if cfg5.get("k1") is not None:
                return [int(cfg5["k1"])], [state_seed]
            best_path = self.out_dir / "artifacts" / "exp3_best_config.json"
            if best_path.exists():
                try:
                    with open(best_path) as f:
                        best = json.load(f)
                    k1 = int(best.get("best_k1"))
                    return [k1], [state_seed]
                except Exception:
                    pass
            if k1_cfg:
                return [max(k1_cfg)], [state_seed]

        # exp4/exp5 only: prefer explicit exp5 k1 override.
        if cfg5.get("k1") is not None:
            return [int(cfg5["k1"])], seeds

        best_path = self.out_dir / "artifacts" / "exp3_best_config.json"
        if best_path.exists():
            try:
                with open(best_path) as f:
                    best = json.load(f)
                k1 = int(best.get("best_k1"))
                return [k1], seeds
            except Exception:
                pass

        # Fallback: just fit the largest K1 for best granularity.
        if k1_cfg:
            return [max(k1_cfg)], seeds
        raise RuntimeError("No exp2_states.k1 values found in config; cannot fit state models.")

    # ---------------------------------------------------------------------
    # Experiment 1: Global baseline
    # ---------------------------------------------------------------------
    def _run_exp1_global(self, du_all: np.ndarray) -> dict:
        self.logger.info("[exp12][exp1] Running global baseline...")
        exp_cfg = self.cfg["exp1_global"]
        k_list = [int(k) for k in exp_cfg["k_global"]]
        hist_bin = int(exp_cfg.get("hist_bin_deg", 5))
        seeds = [int(s) for s in self.cfg.get("seeds", [int(self.cfg.get("seed", 42))])]
        kmeans_cfg = self.cfg["kmeans"]

        rows = []
        for k in k_list:
            for sd in seeds:
                r_data = _global_eval(
                    x_unit=du_all,
                    k=k,
                    seed=sd,
                    kmeans_cfg=kmeans_cfg,
                    hist_bin_deg=hist_bin,
                )
                r_data["kind"] = "data"
                rows.append(r_data)

                x_rand = _random_unit_vectors(du_all.shape[0], du_all.shape[1], seed=sd + 10_000 + k)
                r_rand = _global_eval(
                    x_unit=x_rand,
                    k=k,
                    seed=sd,
                    kmeans_cfg=kmeans_cfg,
                    hist_bin_deg=hist_bin,
                )
                r_rand["kind"] = "random"
                rows.append(r_rand)

            # Plot overlay using the first seed's hist (deterministic and lightweight)
            ha = next(r for r in rows if r["k"] == k and r["seed"] == seeds[0] and r["kind"] == "data")["hist"]
            hb = next(r for r in rows if r["k"] == k and r["seed"] == seeds[0] and r["kind"] == "random")["hist"]
            _plot_hist_overlay(
                self.out_dir / "plots" / f"exp1_hist_k{k:04d}.png",
                ha,
                hb,
                title=f"Exp1 Global: K={k}",
                label_a="data",
                label_b="random",
            )

        df = pd.DataFrame(rows).drop(columns=["hist"])
        out_csv = self.out_dir / "tables" / "exp1_global.csv"
        df.to_csv(out_csv, index=False)
        self.logger.info(f"[exp12][exp1] Wrote {out_csv}")

        # Primary deliverable table: K | mean/median (data/random) | gaps.
        piv_mean = df.pivot_table(index=["k", "seed"], columns="kind", values="theta_mean_deg", aggfunc="mean").reset_index()
        piv_med = df.pivot_table(index=["k", "seed"], columns="kind", values="theta_median_deg", aggfunc="mean").reset_index()
        piv = piv_mean.merge(
            piv_med.rename(columns={"data": "median_data", "random": "median_random"}),
            on=["k", "seed"],
            how="left",
        ).rename(columns={"data": "mean_data", "random": "mean_random"})
        piv["gap_mean_deg"] = piv["mean_random"] - piv["mean_data"]
        piv["gap_median_deg"] = piv["median_random"] - piv["median_data"]

        primary = piv.groupby("k")[["mean_data", "mean_random", "gap_mean_deg", "median_data", "median_random"]].agg(["mean", "std"]).reset_index()
        primary_path = self.out_dir / "tables" / "exp1_primary_table.csv"
        primary.to_csv(primary_path, index=False)
        self.logger.info(f"[exp12][exp1] Wrote {primary_path}")

        # Key metric: global gap at max K.
        k_max = max(k_list)
        gap = float(primary[primary["k"] == k_max][("gap_mean_deg", "mean")].iloc[0])
        return {"key_metrics": {"exp1_gap_deg_kmax": gap, "exp1_kmax": k_max}}

    # ---------------------------------------------------------------------
    # Experiment 2: State quantization
    # ---------------------------------------------------------------------
    def _run_exp2_states(
        self,
        z_all: np.ndarray,
        src_idx_all: np.ndarray,
        *,
        stages: list[StageName],
        k1_list: list[int] | None = None,
        seed_list: list[int] | None = None,
    ) -> dict:
        self.logger.info("[exp12][exp2] Running state quantization...")
        exp_cfg = self.cfg["exp2_states"]
        k1_list = [int(k) for k in (k1_list if k1_list is not None else exp_cfg["k1"])]
        seeds = [int(s) for s in (seed_list if seed_list is not None else self.cfg.get("seeds", [int(self.cfg.get("seed", 42))]))]
        kmeans_cfg = self.cfg["kmeans"]
        min_deltas = int(self.cfg["preprocess"].get("min_deltas_per_cluster", 50))

        # z_all may be memmap; only take as array when needed by sklearn.
        z_np = np.array(z_all, dtype=np.float32, copy=False)

        models: dict[tuple[int, int], dict] = {}
        rows = []
        artifacts_dir = _ensure_dir(self.out_dir / "artifacts")
        for k1 in k1_list:
            for sd in seeds:
                labels_path = artifacts_dir / f"exp2_state_labels_k1_{int(k1):04d}_seed_{int(sd):03d}.npy"
                centers_path = artifacts_dir / f"exp2_state_centers_k1_{int(k1):04d}_seed_{int(sd):03d}.npy"

                # For resume runs (exp4/exp5 only), reuse stored labels/centers when available.
                reuse_ok = ("exp2" not in stages) and labels_path.exists() and centers_path.exists()
                if reuse_ok:
                    self.logger.info(f"[exp12][exp2] Reusing state artifacts for k1={k1} seed={sd}")
                    labels_frames = np.load(labels_path, mmap_mode="r")
                    centers = np.load(centers_path, mmap_mode="r")
                else:
                    set_seed(sd)
                    km = MiniBatchKMeans(
                        n_clusters=int(k1),
                        init="k-means++",
                        n_init=int(kmeans_cfg["n_init"]),
                        max_iter=int(kmeans_cfg["max_iter"]),
                        batch_size=int(kmeans_cfg["batch_size"]),
                        random_state=int(sd),
                    )
                    km.fit(z_np)
                    centers = km.cluster_centers_.astype(np.float32, copy=False)
                    labels_frames = km.labels_.astype(np.int32, copy=False)

                    # Persist for resume without refitting.
                    np.save(centers_path, np.array(centers, dtype=np.float32, copy=True))
                    np.save(labels_path, np.array(labels_frames, dtype=np.int32, copy=True))

                mse = _chunked_mean_sq_dist(z_np, centers, labels_frames)
                counts = np.bincount(labels_frames, minlength=int(k1))

                # Delta counts: deltas are associated to their source frame.
                s_delta = labels_frames[src_idx_all]
                delta_counts = np.bincount(s_delta, minlength=int(k1))
                frac_small = float(np.mean(delta_counts < min_deltas))

                rows.append(
                    {
                        "k1": int(k1),
                        "seed": int(sd),
                        "quant_mse": float(mse),
                        "cluster_frames_min": int(counts.min()) if counts.size else 0,
                        "cluster_frames_median": float(np.median(counts)) if counts.size else 0.0,
                        "cluster_frames_mean": float(np.mean(counts)) if counts.size else 0.0,
                        "cluster_frames_max": int(counts.max()) if counts.size else 0,
                        "clusters_lt_min_deltas_frac": frac_small,
                        "clusters_lt_min_deltas_n": int(np.sum(delta_counts < min_deltas)),
                    }
                )

                models[(k1, sd)] = {
                    "k1": k1,
                    "seed": sd,
                    "centers": centers,
                    "labels_frames": labels_frames,
                }

            # Plot cluster size distribution for first seed only.
            first_sd = seeds[0]
            labels = models[(k1, first_sd)]["labels_frames"]
            counts = np.bincount(labels, minlength=int(k1))
            import matplotlib.pyplot as plt

            plt.figure(figsize=(7, 4))
            plt.hist(counts, bins=50)
            plt.title(f"Exp2 State cluster sizes (K1={k1}, seed={first_sd})")
            plt.xlabel("Frames per cluster")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(self.out_dir / "plots" / f"exp2_state_cluster_sizes_k1_{k1:04d}.png")
            plt.close()

        df = pd.DataFrame(rows)
        # Only write exp2 tables if exp2 explicitly requested; otherwise this
        # might be a partial (K1 subset) fit for resuming exp4/exp5.
        if "exp2" in stages:
            out_csv = self.out_dir / "tables" / "exp2_states.csv"
            df.to_csv(out_csv, index=False)
            self.logger.info(f"[exp12][exp2] Wrote {out_csv}")

        # If exp2 wasn't explicitly requested, keep models but skip extra logging.
        if "exp2" in stages:
            summary = df.groupby("k1")[["quant_mse", "clusters_lt_min_deltas_frac"]].agg(["mean", "std"]).reset_index()
            summary_path = self.out_dir / "tables" / "exp2_states_summary.csv"
            summary.to_csv(summary_path, index=False)
            self.logger.info(f"[exp12][exp2] Wrote {summary_path}")

        return models

    # ---------------------------------------------------------------------
    # Experiment 3: State-conditioned directional analysis
    # ---------------------------------------------------------------------
    def _run_exp3_conditional(self, du_all: np.ndarray, src_idx_all: np.ndarray, state_models: dict) -> dict:
        self.logger.info("[exp12][exp3] Running state-conditioned directional analysis...")
        k2_list = [int(k) for k in self.cfg["exp3_conditional"]["k2"]]
        k1_list = sorted({int(k1) for (k1, _sd) in state_models.keys()})
        seeds = [int(s) for s in self.cfg.get("seeds", [int(self.cfg.get("seed", 42))])]
        repeats = int(self.cfg["exp3_conditional"].get("shuffled_null_repeats", 5))
        min_deltas = int(self.cfg["preprocess"].get("min_deltas_per_cluster", 50))
        n_workers = int(self.cfg.get("compute", {}).get("n_workers", 0))
        kmeans_cfg = self.cfg["kmeans"]

        rows = []

        def per_state_fit_mean_angle(x: np.ndarray, k2: int, seed: int) -> float:
            if x.shape[0] < max(min_deltas, k2):
                return float("nan")
            centers, labels = _fit_kmeans_unit(
                x,
                k2,
                seed,
                n_init=kmeans_cfg["n_init"],
                max_iter=kmeans_cfg["max_iter"],
                batch_size=kmeans_cfg["batch_size"],
            )
            return _mean_angle_unit_to_centroid(x, centers, labels)

        for k1 in k1_list:
            for sd in seeds:
                labels_frames = state_models[(k1, sd)]["labels_frames"]
                s_delta = labels_frames[src_idx_all]  # [N]
                groups = _group_indices_by_label(s_delta, int(k1))

                # Data: per-state kmeans on actual conditional deltas.
                for k2 in k2_list:
                    if n_workers and n_workers > 1:
                        from concurrent.futures import ThreadPoolExecutor

                        def _job(j: int) -> tuple[int, float, int]:
                            idx = groups.indices(j)
                            n = int(idx.size)
                            if n < max(min_deltas, k2):
                                return j, float("nan"), int(idx.size)
                            ang = per_state_fit_mean_angle(du_all[idx], k2, sd + 1000 * j + 17)
                            return j, float(ang), n

                        with ThreadPoolExecutor(max_workers=n_workers) as ex:
                            per_state = list(ex.map(_job, range(int(k1))))
                    else:
                        per_state = []
                        for j in range(int(k1)):
                            idx = groups.indices(j)
                            ang = float("nan")
                            if idx.size >= max(min_deltas, k2):
                                ang = per_state_fit_mean_angle(du_all[idx], k2, sd + 1000 * j + 17)
                            per_state.append((j, ang, int(idx.size)))

                    # Aggregate
                    used = [(ang, n) for (_j, ang, n) in per_state if not math.isnan(ang)]
                    if not used:
                        theta_w = float("nan")
                        theta_u = float("nan")
                        n_used = 0
                        n_states_used = 0
                    else:
                        angs = np.array([a for a, _n in used], dtype=np.float32)
                        ns = np.array([_n for _a, _n in used], dtype=np.int64)
                        theta_w = float(np.sum(angs * ns) / float(np.sum(ns)))
                        theta_u = float(np.mean(angs))
                        n_used = int(np.sum(ns))
                        n_states_used = int(len(angs))

                    rows.append(
                        {
                            "k1": int(k1),
                            "k2": int(k2),
                            "seed": int(sd),
                            "kind": "data",
                            "theta_weighted_mean_deg": theta_w,
                            "theta_unweighted_mean_deg": theta_u,
                            "n_deltas_used": n_used,
                            "n_states_used": n_states_used,
                        }
                    )

                    # Shuffled null: shuffle delta->state assignment, preserve histogram.
                    rng = np.random.default_rng(sd + 999 + k1 + 10 * k2)
                    for r in range(repeats):
                        s_shuf = s_delta.copy()
                        rng.shuffle(s_shuf)
                        shuf_groups = _group_indices_by_label(s_shuf, int(k1))

                        if n_workers and n_workers > 1:
                            from concurrent.futures import ThreadPoolExecutor

                            def _job_shuf(j: int) -> tuple[int, float, int]:
                                idx = shuf_groups.indices(j)
                                n = int(idx.size)
                                if n < max(min_deltas, k2):
                                    return j, float("nan"), n
                                ang = per_state_fit_mean_angle(du_all[idx], k2, sd + 2000 * j + 31 + r * 7)
                                return j, float(ang), n

                            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                                per_state_shuf = list(ex.map(_job_shuf, range(int(k1))))
                        else:
                            per_state_shuf = []
                            for j in range(int(k1)):
                                idx = shuf_groups.indices(j)
                                ang = float("nan")
                                if idx.size >= max(min_deltas, k2):
                                    ang = per_state_fit_mean_angle(du_all[idx], k2, sd + 2000 * j + 31 + r * 7)
                                per_state_shuf.append((j, ang, int(idx.size)))

                        used_s = [(ang, n) for (_j, ang, n) in per_state_shuf if not math.isnan(ang)]
                        if not used_s:
                            theta_w_s = float("nan")
                            theta_u_s = float("nan")
                            n_used_s = 0
                            n_states_used_s = 0
                        else:
                            angs_s = np.array([a for a, _n in used_s], dtype=np.float32)
                            ns_s = np.array([_n for _a, _n in used_s], dtype=np.int64)
                            theta_w_s = float(np.sum(angs_s * ns_s) / float(np.sum(ns_s)))
                            theta_u_s = float(np.mean(angs_s))
                            n_used_s = int(np.sum(ns_s))
                            n_states_used_s = int(len(angs_s))

                        rows.append(
                            {
                                "k1": int(k1),
                                "k2": int(k2),
                                "seed": int(sd),
                                "repeat": int(r),
                                "kind": "shuffled",
                                "theta_weighted_mean_deg": theta_w_s,
                                "theta_unweighted_mean_deg": theta_u_s,
                                "n_deltas_used": n_used_s,
                                "n_states_used": n_states_used_s,
                            }
                        )

                    # Random null: per-state geometric baseline.
                    # This is expensive; still implement per spec.
                    per_state_rand = []
                    for j in range(int(k1)):
                        idx = groups.indices(j)
                        if idx.size < max(min_deltas, k2):
                            per_state_rand.append((j, float("nan"), int(idx.size)))
                            continue
                        x_r = _random_unit_vectors(int(idx.size), du_all.shape[1], seed=sd + 3000 * j + 13 + k2)
                        ang = per_state_fit_mean_angle(x_r, k2, sd + 4000 * j + 9)
                        per_state_rand.append((j, float(ang), int(idx.size)))
                    used_r = [(ang, n) for (_j, ang, n) in per_state_rand if not math.isnan(ang)]
                    if not used_r:
                        theta_w_r = float("nan")
                        theta_u_r = float("nan")
                        n_used_r = 0
                        n_states_used_r = 0
                    else:
                        angs_r = np.array([a for a, _n in used_r], dtype=np.float32)
                        ns_r = np.array([_n for _a, _n in used_r], dtype=np.int64)
                        theta_w_r = float(np.sum(angs_r * ns_r) / float(np.sum(ns_r)))
                        theta_u_r = float(np.mean(angs_r))
                        n_used_r = int(np.sum(ns_r))
                        n_states_used_r = int(len(angs_r))
                    rows.append(
                        {
                            "k1": int(k1),
                            "k2": int(k2),
                            "seed": int(sd),
                            "kind": "random",
                            "theta_weighted_mean_deg": theta_w_r,
                            "theta_unweighted_mean_deg": theta_u_r,
                            "n_deltas_used": n_used_r,
                            "n_states_used": n_states_used_r,
                        }
                    )

                self.logger.info(f"[exp12][exp3] Done k1={k1} seed={sd}")

        df = pd.DataFrame(rows)
        out_csv = self.out_dir / "tables" / "exp3_conditional.csv"
        df.to_csv(out_csv, index=False)
        self.logger.info(f"[exp12][exp3] Wrote {out_csv}")

        # Primary result table: aggregate across seeds and repeats.
        data_df = df[df["kind"] == "data"].copy()
        shuf_df = df[df["kind"] == "shuffled"].copy()
        rand_df = df[df["kind"] == "random"].copy()

        # Average shuffled per (k1,k2,seed) over repeats, then average over seeds.
        shuf_mean = (
            shuf_df.groupby(["k1", "k2", "seed"])["theta_weighted_mean_deg"].mean().reset_index()
        )
        data_mean = data_df[["k1", "k2", "seed", "theta_weighted_mean_deg"]].rename(
            columns={"theta_weighted_mean_deg": "theta_data"}
        )
        shuf_mean = shuf_mean.rename(columns={"theta_weighted_mean_deg": "theta_shuffled"})
        rand_mean = rand_df[["k1", "k2", "seed", "theta_weighted_mean_deg"]].rename(
            columns={"theta_weighted_mean_deg": "theta_random"}
        )

        merged = data_mean.merge(shuf_mean, on=["k1", "k2", "seed"], how="left").merge(
            rand_mean, on=["k1", "k2", "seed"], how="left"
        )
        merged["delta_structure_deg"] = merged["theta_shuffled"] - merged["theta_data"]
        agg = merged.groupby(["k1", "k2"])[["theta_data", "theta_shuffled", "theta_random", "delta_structure_deg"]].agg(
            ["mean", "std"]
        ).reset_index()
        agg_path = self.out_dir / "tables" / "exp3_primary_table.csv"
        agg.to_csv(agg_path, index=False)
        self.logger.info(f"[exp12][exp3] Wrote {agg_path}")

        # Mean-only primary table (easy to consume).
        means = merged.groupby(["k1", "k2"])[["theta_data", "theta_shuffled", "theta_random", "delta_structure_deg"]].mean().reset_index()
        means_path = self.out_dir / "tables" / "exp3_primary_table_means.csv"
        means.to_csv(means_path, index=False)
        self.logger.info(f"[exp12][exp3] Wrote {means_path}")

        # Heatmap of structure gain (mean).
        heat_df = merged.groupby(["k1", "k2"])["delta_structure_deg"].mean().reset_index()
        _plot_heatmap(
            self.out_dir / "plots" / "exp3_heatmap_delta_structure.png",
            heat_df,
            value_col="delta_structure_deg",
            title="Exp3 Δ_structure = θ(shuffled) - θ(data) (deg)",
        )

        # Select best config by maximum Δ_structure at smallest K2 preference.
        best = heat_df.sort_values(["delta_structure_deg", "k2"], ascending=[False, True]).iloc[0].to_dict()
        best_k1 = int(best["k1"])
        best_k2 = int(best["k2"])
        best_delta = float(best["delta_structure_deg"])
        best_path = self.out_dir / "artifacts" / "exp3_best_config.json"
        _write_json(best_path, {"best_k1": best_k1, "best_k2": best_k2, "best_delta_structure_deg": best_delta})
        self.logger.info(f"[exp12][exp3] Best config: K1={best_k1} K2={best_k2} Δ={best_delta:.2f}°")

        # Per-state θ̄_j distribution for best config (first seed), plus one shuffled null draw.
        try:
            sd0 = seeds[0]
            labels_frames = state_models[(best_k1, sd0)]["labels_frames"]
            s_delta = labels_frames[src_idx_all]
            groups = _group_indices_by_label(s_delta, best_k1)

            per_state_theta = np.full((best_k1,), np.nan, dtype=np.float32)
            per_state_n = np.zeros((best_k1,), dtype=np.int64)
            for j in range(best_k1):
                idx = groups.indices(j)
                per_state_n[j] = int(idx.size)
                if idx.size < max(min_deltas, best_k2):
                    continue
                ang = per_state_fit_mean_angle(du_all[idx], best_k2, sd0 + 1000 * j + 17)
                per_state_theta[j] = float(ang)

            # Shuffled: single shuffle draw with same histogram.
            rng = np.random.default_rng(sd0 + 999 + best_k1 + 10 * best_k2)
            s_shuf = s_delta.copy()
            rng.shuffle(s_shuf)
            shuf_groups = _group_indices_by_label(s_shuf, best_k1)
            per_state_theta_shuf = np.full((best_k1,), np.nan, dtype=np.float32)
            for j in range(best_k1):
                idx = shuf_groups.indices(j)
                if idx.size < max(min_deltas, best_k2):
                    continue
                ang = per_state_fit_mean_angle(du_all[idx], best_k2, sd0 + 2000 * j + 31)
                per_state_theta_shuf[j] = float(ang)

            per_state_df = pd.DataFrame(
                {
                    "state": np.arange(best_k1, dtype=np.int32),
                    "n_deltas": per_state_n.astype(np.int64),
                    "theta_mean_deg": per_state_theta.astype(np.float32),
                    "theta_mean_deg_shuffled": per_state_theta_shuf.astype(np.float32),
                }
            )
            per_state_path = self.out_dir / "tables" / "exp3_best_per_state_theta.csv"
            per_state_df.to_csv(per_state_path, index=False)

            # Plot histogram
            import matplotlib.pyplot as plt

            x = per_state_theta[np.isfinite(per_state_theta)]
            x_s = per_state_theta_shuf[np.isfinite(per_state_theta_shuf)]
            plt.figure(figsize=(7, 4))
            plt.hist(x, bins=40, alpha=0.6, label="data")
            plt.hist(x_s, bins=40, alpha=0.6, label="shuffled (1 draw)")
            if x.size:
                plt.axvline(float(np.mean(x)), color="k", linestyle="--", linewidth=1)
            if x_s.size:
                plt.axvline(float(np.mean(x_s)), color="k", linestyle=":", linewidth=1)
            plt.title(f"Exp3 per-state θ̄_j (K1={best_k1}, K2={best_k2}, seed={sd0})")
            plt.xlabel("Mean angular distance (deg)")
            plt.ylabel("States")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.out_dir / "plots" / "exp3_best_per_state_hist.png")
            plt.close()
        except Exception as e:
            self.logger.warning(f"[exp12][exp3] Failed best per-state histogram: {e}")

        return {
            "key_metrics": {"exp3_best_delta_structure_deg": best_delta, "exp3_best_k1": best_k1, "exp3_best_k2": best_k2},
            "best": {"k1": best_k1, "k2": best_k2},
        }

    # ---------------------------------------------------------------------
    # Experiment 4: Bigram conditioning
    # ---------------------------------------------------------------------
    def _run_exp4_bigram(
        self,
        du_all: np.ndarray,
        src_idx_all: np.ndarray,
        prev_idx_all: np.ndarray,
        res3: dict,
        state_models: dict,
    ) -> None:
        if not res3 or "best" not in res3:
            self.logger.warning("[exp12][exp4] Skipping: no exp3 best config found.")
            return
        best_k1 = int(res3["best"]["k1"])
        best_k2 = int(res3["best"]["k2"])
        seeds = [int(s) for s in self.cfg.get("seeds", [int(self.cfg.get("seed", 42))])]
        repeats = int(self.cfg["exp3_conditional"].get("shuffled_null_repeats", 5))
        min_deltas = int(self.cfg["preprocess"].get("min_deltas_per_cluster", 50))
        kmeans_cfg = self.cfg["kmeans"]

        rows = []

        for sd in seeds:
            labels_frames = state_models[(best_k1, sd)]["labels_frames"]
            s_t = labels_frames[src_idx_all]
            s_prev = np.full_like(s_t, -1)
            ok_prev = prev_idx_all >= 0
            s_prev[ok_prev] = labels_frames[prev_idx_all[ok_prev]]

            ok = ok_prev
            s_t_ok = s_t[ok]
            s_prev_ok = s_prev[ok]
            du_ok = du_all[ok]

            # Encode bigram id as int64: b = s_prev * K1 + s_t
            bigram = (s_prev_ok.astype(np.int64) * int(best_k1) + s_t_ok.astype(np.int64))

            def eval_bigrams(assign: np.ndarray, seed_offset: int) -> tuple[float, int, int]:
                uniq, inv = np.unique(assign, return_inverse=True)
                n_groups = int(uniq.size)
                if n_groups == 0:
                    return float("nan"), 0, 0

                order = np.argsort(inv, kind="mergesort")
                inv_sorted = inv[order]
                group_ids = np.arange(n_groups, dtype=np.int32)
                starts = np.searchsorted(inv_sorted, group_ids, side="left")
                ends = np.searchsorted(inv_sorted, group_ids, side="right")

                total_used = 0
                total_weighted = 0.0
                n_groups_used = 0
                for g in range(n_groups):
                    idx = order[starts[g] : ends[g]]
                    n = int(idx.size)
                    if n < max(min_deltas, best_k2):
                        continue
                    centers, labels = _fit_kmeans_unit(
                        du_ok[idx],
                        best_k2,
                        seed=sd + seed_offset + int(g),
                        n_init=kmeans_cfg["n_init"],
                        max_iter=kmeans_cfg["max_iter"],
                        batch_size=kmeans_cfg["batch_size"],
                    )
                    ang = _mean_angle_unit_to_centroid(du_ok[idx], centers, labels)
                    if math.isnan(ang):
                        continue
                    total_weighted += float(ang) * float(n)
                    total_used += n
                    n_groups_used += 1
                if total_used == 0:
                    return float("nan"), 0, 0
                return float(total_weighted / float(total_used)), int(total_used), int(n_groups_used)

            theta_data, n_used, n_groups = eval_bigrams(bigram, seed_offset=10_000)
            rows.append(
                {"k1": best_k1, "k2": best_k2, "seed": sd, "kind": "data", "theta_weighted_mean_deg": theta_data, "n_used": n_used, "n_groups": n_groups}
            )

            # Shuffled null: shuffle frame states then recompute bigrams.
            rng = np.random.default_rng(sd + 777)
            for r in range(repeats):
                perm = labels_frames.copy()
                rng.shuffle(perm)
                s_t_s = perm[src_idx_all]
                s_prev_s = np.full_like(s_t_s, -1)
                s_prev_s[ok_prev] = perm[prev_idx_all[ok_prev]]
                ok_s = ok_prev
                bigram_s = (s_prev_s[ok_s].astype(np.int64) * int(best_k1) + s_t_s[ok_s].astype(np.int64))
                theta_sh, n_used_sh, n_groups_sh = eval_bigrams(bigram_s, seed_offset=20_000 + 37 * r)
                rows.append(
                    {"k1": best_k1, "k2": best_k2, "seed": sd, "repeat": r, "kind": "shuffled", "theta_weighted_mean_deg": theta_sh, "n_used": n_used_sh, "n_groups": n_groups_sh}
                )

            # Random null: geometric baseline per group size.
            # Approximate by matching group sizes from data bigrams.
            uniq, counts = np.unique(bigram, return_counts=True)
            total_weighted = 0.0
            total_used_r = 0
            groups_used = 0
            for i_bi, n in enumerate(counts.tolist()):
                if n < max(min_deltas, best_k2):
                    continue
                x_r = _random_unit_vectors(int(n), du_ok.shape[1], seed=sd + 12345 + i_bi)
                centers, labels = _fit_kmeans_unit(
                    x_r,
                    best_k2,
                    seed=sd + 30000 + i_bi,
                    n_init=kmeans_cfg["n_init"],
                    max_iter=kmeans_cfg["max_iter"],
                    batch_size=kmeans_cfg["batch_size"],
                )
                ang = _mean_angle_unit_to_centroid(x_r, centers, labels)
                total_weighted += float(ang) * float(n)
                total_used_r += int(n)
                groups_used += 1
            theta_rand = float(total_weighted / float(total_used_r)) if total_used_r else float("nan")
            rows.append({"k1": best_k1, "k2": best_k2, "seed": sd, "kind": "random", "theta_weighted_mean_deg": theta_rand, "n_used": total_used_r, "n_groups": groups_used})

        df = pd.DataFrame(rows)
        out_csv = self.out_dir / "tables" / "exp4_bigram.csv"
        df.to_csv(out_csv, index=False)
        self.logger.info(f"[exp12][exp4] Wrote {out_csv}")

        # Summary comparison against unigram (Exp3 best cell).
        unigram_path = self.out_dir / "tables" / "exp3_primary_table_means.csv"
        if unigram_path.exists():
            uni = pd.read_csv(unigram_path)
            uni = uni[(uni["k1"] == best_k1) & (uni["k2"] == best_k2)]
            if not uni.empty:
                uni_row = uni.iloc[0].to_dict()
                summary_rows = []
                for sd in seeds:
                    d_seed = df[(df["seed"] == sd) & (df["kind"] == "data")]["theta_weighted_mean_deg"]
                    s_seed = df[(df["seed"] == sd) & (df["kind"] == "shuffled")]["theta_weighted_mean_deg"]
                    r_seed = df[(df["seed"] == sd) & (df["kind"] == "random")]["theta_weighted_mean_deg"]
                    summary_rows.append(
                        {
                            "k1": best_k1,
                            "k2": best_k2,
                            "seed": int(sd),
                            "theta_bigram_data": float(d_seed.iloc[0]) if len(d_seed) else float("nan"),
                            "theta_bigram_shuffled_mean": float(np.mean(s_seed)) if len(s_seed) else float("nan"),
                            "theta_bigram_random": float(r_seed.iloc[0]) if len(r_seed) else float("nan"),
                            "theta_unigram_data_mean": float(uni_row.get("theta_data", float("nan"))),
                            "theta_unigram_shuffled_mean": float(uni_row.get("theta_shuffled", float("nan"))),
                            "theta_unigram_random_mean": float(uni_row.get("theta_random", float("nan"))),
                        }
                    )
                summary = pd.DataFrame(summary_rows)
                summary_path = self.out_dir / "tables" / "exp4_bigram_summary.csv"
                summary.to_csv(summary_path, index=False)
                self.logger.info(f"[exp12][exp4] Wrote {summary_path}")

    # ---------------------------------------------------------------------
    # Experiment 5: Perceptual validation
    # ---------------------------------------------------------------------
    def _run_exp5_perceptual(self, res3: dict, state_models: dict, cache: dict) -> None:
        cfg5 = self.cfg["exp5_perceptual"]
        seed = int(self.cfg.get("seeds", [int(self.cfg.get("seed", 42))])[0])

        # Choose K1/K2 for perceptual eval.
        # Priority:
        #   1) explicit exp5_perceptual.k1 and exp5_perceptual.k2_values
        #   2) exp3 best config from artifacts or res3
        k1 = int(cfg5.get("k1") or 0)
        k2_values = cfg5.get("k2_values")
        state_seed = int(cfg5.get("state_seed") or seed)

        best_art = self.out_dir / "artifacts" / "exp3_best_config.json"
        best_cfg = None
        if best_art.exists():
            try:
                with open(best_art) as f:
                    best_cfg = json.load(f)
            except Exception:
                best_cfg = None
        if best_cfg is None and res3 and "best" in res3:
            best_cfg = {"best_k1": int(res3["best"]["k1"]), "best_k2": int(res3["best"]["k2"])}

        if k1 <= 0:
            if best_cfg is None:
                self.logger.warning("[exp12][exp5] Skipping: no exp3 best config found and no exp5 k1 override.")
                return
            k1 = int(best_cfg["best_k1"])

        if k2_values is None:
            if best_cfg is None:
                self.logger.warning("[exp12][exp5] Skipping: no exp3 best config found and no exp5 k2_values override.")
                return
            k2_values = [int(best_cfg["best_k2"])]
        k2_list = [int(k) for k in k2_values]
        min_deltas = int(self.cfg["preprocess"].get("min_deltas_per_cluster", 50))
        kmeans_cfg = self.cfg["kmeans"]
        eps = float(cache["near_zero_eps"])

        # Fit codebooks on train utterances only (held-out eval utterances for scoring).
        splits = load_splits(self.cfg["data"]["splits_dir"])
        train_utts = set(splits.train_utterances)
        utt_table: pd.DataFrame = cache["utterances"].sort_values("z_offset").reset_index(drop=True)

        du_full = np.array(cache["delta_unit"][cache["valid_delta_mask"]], dtype=np.float32, copy=True)
        src_full = np.array(cache["delta_src_idx"][cache["valid_delta_mask"]], dtype=np.int32, copy=True)

        # Map each delta's source frame index -> utterance by searching z_offset intervals.
        starts = utt_table["z_offset"].to_numpy(dtype=np.int64)
        ends = (utt_table["z_offset"] + utt_table["n_frames"]).to_numpy(dtype=np.int64)
        utt_ids = utt_table["utterance_id"].astype(str).to_numpy()

        utt_pos = np.searchsorted(starts, src_full.astype(np.int64), side="right") - 1
        utt_pos = np.clip(utt_pos, 0, len(starts) - 1)
        in_range = (src_full.astype(np.int64) >= starts[utt_pos]) & (src_full.astype(np.int64) < ends[utt_pos])
        utt_for_delta = utt_ids[utt_pos]
        is_train = np.isin(utt_for_delta, np.array(sorted(train_utts), dtype=utt_for_delta.dtype)) & in_range

        du = du_full[is_train]
        src = src_full[is_train]
        if du.shape[0] == 0:
            raise RuntimeError("Exp5: no training deltas available after filtering; cannot fit codebooks.")
        if (k1, state_seed) not in state_models:
            raise RuntimeError(
                f"Exp5 requires state model (k1={k1}, seed={state_seed}) but it was not fit. "
                "Run exp2 for that k1/seed or run exp5 without stage filtering."
            )
        labels_frames = state_models[(k1, state_seed)]["labels_frames"]
        s_delta = labels_frames[src]

        # Global direction codebook (train deltas) is shared across K2.
        global_k = int(cfg5.get("global_k", 1024))
        global_centers, _ = _fit_kmeans_unit(
            du,
            global_k,
            seed=seed + 333,
            n_init=kmeans_cfg["n_init"],
            max_iter=kmeans_cfg["max_iter"],
            batch_size=kmeans_cfg["batch_size"],
        )

        # Select held-out utterances: prefer eval utterances from splits and those present in cache.
        eval_utts_avail = [u for u in splits.eval_utterances if u in set(utt_table["utterance_id"].tolist())]
        if not eval_utts_avail:
            eval_utts_avail = utt_table["utterance_id"].tolist()
        eval_utts_avail = sorted(eval_utts_avail)
        n_utts = int(cfg5.get("n_utterances", 20))
        rng = np.random.default_rng(seed + 2025)
        chosen = rng.choice(np.array(eval_utts_avail), size=min(n_utts, len(eval_utts_avail)), replace=False).tolist()

        # Build decoder from exp9 checkpoint (no pretrained downloads required).
        device_str = str(self.cfg.get("train", {}).get("device", "auto"))
        device = _device_from_config(device_str)
        ckpt_path = Path(cfg5["vae_checkpoint"])
        vae = _load_pocket_mimi_vae_from_exp9_checkpoint(ckpt_path, device=device, logger=self.logger)
        vae.eval()

        lat_store = LatentStore(self.cfg["data"]["latents_dir"])
        save_audio = bool(cfg5.get("save_audio", True))
        out_rows = []

        metrics_sr = int(cfg5.get("metrics_sample_rate", 16000))
        out_sr = int(cfg5.get("output_sample_rate", 48000))
        n_mels = int(cfg5.get("mel_n_mels", 80))

        # Prepare per-utterance materialization once (latents + audio).
        utt_payloads = []
        for utt_id in chosen:
            row = utt_table[utt_table["utterance_id"] == utt_id].iloc[0].to_dict()
            audio_path = row["audio_path"]
            z = np.array(lat_store.get_latents(utt_id), dtype=np.float32, copy=True)  # [T, D]
            if z.shape[0] < 2:
                continue
            wav_t, _sr = load_audio(audio_path, target_sr=int(vae.mimi.sample_rate))
            ref_24k = wav_t.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
            ref_24k = np.clip(ref_24k, -1.0, 1.0)
            length = int(ref_24k.shape[0])
            z_off = int(row["z_offset"])
            z_idx = np.arange(z_off, z_off + z.shape[0], dtype=np.int64)
            s_frames = labels_frames[z_idx]
            utt_payloads.append((utt_id, z, s_frames, ref_24k, length))

        # Fit + score for each K2.
        groups = _group_indices_by_label(s_delta, k1)
        for k2 in k2_list:
            self.logger.info(f"[exp12][exp5] Fitting per-state delta codebooks (K1={k1}, K2={k2})...")
            state_centroids = np.zeros((k1, k2, du.shape[1]), dtype=np.float32)
            state_has = np.zeros((k1,), dtype=bool)
            for j in range(k1):
                idx = groups.indices(j)
                if idx.size < max(min_deltas, k2):
                    continue
                centers, _labels = _fit_kmeans_unit(
                    du[idx],
                    k2,
                    seed=seed + 9100 + 97 * k2 + j,
                    n_init=kmeans_cfg["n_init"],
                    max_iter=kmeans_cfg["max_iter"],
                    batch_size=kmeans_cfg["batch_size"],
                )
                state_centroids[j] = centers
                state_has[j] = True

            out_audio_dir = _ensure_dir(self.out_dir / "artifacts" / "exp5_audio" / f"k2_{k2:03d}") if save_audio else None

            for utt_id, z, s_frames, ref_24k, length in utt_payloads:
                def reconstruct(condition: str) -> np.ndarray:
                    z_q = np.zeros_like(z)
                    z_q[0] = z[0]
                    rng_local = np.random.default_rng(seed + 5555)
                    for t in range(1, z.shape[0]):
                        dx = z[t] - z[t - 1]
                        mag = float(np.linalg.norm(dx))
                        if mag < eps:
                            z_q[t] = z_q[t - 1]
                            continue
                        if condition == "state_cond":
                            s = int(s_frames[t - 1])
                            dirs = state_centroids[s] if state_has[s] else global_centers
                            direction = dx / max(mag, 1e-8)
                            k_star = int(np.argmax(dirs @ direction))
                            z_q[t] = z_q[t - 1] + mag * dirs[k_star]
                        elif condition == "global":
                            direction = dx / max(mag, 1e-8)
                            k_star = int(np.argmax(global_centers @ direction))
                            z_q[t] = z_q[t - 1] + mag * global_centers[k_star]
                        elif condition == "random":
                            r = rng_local.standard_normal(size=(z.shape[1],)).astype(np.float32)
                            r = r / float(np.linalg.norm(r) + 1e-8)
                            z_q[t] = z_q[t - 1] + mag * r
                        elif condition == "gt":
                            z_q[t] = z[t]
                        else:
                            raise ValueError(condition)
                    return z_q

                conds = [
                    ("gt", "GroundTruth"),
                    ("state_cond", "StateCondVQ"),
                    ("global", "GlobalVQ"),
                    ("random", "RandomDir"),
                ]

                for cond_key, cond_name in conds:
                    z_use = z if cond_key == "gt" else reconstruct(cond_key)
                    z_torch = torch.from_numpy(z_use.T.copy()).unsqueeze(0).float().to(device)  # [1, D, T]
                    with torch.inference_mode():
                        audio_hat = (
                            vae.decode(z_torch, length=length)
                            .squeeze(0)
                            .squeeze(0)
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                    audio_hat = np.clip(audio_hat[:length], -1.0, 1.0)

                    if save_audio and out_audio_dir is not None:
                        out_wav = _resample_np(audio_hat, int(vae.mimi.sample_rate), out_sr)
                        torchaudio.save(
                            str(out_audio_dir / f"{utt_id}_{cond_name}.wav"),
                            torch.from_numpy(out_wav[None, :]),
                            out_sr,
                        )

                    # Metrics at metrics_sr
                    ref_m = _resample_np(ref_24k, int(vae.mimi.sample_rate), metrics_sr)
                    deg_m = _resample_np(audio_hat, int(vae.mimi.sample_rate), metrics_sr)
                    mlen = min(ref_m.shape[0], deg_m.shape[0])
                    ref_m = np.clip(ref_m[:mlen], -1.0, 1.0)
                    deg_m = np.clip(deg_m[:mlen], -1.0, 1.0)

                    pesq_v = _safe_pesq(ref_m, deg_m, metrics_sr)
                    stoi_v = _safe_stoi(ref_m, deg_m, metrics_sr)
                    mel_v = _mel_l1(
                        torch.from_numpy(ref_m[None, :]).float(),
                        torch.from_numpy(deg_m[None, :]).float(),
                        metrics_sr,
                        n_mels,
                    )

                    out_rows.append(
                        {
                            "utterance_id": utt_id,
                            "k1": int(k1),
                            "k2": int(k2),
                            "condition": cond_name,
                            "pesq": pesq_v,
                            "stoi": stoi_v,
                            "mel_l1": mel_v,
                        }
                    )

                self.logger.info(f"[exp12][exp5] Scored {utt_id} (K2={k2})")

        df = pd.DataFrame(out_rows)
        suffix = "k2sweep" if len(k2_list) > 1 else None
        out_csv = self.out_dir / "tables" / ("exp5_perceptual.csv" if suffix is None else f"exp5_perceptual_{suffix}.csv")
        df.to_csv(out_csv, index=False)
        self.logger.info(f"[exp12][exp5] Wrote {out_csv}")

        summary = df.groupby(["k2", "condition"])[["pesq", "stoi", "mel_l1"]].agg(["mean", "std"]).reset_index()
        summary_path = self.out_dir / "tables" / ("exp5_perceptual_summary.csv" if suffix is None else f"exp5_perceptual_{suffix}_summary.csv")
        summary.to_csv(summary_path, index=False)
        self.logger.info(f"[exp12][exp5] Wrote {summary_path}")

    # ---------------------------------------------------------------------
    # Experiment 6: Scaling diagnostic
    # ---------------------------------------------------------------------
    def _run_exp6_scaling(self) -> None:
        means_path = self.out_dir / "tables" / "exp3_primary_table_means.csv"
        agg_path = self.out_dir / "tables" / "exp3_primary_table.csv"

        if means_path.exists():
            df = pd.read_csv(means_path)
            c_delta = "delta_structure_deg"
            c_data = "theta_data"
            c_shuf = "theta_shuffled"
        elif agg_path.exists():
            # This CSV is written with a 2-row header (columns + mean/std row).
            df_multi = pd.read_csv(agg_path, header=[0, 1])
            # Flatten multiindex columns: (metric, stat)
            df_multi.columns = ["k1" if c[0] == "k1" else "k2" if c[0] == "k2" else f"{c[0]}__{c[1]}" for c in df_multi.columns]
            df = df_multi
            c_delta = "delta_structure_deg__mean"
            c_data = "theta_data__mean"
            c_shuf = "theta_shuffled__mean"
            if c_delta not in df.columns or c_data not in df.columns or c_shuf not in df.columns:
                self.logger.warning("[exp12][exp6] Skipping: exp3 primary table missing expected mean columns.")
                return
        else:
            self.logger.warning("[exp12][exp6] Skipping: exp3 tables not found.")
            return

        df["k1"] = df["k1"].astype(int)
        df["k2"] = df["k2"].astype(int)
        df["k1k2"] = df["k1"] * df["k2"]

        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4))
        plt.plot(df["k1k2"], df[c_data], "o-", label="data")
        plt.plot(df["k1k2"], df[c_shuf], "o-", label="shuffled")
        plt.xlabel("Total entries (K1*K2)")
        plt.ylabel("θ̄_cond (deg)")
        plt.title("Exp6 Scaling diagnostic: θ̄_cond vs K1*K2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.out_dir / "plots" / "exp6_scaling_theta.png")
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(df["k1k2"], df[c_delta], "o-")
        plt.xlabel("Total entries (K1*K2)")
        plt.ylabel("Δ_structure (deg)")
        plt.title("Exp6 Scaling diagnostic: Δ_structure vs K1*K2")
        plt.tight_layout()
        plt.savefig(self.out_dir / "plots" / "exp6_scaling_delta_structure.png")
        plt.close()

        self.logger.info("[exp12][exp6] Wrote scaling plots.")

    # ---------------------------------------------------------------------
    # Reporting
    # ---------------------------------------------------------------------
    def _write_report(self) -> None:
        report = []
        report.append("# Tier4 Exp12 Report\n")
        report.append(f"Generated: {_now_iso()}\n")
        stats_path = _cache_paths(self.out_dir).stats_path
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            report.append("## Dataset\n")
            report.append(f"- Utterances: {stats.get('n_utterances')}\n")
            report.append(f"- Total frames (M): {stats.get('n_frames_total')}\n")
            report.append(f"- Total deltas (N): {stats.get('n_deltas_total')}\n")
            report.append(f"- Latent dim (D): {stats.get('latent_dim')}\n")
            report.append(f"- Near-zero eps: {stats.get('near_zero_eps'):.3e}\n")

        best_path = self.out_dir / "artifacts" / "exp3_best_config.json"
        if best_path.exists():
            with open(best_path) as f:
                best = json.load(f)
            report.append("\n## Best Exp3 Config\n")
            report.append(f"- K1: {best.get('best_k1')}\n")
            report.append(f"- K2: {best.get('best_k2')}\n")
            report.append(f"- Δ_structure (deg): {best.get('best_delta_structure_deg'):.2f}\n")

        report.append("\n## Outputs\n")
        report.append("- Tables: `tables/`\n")
        report.append("- Plots: `plots/`\n")
        report.append("- Artifacts: `artifacts/`\n")
        (self.out_dir / "report.md").write_text("".join(report))


def _load_pocket_mimi_vae_from_exp9_checkpoint(ckpt_path: Path, *, device: torch.device, logger):
    """
    Load PocketMimiVAE from an exp9 checkpoint.

    exp9 checkpoints include a self-contained Mimi state ("mimi_full") so we can
    build the architecture with load_pretrained_mimi=False (no downloads) and
    then load weights strictly.
    """
    ckpt = torch.load(str(ckpt_path), map_location=device)
    latent_dim = int(ckpt.get("latent_dim", 32))
    dec_hidden_dim = int(ckpt.get("dec_hidden_dim", 256))

    vae = build_pocket_mimi_vae(
        latent_dim=latent_dim,
        dec_hidden_dim=dec_hidden_dim,
        freeze_encoder=False,
        freeze_decoder=False,
        device=device,
        allow_partial_pretrained_load=False,
        load_pretrained_mimi=False,
    )
    if "mimi_full" in ckpt:
        vae.mimi.load_state_dict(ckpt["mimi_full"], strict=True)
    else:
        raise KeyError("Exp9 checkpoint missing 'mimi_full'; cannot load offline.")
    vae.mu_proj.load_state_dict(ckpt["vae_bottleneck"]["mu_proj"], strict=True)
    vae.logvar_proj.load_state_dict(ckpt["vae_bottleneck"]["logvar_proj"], strict=True)
    vae.dec_proj.load_state_dict(ckpt["vae_bottleneck"]["dec_proj"], strict=True)
    logger.info(f"[exp12][exp5] Loaded PocketMimiVAE from {ckpt_path}")
    return vae
