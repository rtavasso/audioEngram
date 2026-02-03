#!/usr/bin/env python3
"""
Phase 4: Fit an Engram-style memory on exported z_dyn transitions.

Fits MiniBatchKMeans on z_prev keys, then computes per-centroid Δz mean/var.

Usage:
  uv run python scripts/30_phase4_fit_memory.py --config configs/phase4.yaml
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import yaml

from phase0.utils.logging import setup_logging
from phase4.data import iter_zdyn_pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit Phase 4 k-means memory over z_dyn transitions")
    parser.add_argument("--config", type=str, default="configs/phase4.yaml")
    args = parser.parse_args()

    logger = setup_logging(name="phase4-memory")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_npz = Path(cfg["memory"]["output_npz"])
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    max_pairs = int(cfg["memory"]["max_fit_pairs"])
    logger.info(f"[phase4] Collecting up to {max_pairs} train pairs for k-means fit")

    z_list: list[np.ndarray] = []
    dz_list: list[np.ndarray] = []
    for p in iter_zdyn_pairs(
        zdyn_dir=cfg["data"]["zdyn_dir"],
        zdyn_index_path=cfg["data"]["zdyn_index"],
        splits_dir=cfg["data"]["splits_dir"],
        split="train",
        min_duration_sec=float(cfg["data"]["min_duration_sec"]),
        seed=int(cfg["memory"]["seed"]),
        max_pairs=max_pairs,
        sample_prob=1.0,
    ):
        z_list.append(p.z_prev)
        dz_list.append(p.dz)

    if not z_list:
        raise RuntimeError("No training pairs found. Did you export z_dyn with --split all?")

    z = np.stack(z_list, axis=0).astype(np.float32, copy=False)
    dz = np.stack(dz_list, axis=0).astype(np.float32, copy=False)
    logger.info(f"[phase4] Fit arrays: z_prev={z.shape} dz={dz.shape}")

    from sklearn.cluster import MiniBatchKMeans

    n_clusters = int(cfg["memory"]["n_clusters"])
    mb = int(cfg["memory"]["minibatch_size"])
    seed = int(cfg["memory"]["seed"])

    logger.info(f"[phase4] Fitting MiniBatchKMeans: K={n_clusters} minibatch={mb}")
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=mb,
        random_state=seed,
        n_init="auto",
        reassignment_ratio=0.01,
    )
    km.fit(z)
    centroids = km.cluster_centers_.astype(np.float32, copy=False)  # [K,D]

    # Compute per-centroid mean/var for dz
    labels = km.predict(z)
    k = int(n_clusters)
    d = int(dz.shape[1])

    counts = np.zeros((k,), dtype=np.int64)
    sum_dz = np.zeros((k, d), dtype=np.float64)
    sum_dz2 = np.zeros((k, d), dtype=np.float64)
    for i in range(z.shape[0]):
        lab = int(labels[i])
        counts[lab] += 1
        v = dz[i].astype(np.float64, copy=False)
        sum_dz[lab] += v
        sum_dz2[lab] += v * v

    eps = 1e-8
    dz_mean = (sum_dz / np.maximum(counts[:, None], 1)).astype(np.float32, copy=False)
    dz_var = (sum_dz2 / np.maximum(counts[:, None], 1) - (dz_mean.astype(np.float64) ** 2)).astype(np.float32, copy=False)
    dz_var = np.maximum(dz_var, eps).astype(np.float32, copy=False)

    n_empty = int((counts == 0).sum())
    if n_empty > 0:
        logger.warning(f"[phase4] {n_empty}/{k} empty clusters; filling with global dz stats")
        global_mean = dz.mean(axis=0)
        global_var = dz.var(axis=0) + eps
        empty = counts == 0
        dz_mean[empty] = global_mean
        dz_var[empty] = global_var

    np.savez_compressed(
        str(out_npz),
        centroids=centroids,
        dz_mean=dz_mean,
        dz_var=dz_var,
        counts=counts.astype(np.int64),
    )
    logger.info(f"[phase4] Wrote memory npz: {out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

