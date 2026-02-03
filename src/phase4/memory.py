from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class KMeansDeltaMemory:
    """
    Engram-style memory: nearest-centroid lookup for Δz statistics.

    Stores:
      - centroids: [K, D]
      - dz_mean:   [K, D]
      - dz_var:    [K, D]  (diagonal variance)
      - counts:    [K]
    """

    centroids: torch.Tensor
    dz_mean: torch.Tensor
    dz_var: torch.Tensor
    counts: torch.Tensor

    @property
    def n_clusters(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def dim(self) -> int:
        return int(self.centroids.shape[1])

    @staticmethod
    def load(path: str | Path, *, device: torch.device) -> "KMeansDeltaMemory":
        d = np.load(str(path))
        centroids = torch.from_numpy(d["centroids"]).to(device=device, dtype=torch.float32)
        dz_mean = torch.from_numpy(d["dz_mean"]).to(device=device, dtype=torch.float32)
        dz_var = torch.from_numpy(d["dz_var"]).to(device=device, dtype=torch.float32)
        counts = torch.from_numpy(d["counts"]).to(device=device, dtype=torch.int64)
        return KMeansDeltaMemory(centroids=centroids, dz_mean=dz_mean, dz_var=dz_var, counts=counts)

    def nearest_index(self, z_prev: torch.Tensor) -> torch.Tensor:
        """
        z_prev: [B,D] -> idx: [B] int64
        """
        # Squared L2 distance via ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        z2 = (z_prev * z_prev).sum(dim=-1, keepdim=True)  # [B,1]
        c2 = (self.centroids * self.centroids).sum(dim=-1).unsqueeze(0)  # [1,K]
        sim = z_prev @ self.centroids.t()  # [B,K]
        dist2 = z2 + c2 - 2.0 * sim
        return torch.argmin(dist2, dim=-1)

    def predict_mean(self, z_prev: torch.Tensor) -> torch.Tensor:
        idx = self.nearest_index(z_prev)
        return self.dz_mean.index_select(0, idx)

    def predict_var(self, z_prev: torch.Tensor) -> torch.Tensor:
        idx = self.nearest_index(z_prev)
        return self.dz_var.index_select(0, idx)

