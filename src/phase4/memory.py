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
        if "dz_mean" in d and "dz_var" in d:
            dz_mean_np = d["dz_mean"]
            dz_var_np = d["dz_var"]
        elif "res_mean" in d and "res_var" in d:
            dz_mean_np = d["res_mean"]
            dz_var_np = d["res_var"]
        elif "value_mean" in d and "value_var" in d:
            dz_mean_np = d["value_mean"]
            dz_var_np = d["value_var"]
        else:
            raise KeyError("Expected dz_mean/dz_var or res_mean/res_var or value_mean/value_var in memory npz")

        dz_mean = torch.from_numpy(dz_mean_np).to(device=device, dtype=torch.float32)
        dz_var = torch.from_numpy(dz_var_np).to(device=device, dtype=torch.float32)
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

    def topk_indices_and_weights(self, z_prev: torch.Tensor, *, topk: int, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          idx:     [B,K] int64
          weights: [B,K] float32 (sum to 1)
        """
        k = int(topk)
        if k <= 0:
            raise ValueError("topk must be > 0")
        k = min(k, self.n_clusters)
        temp = float(temperature)
        if temp <= 0:
            raise ValueError("temperature must be > 0")

        z2 = (z_prev * z_prev).sum(dim=-1, keepdim=True)  # [B,1]
        c2 = (self.centroids * self.centroids).sum(dim=-1).unsqueeze(0)  # [1,N]
        sim = z_prev @ self.centroids.t()  # [B,N]
        dist2 = z2 + c2 - 2.0 * sim

        vals, idx = torch.topk(dist2, k=k, dim=-1, largest=False)
        logits = -vals / temp
        weights = torch.softmax(logits, dim=-1).to(dtype=torch.float32)
        return idx.to(dtype=torch.int64), weights

    def predict_mean(self, z_prev: torch.Tensor, *, topk: int = 1, temperature: float = 1.0) -> torch.Tensor:
        idx, w = self.topk_indices_and_weights(z_prev, topk=topk, temperature=temperature)  # [B,K], [B,K]
        means = self.dz_mean.index_select(0, idx.reshape(-1)).reshape(z_prev.shape[0], -1, self.dim)  # [B,K,D]
        return (means * w.unsqueeze(-1)).sum(dim=1)

    def predict_var(self, z_prev: torch.Tensor, *, topk: int = 1, temperature: float = 1.0) -> torch.Tensor:
        """
        Weighted diagonal variance (mixture moment match): E[var] + E[mu^2] - (E[mu])^2.
        """
        idx, w = self.topk_indices_and_weights(z_prev, topk=topk, temperature=temperature)
        means = self.dz_mean.index_select(0, idx.reshape(-1)).reshape(z_prev.shape[0], -1, self.dim)  # [B,K,D]
        vars_ = self.dz_var.index_select(0, idx.reshape(-1)).reshape(z_prev.shape[0], -1, self.dim)  # [B,K,D]
        m1 = (means * w.unsqueeze(-1)).sum(dim=1)  # [B,D]
        m2 = ((vars_ + means * means) * w.unsqueeze(-1)).sum(dim=1)  # [B,D]
        var = m2 - m1 * m1
        return var.clamp_min(1e-8)
