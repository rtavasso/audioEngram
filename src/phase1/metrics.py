"""
Phase 1 evaluation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class MetricAgg:
    n: int = 0
    nll_sum: float = 0.0
    nll_baseline_sum: float = 0.0
    cos_sum: float = 0.0
    logmag_mse_sum: float = 0.0
    logmag_true_sum: float = 0.0
    logmag_true_sq_sum: float = 0.0

    def update(
        self,
        *,
        nll: torch.Tensor,
        nll_baseline: torch.Tensor,
        pred_mean: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-8,
    ) -> None:
        # All are [B] or [B,D]
        b = int(nll.shape[0])
        self.n += b
        self.nll_sum += float(nll.sum().item())
        self.nll_baseline_sum += float(nll_baseline.sum().item())

        # Direction cosine
        p = pred_mean
        y = target
        p_norm = torch.linalg.vector_norm(p, dim=-1).clamp_min(eps)
        y_norm = torch.linalg.vector_norm(y, dim=-1).clamp_min(eps)
        cos = (p * y).sum(dim=-1) / (p_norm * y_norm)
        self.cos_sum += float(cos.sum().item())

        # log-magnitude MSE + R2 ingredients
        logm_y = torch.log(y_norm)
        logm_p = torch.log(p_norm)
        err = (logm_p - logm_y) ** 2
        self.logmag_mse_sum += float(err.sum().item())
        self.logmag_true_sum += float(logm_y.sum().item())
        self.logmag_true_sq_sum += float((logm_y * logm_y).sum().item())

    def finalize(self, *, dim: int) -> dict:
        if self.n == 0:
            return {
                "n": 0,
                "nll": float("nan"),
                "nll_per_dim": float("nan"),
                "nll_baseline": float("nan"),
                "nll_baseline_per_dim": float("nan"),
                "dnll": float("nan"),
                "dnll_per_dim": float("nan"),
                "direction_cosine": float("nan"),
                "logmag_mse": float("nan"),
                "logmag_r2": float("nan"),
            }

        n = float(self.n)
        nll = self.nll_sum / n
        nll_b = self.nll_baseline_sum / n
        dnll = nll - nll_b
        cos = self.cos_sum / n
        logmag_mse = self.logmag_mse_sum / n

        # R2 for log magnitude
        mean_y = self.logmag_true_sum / n
        ss_tot = self.logmag_true_sq_sum - n * (mean_y * mean_y)
        ss_res = self.logmag_mse_sum
        logmag_r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else float("nan")

        return {
            "n": int(self.n),
            "nll": float(nll),
            "nll_per_dim": float(nll / dim),
            "nll_baseline": float(nll_b),
            "nll_baseline_per_dim": float(nll_b / dim),
            "dnll": float(dnll),
            "dnll_per_dim": float(dnll / dim),
            "direction_cosine": float(cos),
            "logmag_mse": float(logmag_mse),
            "logmag_r2": float(logmag_r2),
        }

