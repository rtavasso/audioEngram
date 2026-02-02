"""
Debug tooling for Phase 1 metric discrepancies.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch

from .data import Phase1Sample
from .mdn import MDN
from .stats import DiagGaussianBaseline


@dataclass(frozen=True)
class WorstSample:
    dnll: float
    nll: float
    nll_baseline: float
    utterance_id: str
    speaker_id: int
    t: int
    delta_l2: float
    context_l2: float


def _l2(x: np.ndarray) -> float:
    return float(np.sqrt(np.sum(x.astype(np.float64) ** 2)))


@torch.no_grad()
def eval_stream_with_debug(
    *,
    model: MDN,
    baseline: DiagGaussianBaseline,
    samples: Iterable[Phase1Sample],
    device: torch.device,
    max_samples: Optional[int] = None,
    reservoir_size: int = 50000,
    top_worst: int = 50,
) -> dict:
    """
    Evaluate NLL metrics on a raw sample stream while collecting:
    - approximate quantiles via reservoir sampling
    - top outliers by dnll (conditional - baseline)
    """
    model.eval()
    rng = np.random.default_rng(0)

    n = 0
    nll_sum = 0.0
    nll_b_sum = 0.0

    reservoir_nll: list[float] = []
    reservoir_dnll: list[float] = []

    worst_heap: list[tuple[float, WorstSample]] = []  # min-heap by dnll

    for s in samples:
        ctx = torch.from_numpy(s.context_flat).unsqueeze(0).to(device)
        dx = torch.from_numpy(s.delta).unsqueeze(0).to(device)
        nll = float(model.nll(ctx, dx).item())
        nll_b = float(baseline.nll(dx).item())
        dnll = nll - nll_b

        n += 1
        nll_sum += nll
        nll_b_sum += nll_b

        # Reservoir sample
        if len(reservoir_nll) < reservoir_size:
            reservoir_nll.append(nll)
            reservoir_dnll.append(dnll)
        else:
            j = int(rng.integers(0, n))
            if j < reservoir_size:
                reservoir_nll[j] = nll
                reservoir_dnll[j] = dnll

        # Worst samples by dnll
        ws = WorstSample(
            dnll=dnll,
            nll=nll,
            nll_baseline=nll_b,
            utterance_id=s.utterance_id,
            speaker_id=s.speaker_id,
            t=s.t,
            delta_l2=_l2(s.delta),
            context_l2=_l2(s.context_flat),
        )
        if len(worst_heap) < top_worst:
            heapq.heappush(worst_heap, (dnll, ws))
        else:
            if dnll > worst_heap[0][0]:
                heapq.heapreplace(worst_heap, (dnll, ws))

        if max_samples is not None and n >= int(max_samples):
            break

    if n == 0:
        return {"n": 0}

    def q(arr: list[float], ps=(0.5, 0.9, 0.99, 0.999)) -> dict:
        a = np.array(arr, dtype=np.float64)
        return {f"p{int(p*1000):03d}": float(np.quantile(a, p)) for p in ps}

    worst = [ws for _, ws in sorted(worst_heap, key=lambda x: x[0], reverse=True)]
    dim = model.output_dim
    nll_mean = nll_sum / n
    nll_b_mean = nll_b_sum / n

    return {
        "n": int(n),
        "nll_mean": float(nll_mean),
        "nll_mean_per_dim": float(nll_mean / dim),
        "nll_baseline_mean": float(nll_b_mean),
        "nll_baseline_mean_per_dim": float(nll_b_mean / dim),
        "dnll_mean": float(nll_mean - nll_b_mean),
        "dnll_mean_per_dim": float((nll_mean - nll_b_mean) / dim),
        "nll_quantiles": q(reservoir_nll),
        "dnll_quantiles": q(reservoir_dnll),
        "worst_by_dnll": [ws.__dict__ for ws in worst],
    }

