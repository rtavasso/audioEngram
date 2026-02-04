"""
Phase 1 checkpoint load/save helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .mdn import MDN
from .vmf import VmfLogNormal


def load_phase1_checkpoint(path: str | Path, *, device: torch.device) -> tuple[object, dict[str, Any]]:
    ckpt = torch.load(str(path), map_location=device)
    model_type = str(ckpt.get("model_type", "mdn")).lower().strip()
    model_kwargs = ckpt.get("model_kwargs")
    if not isinstance(model_kwargs, dict):
        raise ValueError(
            "Checkpoint is missing model_kwargs (older format). "
            "Re-train with the updated code or provide a checkpoint that includes model_kwargs."
        )

    if model_type == "mdn":
        model = MDN(**model_kwargs)
    elif model_type == "vmf":
        model = VmfLogNormal(**model_kwargs)
    else:
        raise ValueError(f"Unknown model_type in checkpoint: {model_type}")

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt

