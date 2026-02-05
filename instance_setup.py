#!/usr/bin/env python3
"""
Static one-command setup for this project.

Run:
  python3 instance_setup.py

What it does (idempotent-ish):
  - clones the repo (if not already inside it)
  - initializes submodules
  - installs uv (if missing), creates venv, installs deps
  - installs moshi editable
  - downloads LibriSpeech train-clean-100 (if missing)
  - creates speaker splits
  - extracts Mimi latents
  - builds the Phase 0 frames index
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


# -------- project constants (keep this file "static") --------
REPO_URL = "https://github.com/rtavasso/audioEngram.git"
REPO_DIRNAME = "audioEngram"

PHASE0_CONFIG = "configs/phase0.yaml"
DATA_DIR = "./data"

# Mimi latent extraction defaults (matches the original notebook snippet)
INFER_DEVICE_PREFERRED = "cuda"
INFER_BATCH_SIZE = 16
INFER_NUM_WORKERS = 4
INFER_PREFETCH_FACTOR = 2
INFER_AMP_DTYPE = "fp16"  # "bf16" if your GPU supports it


def _run(cmd: list[str], *, cwd: Path) -> None:
    print(f"[instance_setup] $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _run_shell(script: str, *, cwd: Path) -> None:
    print(f"[instance_setup] $ bash -lc {script!r}")
    subprocess.run(["bash", "-lc", script], cwd=str(cwd), check=True)


def _is_repo_root(path: Path) -> bool:
    return (path / "pyproject.toml").exists() and (path / "scripts").is_dir() and (path / "src").is_dir()


def _ensure_uv_available(*, cwd: Path) -> None:
    if shutil.which("uv") is not None:
        return
    # Install uv into the current user's home (~/.local/bin).
    _run_shell("curl -LsSf https://astral.sh/uv/install.sh | sh", cwd=cwd)
    os.environ["PATH"] = f"{Path.home() / '.local' / 'bin'}:{os.environ.get('PATH','')}"
    if shutil.which("uv") is None:
        raise SystemExit("uv install completed but `uv` is still not on PATH. Open a new shell and re-run.")


def _infer_device() -> str:
    # Avoid importing torch here (deps might not be installed yet).
    if shutil.which("nvidia-smi") is not None:
        return INFER_DEVICE_PREFERRED
    return "cpu"


def main() -> int:
    here = Path.cwd().resolve()

    # Clone if we're not already inside the repo.
    repo_root: Path
    if _is_repo_root(here):
        repo_root = here
    elif _is_repo_root(here / REPO_DIRNAME):
        repo_root = (here / REPO_DIRNAME).resolve()
    else:
        repo_root = (here / REPO_DIRNAME).resolve()
        if not repo_root.exists():
            _run(["git", "clone", "--recurse-submodules", REPO_URL, str(repo_root)], cwd=here)

    # Submodules (harmless if already init'd).
    _run(["git", "submodule", "update", "--init", "--recursive"], cwd=repo_root)

    # Install uv if needed; then venv + deps.
    _ensure_uv_available(cwd=repo_root)
    _run(["uv", "venv"], cwd=repo_root)
    _run(["uv", "sync"], cwd=repo_root)
    _run(["uv", "pip", "install", "-e", "./moshi/moshi", "--python", ".venv/bin/python"], cwd=repo_root)

    # Sanity check.
    _run(["uv", "run", "python", "-c", "import torch; print('CUDA available:', torch.cuda.is_available())"], cwd=repo_root)

    # Data download + Phase 0 artifacts
    _run(["chmod", "+x", "./scripts/00_download_librispeech.sh"], cwd=repo_root)
    _run(["bash", "./scripts/00_download_librispeech.sh", DATA_DIR], cwd=repo_root)

    _run(["uv", "run", "python", "scripts/01_make_speaker_splits.py", "--config", PHASE0_CONFIG], cwd=repo_root)

    device = _infer_device()
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/02_infer_latents.py",
            "--config",
            PHASE0_CONFIG,
            "--device",
            device,
            "--batch-size",
            str(INFER_BATCH_SIZE),
            "--num-workers",
            str(INFER_NUM_WORKERS),
            "--prefetch-factor",
            str(INFER_PREFETCH_FACTOR),
            "--amp",
            "--amp-dtype",
            INFER_AMP_DTYPE,
            "--tf32",
            "--verify-mode",
            "sample",
            "--verify-max-utts",
            "200",
        ],
        cwd=repo_root,
    )

    _run(["uv", "run", "python", "scripts/03_build_phase0_dataset.py", "--config", PHASE0_CONFIG], cwd=repo_root)

    print("[instance_setup] Done. You can now run Tier1 scripts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

