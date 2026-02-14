"""
Lightweight experiment tracking.

Provides run registration, provenance capture, and manifest logging
so every experiment run is reproducible and discoverable.

Usage in experiment scripts:

    from experiment import register_run, finalize_run

    run = register_run(
        experiment="exp1_vmf", run_id=run_id, config_path=args.config,
        config=cfg, cli_args=sys.argv[1:], out_dir=out_dir,
    )
    # ... experiment code ...
    finalize_run(run, key_metrics={"eval_dnll": -2.12})
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Manifest path — shared across all experiments
# ---------------------------------------------------------------------------

_MANIFEST_PATH = Path(__file__).parent.parent / "outputs" / "manifest.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_git_commit() -> str:
    """Return short git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_manifest(record: dict) -> None:
    """Append a single JSON line to the manifest file."""
    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_MANIFEST_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# RunInfo dataclass
# ---------------------------------------------------------------------------


@dataclass
class RunInfo:
    experiment: str
    run_id: str
    config_path: str
    out_dir: Path
    git_commit: str
    start_time: str
    log_handler: Optional[logging.FileHandler] = field(default=None, repr=False)
    log_name: str = "phase0"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_run(
    *,
    experiment: str,
    run_id: str,
    config_path: str,
    config: dict,
    cli_args: list[str],
    out_dir: Path,
    log_name: str = "phase0",
) -> RunInfo:
    """
    Register a new experiment run. Call at script start.

    Creates:
      - out_dir/config.yaml   (snapshot of the config used)
      - out_dir/run_info.json  (provenance metadata)
      - out_dir/run.log        (file handler added to logger)
      - outputs/manifest.jsonl (append "started" line)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    git_commit = get_git_commit()
    start_time = _now_iso()

    # Capture environment
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_device = torch.cuda.get_device_name(0) if cuda_available else None
    except Exception:
        torch_version = "unknown"
        cuda_available = False
        cuda_device = None

    env_info = {
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
        "platform": platform.platform(),
    }

    # Save config snapshot
    config_snapshot_path = out_dir / "config.yaml"
    with open(config_snapshot_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Save run info
    run_info_dict = {
        "experiment": experiment,
        "run_id": run_id,
        "config_path": config_path,
        "cli_args": cli_args,
        "git_commit": git_commit,
        "start_time": start_time,
        "status": "started",
        "environment": env_info,
        "output_dir": str(out_dir),
    }
    run_info_path = out_dir / "run_info.json"
    with open(run_info_path, "w") as f:
        json.dump(run_info_dict, f, indent=2)

    # Add file handler to logger
    log_handler = None
    try:
        logger = logging.getLogger(log_name)
        log_path = out_dir / "run.log"
        fh = logging.FileHandler(str(log_path))
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)
        log_handler = fh
    except Exception:
        pass

    # Append to manifest
    _append_manifest({
        "experiment": experiment,
        "run_id": run_id,
        "status": "started",
        "timestamp": start_time,
        "git_commit": git_commit,
        "config_path": config_path,
        "output_dir": str(out_dir),
    })

    logger = logging.getLogger(log_name)
    logger.info(
        f"[tracking] Registered run: experiment={experiment} run_id={run_id} "
        f"git={git_commit} out={out_dir}"
    )

    return RunInfo(
        experiment=experiment,
        run_id=run_id,
        config_path=config_path,
        out_dir=out_dir,
        git_commit=git_commit,
        start_time=start_time,
        log_handler=log_handler,
        log_name=log_name,
    )


def finalize_run(
    run: RunInfo,
    *,
    status: str = "completed",
    key_metrics: Optional[dict] = None,
) -> None:
    """
    Finalize an experiment run. Call at script end.

    Updates:
      - out_dir/run_info.json  (adds end_time, duration, status, key_metrics)
      - outputs/manifest.jsonl (append "completed" line)
    """
    end_time = _now_iso()

    # Compute duration
    try:
        start_dt = datetime.fromisoformat(run.start_time)
        end_dt = datetime.fromisoformat(end_time)
        duration_sec = (end_dt - start_dt).total_seconds()
    except Exception:
        duration_sec = None

    # Update run_info.json
    run_info_path = run.out_dir / "run_info.json"
    try:
        with open(run_info_path) as f:
            run_info_dict = json.load(f)
    except Exception:
        run_info_dict = {}

    run_info_dict["status"] = status
    run_info_dict["end_time"] = end_time
    run_info_dict["duration_sec"] = duration_sec
    run_info_dict["key_metrics"] = key_metrics

    with open(run_info_path, "w") as f:
        json.dump(run_info_dict, f, indent=2)

    # Append to manifest
    manifest_record = {
        "experiment": run.experiment,
        "run_id": run.run_id,
        "status": status,
        "timestamp": end_time,
        "duration_sec": duration_sec,
        "git_commit": run.git_commit,
        "output_dir": str(run.out_dir),
    }
    if key_metrics:
        manifest_record["key_metrics"] = key_metrics
    _append_manifest(manifest_record)

    # Log summary
    logger = logging.getLogger(run.log_name)
    metrics_str = ""
    if key_metrics:
        metrics_str = " ".join(f"{k}={v}" for k, v in key_metrics.items())
    logger.info(
        f"[tracking] Finalized run: experiment={run.experiment} run_id={run.run_id} "
        f"status={status} duration={duration_sec:.0f}s {metrics_str}"
    )

    # Clean up file handler
    if run.log_handler is not None:
        try:
            logger.removeHandler(run.log_handler)
            run.log_handler.close()
        except Exception:
            pass
