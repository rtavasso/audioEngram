#!/usr/bin/env python3
"""
Collect experiment results from the manifest into consolidated tables.

Reads outputs/manifest.jsonl and produces:
  - outputs/results_table.csv   (for analysis)
  - outputs/results_table.md    (for paper drafts)
  - Console summary

Usage:
  uv run python scripts/collect_results.py                          # all runs
  uv run python scripts/collect_results.py --experiment exp1_vmf    # filter
  uv run python scripts/collect_results.py --latest                 # most recent per experiment
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd

MANIFEST_PATH = project_root / "outputs" / "manifest.jsonl"
OUTPUT_DIR = project_root / "outputs"


def load_manifest() -> list[dict]:
    """Load all lines from manifest.jsonl."""
    if not MANIFEST_PATH.exists():
        return []
    records = []
    with open(MANIFEST_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def build_runs_table(records: list[dict]) -> pd.DataFrame:
    """
    Build a table of completed runs from manifest records.

    Each completed run produces one row. Key metrics are flattened into columns.
    """
    # Group by (experiment, run_id) and take the latest record
    completed = {}
    for r in records:
        if r.get("status") != "completed":
            continue
        key = (r.get("experiment", ""), r.get("run_id", ""))
        completed[key] = r

    rows = []
    for (experiment, run_id), r in sorted(completed.items()):
        row = {
            "experiment": experiment,
            "run_id": run_id,
            "timestamp": r.get("timestamp", ""),
            "duration_sec": r.get("duration_sec"),
            "git_commit": r.get("git_commit", ""),
            "output_dir": r.get("output_dir", ""),
        }
        # Flatten key_metrics
        km = r.get("key_metrics") or {}
        for k, v in km.items():
            row[k] = v
        rows.append(row)

    return pd.DataFrame(rows)


def find_crashed_runs(records: list[dict]) -> list[dict]:
    """Find runs that started but never completed."""
    started = {}
    completed = set()
    for r in records:
        key = (r.get("experiment", ""), r.get("run_id", ""))
        if r.get("status") == "started":
            started[key] = r
        elif r.get("status") == "completed":
            completed.add(key)

    return [r for key, r in started.items() if key not in completed]


def df_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table string."""
    if df.empty:
        return "(no results)\n"

    # Format numeric columns
    formatted = df.copy()
    for col in formatted.columns:
        if formatted[col].dtype in ("float64", "float32"):
            formatted[col] = formatted[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else ""
            )

    lines = []
    headers = list(formatted.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in formatted.iterrows():
        cells = [str(row[h]) if pd.notna(row[h]) else "" for h in headers]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Collect experiment results")
    p.add_argument("--experiment", type=str, default=None, help="Filter by experiment name")
    p.add_argument("--latest", action="store_true", help="Show only most recent run per experiment")
    p.add_argument("--no-save", action="store_true", help="Only print, don't save files")
    args = p.parse_args()

    records = load_manifest()
    if not records:
        print(f"No manifest found at {MANIFEST_PATH}")
        print("Run an experiment first to populate the manifest.")
        return 0

    df = build_runs_table(records)
    if df.empty:
        print("No completed runs found in manifest.")
        crashed = find_crashed_runs(records)
        if crashed:
            print(f"\nFound {len(crashed)} started-but-not-completed runs:")
            for r in crashed:
                print(f"  {r.get('experiment')} / {r.get('run_id')} — started {r.get('timestamp')}")
        return 0

    # Filter
    if args.experiment:
        df = df[df["experiment"] == args.experiment]
        if df.empty:
            print(f"No completed runs for experiment '{args.experiment}'")
            return 0

    # Latest only
    if args.latest:
        df = df.sort_values("timestamp").groupby("experiment").tail(1).reset_index(drop=True)

    # Console output
    print("\n=== Experiment Results ===\n")
    # Show a clean summary without output_dir to fit terminal
    display_cols = [c for c in df.columns if c != "output_dir"]
    print(df[display_cols].to_string(index=False))

    # Crashed runs
    crashed = find_crashed_runs(records)
    if crashed:
        print(f"\n{len(crashed)} incomplete run(s):")
        for r in crashed:
            print(f"  {r.get('experiment')} / {r.get('run_id')} — started {r.get('timestamp')}")

    # Save
    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        csv_path = OUTPUT_DIR / "results_table.csv"
        df.to_csv(str(csv_path), index=False)
        print(f"\nWrote: {csv_path}")

        md_path = OUTPUT_DIR / "results_table.md"
        md_content = "# Experiment Results\n\n"
        md_content += f"Generated: {pd.Timestamp.now().isoformat(timespec='seconds')}\n\n"

        # Group by experiment for readability
        for exp_name, exp_df in df.groupby("experiment"):
            md_content += f"## {exp_name}\n\n"
            md_content += df_to_markdown(exp_df)
            md_content += "\n"

        with open(md_path, "w") as f:
            f.write(md_content)
        print(f"Wrote: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
