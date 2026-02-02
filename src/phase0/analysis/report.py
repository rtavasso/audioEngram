"""
Report generation for Phase 0 analysis.

Creates decision matrix and summary paragraph.

Decision criteria (from SPEC):
- Pass if ALL:
  - Variance ratio < 0.6 on BOTH train and eval (held-out speakers)
  - Cross-speaker degradation < 20% (eval worse than train)
  - Structure persists across confound slices
  - Random baseline ≈ 1.0 (failure if deviation > tolerance)
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd


def make_decision(
    metrics: dict | list,
    variance_ratio_threshold: float = 0.6,
    cross_speaker_degradation_max: float = 0.20,
    random_baseline_tolerance: float = 0.05,
) -> dict:
    """
    Make pass/fail decision based on metrics.

    Decision criteria (from SPEC):
    - Pass if ALL:
      - Variance ratio < 0.6 on BOTH train and eval (critical: eval tests generalization)
      - Cross-speaker degradation < 20% (only penalize if eval is WORSE, not better)
      - Structure persists across confound checks
      - Random baseline ≈ 1.0 (FAIL if deviation > tolerance, not just warning)

    - Fail if ANY:
      - Variance ratio > threshold on train OR eval
      - Eval variance ratio > train + degradation_max (worse on held-out)
      - Structure disappears when confounds controlled
      - Random baseline deviates significantly from 1.0

    Args:
        metrics: List of result dicts from run_full_analysis
        variance_ratio_threshold: Target variance ratio (default 0.6)
        cross_speaker_degradation_max: Max acceptable degradation (default 0.20)
        random_baseline_tolerance: Tolerance for random baseline from 1.0

    Returns:
        Decision dict with pass/fail and reasoning
    """
    if isinstance(metrics, dict):
        metrics = metrics.get("results", [])

    decision = {
        "pass": False,
        "reasons": [],
        "warnings": [],
        "best_condition": None,
        "summary": {},
    }

    if not metrics:
        decision["reasons"].append("No metrics available")
        return decision

    # Group by condition
    by_condition = {}
    for m in metrics:
        cond = m["condition"]
        if cond not in by_condition:
            by_condition[cond] = []
        by_condition[cond].append(m)

    # Evaluate each condition
    passing_conditions = []

    for cond_name, cond_metrics in by_condition.items():
        cond_pass = True
        cond_reasons = []

        # Check variance ratio (use "all" slice, lag=1 as primary)
        primary = [m for m in cond_metrics if m["slice"] == "all" and m["lag"] == 1]
        if primary:
            p = primary[0]
            train_vr = p["train"]["variance_ratio"]
            eval_vr = p["eval"]["variance_ratio"]

            # Check BOTH train and eval variance ratios (eval is critical for generalization)
            if train_vr > variance_ratio_threshold:
                cond_pass = False
                cond_reasons.append(
                    f"Train variance ratio {train_vr:.3f} > {variance_ratio_threshold}"
                )

            if eval_vr > variance_ratio_threshold:
                cond_pass = False
                cond_reasons.append(
                    f"Eval variance ratio {eval_vr:.3f} > {variance_ratio_threshold}"
                )

            # Cross-speaker degradation: only penalize if eval is WORSE (positive degradation)
            # Negative degradation means eval is better, which is fine
            degradation = p.get("cross_speaker_degradation", 0)
            if degradation > cross_speaker_degradation_max:
                cond_pass = False
                cond_reasons.append(
                    f"Cross-speaker degradation {degradation:.3f} > {cross_speaker_degradation_max}"
                )

            # Random baseline check - FAIL if not ~1.0 (not just a warning)
            # This is a critical sanity check per SPEC acceptance tests
            random_vr = p.get("random_baseline_variance_ratio", 1.0)
            random_eval_vr = p.get("random_baseline_eval_variance_ratio", 1.0)

            if abs(random_vr - 1.0) > random_baseline_tolerance:
                cond_pass = False
                cond_reasons.append(
                    f"Random baseline (train) {random_vr:.3f} differs from 1.0 - "
                    f"possible bug or filtering artifact"
                )

            if abs(random_eval_vr - 1.0) > random_baseline_tolerance:
                cond_pass = False
                cond_reasons.append(
                    f"Random baseline (eval) {random_eval_vr:.3f} differs from 1.0 - "
                    f"possible bug or filtering artifact"
                )

        # Check confound robustness (structure must persist)
        high_energy = [m for m in cond_metrics if m["slice"] == "high_energy" and m["lag"] == 1]
        medial = [m for m in cond_metrics if m["slice"] == "utterance_medial" and m["lag"] == 1]

        for confound, label in [(high_energy, "high_energy"), (medial, "utterance_medial")]:
            if confound:
                c = confound[0]
                # Check both train AND eval for confound slices
                train_vr = c["train"]["variance_ratio"]
                eval_vr = c["eval"]["variance_ratio"]

                if train_vr > variance_ratio_threshold + 0.05:
                    cond_pass = False
                    cond_reasons.append(
                        f"Structure disappears in {label} slice (train VR={train_vr:.3f})"
                    )

                if eval_vr > variance_ratio_threshold + 0.05:
                    cond_pass = False
                    cond_reasons.append(
                        f"Structure disappears in {label} slice (eval VR={eval_vr:.3f})"
                    )

        if cond_pass:
            passing_conditions.append(cond_name)
        else:
            decision["reasons"].extend([f"[{cond_name}] {r}" for r in cond_reasons])

    # Overall decision
    if passing_conditions:
        decision["pass"] = True
        decision["best_condition"] = passing_conditions[0]
        decision["passing_conditions"] = passing_conditions

    # Build summary
    decision["summary"] = {
        "n_conditions_tested": len(by_condition),
        "n_conditions_passing": len(passing_conditions),
        "passing_conditions": passing_conditions,
    }

    return decision


def generate_report(
    metrics_path: str | Path,
    output_path: str | Path,
    config: Optional[dict] = None,
) -> str:
    """
    Generate a text report from metrics.

    Args:
        metrics_path: Path to metrics.json
        output_path: Path to write report
        config: Optional config dict for thresholds

    Returns:
        Report text
    """
    with open(metrics_path) as f:
        metrics = json.load(f)

    # Get thresholds from config or use defaults
    vr_threshold = 0.6
    degradation_max = 0.20
    if config:
        vr_threshold = config.get("metrics", {}).get("variance_ratio_threshold", 0.6)
        degradation_max = config.get("metrics", {}).get("cross_speaker_degradation_max", 0.20)

    decision = make_decision(metrics, vr_threshold, degradation_max)

    # Build report
    lines = [
        "=" * 70,
        "PHASE 0 ANALYSIS REPORT: Audio Latent Structure Analysis",
        "=" * 70,
        "",
        "DECISION: " + ("PASS" if decision["pass"] else "FAIL"),
        "",
    ]

    if decision["pass"]:
        lines.extend([
            "Audio latents exhibit reusable local structure suitable for",
            "conditional memory mechanisms.",
            "",
            f"Best condition: {decision['best_condition']}",
            f"Passing conditions: {', '.join(decision['passing_conditions'])}",
        ])
    else:
        lines.extend([
            "Audio latents do NOT exhibit sufficient reusable local structure.",
            "Conditional memory mechanisms are unlikely to provide benefit.",
            "",
            "Failure reasons:",
        ])
        for reason in decision["reasons"]:
            lines.append(f"  - {reason}")

    if decision["warnings"]:
        lines.extend(["", "Warnings:"])
        for warning in decision["warnings"]:
            lines.append(f"  - {warning}")

    # Add summary table
    lines.extend([
        "",
        "-" * 70,
        "METRICS SUMMARY (slice=all, lag=1)",
        "-" * 70,
        "",
    ])

    # Build table
    df = pd.DataFrame(metrics)
    if len(df) > 0:
        summary = []
        for m in metrics:
            if m["slice"] == "all" and m["lag"] == 1:
                summary.append({
                    "Condition": m["condition"],
                    "Train VR": f"{m['train']['variance_ratio']:.3f}",
                    "Eval VR": f"{m['eval']['variance_ratio']:.3f}",
                    "Degradation": f"{m.get('cross_speaker_degradation', 0):.3f}",
                    "Random": f"{m.get('random_baseline_variance_ratio', 0):.3f}",
                    "Excl. Mass": f"{m['train'].get('excluded_mass', 0):.1%}",
                })

        if summary:
            summary_df = pd.DataFrame(summary)
            lines.append(summary_df.to_string(index=False))

    # Add per-speaker CI if available
    lines.extend([
        "",
        "-" * 70,
        "PER-SPEAKER CONFIDENCE INTERVALS (95%)",
        "-" * 70,
        "",
    ])

    ci_summary = []
    for m in metrics:
        if m["slice"] == "all" and m["lag"] == 1:
            ci_lower = m["train"].get("speaker_ci_lower", float("nan"))
            ci_upper = m["train"].get("speaker_ci_upper", float("nan"))
            ci_summary.append({
                "Condition": m["condition"],
                "Train CI": f"[{ci_lower:.3f}, {ci_upper:.3f}]",
                "Train Mean": f"{m['train'].get('speaker_mean', float('nan')):.3f}",
                "Train Std": f"{m['train'].get('speaker_std', float('nan')):.3f}",
            })

    if ci_summary:
        ci_df = pd.DataFrame(ci_summary)
        lines.append(ci_df.to_string(index=False))

    lines.extend([
        "",
        "-" * 70,
        "DECISION CRITERIA (from SPEC)",
        "-" * 70,
        "",
        f"Variance ratio threshold: < {vr_threshold} (on BOTH train and eval)",
        f"Cross-speaker degradation max: < {degradation_max} (eval worse than train)",
        "Random baseline: must be ~1.0 (failure indicates bug or artifact)",
        "Confound robustness: structure must persist in high_energy and medial slices",
        "",
        "Pass if ALL criteria met. Fail if ANY violated.",
        "",
        "=" * 70,
    ])

    report = "\n".join(lines)

    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    return report
