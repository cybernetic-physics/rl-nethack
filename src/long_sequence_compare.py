"""
Comparison helpers for long-sequence evaluation reports.
"""

from __future__ import annotations

import json
import os
from typing import Any


def load_eval_report(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _get_metric(report: dict[str, Any], path: list[str], default: float = 0.0) -> float:
    current: Any = report
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    try:
        return float(current)
    except (TypeError, ValueError):
        return default


def compare_eval_reports(named_reports: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """
    Compare multiple long-sequence eval reports and rank them on key metrics.
    """
    rows = []
    for name, report in named_reports:
        summary = report.get("summary", report)
        rows.append(
            {
                "name": name,
                "overall_exact_match_rate": _get_metric(summary, ["overall", "exact_match_rate"]),
                "late_inventory_exact_match_rate": _get_metric(summary, ["focused_behavior_slices", "late_inventory", "exact_match_rate"]),
                "late_stairs_exact_match_rate": _get_metric(summary, ["focused_behavior_slices", "late_stairs", "exact_match_rate"]),
                "danger_recovery_exact_match_rate": _get_metric(summary, ["recovery_after_dangerous_message", "post_danger_1", "exact_match_rate"]),
                "dangerous_slice_exact_match_rate": _get_metric(summary, ["dangerous_message_slice", "dangerous", "exact_match_rate"]),
            }
        )

    leaderboard = sorted(
        rows,
        key=lambda row: (
            row["overall_exact_match_rate"],
            row["danger_recovery_exact_match_rate"],
            row["late_inventory_exact_match_rate"],
            row["late_stairs_exact_match_rate"],
        ),
        reverse=True,
    )

    best = leaderboard[0] if leaderboard else None
    deltas_vs_best = []
    if best is not None:
        for row in leaderboard:
            deltas_vs_best.append(
                {
                    "name": row["name"],
                    "delta_overall_exact_match_rate": row["overall_exact_match_rate"] - best["overall_exact_match_rate"],
                    "delta_danger_recovery_exact_match_rate": row["danger_recovery_exact_match_rate"] - best["danger_recovery_exact_match_rate"],
                    "delta_late_inventory_exact_match_rate": row["late_inventory_exact_match_rate"] - best["late_inventory_exact_match_rate"],
                    "delta_late_stairs_exact_match_rate": row["late_stairs_exact_match_rate"] - best["late_stairs_exact_match_rate"],
                }
            )

    return {
        "leaderboard": leaderboard,
        "best": best,
        "deltas_vs_best": deltas_vs_best,
    }


def compare_eval_report_paths(named_paths: list[tuple[str, str]]) -> dict[str, Any]:
    reports = [(name, load_eval_report(path)) for name, path in named_paths]
    result = compare_eval_reports(reports)
    result["inputs"] = [{"name": name, "path": path} for name, path in named_paths]
    return result


def save_compare_report(report: dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
