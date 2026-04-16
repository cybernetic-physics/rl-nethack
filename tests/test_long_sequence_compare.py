import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.long_sequence_compare import compare_eval_report_paths, compare_eval_reports


def make_report(overall, recovery, late_inventory, late_stairs):
    return {
        "summary": {
            "overall": {"exact_match_rate": overall},
            "recovery_after_dangerous_message": {"post_danger_1": {"exact_match_rate": recovery}},
            "focused_behavior_slices": {
                "late_inventory": {"exact_match_rate": late_inventory},
                "late_stairs": {"exact_match_rate": late_stairs},
            },
            "dangerous_message_slice": {"dangerous": {"exact_match_rate": recovery}},
        }
    }


def test_compare_eval_reports_ranks_best_first():
    report = compare_eval_reports(
        [
            ("weighted", make_report(0.60, 0.40, 0.30, 0.20)),
            ("pairwise", make_report(0.62, 0.42, 0.35, 0.22)),
            ("kto", make_report(0.59, 0.45, 0.33, 0.25)),
        ]
    )
    assert report["best"]["name"] == "pairwise"
    assert report["leaderboard"][0]["name"] == "pairwise"
    assert len(report["deltas_vs_best"]) == 3


def test_compare_eval_report_paths_reads_files(tmp_path):
    weighted = tmp_path / "weighted.json"
    pairwise = tmp_path / "pairwise.json"
    weighted.write_text(json.dumps(make_report(0.5, 0.3, 0.2, 0.1)))
    pairwise.write_text(json.dumps(make_report(0.7, 0.4, 0.3, 0.2)))
    result = compare_eval_report_paths(
        [("weighted", str(weighted)), ("pairwise", str(pairwise))]
    )
    assert result["best"]["name"] == "pairwise"
    assert result["inputs"][0]["name"] == "weighted"
