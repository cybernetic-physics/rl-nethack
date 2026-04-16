from __future__ import annotations

import argparse
import json


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _planner_metrics(planner: dict) -> dict:
    summary = planner.get("summary")
    if isinstance(summary, dict):
        return {
            "planner_exact_match_rate": summary.get("mean_exact_match_rate"),
            "planner_exact_match_std": summary.get("std_exact_match_rate"),
            "planner_rank_correlation": summary.get("mean_rank_correlation"),
            "planner_rank_correlation_std": summary.get("std_rank_correlation"),
            "planner_teacher_gap": summary.get("mean_teacher_gap"),
            "planner_teacher_gap_std": summary.get("std_teacher_gap"),
        }
    return {
        "planner_exact_match_rate": planner.get("exact_match_rate"),
        "planner_exact_match_std": None,
        "planner_rank_correlation": planner.get("mean_rank_correlation"),
        "planner_rank_correlation_std": None,
        "planner_teacher_gap": planner.get("mean_teacher_gap_from_predicted_best"),
        "planner_teacher_gap_std": None,
    }


def summarize_sequence_benchmark(pairs: list[tuple[str, str]]) -> dict:
    rows = []
    for report_path, planner_eval_path in pairs:
        report = _load_json(report_path)
        planner = _load_json(planner_eval_path)
        val = report.get("val_summary", {})
        planner_metrics = _planner_metrics(planner)
        rows.append(
            {
                "report_path": report_path,
                "planner_eval_path": planner_eval_path,
                "model_path": report_path.replace("_train.json", ".pt") if report_path.endswith("_train.json") else None,
                "feature_mse": val.get("feature_mse"),
                "reward_mae": val.get("reward_mae"),
                "value_mae": val.get("value_mae"),
                "planner_action_mae": val.get("planner_action_mae"),
                **planner_metrics,
            }
        )

    def _best_min(metric: str):
        candidates = [row for row in rows if row.get(metric) is not None]
        return None if not candidates else min(candidates, key=lambda row: float(row[metric]))

    def _best_max(metric: str):
        candidates = [row for row in rows if row.get(metric) is not None]
        return None if not candidates else max(candidates, key=lambda row: float(row[metric]))

    return {
        "models": rows,
        "best_predictive_feature_mse": _best_min("feature_mse"),
        "best_predictive_reward_mae": _best_min("reward_mae"),
        "best_planner_exact_match_rate": _best_max("planner_exact_match_rate"),
        "best_planner_rank_correlation": _best_max("planner_rank_correlation"),
        "best_planner_teacher_gap": _best_min("planner_teacher_gap"),
        "best_planner_rank_correlation_std": _best_min("planner_rank_correlation_std"),
        "best_planner_teacher_gap_std": _best_min("planner_teacher_gap_std"),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Summarize sequence world-model predictive + planner replay metrics")
    parser.add_argument("--pairs", nargs="+", required=True, help="Alternating report.json planner_eval.json paths")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if len(args.pairs) % 2 != 0:
        raise SystemExit("--pairs expects an even number of paths: report planner_eval ...")
    pairs = [(args.pairs[idx], args.pairs[idx + 1]) for idx in range(0, len(args.pairs), 2)]
    print(json.dumps(summarize_sequence_benchmark(pairs), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
