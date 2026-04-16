from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def summarize_sequence_model_reports(report_paths: list[str]) -> dict:
    rows = []
    for path in report_paths:
        payload = _load_json(path)
        val = payload.get("val_summary", {})
        rows.append(
            {
                "report_path": path,
                "model_path": path.replace("_train.json", ".pt") if path.endswith("_train.json") else None,
                "feature_mse": val.get("feature_mse"),
                "reward_mae": val.get("reward_mae"),
                "value_mae": val.get("value_mae"),
                "planner_action_mae": val.get("planner_action_mae"),
                "planner_policy_ce": val.get("planner_policy_ce"),
                "planner_pairwise_loss": val.get("planner_pairwise_loss"),
                "kl_mean": val.get("kl_mean"),
                "overshooting_kl_mean": val.get("overshooting_kl_mean"),
            }
        )
    def _best(metric: str):
        candidates = [row for row in rows if row.get(metric) is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda row: float(row[metric]))

    return {
        "models": rows,
        "best_by_feature_mse": _best("feature_mse"),
        "best_by_reward_mae": _best("reward_mae"),
        "best_by_value_mae": _best("value_mae"),
        "best_by_planner_action_mae": _best("planner_action_mae"),
        "best_by_planner_policy_ce": _best("planner_policy_ce"),
        "best_by_planner_pairwise_loss": _best("planner_pairwise_loss"),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Summarize multiple sequence world-model training reports")
    parser.add_argument("--reports", nargs="+", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(summarize_sequence_model_reports(args.reports), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
