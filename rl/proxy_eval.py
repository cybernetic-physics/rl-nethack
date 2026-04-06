from __future__ import annotations

from collections import Counter
from statistics import mean

from rl.proxy_dataset import load_proxy_rows
from rl.proxy_model import ProxyRewardInference, load_proxy_model


REGRESSION_KEYS = [
    "k_step_progress",
    "k_step_survival",
    "k_step_loop_risk",
    "k_step_resource_value",
    "teacher_margin",
]


def evaluate_proxy_rows(rows: list[dict], inference: ProxyRewardInference) -> dict:
    if not rows:
        raise ValueError("No proxy rows to evaluate")

    predicted_action_counts: Counter[str] = Counter()
    action_matches = 0
    search_rows = 0
    search_matches = 0
    search_tp = search_fp = search_fn = 0
    per_key_errors: dict[str, list[float]] = {key: [] for key in REGRESSION_KEYS}

    for row in rows:
        ranking = inference.rank_actions(row["feature_vector"], row["allowed_actions"])
        predicted = ranking[0]["action"]
        predicted_action_counts[predicted] += 1
        action_matches += int(predicted == row["action"])

        current_prediction = inference.score_action(row["feature_vector"], row["action"])
        for key in REGRESSION_KEYS:
            pred_key = {
                "k_step_progress": "progress",
                "k_step_survival": "survival",
                "k_step_loop_risk": "loop_risk",
                "k_step_resource_value": "resource_value",
                "teacher_margin": "teacher_margin",
            }[key]
            per_key_errors[key].append(abs(float(row[key]) - float(current_prediction[pred_key])))

        if "search" in row["allowed_actions"]:
            search_rows += 1
            search_rank = next(candidate for candidate in ranking if candidate["action"] == "search")
            pred_search = int(search_rank["search_context_prob"] >= 0.5)
            true_search = int(row["search_context_label"])
            search_matches += int(pred_search == true_search)
            search_tp += int(pred_search == 1 and true_search == 1)
            search_fp += int(pred_search == 1 and true_search == 0)
            search_fn += int(pred_search == 0 and true_search == 1)

    return {
        "rows": len(rows),
        "action_top1_accuracy": round(action_matches / max(1, len(rows)), 4),
        "predicted_action_counts": dict(predicted_action_counts),
        "regression_mae": {key: round(mean(errors), 4) for key, errors in per_key_errors.items()},
        "search_context": {
            "rows": search_rows,
            "accuracy": round(search_matches / max(1, search_rows), 4),
            "precision": round(search_tp / max(1, search_tp + search_fp), 4),
            "recall": round(search_tp / max(1, search_tp + search_fn), 4),
        },
    }


def evaluate_proxy_model(path: str, input_path: str) -> dict:
    inference = load_proxy_model(path)
    rows = load_proxy_rows(input_path)
    result = evaluate_proxy_rows(rows, inference)
    result["model_path"] = path
    result["input_path"] = input_path
    return result
