from __future__ import annotations

from rl.proxy_dataset import load_proxy_rows
from rl.proxy_model import load_proxy_model


def build_proxy_report(model_path: str, input_path: str, top_k: int = 5) -> dict:
    inference = load_proxy_model(model_path)
    rows = load_proxy_rows(input_path)
    action_examples = []
    search_examples = []

    for row in rows[: max(top_k * 10, top_k)]:
        ranking = inference.rank_actions(row["feature_vector"], row["allowed_actions"])
        if len(action_examples) < top_k and ranking[0]["action"] != row["action"]:
            action_examples.append(
                {
                    "episode_id": row["episode_id"],
                    "step": row["step"],
                    "teacher_action": row["action"],
                    "predicted_action": ranking[0]["action"],
                    "ranking": ranking[: min(4, len(ranking))],
                }
            )
        if "search" in row["allowed_actions"] and len(search_examples) < top_k:
            search_score = next(candidate for candidate in ranking if candidate["action"] == "search")
            search_examples.append(
                {
                    "episode_id": row["episode_id"],
                    "step": row["step"],
                    "search_context_label": row["search_context_label"],
                    "search_context_prob": round(search_score["search_context_prob"], 4),
                    "prompt": row["prompt"],
                }
            )
        if len(action_examples) >= top_k and len(search_examples) >= top_k:
            break

    return {
        "model_path": model_path,
        "input_path": input_path,
        "top_action_mismatches": action_examples,
        "top_search_examples": search_examples,
    }
