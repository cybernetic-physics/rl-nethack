"""
Evaluation helpers for long-sequence next-action datasets.
"""

from __future__ import annotations

import json
import urllib.request
from collections import defaultdict
from typing import Callable, Optional

from nle_agent.agent_http import parse_action


def load_long_sequence_rows(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_action_text(raw: str) -> str:
    _, action_name = parse_action(raw)
    return action_name


def action_family(action: str) -> str:
    action = normalize_action_text(action)
    if action in {"north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"}:
        return "move"
    if action in {"search"}:
        return "search"
    if action in {"pickup", "drop", "eat", "drink", "read", "wear", "takeoff", "wield", "apply", "zap", "throw"}:
        return "inventory"
    if action in {"up", "down"}:
        return "stairs"
    if action in {"open", "close", "kick"}:
        return "interaction"
    if action in {"wait"}:
        return "wait"
    return "other"


def episode_id_for_row(row: dict) -> str:
    metadata = row.get("metadata", {})
    return str(
        metadata.get("source_episode_id")
        or metadata.get("episode_id")
        or metadata.get("gameid")
        or metadata.get("seed")
        or "episode"
    )


def turn_depth_bucket(turn_index: int) -> str:
    if turn_index < 32:
        return "t000-031"
    if turn_index < 128:
        return "t032-127"
    if turn_index < 512:
        return "t128-511"
    return "t512+"


def is_dangerous_message(text: str) -> bool:
    lower = text.lower()
    cues = [
        "hit", "bites", "misses", "killed", "dies", "die", "poison",
        "fainted", "confused", "blind", "halluc", "attack", "attacks",
        "engulf", "burn", "frozen", "stun",
    ]
    return any(cue in lower for cue in cues)


def extract_current_turn_message(messages: list[dict]) -> str:
    if not messages:
        return ""
    user_content = messages[-1]["content"]
    for line in user_content.splitlines():
        if line.startswith("Message: "):
            return line[len("Message: "):]
    return ""


def query_openai_chat(server_url: str, messages: list[dict], model_name: str = "llama-server") -> str:
    payload = json.dumps(
        {
            "model": model_name,
            "messages": messages,
            "max_tokens": 8,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{server_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def check_server_available(server_url: str) -> bool:
    try:
        req = urllib.request.Request(
            f"{server_url.rstrip('/')}/health",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False


def _metrics_for_rows(scored_rows: list[dict]) -> dict:
    n = len(scored_rows)
    if n == 0:
        return {"n": 0, "exact_match_rate": 0.0, "prediction_rate": 0.0}
    correct = sum(1 for row in scored_rows if row["exact_match"])
    return {
        "n": n,
        "exact_match_rate": correct / n,
        "prediction_rate": n / n,
    }


def _family_metrics(scored_rows: list[dict], family: str) -> dict:
    n = len(scored_rows)
    if n == 0:
        return {
            "n": 0,
            "exact_match_rate": 0.0,
            "target_rate": 0.0,
            "prediction_rate": 0.0,
        }
    exact = sum(1 for row in scored_rows if row["exact_match"])
    target_hits = sum(1 for row in scored_rows if action_family(row["target"]) == family)
    prediction_hits = sum(1 for row in scored_rows if action_family(row["prediction"]) == family)
    return {
        "n": n,
        "exact_match_rate": exact / n,
        "target_rate": target_hits / n,
        "prediction_rate": prediction_hits / n,
    }


def _episode_recovery_windows(scored_rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in scored_rows:
        grouped[episode_id_for_row(row)].append(row)
    for episode_rows in grouped.values():
        episode_rows.sort(key=lambda row: int((row.get("metadata") or {}).get("step_index", 0)))

    post_danger_1 = []
    post_danger_3 = []
    for episode_rows in grouped.values():
        for idx, row in enumerate(episode_rows):
            if idx >= 1 and episode_rows[idx - 1].get("dangerous_message"):
                post_danger_1.append(row)
            window = episode_rows[max(0, idx - 3):idx]
            if window and any(prev.get("dangerous_message") for prev in window):
                post_danger_3.append(row)
    return {
        "post_danger_1": post_danger_1,
        "post_danger_3": post_danger_3,
    }


def summarize_long_sequence_results(scored_rows: list[dict]) -> dict:
    summary = {
        "overall": _metrics_for_rows(scored_rows),
        "by_context_bucket": {},
        "by_outcome": {},
        "by_game_phase": {},
        "by_turn_depth_bucket": {},
        "by_action_family": {},
        "dangerous_message_slice": {},
        "recovery_after_dangerous_message": {},
        "focused_behavior_slices": {},
    }
    by_context_bucket: dict[str, list[dict]] = defaultdict(list)
    by_outcome: dict[str, list[dict]] = defaultdict(list)
    by_game_phase: dict[str, list[dict]] = defaultdict(list)
    by_turn_depth_bucket: dict[str, list[dict]] = defaultdict(list)
    by_action_family: dict[str, list[dict]] = defaultdict(list)
    dangerous_slice: dict[str, list[dict]] = defaultdict(list)
    for row in scored_rows:
        metadata = row.get("metadata", {})
        by_context_bucket[str(metadata.get("target_context_bucket", metadata.get("context_bucket", "unknown")))].append(row)
        by_outcome[str(metadata.get("outcome", "unknown"))].append(row)
        by_game_phase[str(metadata.get("game_phase", "unknown"))].append(row)
        by_turn_depth_bucket[turn_depth_bucket(int(metadata.get("step_index", 0)))].append(row)
        by_action_family[action_family(row["target"])].append(row)
        dangerous_slice["dangerous" if row.get("dangerous_message") else "not_dangerous"].append(row)
    summary["by_context_bucket"] = {k: _metrics_for_rows(v) for k, v in sorted(by_context_bucket.items())}
    summary["by_outcome"] = {k: _metrics_for_rows(v) for k, v in sorted(by_outcome.items())}
    summary["by_game_phase"] = {k: _metrics_for_rows(v) for k, v in sorted(by_game_phase.items())}
    summary["by_turn_depth_bucket"] = {k: _metrics_for_rows(v) for k, v in sorted(by_turn_depth_bucket.items())}
    summary["by_action_family"] = {k: _metrics_for_rows(v) for k, v in sorted(by_action_family.items())}
    summary["dangerous_message_slice"] = {k: _metrics_for_rows(v) for k, v in sorted(dangerous_slice.items())}
    summary["recovery_after_dangerous_message"] = {
        key: _metrics_for_rows(rows)
        for key, rows in sorted(_episode_recovery_windows(scored_rows).items())
    }
    focused_slices = {}
    for family in ("search", "inventory", "stairs", "wait"):
        focused_slices[family] = _family_metrics(scored_rows, family)
    late_rows = [
        row for row in scored_rows
        if str((row.get("metadata") or {}).get("game_phase", "unknown")) in {"late", "endgame", "amulet", "gehennom", "astral", "ascended"}
    ]
    for family in ("search", "inventory", "stairs"):
        focused_slices[f"late_{family}"] = _family_metrics(late_rows, family)
    summary["focused_behavior_slices"] = focused_slices
    return summary


def evaluate_long_sequence_rows(
    rows: list[dict],
    *,
    predict_fn: Callable[[list[dict]], str],
    max_examples: Optional[int] = None,
) -> dict:
    if max_examples is not None:
        rows = rows[:max_examples]
    scored_rows = []
    for row in rows:
        messages = row["conversations"][:-1]
        target = normalize_action_text(row["conversations"][-1]["content"])
        raw_prediction = predict_fn(messages)
        prediction = normalize_action_text(raw_prediction)
        message_text = extract_current_turn_message(messages)
        scored_rows.append(
            {
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "target": target,
                "exact_match": prediction == target,
                "metadata": row.get("metadata", {}),
                "dangerous_message": is_dangerous_message(message_text),
            }
        )
    return {
        "summary": summarize_long_sequence_results(scored_rows),
        "rows": scored_rows,
    }


def evaluate_long_sequence_dataset(
    path: str,
    *,
    server_url: str,
    model_name: str = "llama-server",
    max_examples: Optional[int] = None,
) -> dict:
    rows = load_long_sequence_rows(path)
    server_available = check_server_available(server_url)
    if not server_available:
        return {
            "server_available": False,
            "summary": summarize_long_sequence_results([]),
            "rows": [],
        }

    def predict_fn(messages: list[dict]) -> str:
        return query_openai_chat(server_url, messages, model_name=model_name)

    result = evaluate_long_sequence_rows(rows, predict_fn=predict_fn, max_examples=max_examples)
    result["server_available"] = True
    result["path"] = path
    result["server_url"] = server_url
    result["model_name"] = model_name
    return result
