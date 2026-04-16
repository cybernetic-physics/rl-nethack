"""
Selective positive/negative mining for long-sequence next-action data.
"""

from __future__ import annotations

import json
from collections import defaultdict


WEAK_ACTIONS = {"wait", "search", "pickup", "open"}
ENDGAME_NEGATIVE_ACTIONS = {"wait", "search"}
NEAR_DEATH_ACTIONS = WEAK_ACTIONS | {"wait"}


def extract_current_turn_message(messages: list[dict]) -> str:
    if not messages:
        return ""
    user_content = str(messages[-1].get("content", ""))
    for line in user_content.splitlines():
        if line.startswith("Message: "):
            return line[len("Message: "):]
    return ""


def is_dangerous_message(text: str) -> bool:
    lower = text.lower()
    cues = [
        "hit", "bites", "misses", "killed", "dies", "die", "poison",
        "fainted", "confused", "blind", "halluc", "attack", "attacks",
        "engulf", "burn", "frozen", "stun",
    ]
    return any(cue in lower for cue in cues)


def load_long_sequence_rows(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def episode_id_for_row(row: dict) -> str:
    metadata = row.get("metadata", {})
    return str(
        metadata.get("source_episode_id")
        or metadata.get("episode_id")
        or metadata.get("gameid")
        or metadata.get("seed")
        or "episode"
    )


def assistant_action(row: dict) -> str:
    return str(row["conversations"][-1]["content"]).strip().lower()


def build_kto_style_rows(
    rows: list[dict],
    *,
    positive_limit: int | None = None,
    negative_limit: int | None = None,
    min_repeat_run: int = 3,
    teacher_margin_threshold: float = 0.0,
) -> list[dict]:
    """
    Build selective labeled rows for future KTO / weighted-SFT style training.

    Positive rows:
    - winning-episode actions

    Negative rows:
    - rows from losing episodes that sit inside repeated-action runs
    - especially repeated weak-action runs such as wait/search/open/pickup
    - late losing rows with low HP and weak stalling actions
    - rows that disagree with teacher / relabel metadata when available
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[episode_id_for_row(row)].append(row)
    for episode_rows in grouped.values():
        episode_rows.sort(key=lambda row: int((row.get("metadata") or {}).get("step_index", 0)))

    positives = []
    negatives = []

    for episode_rows in grouped.values():
        outcome = str((episode_rows[0].get("metadata") or {}).get("outcome", "unknown"))
        if outcome == "win":
            for row in episode_rows:
                positives.append(
                    {
                        "messages": row["conversations"][:-1],
                        "completion": row["conversations"][-1]["content"],
                        "label": True,
                        "reason": "winning_episode_action",
                        "metadata": row.get("metadata", {}),
                    }
                )
            continue

        if outcome != "loss":
            continue

        for row in episode_rows:
            metadata = row.get("metadata", {})
            hp = metadata.get("hp")
            hp_max = metadata.get("hp_max")
            action = assistant_action(row)
            teacher_action = metadata.get("teacher_action") or metadata.get("predicted_action")
            teacher_margin = metadata.get("teacher_margin")
            message_text = extract_current_turn_message(row["conversations"][:-1])
            dangerous_message = is_dangerous_message(message_text)
            if hp is not None and hp_max:
                low_hp = float(hp) <= max(1.0, 0.35 * float(hp_max))
                if low_hp and dangerous_message and action in NEAR_DEATH_ACTIONS:
                    negatives.append(
                        {
                            "messages": row["conversations"][:-1],
                            "completion": row["conversations"][-1]["content"],
                            "label": False,
                            "reason": "loss_near_death_mistake",
                            "metadata": {
                                **metadata,
                            },
                        }
                    )
                if low_hp and action in ENDGAME_NEGATIVE_ACTIONS:
                    negatives.append(
                        {
                            "messages": row["conversations"][:-1],
                            "completion": row["conversations"][-1]["content"],
                            "label": False,
                            "reason": "loss_low_hp_stall_action",
                            "metadata": {
                                **metadata,
                            },
                        }
                    )
            if teacher_action is not None and str(teacher_action).strip().lower() != action:
                include = True
                if teacher_margin is not None:
                    try:
                        include = float(teacher_margin) <= teacher_margin_threshold
                    except (TypeError, ValueError):
                        include = True
                if include:
                    negatives.append(
                        {
                            "messages": row["conversations"][:-1],
                            "completion": row["conversations"][-1]["content"],
                            "label": False,
                            "reason": "loss_teacher_disagreement",
                            "metadata": {
                                **metadata,
                            },
                        }
                    )

        run_start = 0
        while run_start < len(episode_rows):
            run_end = run_start + 1
            action = assistant_action(episode_rows[run_start])
            while run_end < len(episode_rows) and assistant_action(episode_rows[run_end]) == action:
                run_end += 1
            run_len = run_end - run_start
            if run_len >= min_repeat_run:
                reason = "loss_repeated_action"
                if action in WEAK_ACTIONS:
                    reason = "loss_repeated_weak_action"
                for idx in range(run_start, run_end):
                    row = episode_rows[idx]
                    negatives.append(
                        {
                            "messages": row["conversations"][:-1],
                            "completion": row["conversations"][-1]["content"],
                            "label": False,
                            "reason": reason,
                            "metadata": {
                                **(row.get("metadata", {})),
                                "repeat_run_length": run_len,
                            },
                        }
                    )
            run_start = run_end

    if positive_limit is not None:
        positives = positives[:positive_limit]
    if negative_limit is not None:
        negatives = negatives[:negative_limit]
    return positives + negatives


def build_pairwise_preference_rows(
    rows: list[dict],
    *,
    negative_limit: int | None = None,
    teacher_margin_threshold: float = 0.0,
) -> list[dict]:
    """
    Build pairwise chosen/rejected rows when teacher alternatives are available.

    Each row represents a teacher-preferred action against the original losing
    action on the same state. This is suitable for future DPO/ORPO/KTO-style
    preference training.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[episode_id_for_row(row)].append(row)
    for episode_rows in grouped.values():
        episode_rows.sort(key=lambda row: int((row.get("metadata") or {}).get("step_index", 0)))

    output = []
    for episode_rows in grouped.values():
        outcome = str((episode_rows[0].get("metadata") or {}).get("outcome", "unknown"))
        if outcome != "loss":
            continue
        for row in episode_rows:
            metadata = row.get("metadata") or {}
            teacher_action = metadata.get("teacher_action") or metadata.get("predicted_action")
            teacher_margin = metadata.get("teacher_margin")
            rejected = assistant_action(row)
            if teacher_action is None:
                continue
            chosen = str(teacher_action).strip().lower()
            if chosen == rejected:
                continue
            include = True
            if teacher_margin is not None:
                try:
                    include = float(teacher_margin) <= teacher_margin_threshold
                except (TypeError, ValueError):
                    include = True
            if not include:
                continue
            output.append(
                {
                    "messages": row["conversations"][:-1],
                    "chosen": chosen,
                    "rejected": row["conversations"][-1]["content"],
                    "reason": "teacher_preferred_alternative",
                    "metadata": {
                        **metadata,
                        "training_objective": "pairwise_preference",
                    },
                }
            )
    if negative_limit is not None:
        output = output[:negative_limit]
    return output


def build_weighted_sft_rows(
    rows: list[dict],
    *,
    positive_limit: int | None = None,
    negative_limit: int | None = None,
    min_repeat_run: int = 3,
    positive_weight: float = 1.0,
    negative_weight: float = -0.25,
    teacher_margin_threshold: float = 0.0,
) -> list[dict]:
    """
    Convert selective positive/negative mined rows into ShareGPT training rows.

    The resulting rows are directly consumable by ``train.py`` and carry a
    ``sample_weight`` field so the trainer can upweight wins and apply a
    negative learning signal to mined losing-segment actions.
    """
    preference_rows = build_kto_style_rows(
        rows,
        positive_limit=positive_limit,
        negative_limit=negative_limit,
        min_repeat_run=min_repeat_run,
        teacher_margin_threshold=teacher_margin_threshold,
    )
    output_rows = []
    for row in preference_rows:
        label = bool(row["label"])
        output_rows.append(
            {
                "conversations": [
                    *row["messages"],
                    {"role": "assistant", "content": row["completion"]},
                ],
                "sample_weight": positive_weight if label else negative_weight,
                "metadata": {
                    **(row.get("metadata") or {}),
                    "preference_label": "positive" if label else "negative",
                    "preference_reason": row["reason"],
                    "training_objective": "weighted_sft",
                },
            }
        )
    return output_rows
