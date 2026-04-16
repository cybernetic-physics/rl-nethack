"""
Back-convert long-sequence rows into episode-style rows for corpus rebuilding.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict


def extract_current_turn_state_text(user_content: str) -> str:
    marker = "\nCurrentTurn:\n"
    suffix = "\nNextAction:"
    if marker not in user_content or suffix not in user_content:
        raise ValueError("Could not locate CurrentTurn / NextAction markers")
    start = user_content.index(marker) + len(marker)
    end = user_content.rindex(suffix)
    return user_content[start:end]


def extract_episode_rows_from_long_sequence_path(input_path: str, output_path: str) -> dict:
    rows = []
    by_episode: dict[str, list[dict]] = defaultdict(list)
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            metadata = row.get("metadata", {})
            episode_id = str(
                metadata.get("source_episode_id")
                or metadata.get("episode_id")
                or metadata.get("gameid")
                or metadata.get("seed")
                or "episode"
            )
            step_index = int(metadata.get("step_index", 0))
            user_content = row["conversations"][-2]["content"]
            state_text = extract_current_turn_state_text(user_content)
            converted = {
                "episode_id": episode_id,
                "step": step_index,
                "state_prompt": state_text,
                "action": row["conversations"][-1]["content"],
                **metadata,
            }
            by_episode[episode_id].append(converted)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        for episode_id in sorted(by_episode):
            episode_rows = sorted(by_episode[episode_id], key=lambda item: int(item["step"]))
            rows.extend(episode_rows)
            for row in episode_rows:
                f.write(json.dumps(row) + "\n")
    return {
        "input_path": input_path,
        "output_path": output_path,
        "episodes": len(by_episode),
        "rows": len(rows),
    }
