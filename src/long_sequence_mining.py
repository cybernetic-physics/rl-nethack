"""
Helpers for mining focused long-sequence shards such as gold winning episodes.
"""

from __future__ import annotations

import json
from collections import defaultdict


def load_long_sequence_rows(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def episode_group_key(row: dict) -> str:
    metadata = row.get("metadata", {})
    return str(
        metadata.get("source_episode_id")
        or metadata.get("episode_id")
        or metadata.get("gameid")
        or metadata.get("seed")
        or "episode"
    )


def compute_episode_score(rows: list[dict]) -> tuple:
    """Higher is better."""
    metadata = [row.get("metadata", {}) for row in rows]
    is_win = max(1 if m.get("is_win") else 0 for m in metadata)
    achieve = max(int(m.get("achieve") or 0) for m in metadata)
    maxlvl = max(int(m.get("maxlvl") or m.get("depth") or 0) for m in metadata)
    turns = max(int(m.get("turns") or 0) for m in metadata)
    n_rows = len(rows)
    return (is_win, achieve, maxlvl, turns, n_rows)


def build_gold_wins_rows(rows: list[dict], *, max_episodes: int | None = None) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[episode_group_key(row)].append(row)

    ranked = sorted(
        grouped.items(),
        key=lambda kv: compute_episode_score(kv[1]),
        reverse=True,
    )
    if max_episodes is not None:
        ranked = ranked[:max_episodes]

    selected = []
    for _, episode_rows in ranked:
        selected.extend(episode_rows)
    return selected
