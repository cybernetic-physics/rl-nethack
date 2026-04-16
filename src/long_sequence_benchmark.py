"""
Deterministic benchmark-shard builder for long-sequence datasets.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from src.long_sequence_eval import action_family, load_long_sequence_rows


def benchmark_key(row: dict) -> tuple:
    metadata = row.get("metadata", {})
    return (
        str(metadata.get("target_context_bucket", metadata.get("context_bucket", "unknown"))),
        str(metadata.get("game_phase", "unknown")),
        action_family(row["conversations"][-1]["content"]),
        str(metadata.get("episode_id", metadata.get("source_episode_id", metadata.get("gameid", metadata.get("seed", "episode"))))),
        int(metadata.get("step_index", 0)),
    )


def build_benchmark_rows(
    rows: list[dict],
    *,
    per_bucket: int = 32,
    per_phase: int = 32,
    per_action_family: int = 32,
) -> list[dict]:
    """Build a deterministic mixed benchmark shard from one long-sequence corpus."""
    rows = sorted(rows, key=benchmark_key)
    selected = []
    seen = set()

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    by_phase: dict[str, list[dict]] = defaultdict(list)
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata", {})
        by_bucket[str(metadata.get("target_context_bucket", metadata.get("context_bucket", "unknown")))].append(row)
        by_phase[str(metadata.get("game_phase", "unknown"))].append(row)
        by_family[action_family(row["conversations"][-1]["content"])].append(row)

    for groups, limit in (
        (by_bucket, per_bucket),
        (by_phase, per_phase),
        (by_family, per_action_family),
    ):
        for _, group_rows in sorted(groups.items()):
            for row in group_rows[:limit]:
                key = benchmark_key(row)
                if key in seen:
                    continue
                selected.append(row)
                seen.add(key)
    return sorted(selected, key=benchmark_key)


def build_benchmark_from_path(
    input_path: str,
    output_path: str,
    *,
    per_bucket: int = 32,
    per_phase: int = 32,
    per_action_family: int = 32,
) -> dict:
    rows = load_long_sequence_rows(input_path)
    benchmark_rows = build_benchmark_rows(
        rows,
        per_bucket=per_bucket,
        per_phase=per_phase,
        per_action_family=per_action_family,
    )
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        for row in benchmark_rows:
            f.write(json.dumps(row) + "\n")
    return {
        "input_rows": len(rows),
        "benchmark_rows": len(benchmark_rows),
        "output_path": output_path,
        "per_bucket": per_bucket,
        "per_phase": per_phase,
        "per_action_family": per_action_family,
    }
