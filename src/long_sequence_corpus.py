"""
Token-budgeted corpus builder for long-sequence NetHack training.

This compiles one or more long-sequence JSONL shards into a mixed corpus that:

- prioritizes full-game winning episodes
- keeps most examples on the long-context end
- reduces near-duplicate overlap with per-episode stride sampling
- stops when a projected token budget is reached
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from src.long_sequence_mining import episode_group_key


@dataclass
class EpisodeSummary:
    key: str
    rows: list[dict]
    source_path: str
    source_name: str
    is_win: bool
    outcome: str
    max_depth: int
    max_turn: int
    max_context_tokens: int
    avg_context_tokens: float
    total_context_tokens: int


def load_long_sequence_rows_with_source(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                metadata = row.setdefault("metadata", {})
                metadata.setdefault("corpus_input_path", path)
                rows.append(row)
    return rows


def _row_context_tokens(row: dict) -> int:
    metadata = row.get("metadata", {})
    return int(
        metadata.get("context_tokens_estimate")
        or metadata.get("target_context_tokens")
        or metadata.get("max_context_tokens")
        or 0
    )


def _row_turn_index(row: dict) -> int:
    metadata = row.get("metadata", {})
    return int(metadata.get("step_index") or metadata.get("turn") or 0)


def _row_depth(row: dict) -> int:
    metadata = row.get("metadata", {})
    return int(metadata.get("depth") or metadata.get("maxlvl") or 0)


def _row_outcome(row: dict) -> str:
    metadata = row.get("metadata", {})
    return str(metadata.get("outcome") or "unknown")


def _row_source_name(row: dict) -> str:
    metadata = row.get("metadata", {})
    return str(metadata.get("source") or metadata.get("corpus_input_path") or "unknown")


def summarize_episodes(rows: list[dict]) -> list[EpisodeSummary]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("metadata", {}).get("corpus_input_path", "unknown"), episode_group_key(row))].append(row)

    summaries: list[EpisodeSummary] = []
    for (source_path, episode_id), episode_rows in grouped.items():
        ordered = sorted(episode_rows, key=_row_turn_index)
        metadata = [row.get("metadata", {}) for row in ordered]
        outcomes = [_row_outcome(row) for row in ordered]
        outcome = "win" if "win" in outcomes else outcomes[-1]
        max_depth = max(int(m.get("depth") or m.get("maxlvl") or 0) for m in metadata)
        max_turn = max(int(m.get("turn") or m.get("turns") or _row_turn_index(row)) for m, row in zip(metadata, ordered))
        contexts = [_row_context_tokens(row) for row in ordered]
        summaries.append(
            EpisodeSummary(
                key=str(episode_id),
                rows=ordered,
                source_path=str(source_path),
                source_name=_row_source_name(ordered[0]),
                is_win=outcome == "win" or any(bool(m.get("is_win")) for m in metadata),
                outcome=outcome,
                max_depth=max_depth,
                max_turn=max_turn,
                max_context_tokens=max(contexts) if contexts else 0,
                avg_context_tokens=(sum(contexts) / len(contexts)) if contexts else 0.0,
                total_context_tokens=sum(contexts),
            )
        )
    return summaries


def is_deep_loss(summary: EpisodeSummary) -> bool:
    if summary.is_win:
        return False
    return (
        summary.max_depth >= 7
        or summary.max_turn >= 750
        or summary.max_context_tokens >= 32_000
        or len(summary.rows) >= 256
    )


def episode_priority(summary: EpisodeSummary) -> tuple:
    return (
        1 if summary.is_win else 0,
        1 if is_deep_loss(summary) else 0,
        summary.max_depth,
        summary.max_turn,
        summary.max_context_tokens,
        len(summary.rows),
    )


def row_tier(
    row: dict,
    *,
    very_long_min_tokens: int,
    long_min_tokens: int,
    medium_min_tokens: int,
) -> str | None:
    tokens = _row_context_tokens(row)
    if tokens >= very_long_min_tokens:
        return "very_long"
    if tokens >= long_min_tokens:
        return "long"
    if tokens >= medium_min_tokens:
        return "medium"
    return None


def row_priority(row: dict) -> tuple:
    metadata = row.get("metadata", {})
    return (
        1 if bool(metadata.get("is_win")) or _row_outcome(row) == "win" else 0,
        _row_depth(row),
        _row_context_tokens(row),
        _row_turn_index(row),
    )


def _row_unique_key(row: dict) -> tuple[str, str, int, str]:
    metadata = row.get("metadata", {})
    return (
        str(metadata.get("corpus_input_path", "unknown")),
        str(episode_group_key(row)),
        int(metadata.get("step_index") or 0),
        str(metadata.get("target_context_tokens") or metadata.get("context_bucket") or ""),
    )


def build_token_budgeted_corpus(
    input_paths: list[str],
    *,
    output_path: str,
    manifest_path: str | None = None,
    target_tokens: int = 1_000_000_000,
    full_episode_fraction: float = 0.40,
    very_long_fraction: float = 0.35,
    long_fraction: float = 0.20,
    medium_fraction: float = 0.05,
    full_episode_winning_share: float = 0.70,
    very_long_min_tokens: int = 65_536,
    long_min_tokens: int = 16_384,
    medium_min_tokens: int = 4_096,
    very_long_stride: int = 16,
    long_stride: int = 32,
    medium_stride: int = 64,
) -> dict[str, Any]:
    if not input_paths:
        raise ValueError("build_token_budgeted_corpus requires at least one input path")
    fractions_total = full_episode_fraction + very_long_fraction + long_fraction + medium_fraction
    if abs(fractions_total - 1.0) > 1e-6:
        raise ValueError("Corpus tier fractions must sum to 1.0")

    rows = load_long_sequence_rows_with_source(input_paths)
    episodes = summarize_episodes(rows)
    ranked_episodes = sorted(episodes, key=episode_priority, reverse=True)

    target_by_tier = {
        "full_episode": int(target_tokens * full_episode_fraction),
        "very_long": int(target_tokens * very_long_fraction),
        "long": int(target_tokens * long_fraction),
        "medium": target_tokens - (
            int(target_tokens * full_episode_fraction)
            + int(target_tokens * very_long_fraction)
            + int(target_tokens * long_fraction)
        ),
    }
    winning_full_target = int(target_by_tier["full_episode"] * full_episode_winning_share)

    selected_rows: list[dict] = []
    selected_keys: set[tuple[str, str, int, str]] = set()
    selected_tokens = 0
    selected_by_tier = Counter()
    selected_by_outcome = Counter()
    selected_by_source = Counter()
    selected_episode_ids: set[tuple[str, str]] = set()

    def add_row(row: dict, tier_name: str) -> bool:
        nonlocal selected_tokens
        key = _row_unique_key(row)
        if key in selected_keys:
            return False
        row_copy = json.loads(json.dumps(row))
        row_copy.setdefault("metadata", {})
        row_copy["metadata"]["corpus_sampling_tier"] = tier_name
        row_copy["metadata"]["corpus_input_path"] = row.get("metadata", {}).get("corpus_input_path")
        selected_rows.append(row_copy)
        selected_keys.add(key)
        tokens = _row_context_tokens(row_copy)
        selected_tokens += tokens
        selected_by_tier[tier_name] += tokens
        selected_by_outcome[_row_outcome(row_copy)] += tokens
        selected_by_source[_row_source_name(row_copy)] += tokens
        selected_episode_ids.add((str(row_copy["metadata"].get("corpus_input_path", "unknown")), str(episode_group_key(row_copy))))
        return True

    def add_episode(summary: EpisodeSummary, tier_name: str, *, token_cap: int | None = None) -> int:
        added = 0
        for row in summary.rows:
            if token_cap is not None and added >= token_cap:
                break
            if add_row(row, tier_name):
                added += _row_context_tokens(row)
        return added

    full_episode_tokens = 0
    full_episode_win_tokens = 0
    for summary in ranked_episodes:
        if full_episode_win_tokens >= winning_full_target:
            break
        if not summary.is_win:
            continue
        full_episode_tokens += add_episode(summary, "full_episode")
        if summary.is_win:
            full_episode_win_tokens = full_episode_tokens
        if full_episode_tokens >= target_by_tier["full_episode"]:
            break

    if full_episode_tokens < target_by_tier["full_episode"]:
        for summary in ranked_episodes:
            episode_key = (summary.source_path, summary.key)
            if episode_key in selected_episode_ids:
                continue
            full_episode_tokens += add_episode(summary, "full_episode")
            if full_episode_tokens >= target_by_tier["full_episode"]:
                break

    stride_by_tier = {
        "very_long": max(1, very_long_stride),
        "long": max(1, long_stride),
        "medium": max(1, medium_stride),
    }

    for tier_name in ("very_long", "long", "medium"):
        tier_tokens = 0
        stride = stride_by_tier[tier_name]
        last_selected_turn_by_episode: dict[tuple[str, str], int] = {}
        candidates: list[tuple[EpisodeSummary, dict]] = []
        for summary in ranked_episodes:
            for row in summary.rows:
                if row_tier(
                    row,
                    very_long_min_tokens=very_long_min_tokens,
                    long_min_tokens=long_min_tokens,
                    medium_min_tokens=medium_min_tokens,
                ) == tier_name:
                    candidates.append((summary, row))
        candidates.sort(key=lambda item: row_priority(item[1]), reverse=True)
        for summary, row in candidates:
            if tier_tokens >= target_by_tier[tier_name]:
                break
            episode_key = (summary.source_path, summary.key)
            turn_index = _row_turn_index(row)
            last_turn = last_selected_turn_by_episode.get(episode_key)
            if last_turn is not None and turn_index - last_turn < stride:
                continue
            if add_row(row, tier_name):
                tier_tokens += _row_context_tokens(row)
                last_selected_turn_by_episode[episode_key] = turn_index

    selected_rows.sort(
        key=lambda row: (
            str(row.get("metadata", {}).get("corpus_input_path", "")),
            str(episode_group_key(row)),
            int(row.get("metadata", {}).get("step_index") or 0),
            str(row.get("metadata", {}).get("corpus_sampling_tier", "")),
        )
    )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        for row in selected_rows:
            f.write(json.dumps(row) + "\n")

    manifest = {
        "input_paths": input_paths,
        "output_path": output_path,
        "target_tokens": target_tokens,
        "selected_tokens": selected_tokens,
        "selected_rows": len(selected_rows),
        "selected_episodes": len(selected_episode_ids),
        "input_rows": len(rows),
        "input_episodes": len(episodes),
        "tier_target_tokens": target_by_tier,
        "tier_selected_tokens": dict(selected_by_tier),
        "outcome_selected_tokens": dict(selected_by_outcome),
        "source_selected_tokens": dict(selected_by_source),
        "config": {
            "full_episode_fraction": full_episode_fraction,
            "very_long_fraction": very_long_fraction,
            "long_fraction": long_fraction,
            "medium_fraction": medium_fraction,
            "full_episode_winning_share": full_episode_winning_share,
            "very_long_min_tokens": very_long_min_tokens,
            "long_min_tokens": long_min_tokens,
            "medium_min_tokens": medium_min_tokens,
            "very_long_stride": very_long_stride,
            "long_stride": long_stride,
            "medium_stride": medium_stride,
        },
        "episode_preview": [
            {
                "episode_id": summary.key,
                "source": summary.source_name,
                "outcome": summary.outcome,
                "is_win": summary.is_win,
                "max_depth": summary.max_depth,
                "max_turn": summary.max_turn,
                "rows": len(summary.rows),
                "max_context_tokens": summary.max_context_tokens,
                "avg_context_tokens": round(summary.avg_context_tokens, 2),
            }
            for summary in ranked_episodes[:20]
        ],
    }

    actual_manifest_path = manifest_path
    if actual_manifest_path is None:
        actual_manifest_path = f"{output_path}.manifest.json"
    with open(actual_manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    manifest["manifest_path"] = actual_manifest_path
    return manifest
