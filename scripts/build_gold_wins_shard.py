#!/usr/bin/env python3
"""Build a ranked gold-wins shard from long-sequence JSONL."""

from __future__ import annotations

import argparse
import json
import os

from src.long_sequence_mining import build_gold_wins_rows, load_long_sequence_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Build a ranked gold-wins shard from long-sequence JSONL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_long_sequence_rows(args.input)
    selected = build_gold_wins_rows(rows, max_episodes=args.max_episodes)
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        for row in selected:
            f.write(json.dumps(row) + "\n")
    episode_ids = {
        (row.get("metadata", {}) or {}).get("source_episode_id")
        or (row.get("metadata", {}) or {}).get("episode_id")
        for row in selected
    }
    print(
        json.dumps(
            {
                "selected_rows": len(selected),
                "selected_episodes": len(episode_ids),
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
