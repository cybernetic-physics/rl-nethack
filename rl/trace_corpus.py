from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from rl.io_utils import atomic_write_json, atomic_write_text


REQUIRED_TRACE_KEYS = {
    "episode_id",
    "step",
    "action",
    "allowed_actions",
    "reward",
    "done",
    "prompt",
    "feature_vector",
}


def load_trace_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_trace_rows(rows: list[dict], *, default_observation_version: str | None = None) -> list[dict]:
    normalized: list[dict] = []
    for row in rows:
        missing = sorted(REQUIRED_TRACE_KEYS - set(row))
        if missing:
            raise ValueError(f"Trace row missing required keys: {missing}")
        feature_vector = row.get("feature_vector")
        if not isinstance(feature_vector, list) or not feature_vector:
            raise ValueError("Trace row feature_vector must be a non-empty list")
        normalized_row = dict(row)
        normalized_row["episode_id"] = str(normalized_row["episode_id"])
        normalized_row["step"] = int(normalized_row["step"])
        if "observation_version" not in normalized_row and default_observation_version is not None:
            normalized_row["observation_version"] = default_observation_version
        normalized.append(normalized_row)
    normalized.sort(key=lambda row: (str(row["episode_id"]), int(row["step"])))
    return normalized


def summarize_trace_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "rows": 0,
            "episodes": 0,
            "observation_versions": [],
            "feature_dims": [],
            "planner_trace_rows": 0,
            "avg_episode_length": 0.0,
            "median_episode_length": 0.0,
            "max_episode_length": 0,
            "action_counts": {},
            "done_fraction": 0.0,
        }

    episodes: dict[str, list[dict]] = defaultdict(list)
    action_counts: Counter[str] = Counter()
    versions = set()
    feature_dims = set()
    planner_trace_rows = 0
    done_count = 0

    for row in rows:
        episodes[str(row["episode_id"])].append(row)
        action_counts[str(row["action"])] += 1
        versions.add(str(row.get("observation_version", "unknown")))
        feature_dims.add(len(row["feature_vector"]))
        planner_trace_rows += int(bool(row.get("planner_trace")))
        done_count += int(bool(row.get("done")))

    episode_lengths = sorted(len(episode_rows) for episode_rows in episodes.values())
    median_idx = len(episode_lengths) // 2
    if len(episode_lengths) % 2 == 1:
        median_episode_length = float(episode_lengths[median_idx])
    else:
        median_episode_length = float(episode_lengths[median_idx - 1] + episode_lengths[median_idx]) / 2.0

    return {
        "rows": len(rows),
        "episodes": len(episodes),
        "observation_versions": sorted(versions),
        "feature_dims": sorted(feature_dims),
        "planner_trace_rows": planner_trace_rows,
        "planner_trace_fraction": float(planner_trace_rows / len(rows)),
        "avg_episode_length": float(sum(episode_lengths) / len(episode_lengths)),
        "median_episode_length": median_episode_length,
        "max_episode_length": max(episode_lengths),
        "action_counts": dict(sorted(action_counts.items())),
        "done_fraction": float(done_count / len(rows)),
    }


def split_trace_rows_by_episode(rows: list[dict], *, eval_fraction: float = 0.15) -> tuple[list[dict], list[dict], dict]:
    if not rows:
        return [], [], {"train_episode_ids": [], "eval_episode_ids": []}
    episodes: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        episodes[str(row["episode_id"])].append(row)
    episode_ids = sorted(episodes)
    eval_count = max(1, int(math.ceil(len(episode_ids) * float(eval_fraction)))) if len(episode_ids) > 1 else 0
    eval_ids = episode_ids[-eval_count:] if eval_count else []
    train_ids = episode_ids[:-eval_count] if eval_count else episode_ids
    train_rows = [row for episode_id in train_ids for row in episodes[episode_id]]
    eval_rows = [row for episode_id in eval_ids for row in episodes[episode_id]]
    return train_rows, eval_rows, {
        "train_episode_ids": train_ids,
        "eval_episode_ids": eval_ids,
    }


def write_trace_rows(path: str, rows: list[dict]) -> str:
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    return atomic_write_text(path, payload)


def audit_trace_corpus(input_path: str) -> dict:
    rows = normalize_trace_rows(load_trace_rows(input_path))
    summary = summarize_trace_rows(rows)
    summary["input_path"] = input_path
    return summary


def split_trace_corpus(
    input_path: str,
    *,
    train_output_path: str,
    eval_output_path: str,
    manifest_output_path: str | None = None,
    eval_fraction: float = 0.15,
    default_observation_version: str | None = None,
) -> dict:
    rows = normalize_trace_rows(load_trace_rows(input_path), default_observation_version=default_observation_version)
    train_rows, eval_rows, split_ids = split_trace_rows_by_episode(rows, eval_fraction=eval_fraction)
    write_trace_rows(train_output_path, train_rows)
    write_trace_rows(eval_output_path, eval_rows)
    manifest = {
        "input_path": input_path,
        "train_output_path": train_output_path,
        "eval_output_path": eval_output_path,
        "eval_fraction": float(eval_fraction),
        "train_summary": summarize_trace_rows(train_rows),
        "eval_summary": summarize_trace_rows(eval_rows),
        **split_ids,
    }
    if manifest_output_path:
        atomic_write_json(manifest_output_path, manifest)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Audit and split full-trajectory trace corpora")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--input", required=True)
    audit_parser.add_argument("--output", default=None)

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--input", required=True)
    split_parser.add_argument("--train-output", required=True)
    split_parser.add_argument("--eval-output", required=True)
    split_parser.add_argument("--manifest-output", default=None)
    split_parser.add_argument("--eval-fraction", type=float, default=0.15)
    split_parser.add_argument("--default-observation-version", default=None)

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "audit":
        result = audit_trace_corpus(args.input)
        if args.output:
            atomic_write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "split":
        result = split_trace_corpus(
            args.input,
            train_output_path=args.train_output,
            eval_output_path=args.eval_output,
            manifest_output_path=args.manifest_output,
            eval_fraction=args.eval_fraction,
            default_observation_version=args.default_observation_version,
        )
        print(json.dumps(result, indent=2))
        return 0
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
