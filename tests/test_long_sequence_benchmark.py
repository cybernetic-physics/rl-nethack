import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.long_sequence_benchmark import build_benchmark_from_path, build_benchmark_rows


def make_row(action: str, *, bucket: str, phase: str, episode_id: str, step_index: int):
    return {
        "conversations": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": action},
        ],
        "metadata": {
            "target_context_bucket": bucket,
            "game_phase": phase,
            "episode_id": episode_id,
            "step_index": step_index,
        },
    }


def test_build_benchmark_rows_is_deterministic_and_deduplicated():
    rows = [
        make_row("north", bucket="128k", phase="early", episode_id="a", step_index=1),
        make_row("search", bucket="128k", phase="early", episode_id="a", step_index=2),
        make_row("down", bucket="256k", phase="late", episode_id="b", step_index=3),
        make_row("east", bucket="256k", phase="late", episode_id="b", step_index=4),
    ]
    first = build_benchmark_rows(rows, per_bucket=1, per_phase=1, per_action_family=1)
    second = build_benchmark_rows(list(reversed(rows)), per_bucket=1, per_phase=1, per_action_family=1)
    assert first == second
    keys = [(row["metadata"]["episode_id"], row["metadata"]["step_index"]) for row in first]
    assert len(keys) == len(set(keys))


def test_build_benchmark_from_path_writes_jsonl(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "benchmark.jsonl"
    rows = [
        make_row("north", bucket="128k", phase="early", episode_id="a", step_index=1),
        make_row("search", bucket="256k", phase="late", episode_id="b", step_index=2),
    ]
    with open(input_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    result = build_benchmark_from_path(str(input_path), str(output_path), per_bucket=1, per_phase=1, per_action_family=1)
    assert result["benchmark_rows"] >= 1
    assert output_path.exists()
