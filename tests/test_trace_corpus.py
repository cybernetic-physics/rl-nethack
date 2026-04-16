from __future__ import annotations

import json

from rl.trace_corpus import normalize_trace_rows, split_trace_corpus, summarize_trace_rows


def _rows():
    rows = []
    for episode in range(4):
        for step in range(3):
            rows.append(
                {
                    "episode_id": f"ep{episode}",
                    "step": step,
                    "action": "east" if step % 2 == 0 else "west",
                    "allowed_actions": ["east", "west", "wait"],
                    "reward": float(step),
                    "done": step == 2,
                    "prompt": f"episode {episode} step {step}",
                    "feature_vector": [float(step), float(episode)],
                    "planner_trace": [{"action": "east", "total": 1.0}, {"action": "west", "total": 0.0}],
                }
            )
    return rows


def test_summarize_trace_rows():
    summary = summarize_trace_rows(normalize_trace_rows(_rows(), default_observation_version="toy"))
    assert summary["rows"] == 12
    assert summary["episodes"] == 4
    assert summary["feature_dims"] == [2]
    assert summary["observation_versions"] == ["toy"]
    assert summary["planner_trace_rows"] == 12
    assert summary["max_episode_length"] == 3


def test_split_trace_corpus(tmp_path):
    input_path = tmp_path / "trace.jsonl"
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    manifest_path = tmp_path / "manifest.json"
    input_path.write_text("".join(json.dumps(row) + "\n" for row in _rows()))

    manifest = split_trace_corpus(
        str(input_path),
        train_output_path=str(train_path),
        eval_output_path=str(eval_path),
        manifest_output_path=str(manifest_path),
        eval_fraction=0.25,
        default_observation_version="toy",
    )

    assert manifest["train_summary"]["episodes"] == 3
    assert manifest["eval_summary"]["episodes"] == 1
    assert train_path.exists()
    assert eval_path.exists()
    assert manifest_path.exists()
