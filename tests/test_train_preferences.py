import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train_preferences import (
    build_preference_texts,
    format_messages_prefix,
    load_preference_data,
    normalize_preference_row,
    parse_args,
    truncate_rows,
)


def test_normalize_preference_row_uses_messages_when_present():
    row = {"messages": [{"role": "system", "content": "sys"}], "chosen": "east", "rejected": "wait"}
    normalized = normalize_preference_row(row)
    assert normalized["messages"][0]["content"] == "sys"


def test_normalize_preference_row_falls_back_to_conversations():
    row = {
        "conversations": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "state"},
            {"role": "assistant", "content": "wait"},
        ],
        "chosen": "east",
        "rejected": "wait",
    }
    normalized = normalize_preference_row(row)
    assert len(normalized["messages"]) == 2
    assert normalized["messages"][-1]["role"] == "user"


def test_format_messages_prefix_and_preference_texts():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "state"},
    ]
    prefix = format_messages_prefix(messages)
    prompt, chosen, rejected = build_preference_texts({"messages": messages, "chosen": "east", "rejected": "wait"})
    assert "<|im_start|>system" in prefix
    assert prompt == prefix
    assert "east" in chosen
    assert "wait" in rejected


def test_truncate_rows_caps_length():
    rows = [{"x": 1}, {"x": 2}, {"x": 3}]
    assert len(truncate_rows(rows, 2)) == 2
    assert len(truncate_rows(rows, None)) == 3


def test_load_preference_data_reads_jsonl(tmp_path):
    path = tmp_path / "prefs.jsonl"
    rows = [
        {"messages": [{"role": "system", "content": "sys"}], "chosen": "east", "rejected": "wait"},
        {"messages": [{"role": "system", "content": "sys"}], "chosen": "south", "rejected": "search"},
    ]
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    loaded = load_preference_data(str(path))
    assert len(loaded) == 2
    assert loaded[0]["chosen"] == "east"


def test_parse_args_supports_pairwise_training_flags():
    args = parse_args([
        "--data", "/tmp/train.jsonl",
        "--output", "/tmp/out",
        "--beta", "0.5",
        "--max-seq-length", "4096",
    ])
    assert args.data == "/tmp/train.jsonl"
    assert args.output == "/tmp/out"
    assert args.beta == 0.5
    assert args.max_seq_length == 4096


def test_cli_help_works():
    project_root = os.path.join(os.path.dirname(__file__), "..")
    result = subprocess.run(
        [sys.executable, "train_preferences.py", "--help"],
        capture_output=True,
        text=True,
        cwd=project_root,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--beta" in result.stdout
