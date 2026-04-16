import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train_kto import (
    build_kto_texts,
    format_messages_prefix,
    load_kto_rows,
    normalize_kto_row,
    parse_args,
    truncate_rows,
)


def test_normalize_kto_row_uses_messages_when_present():
    row = {"messages": [{"role": "system", "content": "sys"}], "completion": "east", "label": True}
    normalized = normalize_kto_row(row)
    assert normalized["messages"][0]["content"] == "sys"
    assert normalized["label"] is True


def test_normalize_kto_row_derives_fields():
    row = {
        "conversations": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "state"},
            {"role": "assistant", "content": "wait"},
        ],
        "sample_weight": -0.25,
    }
    normalized = normalize_kto_row(row)
    assert len(normalized["messages"]) == 2
    assert normalized["completion"] == "wait"
    assert normalized["label"] is False


def test_format_messages_prefix_and_kto_texts():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "state"},
    ]
    prefix = format_messages_prefix(messages)
    prompt, completion = build_kto_texts({"messages": messages, "completion": "east"})
    assert "<|im_start|>system" in prefix
    assert prompt == prefix
    assert "east" in completion


def test_truncate_rows_caps_length():
    rows = [{"x": 1}, {"x": 2}, {"x": 3}]
    assert len(truncate_rows(rows, 2)) == 2
    assert len(truncate_rows(rows, None)) == 3


def test_load_kto_rows_reads_jsonl(tmp_path):
    path = tmp_path / "kto.jsonl"
    rows = [
        {"messages": [{"role": "system", "content": "sys"}], "completion": "east", "label": True},
        {"messages": [{"role": "system", "content": "sys"}], "completion": "wait", "label": False},
    ]
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    loaded = load_kto_rows(str(path))
    assert len(loaded) == 2
    assert loaded[1]["completion"] == "wait"


def test_parse_args_supports_kto_flags():
    args = parse_args([
        "--data", "/tmp/train.jsonl",
        "--output", "/tmp/out",
        "--beta", "0.5",
        "--desirable-weight", "2.0",
        "--undesirable-weight", "0.5",
    ])
    assert args.beta == 0.5
    assert args.desirable_weight == 2.0
    assert args.undesirable_weight == 0.5


def test_cli_help_works():
    project_root = os.path.join(os.path.dirname(__file__), "..")
    result = subprocess.run(
        [sys.executable, "train_kto.py", "--help"],
        capture_output=True,
        text=True,
        cwd=project_root,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--beta" in result.stdout
