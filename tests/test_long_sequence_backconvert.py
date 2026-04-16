import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.long_sequence_backconvert import (
    extract_current_turn_state_text,
    extract_episode_rows_from_long_sequence_path,
)


def test_extract_current_turn_state_text():
    user = "EpisodeId: ep1\nCurrentTurn:\nTurnIndex: 4\nMessage: hello\nBoard:\nabc\nNextAction:"
    text = extract_current_turn_state_text(user)
    assert text == "TurnIndex: 4\nMessage: hello\nBoard:\nabc"


def test_extract_episode_rows_from_long_sequence_path(tmp_path):
    input_path = tmp_path / "long.jsonl"
    output_path = tmp_path / "episodes.jsonl"
    row = {
        "conversations": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "EpisodeId: ep1\nTargetStep: 0\nCurrentTurn:\nTurnIndex: 0\nMessage: hi\nBoard:\nabc\nNextAction:"},
            {"role": "assistant", "content": "east"},
        ],
        "metadata": {
            "episode_id": "ep1",
            "step_index": 0,
            "outcome": "loss",
        },
    }
    with open(input_path, "w") as f:
        f.write(json.dumps(row) + "\n")
    result = extract_episode_rows_from_long_sequence_path(str(input_path), str(output_path))
    assert result["episodes"] == 1
    with open(output_path, "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert rows[0]["state_prompt"] == "TurnIndex: 0\nMessage: hi\nBoard:\nabc"
    assert rows[0]["action"] == "east"
