import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.long_sequence_eval import (
    evaluate_long_sequence_rows,
    normalize_action_text,
)


def make_row(target_action: str, *, bucket: str, outcome: str, phase: str):
    return {
        "conversations": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "state"},
            {"role": "assistant", "content": target_action},
        ],
        "metadata": {
            "episode_id": "ep-1",
            "step_index": 0,
            "target_context_bucket": bucket,
            "outcome": outcome,
            "game_phase": phase,
        },
    }


def test_normalize_action_text_handles_variants():
    assert normalize_action_text("move north") == "north"
    assert normalize_action_text("South.") == "south"
    assert normalize_action_text("search") == "search"


def test_evaluate_long_sequence_rows_summarizes_slices():
    rows = [
        make_row("north", bucket="128k", outcome="win", phase="late"),
        make_row("south", bucket="256k", outcome="loss", phase="early"),
        make_row("west", bucket="256k", outcome="loss", phase="early"),
    ]
    predictions = iter(["north", "east", "west"])

    result = evaluate_long_sequence_rows(
        rows,
        predict_fn=lambda messages: next(predictions),
    )
    summary = result["summary"]
    assert summary["overall"]["n"] == 3
    assert summary["overall"]["exact_match_rate"] == 2 / 3
    assert summary["by_context_bucket"]["128k"]["exact_match_rate"] == 1.0
    assert summary["by_context_bucket"]["256k"]["exact_match_rate"] == 0.5
    assert summary["by_outcome"]["win"]["exact_match_rate"] == 1.0
    assert summary["by_outcome"]["loss"]["exact_match_rate"] == 0.5
    assert summary["by_game_phase"]["early"]["exact_match_rate"] == 0.5
    assert summary["by_action_family"]["move"]["exact_match_rate"] == 2 / 3
    assert summary["by_turn_depth_bucket"]["t000-031"]["n"] == 3
    assert "search" in summary["focused_behavior_slices"]
    assert "late_inventory" in summary["focused_behavior_slices"]


def test_evaluate_long_sequence_rows_detects_dangerous_messages():
    rows = [
        {
            "conversations": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Message: The goblin hits!\nCurrentTurn:\nfoo"},
                {"role": "assistant", "content": "north"},
            ],
            "metadata": {"target_context_bucket": "128k", "outcome": "loss", "game_phase": "early"},
        }
    ]
    result = evaluate_long_sequence_rows(rows, predict_fn=lambda messages: "north")
    assert result["summary"]["dangerous_message_slice"]["dangerous"]["n"] == 1


def test_evaluate_long_sequence_rows_tracks_post_danger_recovery():
    rows = [
        {
            "conversations": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Message: The goblin hits!\nCurrentTurn:\nfoo"},
                {"role": "assistant", "content": "north"},
            ],
            "metadata": {"episode_id": "ep-1", "step_index": 0, "target_context_bucket": "128k", "outcome": "loss", "game_phase": "early"},
        },
        {
            "conversations": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Message: <none>\nCurrentTurn:\nfoo"},
                {"role": "assistant", "content": "search"},
            ],
            "metadata": {"episode_id": "ep-1", "step_index": 1, "target_context_bucket": "128k", "outcome": "loss", "game_phase": "early"},
        },
    ]
    result = evaluate_long_sequence_rows(rows, predict_fn=lambda messages: "search")
    assert result["summary"]["recovery_after_dangerous_message"]["post_danger_1"]["n"] == 1
    assert result["summary"]["recovery_after_dangerous_message"]["post_danger_3"]["n"] == 1
