import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.long_sequence_preferences import build_kto_style_rows
from src.long_sequence_preferences import build_pairwise_preference_rows
from src.long_sequence_preferences import build_weighted_sft_rows


def make_row(episode_id: str, step: int, action: str, outcome: str):
    return {
        "conversations": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": f"state {step}"},
            {"role": "assistant", "content": action},
        ],
        "metadata": {
            "episode_id": episode_id,
            "step_index": step,
            "outcome": outcome,
        },
    }


def test_build_kto_style_rows_keeps_win_rows_positive():
    rows = [
        make_row("win-ep", 0, "east", "win"),
        make_row("win-ep", 1, "south", "win"),
    ]
    output = build_kto_style_rows(rows)
    assert len(output) == 2
    assert all(row["label"] is True for row in output)
    assert all(row["reason"] == "winning_episode_action" for row in output)


def test_build_kto_style_rows_marks_repeated_weak_loss_actions_negative():
    rows = [
        make_row("loss-ep", 0, "wait", "loss"),
        make_row("loss-ep", 1, "wait", "loss"),
        make_row("loss-ep", 2, "wait", "loss"),
        make_row("loss-ep", 3, "east", "loss"),
    ]
    output = build_kto_style_rows(rows, min_repeat_run=3)
    negatives = [row for row in output if row["label"] is False]
    assert len(negatives) == 3
    assert all(row["reason"] == "loss_repeated_weak_action" for row in negatives)
    assert all(row["metadata"]["repeat_run_length"] == 3 for row in negatives)


def test_build_kto_style_rows_marks_low_hp_stall_actions_negative():
    rows = [
        {
            "conversations": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "state"},
                {"role": "assistant", "content": "wait"},
            ],
            "metadata": {
                "episode_id": "loss-ep",
                "step_index": 5,
                "outcome": "loss",
                "hp": 3,
                "hp_max": 20,
            },
        }
    ]
    output = build_kto_style_rows(rows)
    negatives = [row for row in output if row["label"] is False]
    assert len(negatives) == 1
    assert negatives[0]["reason"] == "loss_low_hp_stall_action"


def test_build_kto_style_rows_marks_near_death_mistakes_negative():
    rows = [
        {
            "conversations": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Message: The goblin hits!\nCurrentTurn:\nfoo"},
                {"role": "assistant", "content": "wait"},
            ],
            "metadata": {
                "episode_id": "loss-ep",
                "step_index": 6,
                "outcome": "loss",
                "hp": 2,
                "hp_max": 20,
            },
        }
    ]
    output = build_kto_style_rows(rows)
    negatives = [row for row in output if row["label"] is False]
    assert any(row["reason"] == "loss_near_death_mistake" for row in negatives)


def test_build_weighted_sft_rows_emits_sharegpt_training_rows():
    rows = [
        make_row("win-ep", 0, "east", "win"),
        make_row("loss-ep", 0, "wait", "loss"),
        make_row("loss-ep", 1, "wait", "loss"),
        make_row("loss-ep", 2, "wait", "loss"),
    ]
    output = build_weighted_sft_rows(rows, min_repeat_run=3, positive_weight=1.0, negative_weight=-0.5)
    assert output[0]["conversations"][-1]["role"] == "assistant"
    assert output[0]["sample_weight"] == 1.0
    negatives = [row for row in output if row["sample_weight"] < 0]
    assert len(negatives) == 3
    assert all(row["metadata"]["training_objective"] == "weighted_sft" for row in output)


def test_build_kto_style_rows_marks_teacher_disagreement_negative():
    rows = [
        {
            "conversations": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "state"},
                {"role": "assistant", "content": "wait"},
            ],
            "metadata": {
                "episode_id": "loss-ep",
                "step_index": 1,
                "outcome": "loss",
                "teacher_action": "east",
                "teacher_margin": -0.2,
            },
        }
    ]
    output = build_kto_style_rows(rows, teacher_margin_threshold=0.0)
    negatives = [row for row in output if row["label"] is False]
    assert len(negatives) == 1
    assert negatives[0]["reason"] == "loss_teacher_disagreement"


def test_build_pairwise_preference_rows_uses_teacher_preferred_action():
    rows = [
        {
            "conversations": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "state"},
                {"role": "assistant", "content": "wait"},
            ],
            "metadata": {
                "episode_id": "loss-ep",
                "step_index": 1,
                "outcome": "loss",
                "teacher_action": "east",
                "teacher_margin": -0.2,
            },
        }
    ]
    output = build_pairwise_preference_rows(rows, teacher_margin_threshold=0.0)
    assert len(output) == 1
    assert output[0]["chosen"] == "east"
    assert output[0]["rejected"] == "wait"
    assert output[0]["reason"] == "teacher_preferred_alternative"
