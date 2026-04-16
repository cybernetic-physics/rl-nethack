import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.policy_actions import canonicalize_action, normalize_action_text
from src.policy_replay import (
    classify_modal_screen,
    is_dangerous_message,
    parse_tty_screen,
    render_policy_state_from_text,
)


def test_canonicalize_action_keeps_runtime_actions():
    action = canonicalize_action("Move north")
    assert action.normalized == "north"
    assert action.should_keep is True
    assert action.is_runtime_action is True
    assert action.action_class == "move"


def test_canonicalize_action_drops_modal_actions_by_default():
    action = canonicalize_action("space")
    assert action.normalized == "space"
    assert action.is_modal_action is True
    assert action.should_keep is False
    assert action.drop_reason == "modal_action"


def test_normalize_action_text_can_keep_modal_actions():
    assert normalize_action_text("enter", keep_modal_actions=True) == "more"


def test_classify_modal_screen_detects_more_prompt():
    modal = classify_modal_screen("The goblin hits!\n--More--")
    assert modal.modal_type == "more_prompt"
    assert modal.disposition == "keep_modal"


def test_parse_tty_screen_extracts_message_board_and_status_lines():
    screen = "\n".join(
        [
            "You see here a potion.",
            "-----",
            "|.@.|",
            "-----",
        ]
        + [""] * 18
        + ["Dlvl:1  HP:14(14)", "Str:16 Dex:14"]
    )
    parsed = parse_tty_screen(screen)
    assert parsed["message"] == "You see here a potion."
    assert parsed["board_rows"][0] == "-----"
    assert parsed["status_lines"] == ["Dlvl:1  HP:14(14)", "Str:16 Dex:14"]


def test_render_policy_state_from_text_uses_canonical_fields():
    screen = "\n".join(
        [
            "You see here a potion.",
            "-----",
            "|.@.|",
            "-----",
        ]
        + [""] * 18
        + ["Dlvl:1  HP:14(14)", "Str:16 Dex:14"]
    )
    rendered = render_policy_state_from_text(screen, state_index=3, board_mode="tokenized")
    assert "TurnIndex: 3" in rendered.text
    assert "RenderSource: tty_text" in rendered.text
    assert "Message: You see here a potion." in rendered.text
    assert "StatusLines:" in rendered.text
    assert "BoardMode: tokenized" in rendered.text
    assert rendered.modal_state.modal_type == "none"


def test_is_dangerous_message_flags_combat_text():
    assert is_dangerous_message("The goblin hits!") is True
    assert is_dangerous_message("You see here a potion.") is False
