import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.long_sequence_dataset import (
    EpisodeActionStep,
    build_long_sequence_examples_from_episode,
    convert_episode_jsonl_to_long_sequence_dataset,
    context_bucket,
    infer_game_phase,
    infer_outcome_from_nle_info,
    infer_outcome_label,
    render_state_block,
)
from src.policy_actions import canonicalize_action
from src.policy_replay import ModalState, ReplayedEpisodeStep
from src.state_encoder import StateEncoder


def make_obs(turn: int, player_x: int = 33, player_y: int = 11, message: str = ""):
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    chars[10, 30:36] = ord("-")
    chars[12, 30:36] = ord("-")
    chars[11, 30] = ord("|")
    chars[11, 35] = ord("|")
    chars[11, 31:35] = ord(".")
    chars[player_y, player_x] = ord("@")
    chars[10, 33] = ord("+")

    blstats = np.zeros(27, dtype=np.int64)
    blstats[0] = player_x
    blstats[1] = player_y
    blstats[3] = 16
    blstats[4] = 14
    blstats[10] = 14
    blstats[11] = 14
    blstats[12] = 1
    blstats[13] = 0
    blstats[16] = 5
    blstats[20] = turn

    encoded_message = (message.encode("ascii", errors="replace") + b"\x00" * 256)[:256]
    return {
        "chars": chars,
        "blstats": blstats,
        "message": np.frombuffer(encoded_message, dtype=np.uint8).copy(),
    }


def test_render_state_block_uses_tokenized_board_mode():
    encoder = StateEncoder()
    text, tokens = render_state_block(make_obs(turn=1), encoder=encoder, state_index=0, board_mode="tokenized")
    assert "BoardMode: tokenized" in text
    assert "Board:" in text
    assert "r00|" in text
    assert tokens > 0


def test_build_examples_from_episode_trims_history_by_budget():
    encoder = StateEncoder()
    episode = [
        EpisodeActionStep(turn_index=0, obs=make_obs(turn=1, message="start"), state_text=None, action="east"),
        EpisodeActionStep(turn_index=1, obs=make_obs(turn=2, player_x=34, message="move"), state_text=None, action="south"),
        EpisodeActionStep(turn_index=2, obs=make_obs(turn=3, player_x=34, player_y=12, message="descend"), state_text=None, action="wait"),
    ]
    examples = build_long_sequence_examples_from_episode(
        episode,
        encoder=encoder,
        episode_id="ep-1",
        max_context_tokens=140,
        board_mode="tokenized",
    )
    assert len(examples) == 3
    final = examples[-1]
    metadata = final["metadata"]
    assert metadata["history_steps_available"] == 2
    assert metadata["history_steps_included"] < metadata["history_steps_available"]
    assert metadata["context_budget_exceeded"] is True


def test_build_examples_from_episode_can_persist_dual_views():
    encoder = StateEncoder()
    episode = [
        EpisodeActionStep(turn_index=0, obs=make_obs(turn=1, message="start"), state_text=None, action="east"),
    ]
    examples = build_long_sequence_examples_from_episode(
        episode,
        encoder=encoder,
        episode_id="ep-dual",
        max_context_tokens=256,
        board_mode="tokenized",
        persist_dual_views=True,
    )
    assert len(examples) == 1
    assert examples[0]["metadata"]["has_dual_views"] is True
    assert "board_views" in examples[0]
    assert "ascii_board" in examples[0]["board_views"]
    assert "tokenized_board" in examples[0]["board_views"]
    assert examples[0]["board_views"]["width"] == 79


def test_build_examples_from_replayed_steps_preserves_canonical_history_text():
    encoder = StateEncoder()
    step0 = ReplayedEpisodeStep(
        turn_index=0,
        state_text=(
            "TurnIndex: 0\n"
            "RenderSource: obs\n"
            "Message: start\n"
            "BoardMode: tokenized\n"
            "BoardShape: 2x2\n"
            "StatusLines:\n"
            "Stats: HP=1/1 AC=10 Str=10 Dex=10 Gold=0 Depth=1 Pos=(1, 1) Clock=1\n"
            "Board:\n"
            "r00|raw|\"..\"\n"
            "r01|raw|\".@\""
        ),
        token_estimate=32,
        action=canonicalize_action("east"),
        render_source="obs",
        board_mode="tokenized",
        board_views=None,
        modal_state=ModalState("keep_gameplay", "none", "obs_gameplay"),
        extra_metadata={},
    )
    step1 = ReplayedEpisodeStep(
        turn_index=1,
        state_text=(
            "TurnIndex: 1\n"
            "RenderSource: obs\n"
            "Message: moved\n"
            "BoardMode: tokenized\n"
            "BoardShape: 2x2\n"
            "StatusLines:\n"
            "Stats: HP=1/1 AC=10 Str=10 Dex=10 Gold=0 Depth=1 Pos=(2, 1) Clock=2\n"
            "Board:\n"
            "r00|raw|\"..\"\n"
            "r01|raw|\".@\""
        ),
        token_estimate=32,
        action=canonicalize_action("south"),
        render_source="obs",
        board_mode="tokenized",
        board_views=None,
        modal_state=ModalState("keep_gameplay", "none", "obs_gameplay"),
        extra_metadata={},
    )

    examples = build_long_sequence_examples_from_episode(
        [step0, step1],
        encoder=encoder,
        episode_id="ep-replayed",
        max_context_tokens=4096,
    )

    user_content = examples[1]["conversations"][1]["content"]
    assert "HistoryTurns:\nTurnIndex: 0\nRenderSource: obs\nMessage: start" in user_content
    assert 'r00|raw|"RenderSource: obs"' not in user_content


def test_context_bucket_labels_large_windows():
    assert context_bucket(128_000) == "128k"
    assert context_bucket(256_000) == "256k"
    assert context_bucket(512_000) == "512k"
    assert context_bucket(1_000_000) == "1M"


def test_phase_and_outcome_heuristics():
    assert infer_game_phase(depth=3) == "early"
    assert infer_game_phase(depth=10) == "mid"
    assert infer_game_phase(depth=20) == "late"
    assert infer_game_phase(achieve=0x0020) == "amulet"
    assert infer_game_phase(achieve=0x0100) == "ascended"
    assert infer_outcome_label(death="ascended") == "win"
    assert infer_outcome_label(death="killed by a newt") == "loss"
    assert infer_outcome_label(achieve=0x0100) == "win"
    assert infer_outcome_from_nle_info({"is_ascended": True}, terminated=True, truncated=False) == "win"
    assert infer_outcome_from_nle_info({"is_ascended": False}, terminated=True, truncated=False) == "loss"
    assert infer_outcome_from_nle_info({}, terminated=False, truncated=True) == "truncated"


def test_convert_episode_jsonl_to_long_sequence_dataset(tmp_path):
    encoder = StateEncoder()
    input_path = tmp_path / "episode_rows.jsonl"
    output_path = tmp_path / "long_sequences.jsonl"
    rows = [
        {
            "episode_id": "ep-a",
            "step": 0,
            "state_prompt": "HP:14/14\nBoard: room",
            "action": "east",
            "depth": 1,
            "death": "killed by a jackal",
        },
        {
            "episode_id": "ep-a",
            "step": 1,
            "state_prompt": "HP:12/14\nBoard: corridor",
            "action": "south",
            "depth": 2,
            "death": "killed by a jackal",
        },
    ]
    with open(input_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    result = convert_episode_jsonl_to_long_sequence_dataset(
        str(input_path),
        str(output_path),
        encoder=encoder,
        max_context_tokens=256,
    )
    assert result["episodes"] == 1
    assert result["examples"] == 2
    assert os.path.isfile(result["validation_path"])

    with open(output_path, "r") as f:
        converted = [json.loads(line) for line in f if line.strip()]
    assert len(converted) == 2
    assert converted[0]["metadata"]["outcome"] == "loss"
    assert converted[0]["metadata"]["game_phase"] == "early"
    assert converted[1]["metadata"]["history_steps_available"] == 1
    assert converted[0]["metadata"]["has_dual_views"] is False
    assert converted[0]["metadata"]["render_source"] in {"tty_text", "fallback_text"}


def test_convert_episode_jsonl_to_long_sequence_dataset_drops_modal_rows(tmp_path):
    encoder = StateEncoder()
    input_path = tmp_path / "episode_rows_modal.jsonl"
    output_path = tmp_path / "long_sequences_modal.jsonl"
    rows = [
        {
            "episode_id": "ep-modal",
            "step": 0,
            "state_prompt": "You see here a potion.\n--More--",
            "action": "more",
        },
        {
            "episode_id": "ep-modal",
            "step": 1,
            "state_prompt": "You see here a potion.\n-----\n|.@.|\n-----",
            "action": "east",
        },
    ]
    with open(input_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    result = convert_episode_jsonl_to_long_sequence_dataset(
        str(input_path),
        str(output_path),
        encoder=encoder,
        max_context_tokens=256,
    )
    assert result["examples"] == 1
    with open(result["validation_path"], "r") as f:
        report = json.load(f)
    assert report["rows_before_replay"] == 2
    assert report["rows_after_replay"] == 1
    assert report["dropped_by_reason"]["more_prompt"] == 1


def test_convert_episode_jsonl_to_long_sequence_dataset_supports_danger_only(tmp_path):
    encoder = StateEncoder()
    input_path = tmp_path / "episode_rows_danger.jsonl"
    output_path = tmp_path / "long_sequences_danger.jsonl"
    rows = [
        {
            "episode_id": "ep-danger",
            "step": 0,
            "state_prompt": "You are hit!\n-----\n|.@.|\n-----",
            "action": "west",
            "depth": 1,
        },
        {
            "episode_id": "ep-danger",
            "step": 1,
            "state_prompt": "You see here a potion.\n-----\n|.@.|\n-----",
            "action": "east",
            "depth": 1,
        },
        {
            "episode_id": "ep-danger",
            "step": 2,
            "state_prompt": "You see here a potion.\n-----\n|.@.|\n-----",
            "action": "north",
            "depth": 1,
        },
    ]
    with open(input_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    result = convert_episode_jsonl_to_long_sequence_dataset(
        str(input_path),
        str(output_path),
        encoder=encoder,
        max_context_tokens=256,
        danger_only=True,
        danger_window=1,
    )
    assert result["examples"] == 2
    with open(result["validation_path"], "r") as f:
        report = json.load(f)
    assert report["dropped_by_reason"]["not_in_danger_window"] == 1
