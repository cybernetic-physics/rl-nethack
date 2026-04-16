import os
import sys
import json

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.nld_long_sequence_import import (
    build_episode_rows_from_ttyrec_game,
    canonical_action_name_from_keypress,
    export_hdf5_nld_episodes_to_jsonl,
    rank_nld_game_metadata,
    render_tty_chars_state,
    select_nld_gameids,
)


def test_canonical_action_name_from_keypress_maps_common_actions():
    assert canonical_action_name_from_keypress(46) == "wait"
    assert canonical_action_name_from_keypress(107) == "north"
    assert canonical_action_name_from_keypress(13) == "more"


def test_render_tty_chars_state_renders_ascii_grid():
    tty = np.array(
        [
            [ord("."), ord("@"), ord(" "), ord(" ")],
            [ord("-"), ord("|"), ord(" "), ord(" ")],
        ],
        dtype=np.uint8,
    )
    assert render_tty_chars_state(tty) == ".@\n-|"


def test_rank_nld_game_metadata_prefers_wins_then_depth_and_turns():
    rows = [
        {"gameid": 1, "death": "killed by a newt", "achieve": 0, "maxlvl": 20, "turns": 2000, "points": 1000},
        {"gameid": 2, "death": "ascended", "achieve": 0x0100, "maxlvl": 10, "turns": 500, "points": 5000},
        {"gameid": 3, "death": "quit", "achieve": 0, "maxlvl": 25, "turns": 4000, "points": 2000},
    ]
    ranked = rank_nld_game_metadata(rows)
    assert ranked[0]["gameid"] == 2
    assert ranked[0]["is_win"] is True
    selected = select_nld_gameids(rows, wins_only=False, min_turns=1000)
    assert selected == [3, 1]


def test_build_episode_rows_from_ttyrec_game_decodes_minibatch():
    minibatch = {
        "gameids": np.array([[42, 42, 0]], dtype=np.int32),
        "keypresses": np.array([[107, 46, 0]], dtype=np.uint8),
        "tty_chars": np.array(
            [
                [
                    [[ord("@"), ord(" "), ord(" ")], [ord("-"), ord(" "), ord(" ")]],
                    [[ord("."), ord("@"), ord(" ")], [ord("-"), ord(" "), ord(" ")]],
                    [[0, 0, 0], [0, 0, 0]],
                ]
            ],
            dtype=np.uint8,
        ),
        "done": np.array([[0, 1, 0]], dtype=np.uint8),
        "scores": np.array([[10, 11, 0]], dtype=np.int32),
    }
    rows = build_episode_rows_from_ttyrec_game(
        [minibatch],
        dataset_name="nld-nao",
        gameid=42,
        metadata={"death": "ascended", "turns": 12},
    )
    assert len(rows) == 2
    assert rows[0]["action"] == "north"
    assert rows[1]["action"] == "wait"
    assert rows[1]["done"] is True
    assert rows[0]["episode_id"] == "nld-nao-42"
    assert rows[0]["score"] == 10


def test_export_hdf5_nld_episodes_to_jsonl(tmp_path):
    h5py = pytest.importorskip("h5py")

    hdf5_path = tmp_path / "sample.hdf5"
    metadata_path = tmp_path / "metadata.json"
    output_path = tmp_path / "episodes.jsonl"

    with h5py.File(hdf5_path, "w") as h5:
        group = h5.create_group("42")
        tty = np.array(
            [
                [[ord("@"), ord(" "), ord(" ")], [ord("-"), ord(" "), ord(" ")]],
                [[ord("."), ord("@"), ord(" ")], [ord("-"), ord(" "), ord(" ")]],
            ],
            dtype=np.uint8,
        )
        group.create_dataset("tty_chars", data=tty)
        group.create_dataset("actions", data=np.array([107, 46], dtype=np.int16))
        group.create_dataset("dones", data=np.array([0, 1], dtype=np.bool_))
        group.create_dataset("rewards", data=np.array([3, 4], dtype=np.int32))

    metadata_path.write_text(json.dumps([
        {"gameid": 42, "death": "ascended", "achieve": 0x0100, "turns": 99, "maxlvl": 5}
    ]))

    result = export_hdf5_nld_episodes_to_jsonl(
        str(output_path),
        hdf5_path=str(hdf5_path),
        metadata_json_path=str(metadata_path),
        dataset_name="hf-nld",
        max_games=1,
    )

    assert result["games_exported"] == 1
    assert result["rows_exported"] == 2
    with open(output_path) as f:
        rows = [json.loads(line) for line in f]
    assert rows[0]["episode_id"] == "hf-nld-42"
    assert rows[0]["action"] == "north"
    assert rows[1]["done"] is True
    assert rows[1]["reward"] == 4
