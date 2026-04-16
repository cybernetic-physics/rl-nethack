import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.board_view import (
    build_board_view,
    build_nth_board_view,
    decode_tokenized_board,
    render_ascii_board,
    render_tokenized_board,
)


def make_obs(chars_shape=(21, 79)):
    chars = np.full(chars_shape, ord(" "), dtype=np.uint8)
    chars[10, 30:36] = ord("-")
    chars[12, 30:36] = ord("-")
    chars[11, 30] = ord("|")
    chars[11, 35] = ord("|")
    chars[11, 31:35] = ord(".")
    chars[11, 33] = ord("@")
    chars[10, 33] = ord("+")
    return {"chars": chars}


def test_render_ascii_board_preserves_full_dimensions():
    obs = make_obs()
    board = render_ascii_board(obs)
    rows = board.splitlines()
    assert len(rows) == 21
    assert all(len(row) == 79 for row in rows)
    assert rows[11][33] == "@"
    assert rows[10][33] == "+"


def test_tokenized_board_round_trips_exactly():
    obs = make_obs()
    tokenized = render_tokenized_board(obs)
    decoded = decode_tokenized_board(tokenized)
    assert decoded == render_ascii_board(obs)


def test_build_board_view_reports_expected_metadata():
    obs = make_obs()
    view = build_board_view(obs, state_index=7)
    assert view.state_index == 7
    assert view.height == 21
    assert view.width == 79
    assert view.ascii_char_count == len(view.ascii_board)
    assert view.tokenized_char_count == len(view.tokenized_board)
    assert view.ascii_token_estimate > 0
    assert view.tokenized_token_estimate > 0


def test_tokenized_view_is_more_compact_for_sparse_board():
    obs = make_obs()
    view = build_board_view(obs)
    assert view.tokenized_char_count < view.ascii_char_count


def test_build_nth_board_view_selects_requested_observation():
    first = make_obs()
    second = make_obs()
    second["chars"][11, 33] = ord(".")
    second["chars"][11, 34] = ord("@")
    view = build_nth_board_view([first, second], 1)
    rows = view.ascii_board.splitlines()
    assert rows[11][34] == "@"
    assert rows[11][33] == "."
