"""
Utilities for rendering a full NetHack board view from an observation.

This module provides two complementary representations:

- exact full-board ASCII, preserving the entire chars grid
- a reversible, token-aware serialization that adaptively chooses raw or
  run-length encoded rows to reduce prompt footprint on sparse boards

The API is observation-centric so the same helpers can be reused from data
generation, debugging, evaluation, and reporting code.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


def _to_chars_grid(obs: dict[str, Any]) -> np.ndarray:
    chars = np.asarray(obs["chars"], dtype=np.uint8)
    if chars.ndim != 2:
        raise ValueError(f"Expected obs['chars'] to be 2D, got shape {chars.shape}")
    return chars


def _base36(n: int) -> str:
    if n < 0:
        raise ValueError("base36 input must be non-negative")
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if n == 0:
        return "0"
    out = []
    while n:
        n, rem = divmod(n, 36)
        out.append(digits[rem])
    return "".join(reversed(out))


def _from_base36(value: str) -> int:
    return int(value, 36)


def _encode_row_rle(row: str) -> str:
    if not row:
        return ""
    parts: list[str] = []
    current = row[0]
    run_len = 1
    for ch in row[1:]:
        if ch == current:
            run_len += 1
            continue
        parts.append(f"{_base36(run_len)}x{ord(current):02x}")
        current = ch
        run_len = 1
    parts.append(f"{_base36(run_len)}x{ord(current):02x}")
    return ",".join(parts)


def _decode_row_rle(encoded: str) -> str:
    if not encoded:
        return ""
    chars: list[str] = []
    for part in encoded.split(","):
        count_text, hex_text = part.split("x", 1)
        chars.append(chr(int(hex_text, 16)) * _from_base36(count_text))
    return "".join(chars)


def render_ascii_board(obs: dict[str, Any]) -> str:
    """Return the exact chars grid as ASCII, preserving full board width."""
    chars = _to_chars_grid(obs)
    rows = [bytes(chars[y]).decode("ascii", errors="replace") for y in range(chars.shape[0])]
    return "\n".join(rows)


def render_ascii_board_from_rows(rows: Sequence[str]) -> str:
    """Return an exact ASCII board from pre-rendered text rows."""
    return "\n".join(str(row) for row in rows)


def render_tokenized_board(obs: dict[str, Any]) -> str:
    """Return a reversible, token-aware serialization of the full board."""
    chars = _to_chars_grid(obs)
    rows = [bytes(chars[y]).decode("ascii", errors="replace") for y in range(chars.shape[0])]
    return render_tokenized_board_from_rows(rows)


def render_tokenized_board_from_rows(rows: Sequence[str]) -> str:
    """Return a reversible, token-aware serialization for text rows."""
    lines: list[str] = []
    for y, row in enumerate(rows):
        raw_payload = json.dumps(row, ensure_ascii=True)
        rle_payload = _encode_row_rle(row)
        if len(rle_payload) < len(raw_payload):
            lines.append(f"r{y:02d}|rle|{rle_payload}")
        else:
            lines.append(f"r{y:02d}|raw|{raw_payload}")
    return "\n".join(lines)


def decode_tokenized_board(tokenized_board: str) -> str:
    """Decode a tokenized board back into exact full-board ASCII."""
    rows: list[str] = []
    for line in tokenized_board.splitlines():
        if not line:
            continue
        _, mode, payload = line.split("|", 2)
        if mode == "raw":
            row = json.loads(payload)
        elif mode == "rle":
            row = _decode_row_rle(payload)
        else:
            raise ValueError(f"Unknown row mode: {mode}")
        rows.append(row)
    return "\n".join(rows)


def estimate_text_tokens(
    text: str,
    tokenizer: Any | Callable[[str], Any] | None = None,
) -> int:
    """Estimate token count, or use a provided tokenizer when available."""
    if tokenizer is None:
        return max(1, math.ceil(len(text) / 4)) if text else 0
    if callable(tokenizer):
        encoded = tokenizer(text)
        return len(encoded)
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        return len(encoded)
    raise TypeError("tokenizer must be None, callable, or expose .encode()")


@dataclass(frozen=True)
class BoardView:
    state_index: int
    height: int
    width: int
    ascii_board: str
    tokenized_board: str
    ascii_char_count: int
    tokenized_char_count: int
    ascii_token_estimate: int
    tokenized_token_estimate: int


def build_board_view(
    obs: dict[str, Any],
    *,
    state_index: int = 0,
    tokenizer: Any | Callable[[str], Any] | None = None,
) -> BoardView:
    """Build both full-board views plus rough token estimates."""
    chars = _to_chars_grid(obs)
    ascii_board = render_ascii_board(obs)
    tokenized_board = render_tokenized_board(obs)
    return BoardView(
        state_index=state_index,
        height=int(chars.shape[0]),
        width=int(chars.shape[1]),
        ascii_board=ascii_board,
        tokenized_board=tokenized_board,
        ascii_char_count=len(ascii_board),
        tokenized_char_count=len(tokenized_board),
        ascii_token_estimate=estimate_text_tokens(ascii_board, tokenizer=tokenizer),
        tokenized_token_estimate=estimate_text_tokens(tokenized_board, tokenizer=tokenizer),
    )


def build_board_view_from_rows(
    rows: Sequence[str],
    *,
    state_index: int = 0,
    tokenizer: Any | Callable[[str], Any] | None = None,
) -> BoardView:
    """Build both board views plus token estimates from raw text rows."""
    normalized_rows = [str(row) for row in rows]
    height = len(normalized_rows)
    width = max((len(row) for row in normalized_rows), default=0)
    ascii_board = render_ascii_board_from_rows(normalized_rows)
    tokenized_board = render_tokenized_board_from_rows(normalized_rows)
    return BoardView(
        state_index=state_index,
        height=height,
        width=width,
        ascii_board=ascii_board,
        tokenized_board=tokenized_board,
        ascii_char_count=len(ascii_board),
        tokenized_char_count=len(tokenized_board),
        ascii_token_estimate=estimate_text_tokens(ascii_board, tokenizer=tokenizer),
        tokenized_token_estimate=estimate_text_tokens(tokenized_board, tokenizer=tokenizer),
    )


def build_nth_board_view(
    observations: Sequence[dict[str, Any]],
    index: int,
    *,
    tokenizer: Any | Callable[[str], Any] | None = None,
) -> BoardView:
    """Build a board view for the Nth observation in a sequence."""
    if not observations:
        raise ValueError("observations must not be empty")
    obs = observations[index]
    return build_board_view(obs, state_index=index, tokenizer=tokenizer)
