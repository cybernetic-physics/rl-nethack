"""
NLD / ttyrec import helpers for long-sequence next-action training.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from typing import Any, Iterable

import numpy as np

from nle_agent.agent_http import _build_action_map
from src.long_sequence_dataset import convert_episode_jsonl_to_long_sequence_dataset
from src.state_encoder import StateEncoder


def _parse_int(value: Any, default: int = 0) -> int:
    """Parse NLD metadata integers, accepting decimal or 0x-prefixed strings."""
    if value is None or value == "":
        return default
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _reverse_action_map() -> dict[int, str]:
    action_map = _build_action_map()
    return {index: name for name, index in action_map.items()}


def build_keypress_action_name_map() -> dict[int, str]:
    """Map ttyrec keypress integers to canonical action names when possible."""
    import nle.nethack as nh

    reverse_action_map = _reverse_action_map()
    keypress_to_name: dict[int, str] = {}
    for index, action in enumerate(nh.ACTIONS):
        keypress = int(action)
        canonical = reverse_action_map.get(index)
        if canonical is None:
            enum_name = getattr(action, "name", None)
            canonical = enum_name.lower() if enum_name else f"keypress_{keypress}"
        keypress_to_name[keypress] = canonical
    return keypress_to_name


_KEYPRESS_ACTION_NAME_MAP = build_keypress_action_name_map()


def canonical_action_name_from_keypress(keypress: int) -> str:
    """Convert a ttyrec keypress to a canonical action name."""
    return _KEYPRESS_ACTION_NAME_MAP.get(int(keypress), f"keypress_{int(keypress)}")


def render_tty_chars_state(tty_chars: np.ndarray) -> str:
    """Render a ttyrec character grid into newline-delimited ASCII."""
    lines = []
    for row in tty_chars:
        text = bytes(np.asarray(row, dtype=np.uint8)).decode("ascii", errors="replace")
        lines.append(text.rstrip())
    return "\n".join(lines)


def _dataset_registration_row(dataset_name: str, dbfilename: str) -> tuple[str, int] | None:
    with sqlite3.connect(dbfilename) as conn:
        row = conn.execute(
            "SELECT root, ttyrec_version FROM roots WHERE dataset_name = ?",
            (dataset_name,),
        ).fetchone()
    return row


def register_nld_directory(
    root_path: str,
    dataset_name: str,
    *,
    dataset_type: str = "altorg",
    dbfilename: str = "ttyrecs.db",
) -> dict[str, Any]:
    """Register an NLD-compatible root in the local ttyrec sqlite database."""
    import nle.dataset as nld

    existing = _dataset_registration_row(dataset_name, dbfilename)
    if existing is not None:
        return {
            "dataset_name": dataset_name,
            "root_path": existing[0],
            "ttyrec_version": existing[1],
            "registered": False,
            "already_present": True,
        }

    os.makedirs(os.path.dirname(dbfilename) if os.path.dirname(dbfilename) else ".", exist_ok=True)
    if dataset_type == "nledata":
        nld.add_nledata_directory(root_path, dataset_name, filename=dbfilename)
    elif dataset_type == "altorg":
        nld.add_altorg_directory(root_path, dataset_name, filename=dbfilename)
    else:
        raise ValueError(f"Unsupported dataset_type {dataset_type!r}")

    row = _dataset_registration_row(dataset_name, dbfilename)
    return {
        "dataset_name": dataset_name,
        "root_path": row[0] if row else root_path,
        "ttyrec_version": row[1] if row else None,
        "registered": True,
        "already_present": False,
    }


def list_nld_game_metadata(dataset_name: str, *, dbfilename: str = "ttyrecs.db") -> list[dict]:
    """Load per-game metadata rows for one registered NLD dataset."""
    with sqlite3.connect(dbfilename) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT games.*
            FROM games
            INNER JOIN datasets ON games.gameid = datasets.gameid
            WHERE datasets.dataset_name = ?
            """,
            (dataset_name,),
        ).fetchall()
    return [dict(row) for row in rows]


def infer_metadata_outcome(row: dict) -> str:
    death_text = str(row.get("death", "") or "").lower()
    if "ascended" in death_text or "escaped" in death_text:
        return "win"
    if death_text:
        return "loss"
    achieve = _parse_int(row.get("achieve", 0), 0)
    if achieve & 0x0100:
        return "win"
    return "unknown"


def rank_nld_game_metadata(rows: list[dict]) -> list[dict]:
    """Attach a heuristic rank score and return rows sorted best-first."""

    def score(row: dict) -> tuple:
        outcome = infer_metadata_outcome(row)
        is_win = outcome == "win"
        achieve = _parse_int(row.get("achieve", 0), 0)
        maxlvl = _parse_int(row.get("maxlvl", 0), 0)
        turns = _parse_int(row.get("turns", 0), 0)
        points = _parse_int(row.get("points", 0), 0)
        return (
            1 if is_win else 0,
            achieve,
            maxlvl,
            turns,
            points,
            -_parse_int(row.get("gameid", 0), 0),
        )

    ranked = []
    for row in rows:
        enriched = dict(row)
        enriched["outcome"] = infer_metadata_outcome(row)
        enriched["is_win"] = enriched["outcome"] == "win"
        enriched["rank_score"] = list(score(row))
        ranked.append(enriched)
    ranked.sort(key=score, reverse=True)
    return ranked


def select_nld_gameids(
    rows: list[dict],
    *,
    max_games: int | None = None,
    wins_only: bool = False,
    min_turns: int = 0,
    min_maxlvl: int = 0,
) -> list[int]:
    """Filter and rank NLD games, returning the selected gameids."""
    filtered = []
    for row in rank_nld_game_metadata(rows):
        if wins_only and not row["is_win"]:
            continue
        if _parse_int(row.get("turns", 0), 0) < min_turns:
            continue
        if _parse_int(row.get("maxlvl", 0), 0) < min_maxlvl:
            continue
        filtered.append(_parse_int(row["gameid"], 0))
    if max_games is not None:
        filtered = filtered[:max_games]
    return filtered


def build_episode_rows_from_ttyrec_game(
    minibatches: Iterable[dict],
    *,
    dataset_name: str,
    gameid: int,
    metadata: dict | None = None,
    max_steps_per_game: int | None = None,
) -> list[dict]:
    """Convert ttyrec minibatches for one game into episode-style JSONL rows."""
    metadata = dict(metadata or {})
    rows = []
    step_index = 0
    for minibatch in minibatches:
        gameids = np.asarray(minibatch["gameids"])
        if gameids.ndim != 2 or gameids.shape[0] != 1:
            raise ValueError("build_episode_rows_from_ttyrec_game expects batch_size=1")
        for t in range(gameids.shape[1]):
            gid = int(gameids[0, t])
            if gid == 0:
                continue
            if gid != int(gameid):
                continue
            keypress = int(np.asarray(minibatch["keypresses"])[0, t])
            tty_chars = np.asarray(minibatch["tty_chars"])[0, t]
            row = {
                "episode_id": f"{dataset_name}-{gameid}",
                "gameid": int(gameid),
                "step": step_index,
                "action": canonical_action_name_from_keypress(keypress),
                "keypress": keypress,
                "state_text": render_tty_chars_state(tty_chars),
                "dataset_name": dataset_name,
                "source": dataset_name,
                "done": bool(int(np.asarray(minibatch["done"])[0, t])),
                **metadata,
            }
            if "scores" in minibatch:
                row["score"] = int(np.asarray(minibatch["scores"])[0, t])
            rows.append(row)
            step_index += 1
            if max_steps_per_game is not None and step_index >= max_steps_per_game:
                return rows
    return rows


def export_nld_episodes_to_jsonl(
    output_path: str,
    *,
    dataset_name: str,
    dbfilename: str = "ttyrecs.db",
    root_path: str | None = None,
    dataset_type: str = "altorg",
    max_games: int | None = None,
    wins_only: bool = False,
    min_turns: int = 0,
    min_maxlvl: int = 0,
    rows: int = 24,
    cols: int = 80,
    seq_length: int = 64,
    max_steps_per_game: int | None = None,
) -> dict[str, Any]:
    """Export selected NLD games into a flexible episode-style JSONL file."""
    import nle.dataset as nld

    registration = None
    if root_path:
        registration = register_nld_directory(
            root_path,
            dataset_name,
            dataset_type=dataset_type,
            dbfilename=dbfilename,
        )

    metadata_rows = list_nld_game_metadata(dataset_name, dbfilename=dbfilename)
    selected_gameids = select_nld_gameids(
        metadata_rows,
        max_games=max_games,
        wins_only=wins_only,
        min_turns=min_turns,
        min_maxlvl=min_maxlvl,
    )
    metadata_by_gameid = {int(row["gameid"]): rank_nld_game_metadata([row])[0] for row in metadata_rows}

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    exported_rows = 0
    with open(output_path, "w") as f:
        for gameid in selected_gameids:
            dataset = nld.TtyrecDataset(
                dataset_name,
                batch_size=1,
                seq_length=seq_length,
                rows=rows,
                cols=cols,
                dbfilename=dbfilename,
                gameids=[gameid],
                shuffle=False,
                loop_forever=False,
            )
            episode_rows = build_episode_rows_from_ttyrec_game(
                dataset,
                dataset_name=dataset_name,
                gameid=gameid,
                metadata=metadata_by_gameid.get(gameid, {}),
                max_steps_per_game=max_steps_per_game,
            )
            for row in episode_rows:
                f.write(json.dumps(row) + "\n")
            exported_rows += len(episode_rows)
    return {
        "dataset_name": dataset_name,
        "output_path": output_path,
        "games_considered": len(metadata_rows),
        "games_exported": len(selected_gameids),
        "rows_exported": exported_rows,
        "wins_only": wins_only,
        "min_turns": min_turns,
        "min_maxlvl": min_maxlvl,
        "registration": registration,
    }


def import_nld_to_long_sequence_dataset(
    *,
    output_path: str,
    dataset_name: str,
    encoder: StateEncoder,
    dbfilename: str = "ttyrecs.db",
    root_path: str | None = None,
    dataset_type: str = "altorg",
    max_games: int | None = None,
    wins_only: bool = False,
    min_turns: int = 0,
    min_maxlvl: int = 0,
    rows: int = 24,
    cols: int = 80,
    seq_length: int = 64,
    max_steps_per_game: int | None = None,
    max_context_tokens: int = 128_000,
    reserve_output_tokens: int = 16,
    source: str | None = None,
) -> dict[str, Any]:
    """Register/select/export NLD games and convert them into long-sequence JSONL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        episode_path = os.path.join(tmpdir, f"{dataset_name}_episodes.jsonl")
        export_result = export_nld_episodes_to_jsonl(
            episode_path,
            dataset_name=dataset_name,
            dbfilename=dbfilename,
            root_path=root_path,
            dataset_type=dataset_type,
            max_games=max_games,
            wins_only=wins_only,
            min_turns=min_turns,
            min_maxlvl=min_maxlvl,
            rows=rows,
            cols=cols,
            seq_length=seq_length,
            max_steps_per_game=max_steps_per_game,
        )
        convert_result = convert_episode_jsonl_to_long_sequence_dataset(
            episode_path,
            output_path,
            encoder=encoder,
            max_context_tokens=max_context_tokens,
            board_mode="tokenized",
            reserve_output_tokens=reserve_output_tokens,
            source=source or f"nld:{dataset_name}",
        )
    return {
        "dataset_name": dataset_name,
        "episode_export": export_result,
        "long_sequence_output": convert_result,
    }
