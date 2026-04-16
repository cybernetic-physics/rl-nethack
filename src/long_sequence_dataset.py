"""
Long-context sequence dataset builder for NetHack next-action training.

This module builds ShareGPT-style JSONL examples where the user message contains
as much recent game history as possible under a token budget, and the assistant
target is the next action.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional

import numpy as np

from nle_agent.agent_http import _build_action_map
from src.board_view import build_board_view, estimate_text_tokens
from src.state_encoder import StateEncoder

SYSTEM_PROMPT = (
    "You are choosing the next NetHack action from a long game history. "
    "Read the prior turns and current full-board state carefully, then reply "
    "with exactly one action word and nothing else."
)

_ACTION_MAP: Optional[Dict[str, int]] = None
_DIRECTIONS = ["north", "south", "east", "west"]


def _get_action_map() -> Dict[str, int]:
    global _ACTION_MAP
    if _ACTION_MAP is None:
        _ACTION_MAP = _build_action_map()
    return _ACTION_MAP


def _decode_message(msg_array: np.ndarray) -> str:
    raw = bytes(msg_array).decode("ascii", errors="replace")
    return raw.strip().rstrip("\x00").strip()


def _copy_obs(obs: dict) -> dict:
    copied = {}
    for key in ("chars", "blstats", "message"):
        copied[key] = np.array(obs[key], copy=True)
    return copied


def wall_avoidance_policy(adjacent: dict, rng: random.Random) -> str:
    open_dirs = [
        d for d in _DIRECTIONS
        if adjacent.get(d, "unseen") not in ("wall", "unseen")
    ]
    if open_dirs:
        return rng.choice(open_dirs)
    return "wait"


def context_bucket(max_context_tokens: int) -> str:
    if max_context_tokens >= 1_000_000:
        return "1M"
    if max_context_tokens >= 512_000:
        return "512k"
    if max_context_tokens >= 256_000:
        return "256k"
    if max_context_tokens >= 128_000:
        return "128k"
    return str(max_context_tokens)


def infer_game_phase(
    *,
    depth: int | None = None,
    maxlvl: int | None = None,
    achieve: int | str | None = None,
) -> str:
    """Heuristic game-phase label for metadata."""
    depth_value = int(depth if depth is not None else (maxlvl or 0))
    achieve_int = 0
    if isinstance(achieve, str):
        try:
            achieve_int = int(achieve, 0)
        except ValueError:
            achieve_int = 0
    elif achieve is not None:
        achieve_int = int(achieve)
    if achieve_int & 0x0100:
        return "ascended"
    if achieve_int & 0x0080:
        return "astral"
    if achieve_int & 0x0040:
        return "endgame"
    if achieve_int & 0x0020:
        return "amulet"
    if achieve_int & 0x0010:
        return "invocation"
    if achieve_int & 0x0002:
        return "gehennom"
    if depth_value >= 15:
        return "late"
    if depth_value >= 7:
        return "mid"
    return "early"


def infer_outcome_label(*, death: str | None = None, achieve: int | str | None = None) -> str:
    """Heuristic episode-outcome label for metadata."""
    phase = infer_game_phase(achieve=achieve)
    if phase == "ascended":
        return "win"
    death_text = (death or "").lower()
    if death_text:
        if "ascended" in death_text or "escaped" in death_text:
            return "win"
        return "loss"
    return "unknown"


def infer_outcome_from_nle_info(info: dict | None, *, terminated: bool, truncated: bool) -> str:
    """Infer a coarse outcome label from NLE episode termination info."""
    info = info or {}
    if bool(info.get("is_ascended")):
        return "win"
    if terminated:
        return "loss"
    if truncated:
        return "truncated"
    return "unknown"


@dataclass(frozen=True)
class RenderedTurn:
    turn_index: int
    state_text: str
    state_token_estimate: int
    turn_text: str
    turn_token_estimate: int
    action: str


@dataclass(frozen=True)
class EpisodeActionStep:
    turn_index: int
    obs: Optional[dict]
    state_text: Optional[str]
    action: str
    extra_metadata: Optional[dict[str, Any]] = None


def render_state_block(
    obs: dict,
    *,
    encoder: StateEncoder,
    state_index: int,
    board_mode: str = "tokenized",
    tokenizer=None,
) -> tuple[str, int]:
    """Render one full-state block with a whole-board view."""
    state = encoder.encode_full(obs)
    board_view = build_board_view(obs, state_index=state_index, tokenizer=tokenizer)
    board_text = (
        board_view.tokenized_board if board_mode == "tokenized" else board_view.ascii_board
    )
    lines = [
        f"TurnIndex: {state_index}",
        (
            f"Stats: HP={state['hp']}/{state['hp_max']} AC={state['ac']} "
            f"Str={state['strength']} Dex={state['dexterity']} Gold={state['gold']} "
            f"Depth={state['depth']} Pos={state['position']} Clock={state['turn']}"
        ),
        f"Message: {_decode_message(obs['message']) or '<none>'}",
        f"BoardMode: {board_mode}",
        f"BoardShape: {board_view.height}x{board_view.width}",
        "Board:",
        board_text,
    ]
    text = "\n".join(lines)
    tokens = estimate_text_tokens(text, tokenizer=tokenizer)
    return text, tokens


def render_state_views(
    obs: dict,
    *,
    state_index: int,
    tokenizer=None,
) -> dict[str, object]:
    """Render both exact and compact board views for optional persistence."""
    board_view = build_board_view(obs, state_index=state_index, tokenizer=tokenizer)
    return {
        "ascii_board": board_view.ascii_board,
        "tokenized_board": board_view.tokenized_board,
        "ascii_chars": board_view.ascii_char_count,
        "tokenized_chars": board_view.tokenized_char_count,
        "ascii_tokens_estimate": board_view.ascii_token_estimate,
        "tokenized_tokens_estimate": board_view.tokenized_token_estimate,
        "height": board_view.height,
        "width": board_view.width,
        "state_index": state_index,
    }


def render_text_state_block(
    state_text: str,
    *,
    state_index: int,
    tokenizer=None,
) -> tuple[str, int]:
    text = "\n".join(
        [
            f"TurnIndex: {state_index}",
            "BoardMode: external_text",
            "State:",
            state_text,
        ]
    )
    tokens = estimate_text_tokens(text, tokenizer=tokenizer)
    return text, tokens


def render_turn(
    *,
    obs: Optional[dict],
    state_text: Optional[str],
    action: str,
    encoder: StateEncoder,
    turn_index: int,
    board_mode: str = "tokenized",
    tokenizer=None,
) -> RenderedTurn:
    if obs is not None:
        rendered_state_text, state_tokens = render_state_block(
            obs,
            encoder=encoder,
            state_index=turn_index,
            board_mode=board_mode,
            tokenizer=tokenizer,
        )
    elif state_text is not None:
        rendered_state_text, state_tokens = render_text_state_block(
            state_text,
            state_index=turn_index,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError("render_turn requires either obs or state_text")
    turn_text = f"{rendered_state_text}\nAction: {action}"
    turn_tokens = estimate_text_tokens(turn_text, tokenizer=tokenizer)
    return RenderedTurn(
        turn_index=turn_index,
        state_text=rendered_state_text,
        state_token_estimate=state_tokens,
        turn_text=turn_text,
        turn_token_estimate=turn_tokens,
        action=action,
    )


def build_messages(
    *,
    history_turns: list[RenderedTurn],
    current_state_text: str,
    current_state_tokens: int,
    target_action: str,
    episode_id: str,
    step_index: int,
    max_context_tokens: int,
    reserve_output_tokens: int = 16,
    tokenizer=None,
) -> tuple[list[dict], dict]:
    """Build a trimmed history prompt plus metadata."""
    prefix = (
        f"EpisodeId: {episode_id}\n"
        f"TargetStep: {step_index}\n"
        "Task: Predict the next NetHack action from the longest recent history that fits.\n"
    )
    suffix = "\nCurrentTurn:\n{current}\nNextAction:"
    system_tokens = estimate_text_tokens(SYSTEM_PROMPT, tokenizer=tokenizer)
    prefix_tokens = estimate_text_tokens(prefix, tokenizer=tokenizer)
    suffix_tokens = estimate_text_tokens(
        suffix.format(current=current_state_text),
        tokenizer=tokenizer,
    )

    available = max(
        0,
        max_context_tokens - system_tokens - prefix_tokens - suffix_tokens - reserve_output_tokens,
    )
    selected: list[RenderedTurn] = []
    used = 0
    for turn in reversed(history_turns):
        if selected and used + turn.turn_token_estimate > available:
            break
        if not selected and turn.turn_token_estimate > available:
            break
        selected.append(turn)
        used += turn.turn_token_estimate
    selected.reverse()

    history_lines = ["HistoryTurns:"]
    if selected:
        history_lines.extend(turn.turn_text for turn in selected)
    else:
        history_lines.append("<none>")
    user_content = prefix + "\n".join(history_lines) + suffix.format(current=current_state_text)
    total_tokens = (
        system_tokens
        + estimate_text_tokens(user_content, tokenizer=tokenizer)
        + estimate_text_tokens(target_action, tokenizer=tokenizer)
    )
    metadata = {
        "episode_id": episode_id,
        "step_index": step_index,
        "history_steps_available": len(history_turns),
        "history_steps_included": len(selected),
        "history_turn_start": selected[0].turn_index if selected else step_index,
        "history_turn_end": selected[-1].turn_index if selected else step_index - 1,
        "current_state_tokens_estimate": current_state_tokens,
        "context_tokens_estimate": total_tokens,
        "max_context_tokens": max_context_tokens,
        "target_context_tokens": max_context_tokens,
        "context_bucket": context_bucket(max_context_tokens),
        "context_budget_exceeded": total_tokens > max_context_tokens,
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": target_action},
    ]
    return messages, metadata


def build_long_sequence_examples_from_episode(
    episode_steps: list[EpisodeActionStep],
    *,
    encoder: StateEncoder,
    episode_id: str,
    max_context_tokens: int = 128_000,
    board_mode: str = "tokenized",
    persist_dual_views: bool = False,
    reserve_output_tokens: int = 16,
    source: str = "episode",
    tokenizer=None,
) -> list[dict]:
    """Convert a full episode into long-history next-action examples."""
    history_turns: list[RenderedTurn] = []
    examples: list[dict] = []
    for step in episode_steps:
        if step.obs is not None:
            current_state_text, current_state_tokens = render_state_block(
                step.obs,
                encoder=encoder,
                state_index=step.turn_index,
                board_mode=board_mode,
                tokenizer=tokenizer,
            )
        elif step.state_text is not None:
            current_state_text, current_state_tokens = render_text_state_block(
                step.state_text,
                state_index=step.turn_index,
                tokenizer=tokenizer,
            )
        else:
            raise ValueError("EpisodeActionStep requires obs or state_text")
        messages, metadata = build_messages(
            history_turns=history_turns,
            current_state_text=current_state_text,
            current_state_tokens=current_state_tokens,
            target_action=step.action,
            episode_id=episode_id,
            step_index=step.turn_index,
            max_context_tokens=max_context_tokens,
            reserve_output_tokens=reserve_output_tokens,
            tokenizer=tokenizer,
        )
        examples.append(
            {
                "conversations": messages,
                "metadata": {
                    **metadata,
                    "source": source,
                    "board_mode": board_mode,
                    "has_dual_views": persist_dual_views,
                    **(step.extra_metadata or {}),
                },
                **(
                    {
                        "board_views": render_state_views(
                            step.obs,
                            state_index=step.turn_index,
                            tokenizer=tokenizer,
                        )
                    }
                    if persist_dual_views and step.obs is not None
                    else {}
                ),
            }
        )
        history_turns.append(
            render_turn(
                obs=step.obs,
                state_text=step.state_text,
                action=step.action,
                encoder=encoder,
                turn_index=step.turn_index,
                board_mode=board_mode,
                tokenizer=tokenizer,
            )
        )
    return examples


def build_long_sequence_examples_from_episode_multi_budget(
    episode_steps: list[EpisodeActionStep],
    *,
    encoder: StateEncoder,
    episode_id: str,
    context_budgets: list[int],
    board_mode: str = "tokenized",
    persist_dual_views: bool = False,
    reserve_output_tokens: int = 16,
    source: str = "episode",
    tokenizer=None,
) -> list[dict]:
    """Emit one set of long-sequence examples per requested context budget."""
    all_examples: list[dict] = []
    for budget in context_budgets:
        examples = build_long_sequence_examples_from_episode(
            episode_steps,
            encoder=encoder,
            episode_id=episode_id,
            max_context_tokens=budget,
            board_mode=board_mode,
            persist_dual_views=persist_dual_views,
            reserve_output_tokens=reserve_output_tokens,
            source=source,
            tokenizer=tokenizer,
        )
        for row in examples:
            row["metadata"]["target_context_tokens"] = budget
            row["metadata"]["target_context_bucket"] = context_bucket(budget)
        all_examples.extend(examples)
    return all_examples


def generate_long_sequence_game(
    seed: int,
    max_steps: int,
    encoder: StateEncoder,
    *,
    policy: Optional[Callable] = None,
    max_context_tokens: int = 128_000,
    board_mode: str = "tokenized",
    persist_dual_views: bool = False,
    reserve_output_tokens: int = 16,
    source: str = "nle_generated",
    tokenizer=None,
) -> Iterator[str]:
    """Play one game and emit long-history next-action examples."""
    import nle.env

    if policy is None:
        policy = wall_avoidance_policy

    rng = random.Random(seed)
    action_map = _get_action_map()
    env = nle.env.NLE()
    obs, _ = env.reset(seed=seed)
    episode_id = f"{source}-seed-{seed}"
    episode_steps: list[EpisodeActionStep] = []
    final_outcome = "unknown"
    final_is_win = False
    max_depth_seen = 0

    for step in range(max_steps):
        obs_copy = _copy_obs(obs)
        state = encoder.encode_full(obs_copy)
        max_depth_seen = max(max_depth_seen, int(state["depth"]))
        action_name = policy(state["adjacent"], rng)
        action_idx = action_map.get(action_name, action_map.get("wait", 18))
        if action_name not in action_map:
            action_name = "wait"
        episode_steps.append(
            EpisodeActionStep(
                turn_index=step,
                obs=obs_copy,
                state_text=None,
                action=action_name,
                extra_metadata={
                    "hp": state["hp"],
                    "hp_max": state["hp_max"],
                    "depth": state["depth"],
                    "turn": state["turn"],
                    "game_phase": infer_game_phase(depth=state["depth"]),
                    "outcome": "unknown",
                    "is_win": False,
                },
            )
        )

        obs_after, _, terminated, truncated, info = env.step(action_idx)
        obs = obs_after
        if terminated or truncated:
            final_outcome = infer_outcome_from_nle_info(info, terminated=terminated, truncated=truncated)
            final_is_win = final_outcome == "win"
            break

    env.close()
    final_phase = infer_game_phase(depth=max_depth_seen)
    if final_is_win:
        final_phase = "ascended"
    finalized_steps = []
    for step in episode_steps:
        extra = dict(step.extra_metadata or {})
        extra["outcome"] = final_outcome
        extra["is_win"] = final_is_win
        extra["game_phase"] = final_phase
        finalized_steps.append(
            EpisodeActionStep(
                turn_index=step.turn_index,
                obs=step.obs,
                state_text=step.state_text,
                action=step.action,
                extra_metadata=extra,
            )
        )
    examples = build_long_sequence_examples_from_episode(
        finalized_steps,
        encoder=encoder,
        episode_id=episode_id,
        max_context_tokens=max_context_tokens,
        board_mode=board_mode,
        persist_dual_views=persist_dual_views,
        reserve_output_tokens=reserve_output_tokens,
        source=source,
        tokenizer=tokenizer,
    )
    for payload in examples:
        payload["metadata"]["seed"] = seed
        yield json.dumps(payload)


def generate_long_sequence_dataset(
    output_path: str,
    *,
    num_games: int,
    max_steps: int,
    seed_start: int,
    encoder: StateEncoder,
    max_context_tokens: int = 128_000,
    board_mode: str = "tokenized",
    persist_dual_views: bool = False,
    eval_path: Optional[str] = None,
    eval_fraction: float = 0.2,
    policy: Optional[Callable] = None,
    reserve_output_tokens: int = 16,
    source: str = "nle_generated",
    tokenizer=None,
) -> dict:
    """Generate long-sequence JSONL train/eval files."""
    num_eval = int(num_games * eval_fraction) if eval_path else 0
    num_train = num_games - num_eval
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    if eval_path:
        os.makedirs(os.path.dirname(eval_path) if os.path.dirname(eval_path) else ".", exist_ok=True)

    train_lines: list[str] = []
    eval_lines: list[str] = []
    total_examples = 0
    for i in range(num_games):
        seed = seed_start + i
        lines = list(
            generate_long_sequence_game(
                seed,
                max_steps,
                encoder,
                policy=policy,
                max_context_tokens=max_context_tokens,
                board_mode=board_mode,
                persist_dual_views=persist_dual_views,
                reserve_output_tokens=reserve_output_tokens,
                source=source,
                tokenizer=tokenizer,
            )
        )
        total_examples += len(lines)
        if i < num_train:
            train_lines.extend(lines)
        else:
            eval_lines.extend(lines)

    with open(output_path, "w") as f:
        for line in train_lines:
            f.write(line + "\n")

    actual_eval_path = None
    if eval_path and eval_lines:
        with open(eval_path, "w") as f:
            for line in eval_lines:
                f.write(line + "\n")
        actual_eval_path = eval_path

    return {
        "total_games": num_games,
        "total_examples": total_examples,
        "train_examples": len(train_lines),
        "eval_examples": len(eval_lines),
        "train_path": output_path,
        "eval_path": actual_eval_path,
        "max_context_tokens": max_context_tokens,
        "context_bucket": context_bucket(max_context_tokens),
        "board_mode": board_mode,
        "persist_dual_views": persist_dual_views,
    }


def generate_long_sequence_dataset_multi_budget(
    output_path: str,
    *,
    num_games: int,
    max_steps: int,
    seed_start: int,
    encoder: StateEncoder,
    context_budgets: list[int],
    board_mode: str = "tokenized",
    persist_dual_views: bool = False,
    eval_path: Optional[str] = None,
    eval_fraction: float = 0.2,
    policy: Optional[Callable] = None,
    reserve_output_tokens: int = 16,
    source: str = "nle_generated",
    tokenizer=None,
) -> dict:
    """Generate long-sequence JSONL using multiple target context budgets."""
    num_eval = int(num_games * eval_fraction) if eval_path else 0
    num_train = num_games - num_eval
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    if eval_path:
        os.makedirs(os.path.dirname(eval_path) if os.path.dirname(eval_path) else ".", exist_ok=True)

    train_lines: list[str] = []
    eval_lines: list[str] = []
    total_examples = 0
    for i in range(num_games):
        seed = seed_start + i
        episode_lines: list[str] = []
        for budget in context_budgets:
            lines = list(
                generate_long_sequence_game(
                    seed,
                    max_steps,
                    encoder,
                    policy=policy,
                    max_context_tokens=budget,
                    board_mode=board_mode,
                    persist_dual_views=persist_dual_views,
                    reserve_output_tokens=reserve_output_tokens,
                    source=source,
                    tokenizer=tokenizer,
                )
            )
            for line in lines:
                row = json.loads(line)
                row["metadata"]["target_context_tokens"] = budget
                row["metadata"]["target_context_bucket"] = context_bucket(budget)
                episode_lines.append(json.dumps(row))
        total_examples += len(episode_lines)
        if i < num_train:
            train_lines.extend(episode_lines)
        else:
            eval_lines.extend(episode_lines)

    with open(output_path, "w") as f:
        for line in train_lines:
            f.write(line + "\n")

    actual_eval_path = None
    if eval_path and eval_lines:
        with open(eval_path, "w") as f:
            for line in eval_lines:
                f.write(line + "\n")
        actual_eval_path = eval_path

    return {
        "total_games": num_games,
        "total_examples": total_examples,
        "train_examples": len(train_lines),
        "eval_examples": len(eval_lines),
        "train_path": output_path,
        "eval_path": actual_eval_path,
        "context_budgets": context_budgets,
        "board_mode": board_mode,
        "persist_dual_views": persist_dual_views,
    }


def load_episode_action_steps_from_jsonl(input_path: str) -> dict[str, list[EpisodeActionStep]]:
    """Load episode-grouped action steps from a flexible JSONL schema."""
    episodes: dict[str, list[EpisodeActionStep]] = {}
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            episode_id = str(row.get("episode_id", row.get("gameid", row.get("seed", "episode"))))
            turn_index = int(row.get("step", row.get("turn_index", 0)))
            action = str(row.get("action", row.get("target_action", "wait")))
            state_text = row.get("state_text")
            if state_text is None:
                state_text = row.get("state_prompt")
            if state_text is None:
                state_text = row.get("prompt")
            outcome = infer_outcome_label(death=row.get("death"), achieve=row.get("achieve"))
            metadata = {
                "source_episode_id": episode_id,
                "seed": row.get("seed"),
                "gameid": row.get("gameid"),
                "turns": row.get("turns"),
                "maxlvl": row.get("maxlvl"),
                "hp": row.get("hp"),
                "hp_max": row.get("hp_max"),
                "depth": row.get("depth"),
                "turn": row.get("turn"),
                "death": row.get("death"),
                "achieve": row.get("achieve"),
                "outcome": outcome,
                "is_win": outcome == "win",
                "game_phase": infer_game_phase(
                    depth=row.get("depth"),
                    maxlvl=row.get("maxlvl"),
                    achieve=row.get("achieve"),
                ),
            }
            episodes.setdefault(episode_id, []).append(
                EpisodeActionStep(
                    turn_index=turn_index,
                    obs=None,
                    state_text=state_text,
                    action=action,
                    extra_metadata=metadata,
                )
            )
    for steps in episodes.values():
        steps.sort(key=lambda step: step.turn_index)
    return episodes


def convert_episode_jsonl_to_long_sequence_dataset(
    input_path: str,
    output_path: str,
    *,
    encoder: StateEncoder,
    max_context_tokens: int = 128_000,
    board_mode: str = "tokenized",
    persist_dual_views: bool = False,
    reserve_output_tokens: int = 16,
    source: str = "external_jsonl",
    tokenizer=None,
) -> dict:
    """Convert an episode-style JSONL corpus into long-sequence next-action data."""
    episodes = load_episode_action_steps_from_jsonl(input_path)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    total_examples = 0
    with open(output_path, "w") as f:
        for episode_id, steps in episodes.items():
            examples = build_long_sequence_examples_from_episode(
                steps,
                encoder=encoder,
                episode_id=episode_id,
                max_context_tokens=max_context_tokens,
                board_mode=board_mode,
                persist_dual_views=persist_dual_views,
                reserve_output_tokens=reserve_output_tokens,
                source=source,
                tokenizer=tokenizer,
            )
            for row in examples:
                f.write(json.dumps(row) + "\n")
            total_examples += len(examples)
    return {
        "input_path": input_path,
        "output_path": output_path,
        "episodes": len(episodes),
        "examples": total_examples,
        "context_bucket": context_bucket(max_context_tokens),
        "source": source,
        "persist_dual_views": persist_dual_views,
    }
