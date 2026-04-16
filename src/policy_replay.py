"""
Canonical policy rendering, modal detection, replay, and validation helpers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from src.board_view import build_board_view, build_board_view_from_rows, estimate_text_tokens
from src.policy_actions import CanonicalAction, canonicalize_action


POLICY_SYSTEM_PROMPT = (
    "You are choosing the next NetHack action from a long game history. "
    "Read the prior turns and current full-board state carefully, then reply "
    "with exactly one action word and nothing else."
)

DEFAULT_POLICY_BOARD_MODE = "tokenized"


@dataclass(frozen=True)
class ModalState:
    disposition: str
    modal_type: str
    reason: str


@dataclass(frozen=True)
class RenderedPolicyState:
    text: str
    token_estimate: int
    render_source: str
    board_mode: str
    board_views: dict[str, object] | None
    modal_state: ModalState
    message: str
    status_lines: list[str]


@dataclass(frozen=True)
class ReplayedEpisodeStep:
    turn_index: int
    state_text: str
    token_estimate: int
    action: CanonicalAction
    render_source: str
    board_mode: str
    board_views: dict[str, object] | None
    modal_state: ModalState
    extra_metadata: dict[str, Any]


def is_dangerous_message(text: str) -> bool:
    lower = (text or "").lower()
    cues = [
        "hit", "bites", "misses", "killed", "dies", "die", "poison",
        "fainted", "confused", "blind", "halluc", "attack", "attacks",
        "engulf", "burn", "frozen", "stun",
    ]
    return any(cue in lower for cue in cues)


def classify_modal_screen(text: str) -> ModalState:
    lower = (text or "").lower()
    if not lower.strip():
        return ModalState("keep_gameplay", "none", "empty_text")
    if "rip " in lower or "rest in peace" in lower:
        return ModalState("drop_terminal_screen", "death_screen", "grave_screen")
    if "--more--" in lower:
        return ModalState("keep_modal", "more_prompt", "more_prompt")
    if "what do you want to" in lower or "what would you like to" in lower:
        return ModalState("keep_modal", "command_prompt", "command_prompt")
    if "(end)" in lower or "(1 of" in lower:
        return ModalState("keep_modal", "paged_view", "paged_view")
    if "inventory:" in lower or "pick up what" in lower:
        return ModalState("keep_modal", "inventory_menu", "inventory_menu")
    if "[yn" in lower or "(yes or no)" in lower:
        return ModalState("keep_modal", "yes_no_prompt", "yes_no_prompt")
    return ModalState("keep_gameplay", "none", "gameplay_screen")


def _normalize_screen_lines(screen_text: str) -> list[str]:
    return [line.rstrip("\n") for line in str(screen_text or "").splitlines()]


def parse_tty_screen(screen_text: str) -> dict[str, Any]:
    lines = _normalize_screen_lines(screen_text)
    if not lines:
        return {
            "message": "<none>",
            "board_rows": [],
            "status_lines": [],
            "all_lines": [],
        }
    message = lines[0].strip() or "<none>"
    board_rows = lines[1:22] if len(lines) >= 22 else lines[1:]
    status_lines = lines[22:24] if len(lines) >= 24 else []
    while board_rows and not board_rows[-1].strip():
        board_rows.pop()
    return {
        "message": message,
        "board_rows": board_rows,
        "status_lines": [line.rstrip() for line in status_lines if line.strip()],
        "all_lines": lines,
    }


def _render_state_text(
    *,
    state_index: int,
    message: str,
    board_mode: str,
    board_text: str,
    board_height: int,
    board_width: int,
    status_lines: list[str],
    render_source: str,
) -> str:
    lines = [
        f"TurnIndex: {state_index}",
        f"RenderSource: {render_source}",
        f"Message: {message or '<none>'}",
        f"BoardMode: {board_mode}",
        f"BoardShape: {board_height}x{board_width}",
    ]
    if status_lines:
        lines.append("StatusLines:")
        lines.extend(status_lines)
    lines.extend(["Board:", board_text])
    return "\n".join(lines)


def render_policy_state_from_obs(
    obs: dict,
    *,
    state_index: int,
    encoder,
    board_mode: str = DEFAULT_POLICY_BOARD_MODE,
    persist_dual_views: bool = False,
    tokenizer=None,
) -> RenderedPolicyState:
    state = encoder.encode_full(obs)
    board_view = build_board_view(obs, state_index=state_index, tokenizer=tokenizer)
    board_text = board_view.tokenized_board if board_mode == "tokenized" else board_view.ascii_board
    message_text = state.get("message") or "<none>"
    stats_line = (
        f"Stats: HP={state['hp']}/{state['hp_max']} AC={state['ac']} "
        f"Str={state['strength']} Dex={state['dexterity']} Gold={state['gold']} "
        f"Depth={state['depth']} Pos={state['position']} Clock={state['turn']}"
    )
    text = _render_state_text(
        state_index=state_index,
        message=message_text,
        board_mode=board_mode,
        board_text=board_text,
        board_height=board_view.height,
        board_width=board_view.width,
        status_lines=[stats_line],
        render_source="obs",
    )
    return RenderedPolicyState(
        text=text,
        token_estimate=estimate_text_tokens(text, tokenizer=tokenizer),
        render_source="obs",
        board_mode=board_mode,
        board_views=(
            {
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
            if persist_dual_views
            else None
        ),
        modal_state=ModalState("keep_gameplay", "none", "obs_gameplay"),
        message=message_text,
        status_lines=[stats_line],
    )


def render_policy_state_from_text(
    screen_text: str,
    *,
    state_index: int,
    board_mode: str = DEFAULT_POLICY_BOARD_MODE,
    persist_dual_views: bool = False,
    tokenizer=None,
) -> RenderedPolicyState:
    modal_state = classify_modal_screen(screen_text)
    parsed = parse_tty_screen(screen_text)
    board_rows = parsed["board_rows"] if parsed["board_rows"] else parsed["all_lines"]
    board_view = build_board_view_from_rows(board_rows, state_index=state_index, tokenizer=tokenizer)
    board_text = board_view.tokenized_board if board_mode == "tokenized" else board_view.ascii_board
    text = _render_state_text(
        state_index=state_index,
        message=parsed["message"],
        board_mode=board_mode,
        board_text=board_text,
        board_height=board_view.height,
        board_width=board_view.width,
        status_lines=parsed["status_lines"],
        render_source="tty_text" if parsed["board_rows"] else "fallback_text",
    )
    render_source = "tty_text" if parsed["board_rows"] else "fallback_text"
    return RenderedPolicyState(
        text=text,
        token_estimate=estimate_text_tokens(text, tokenizer=tokenizer),
        render_source=render_source,
        board_mode=board_mode,
        board_views=(
            {
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
            if persist_dual_views
            else None
        ),
        modal_state=modal_state,
        message=parsed["message"],
        status_lines=parsed["status_lines"],
    )


def replay_episode_steps(
    episode_steps: Iterable[Any],
    *,
    encoder,
    board_mode: str = DEFAULT_POLICY_BOARD_MODE,
    persist_dual_views: bool = False,
    keep_modal_actions: bool = False,
    modal_policy: str = "drop_modal",
    stride: int = 1,
    min_turn_index: int = 0,
    max_turn_index: int | None = None,
    min_depth: int | None = None,
    danger_only: bool = False,
    danger_window: int = 0,
    tokenizer=None,
) -> tuple[list[ReplayedEpisodeStep], dict[str, Any]]:
    counters = Counter()
    replayed: list[ReplayedEpisodeStep] = []
    recent_danger_turns: list[int] = []
    for raw_step in sorted(episode_steps, key=lambda step: int(step.turn_index)):
        counters["rows_before_replay"] += 1
        if raw_step.turn_index < min_turn_index:
            counters["dropped_before_min_turn"] += 1
            continue
        if max_turn_index is not None and raw_step.turn_index > max_turn_index:
            counters["dropped_after_max_turn"] += 1
            continue
        if stride > 1 and ((raw_step.turn_index - min_turn_index) % stride != 0):
            counters["dropped_by_stride"] += 1
            continue

        action = canonicalize_action(raw_step.action, keep_modal_actions=keep_modal_actions)
        if raw_step.obs is not None:
            rendered = render_policy_state_from_obs(
                raw_step.obs,
                state_index=raw_step.turn_index,
                encoder=encoder,
                board_mode=board_mode,
                persist_dual_views=persist_dual_views,
                tokenizer=tokenizer,
            )
        else:
            rendered = render_policy_state_from_text(
                raw_step.state_text or "",
                state_index=raw_step.turn_index,
                board_mode=board_mode,
                persist_dual_views=persist_dual_views,
                tokenizer=tokenizer,
            )

        if min_depth is not None:
            depth_value = raw_step.extra_metadata.get("depth") if getattr(raw_step, "extra_metadata", None) else None
            if depth_value is not None and int(depth_value) < int(min_depth):
                counters["dropped_before_min_depth"] += 1
                continue

        if is_dangerous_message(rendered.message):
            recent_danger_turns.append(int(raw_step.turn_index))
        recent_danger_turns = [turn for turn in recent_danger_turns if int(raw_step.turn_index) - turn <= max(0, danger_window)]
        if danger_only and not recent_danger_turns:
            counters["dropped_not_in_danger_window"] += 1
            continue

        disposition = rendered.modal_state.disposition
        if disposition.startswith("drop_"):
            counters[f"dropped_{rendered.modal_state.modal_type}"] += 1
            if modal_policy != "keep_all":
                continue
        elif disposition == "keep_modal" and modal_policy == "drop_modal":
            counters[f"dropped_{rendered.modal_state.modal_type}"] += 1
            continue

        if not action.should_keep:
            counters[f"dropped_action_{action.drop_reason or 'unknown'}"] += 1
            continue

        counters["rows_after_replay"] += 1
        counters[f"render_source_{rendered.render_source}"] += 1
        counters[f"modal_type_{rendered.modal_state.modal_type}"] += 1
        counters[f"action_before_{action.original or '<empty>'}"] += 1
        counters[f"action_after_{action.normalized or '<dropped>'}"] += 1
        replayed.append(
            ReplayedEpisodeStep(
                turn_index=raw_step.turn_index,
                state_text=rendered.text,
                token_estimate=rendered.token_estimate,
                action=action,
                render_source=rendered.render_source,
                board_mode=rendered.board_mode,
                board_views=rendered.board_views,
                modal_state=rendered.modal_state,
                extra_metadata={
                    **dict(raw_step.extra_metadata or {}),
                    "render_source": rendered.render_source,
                    "modal_type": rendered.modal_state.modal_type,
                    "modal_disposition": rendered.modal_state.disposition,
                    "modal_reason": rendered.modal_state.reason,
                    "normalized_action": action.normalized,
                    "original_action": action.original,
                    "action_class": action.action_class,
                },
            )
        )
    report = {
        "rows_before_replay": counters["rows_before_replay"],
        "rows_after_replay": counters["rows_after_replay"],
        "dropped_by_reason": {
            key[len("dropped_"):]: value for key, value in sorted(counters.items()) if key.startswith("dropped_")
        },
        "action_histogram_before": {
            key[len("action_before_"):]: value for key, value in sorted(counters.items()) if key.startswith("action_before_")
        },
        "action_histogram_after": {
            key[len("action_after_"):]: value for key, value in sorted(counters.items()) if key.startswith("action_after_")
        },
        "modal_split": {
            key[len("modal_type_"):]: value for key, value in sorted(counters.items()) if key.startswith("modal_type_")
        },
        "render_source_split": {
            key[len("render_source_"):]: value for key, value in sorted(counters.items()) if key.startswith("render_source_")
        },
        "config": {
            "board_mode": board_mode,
            "persist_dual_views": persist_dual_views,
            "keep_modal_actions": keep_modal_actions,
            "modal_policy": modal_policy,
            "stride": stride,
            "min_turn_index": min_turn_index,
            "max_turn_index": max_turn_index,
            "min_depth": min_depth,
            "danger_only": danger_only,
            "danger_window": danger_window,
        },
    }
    return replayed, report
