"""
Canonical action normalization and classification for long-context policy data.
"""

from __future__ import annotations

from dataclasses import dataclass

from nle_agent.agent_http import ACTION_MAP, parse_action


RUNTIME_ACTIONS = tuple(sorted(ACTION_MAP.keys()))

MODAL_ALIAS_MAP = {
    "": None,
    " ": "space",
    "space": "space",
    "spacebar": "space",
    "esc": "esc",
    "escape": "esc",
    "enter": "more",
    "return": "more",
    "newline": "more",
    "more": "more",
    "--more--": "more",
    "look": "look",
    "whatis": "whatis",
    "what is": "whatis",
    "inventory": "inventory_menu",
    "yes": "yes",
    "no": "no",
}

MODAL_ACTIONS = {
    "space",
    "esc",
    "more",
    "look",
    "whatis",
    "inventory_menu",
    "yes",
    "no",
}


@dataclass(frozen=True)
class CanonicalAction:
    original: str
    normalized: str | None
    action_source: str
    action_class: str
    is_runtime_action: bool
    is_modal_action: bool
    should_keep: bool
    drop_reason: str | None


def _clean_action_text(raw: object) -> str:
    if raw is None:
        return ""
    return str(raw).strip().lower()


def classify_action_family(action: str | None) -> str:
    if action is None:
        return "dropped"
    if action in {"north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"}:
        return "move"
    if action in {"up", "down"}:
        return "stairs"
    if action in {"open", "close", "kick"}:
        return "interaction"
    if action in {"pickup", "drop", "eat", "drink", "read", "wear", "takeoff", "wield", "apply", "zap", "throw", "fire"}:
        return "inventory"
    if action in {"search", "wait"}:
        return action
    if action in MODAL_ACTIONS:
        return "modal"
    return "other"


def canonicalize_action(raw: object, *, keep_modal_actions: bool = False) -> CanonicalAction:
    original = _clean_action_text(raw)
    if original in MODAL_ALIAS_MAP:
        normalized = MODAL_ALIAS_MAP[original]
        if normalized is None:
            return CanonicalAction(
                original=original,
                normalized=None,
                action_source="raw",
                action_class="dropped",
                is_runtime_action=False,
                is_modal_action=False,
                should_keep=False,
                drop_reason="empty_action",
            )
        keep = keep_modal_actions
        return CanonicalAction(
            original=original,
            normalized=normalized,
            action_source="modal_alias",
            action_class=classify_action_family(normalized),
            is_runtime_action=normalized in ACTION_MAP,
            is_modal_action=True,
            should_keep=keep,
            drop_reason=None if keep else "modal_action",
        )

    _, normalized = parse_action(original)
    if normalized == "wait" and original not in {"wait", ".", "rest"}:
        return CanonicalAction(
            original=original,
            normalized=None,
            action_source="parse_fallback",
            action_class="dropped",
            is_runtime_action=False,
            is_modal_action=False,
            should_keep=False,
            drop_reason="unrecognized_action",
        )

    return CanonicalAction(
        original=original,
        normalized=normalized,
        action_source="runtime_parse",
        action_class=classify_action_family(normalized),
        is_runtime_action=normalized in ACTION_MAP,
        is_modal_action=normalized in MODAL_ACTIONS,
        should_keep=True,
        drop_reason=None,
    )


def normalize_action_text(raw: object, *, keep_modal_actions: bool = False) -> str | None:
    return canonicalize_action(raw, keep_modal_actions=keep_modal_actions).normalized
