from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.task_harness import TASK_DIRECTIVES


State = dict
Memory = object


@dataclass(frozen=True)
class SkillOption:
    name: str
    directive: str
    can_start: Callable[[State, Memory], bool]
    should_stop: Callable[[State, Memory, int], bool]
    allowed_actions: Callable[[State, Memory], list[str]]


def _movement_actions(state: State, memory: Memory) -> list[str]:
    del memory
    actions = ["north", "south", "east", "west", "wait", "search"]
    message = state.get("message", "").lower()
    if "see here" in message or any(item["pos"] == state["position"] for item in state.get("visible_items", [])):
        actions.append("pickup")
    return actions


def _combat_actions(state: State, memory: Memory) -> list[str]:
    del memory
    base = ["north", "south", "east", "west", "wait", "search"]
    if state.get("visible_monsters"):
        return base + ["kick"]
    return base


def _resource_actions(state: State, memory: Memory) -> list[str]:
    del memory
    actions = ["north", "south", "east", "west", "wait", "pickup", "search"]
    if "hungry" in state.get("message", "").lower():
        actions.append("eat")
    return actions


def _descend_actions(state: State, memory: Memory) -> list[str]:
    del memory
    actions = _movement_actions(state, memory)
    if "stairs_down" in set(state.get("adjacent", {}).values()):
        actions.append("down")
    return list(dict.fromkeys(actions))


def _always_start(state: State, memory: Memory) -> bool:
    del state, memory
    return True


def _low_hp_start(state: State, memory: Memory) -> bool:
    del memory
    return state.get("hp", 0) <= max(1, state.get("hp_max", 1) // 2)


def _combat_start(state: State, memory: Memory) -> bool:
    del memory
    return bool(state.get("visible_monsters"))


def _descend_start(state: State, memory: Memory) -> bool:
    del memory
    return (
        "stairs_down" in set(state.get("adjacent", {}).values())
        or any("stairs_down" == item.get("type") for item in state.get("visible_items", []))
    )


def _resource_start(state: State, memory: Memory) -> bool:
    del memory
    msg = state.get("message", "").lower()
    return "see here" in msg or bool(state.get("visible_items"))


def _default_stop(state: State, memory: Memory, steps_in_skill: int) -> bool:
    del state, memory
    return steps_in_skill >= 8


def _combat_stop(state: State, memory: Memory, steps_in_skill: int) -> bool:
    del memory
    return steps_in_skill >= 8 or not state.get("visible_monsters")


def _survive_stop(state: State, memory: Memory, steps_in_skill: int) -> bool:
    del memory
    return steps_in_skill >= 8 or state.get("hp", 0) > max(1, state.get("hp_max", 1) // 2)


def build_skill_registry() -> dict[str, SkillOption]:
    return {
        "explore": SkillOption(
            name="explore",
            directive=TASK_DIRECTIVES["explore"],
            can_start=_always_start,
            should_stop=_default_stop,
            allowed_actions=_movement_actions,
        ),
        "survive": SkillOption(
            name="survive",
            directive=TASK_DIRECTIVES["survive"],
            can_start=_low_hp_start,
            should_stop=_survive_stop,
            allowed_actions=_movement_actions,
        ),
        "combat": SkillOption(
            name="combat",
            directive=TASK_DIRECTIVES["combat"],
            can_start=_combat_start,
            should_stop=_combat_stop,
            allowed_actions=_combat_actions,
        ),
        "descend": SkillOption(
            name="descend",
            directive=TASK_DIRECTIVES["descend"],
            can_start=_descend_start,
            should_stop=_default_stop,
            allowed_actions=_descend_actions,
        ),
        "resource": SkillOption(
            name="resource",
            directive=TASK_DIRECTIVES["resource"],
            can_start=_resource_start,
            should_stop=_default_stop,
            allowed_actions=_resource_actions,
        ),
    }

