from __future__ import annotations

import numpy as np

from rl.options import build_skill_registry


ADJ_TILES = [
    "unseen",
    "wall",
    "floor",
    "corridor",
    "door",
    "stairs_down",
    "stairs_up",
    "gold",
    "scroll",
    "potion",
    "wand",
    "ring",
    "gem",
    "amulet",
    "tool",
    "weapon",
    "armor",
    "food",
    "monster",
]

ACTION_SET = [
    "north",
    "south",
    "east",
    "west",
    "wait",
    "search",
    "pickup",
    "up",
    "down",
    "kick",
    "eat",
    "drink",
    "drop",
]

SKILL_SET = list(build_skill_registry().keys())
_TILE_TO_IDX = {name: i for i, name in enumerate(ADJ_TILES)}
_SKILL_TO_IDX = {name: i for i, name in enumerate(SKILL_SET)}
_ACTION_TO_IDX = {name: i for i, name in enumerate(ACTION_SET)}


def observation_dim() -> int:
    return 12 + 4 * len(ADJ_TILES) + len(SKILL_SET) + len(ACTION_SET)


def action_name_to_index(action_name: str) -> int:
    return _ACTION_TO_IDX.get(action_name, _ACTION_TO_IDX["wait"])


def index_to_action_name(index: int) -> str:
    if 0 <= index < len(ACTION_SET):
        return ACTION_SET[index]
    return "wait"


def _encode_adjacent_tile(tile_name: str) -> np.ndarray:
    vec = np.zeros(len(ADJ_TILES), dtype=np.float32)
    if tile_name.startswith("monster_"):
        tile_name = "monster"
    vec[_TILE_TO_IDX.get(tile_name, _TILE_TO_IDX["unseen"])] = 1.0
    return vec


def encode_observation(timestep: dict) -> np.ndarray:
    state = timestep["state"]
    active_skill = timestep["active_skill"]
    allowed_actions = timestep["allowed_actions"]

    hp = float(state["hp"])
    hp_max = float(max(1, state["hp_max"]))
    features = [
        hp / hp_max,
        hp_max / 20.0,
        float(state["gold"]) / 100.0,
        float(state["depth"]) / 10.0,
        float(state["turn"]) / 1000.0,
        float(state["ac"]) / 20.0,
        float(state["strength"]) / 25.0,
        float(state["dexterity"]) / 25.0,
        min(1.0, float(len(state.get("visible_monsters", []))) / 8.0),
        min(1.0, float(len(state.get("visible_items", []))) / 8.0),
        min(1.0, float(timestep["memory_total_explored"]) / 400.0),
        min(1.0, float(timestep["rooms_discovered"]) / 20.0),
    ]

    adj_vectors = []
    for direction in ("north", "south", "east", "west"):
        adj_vectors.append(_encode_adjacent_tile(state["adjacent"].get(direction, "unseen")))

    skill_one_hot = np.zeros(len(SKILL_SET), dtype=np.float32)
    skill_one_hot[_SKILL_TO_IDX.get(active_skill, 0)] = 1.0

    action_mask = np.zeros(len(ACTION_SET), dtype=np.float32)
    for action_name in allowed_actions:
        if action_name in _ACTION_TO_IDX:
            action_mask[_ACTION_TO_IDX[action_name]] = 1.0

    return np.concatenate(
        [np.asarray(features, dtype=np.float32), *adj_vectors, skill_one_hot, action_mask]
    )

