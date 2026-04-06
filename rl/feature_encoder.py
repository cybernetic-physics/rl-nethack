from __future__ import annotations

import numpy as np

from rl.options import build_skill_registry
from src.state_encoder import _tile_name


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
_OBS_VERSION_TO_DIM = {"v1": 106, "v2": 160, "v3": 244, "v4": 302}

_LOCAL_PATCH_CATEGORIES = [
    "unseen",
    "wall",
    "floorish",
    "door",
    "stairs",
    "item",
    "monster",
    "other",
]
_LOCAL_PATCH_TO_IDX = {name: i for i, name in enumerate(_LOCAL_PATCH_CATEGORIES)}


def _normalize_tile_name(tile_name: str) -> str:
    if tile_name.startswith("monster_"):
        return "monster"
    return tile_name


def observation_dim(version: str = "v1") -> int:
    if version not in _OBS_VERSION_TO_DIM:
        raise ValueError(f"Unknown observation version: {version}")
    return _OBS_VERSION_TO_DIM[version]


def action_name_to_index(action_name: str) -> int:
    return _ACTION_TO_IDX.get(action_name, _ACTION_TO_IDX["wait"])


def index_to_action_name(index: int) -> str:
    if 0 <= index < len(ACTION_SET):
        return ACTION_SET[index]
    return "wait"


def _encode_tile_one_hot(tile_name: str) -> np.ndarray:
    vec = np.zeros(len(ADJ_TILES), dtype=np.float32)
    vec[_TILE_TO_IDX.get(_normalize_tile_name(tile_name), _TILE_TO_IDX["unseen"])] = 1.0
    return vec


def _base_scalar_features(timestep: dict) -> list[float]:
    state = timestep["state"]
    hp = float(state["hp"])
    hp_max = float(max(1, state["hp_max"]))
    return [
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


def _adjacent_vectors(state: dict) -> list[np.ndarray]:
    return [
        _encode_tile_one_hot(state["adjacent"].get(direction, "unseen"))
        for direction in ("north", "south", "east", "west")
    ]


def _skill_one_hot(active_skill: str) -> np.ndarray:
    vec = np.zeros(len(SKILL_SET), dtype=np.float32)
    vec[_SKILL_TO_IDX.get(active_skill, 0)] = 1.0
    return vec


def _action_mask(allowed_actions: list[str]) -> np.ndarray:
    vec = np.zeros(len(ACTION_SET), dtype=np.float32)
    for action_name in allowed_actions:
        if action_name in _ACTION_TO_IDX:
            vec[_ACTION_TO_IDX[action_name]] = 1.0
    return vec


def _v2_extra_features(timestep: dict) -> np.ndarray:
    state = timestep["state"]
    allowed_actions = set(timestep["allowed_actions"])
    adjacent = state.get("adjacent", {})
    visible_monsters = state.get("visible_monsters", [])
    visible_items = state.get("visible_items", [])
    recent_action_names = timestep.get("recent_actions", [])
    recent_positions = timestep.get("recent_positions", [])
    px, py = state["position"]

    monster_dx = [monster["pos"][0] - px for monster in visible_monsters[:4]]
    monster_dy = [monster["pos"][1] - py for monster in visible_monsters[:4]]
    item_dx = [item["pos"][0] - px for item in visible_items[:4]]
    item_dy = [item["pos"][1] - py for item in visible_items[:4]]
    monster_dx += [0] * (4 - len(monster_dx))
    monster_dy += [0] * (4 - len(monster_dy))
    item_dx += [0] * (4 - len(item_dx))
    item_dy += [0] * (4 - len(item_dy))

    directional_frontier = []
    for direction in ("north", "south", "east", "west"):
        tile = adjacent.get(direction, "unseen")
        directional_frontier.append(1.0 if tile == "unseen" else 0.0)
        directional_frontier.append(1.0 if tile in {"floor", "corridor", "door", "stairs_down", "stairs_up"} else 0.0)

    recent_action_one_hot = np.zeros(len(ACTION_SET), dtype=np.float32)
    for action_name in recent_action_names[-4:]:
        if action_name in _ACTION_TO_IDX:
            recent_action_one_hot[_ACTION_TO_IDX[action_name]] += 0.25

    extras = [
        float(px) / 79.0,
        float(py) / 21.0,
        min(1.0, float(timestep.get("steps_in_skill", 0)) / 32.0),
        min(1.0, float(timestep.get("repeated_state_count", 0)) / 8.0),
        min(1.0, float(timestep.get("revisited_recent_tile_count", 0)) / 8.0),
        min(1.0, float(timestep.get("repeated_action_count", 0)) / 8.0),
        1.0 if timestep.get("standing_on_down_stairs") else 0.0,
        1.0 if timestep.get("standing_on_up_stairs") else 0.0,
        1.0 if "stairs_down" in set(adjacent.values()) else 0.0,
        1.0 if "stairs_up" in set(adjacent.values()) else 0.0,
        1.0 if "door" in set(adjacent.values()) else 0.0,
        1.0 if "see here" in state.get("message", "").lower() else 0.0,
        1.0 if any(action in allowed_actions for action in ("pickup", "down", "up")) else 0.0,
    ]
    extras.extend(float(dx) / 10.0 for dx in monster_dx)
    extras.extend(float(dy) / 10.0 for dy in monster_dy)
    extras.extend(float(dx) / 10.0 for dx in item_dx)
    extras.extend(float(dy) / 10.0 for dy in item_dy)
    extras.extend(directional_frontier)
    extras.extend(list(recent_action_one_hot))

    recent_pos_vec = np.zeros(4, dtype=np.float32)
    for idx, pos in enumerate(recent_positions[-4:]):
        recent_pos_vec[idx] = 1.0 if tuple(pos) == tuple(state["position"]) else 0.0
    extras.extend(list(recent_pos_vec))

    return np.asarray(extras, dtype=np.float32)


def _local_patch_bucket(tile_name: str) -> str:
    tile_name = _normalize_tile_name(tile_name)
    if tile_name == "unseen":
        return "unseen"
    if tile_name == "wall":
        return "wall"
    if tile_name in {"floor", "corridor"}:
        return "floorish"
    if tile_name == "door":
        return "door"
    if tile_name in {"stairs_down", "stairs_up"}:
        return "stairs"
    if tile_name in {"gold", "scroll", "potion", "wand", "ring", "gem", "amulet", "tool", "weapon", "armor", "food"}:
        return "item"
    if tile_name == "monster":
        return "monster"
    return "other"


def _encode_local_patch(timestep: dict, radius: int = 1) -> np.ndarray:
    obs = timestep["obs"]
    state = timestep["state"]
    px, py = state["position"]
    chars = obs["chars"]
    patch = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny, nx = py + dy, px + dx
            tile_name = "unseen"
            if 0 <= ny < chars.shape[0] and 0 <= nx < chars.shape[1]:
                tile_name = _tile_name(int(chars[ny, nx]))
            bucket = _local_patch_bucket(tile_name)
            vec = np.zeros(len(_LOCAL_PATCH_CATEGORIES), dtype=np.float32)
            vec[_LOCAL_PATCH_TO_IDX[bucket]] = 1.0
            patch.append(vec)
    return np.concatenate(patch)


def _directional_ray_features(timestep: dict, max_len: int = 5) -> np.ndarray:
    obs = timestep["obs"]
    state = timestep["state"]
    px, py = state["position"]
    chars = obs["chars"]
    features: list[float] = []
    deltas = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
    for dx, dy in deltas.values():
        unseen_count = 0
        passable_count = 0
        wall_seen = 0.0
        for step in range(1, max_len + 1):
            nx, ny = px + dx * step, py + dy * step
            if 0 <= ny < chars.shape[0] and 0 <= nx < chars.shape[1]:
                tile_name = _normalize_tile_name(_tile_name(int(chars[ny, nx])))
            else:
                tile_name = "unseen"
            if tile_name == "unseen":
                unseen_count += 1
            if tile_name in {"floor", "corridor", "door", "stairs_down", "stairs_up"}:
                passable_count += 1
            if tile_name == "wall":
                wall_seen = 1.0
                break
        features.extend(
            [
                unseen_count / float(max_len),
                passable_count / float(max_len),
                wall_seen,
            ]
        )
    return np.asarray(features, dtype=np.float32)


def _directional_quadrant_features(timestep: dict, radius: int = 4) -> np.ndarray:
    obs = timestep["obs"]
    state = timestep["state"]
    px, py = state["position"]
    chars = obs["chars"]
    features: list[float] = []
    directions = {
        "north": lambda dx, dy: dy < 0 and abs(dx) <= abs(dy),
        "south": lambda dx, dy: dy > 0 and abs(dx) <= abs(dy),
        "east": lambda dx, dy: dx > 0 and abs(dy) <= abs(dx),
        "west": lambda dx, dy: dx < 0 and abs(dy) <= abs(dx),
    }
    passable = {"floor", "corridor", "door", "stairs_down", "stairs_up"}
    items = {"gold", "scroll", "potion", "wand", "ring", "gem", "amulet", "tool", "weapon", "armor", "food"}
    for predicate in directions.values():
        unseen_count = 0
        passable_count = 0
        wall_count = 0
        item_count = 0
        monster_count = 0
        total = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                if not predicate(dx, dy):
                    continue
                nx, ny = px + dx, py + dy
                total += 1
                tile_name = "unseen"
                if 0 <= ny < chars.shape[0] and 0 <= nx < chars.shape[1]:
                    tile_name = _normalize_tile_name(_tile_name(int(chars[ny, nx])))
                if tile_name == "unseen":
                    unseen_count += 1
                elif tile_name == "wall":
                    wall_count += 1
                elif tile_name in passable:
                    passable_count += 1
                elif tile_name in items:
                    item_count += 1
                elif tile_name == "monster":
                    monster_count += 1
        denom = float(max(1, total))
        features.extend(
            [
                unseen_count / denom,
                passable_count / denom,
                wall_count / denom,
                item_count / denom,
                monster_count / denom,
            ]
        )
    return np.asarray(features, dtype=np.float32)


def _directional_first_hit_features(timestep: dict, max_len: int = 6) -> np.ndarray:
    obs = timestep["obs"]
    state = timestep["state"]
    px, py = state["position"]
    chars = obs["chars"]
    buckets = ["unseen", "wall", "passable", "item", "monster", "other"]
    bucket_to_idx = {name: idx for idx, name in enumerate(buckets)}
    passable = {"floor", "corridor", "door", "stairs_down", "stairs_up"}
    items = {"gold", "scroll", "potion", "wand", "ring", "gem", "amulet", "tool", "weapon", "armor", "food"}
    deltas = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
    all_features = []
    for dx, dy in deltas.values():
        vec = np.zeros(len(buckets) + 1, dtype=np.float32)
        hit_bucket = "unseen"
        hit_distance = float(max_len)
        for step in range(1, max_len + 1):
            nx, ny = px + dx * step, py + dy * step
            tile_name = "unseen"
            if 0 <= ny < chars.shape[0] and 0 <= nx < chars.shape[1]:
                tile_name = _normalize_tile_name(_tile_name(int(chars[ny, nx])))
            if tile_name == "unseen":
                hit_bucket = "unseen"
                hit_distance = float(step)
                break
            if tile_name == "wall":
                hit_bucket = "wall"
                hit_distance = float(step)
                break
            if tile_name in passable:
                hit_bucket = "passable"
                hit_distance = float(step)
                break
            if tile_name in items:
                hit_bucket = "item"
                hit_distance = float(step)
                break
            if tile_name == "monster":
                hit_bucket = "monster"
                hit_distance = float(step)
                break
            hit_bucket = "other"
            hit_distance = float(step)
            break
        vec[bucket_to_idx[hit_bucket]] = 1.0
        vec[-1] = hit_distance / float(max_len)
        all_features.append(vec)
    return np.concatenate(all_features)


def _recent_action_direction_features(timestep: dict) -> np.ndarray:
    recent_actions = timestep.get("recent_actions", [])
    last_action = recent_actions[-1] if recent_actions else None
    reverse_of = {"north": "south", "south": "north", "east": "west", "west": "east"}
    vec = np.zeros(10, dtype=np.float32)
    dir_actions = ["north", "south", "east", "west"]
    if last_action in dir_actions:
        vec[dir_actions.index(last_action)] = 1.0
        reverse = reverse_of[last_action]
        vec[4 + dir_actions.index(reverse)] = 1.0
    allowed = set(timestep.get("allowed_actions", []))
    for idx, action in enumerate(dir_actions):
        vec[8] += 0.25 if action in allowed else 0.0
    vec[9] = min(1.0, float(timestep.get("repeated_action_count", 0)) / 8.0)
    return vec


def _v3_extra_features(timestep: dict) -> np.ndarray:
    extras = [*_v2_extra_features(timestep)]
    local_patch = _encode_local_patch(timestep, radius=1)
    ray = _directional_ray_features(timestep, max_len=5)
    return np.concatenate([np.asarray(extras, dtype=np.float32), local_patch, ray])


def _v4_extra_features(timestep: dict) -> np.ndarray:
    v3 = _v3_extra_features(timestep)
    quadrants = _directional_quadrant_features(timestep, radius=4)
    hits = _directional_first_hit_features(timestep, max_len=6)
    recent = _recent_action_direction_features(timestep)
    return np.concatenate([v3, quadrants, hits, recent])


def encode_observation(timestep: dict, version: str = "v1") -> np.ndarray:
    state = timestep["state"]
    active_skill = timestep["active_skill"]
    allowed_actions = timestep["allowed_actions"]

    parts = [
        np.asarray(_base_scalar_features(timestep), dtype=np.float32),
        *_adjacent_vectors(state),
        _skill_one_hot(active_skill),
        _action_mask(allowed_actions),
    ]
    if version == "v2":
        parts.append(_v2_extra_features(timestep))
    elif version == "v3":
        parts.append(_v3_extra_features(timestep))
    elif version == "v4":
        parts.append(_v4_extra_features(timestep))

    encoded = np.concatenate(parts)
    expected_dim = observation_dim(version)
    if encoded.shape != (expected_dim,):
        raise ValueError(f"Encoded observation has shape {encoded.shape}, expected {(expected_dim,)} for {version}")
    return encoded
