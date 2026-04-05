"""
Task-specific reward shaping for NetHack control and evaluation.

These rewards are repo-defined diagnostics and objectives; they are not the
default NLE reward, which is sparse score delta.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from nle import nethack


DEFAULT_ACTIONS = ("north", "south", "east", "west", "wait", "search")
THREAT_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
USEFUL_ITEM_TILES = {
    "gold", "scroll", "potion", "wand", "ring", "gem", "amulet",
    "tool", "weapon", "armor", "food",
}


@dataclass
class TaskRewardResult:
    total: float
    components: dict[str, float]


@dataclass(frozen=True)
class MemorySnapshot:
    total_explored: int
    rooms: tuple


def observation_hash(obs: dict) -> str:
    """Stable hash for loop / repetition accounting."""
    digest = hashlib.sha256()
    digest.update(obs["chars"].tobytes())
    digest.update(obs["blstats"].tobytes())
    digest.update(obs["message"].tobytes())
    return digest.hexdigest()


def snapshot_memory(memory) -> MemorySnapshot:
    """Capture the small subset of memory state used for reward shaping."""
    return MemorySnapshot(
        total_explored=int(memory.total_explored),
        rooms=tuple(memory.rooms),
    )


def _visible_stairs(obs: dict) -> int:
    chars = obs["chars"]
    return int(((chars == ord(">")) | (chars == ord("<"))).sum())


def _visible_useful_items(state: dict) -> int:
    return sum(1 for item in state.get("visible_items", []) if item["type"] in USEFUL_ITEM_TILES)


def _adjacent_hostile_count(obs: dict, state: dict) -> int:
    px, py = state["position"]
    chars = obs["chars"]
    count = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = py + dy, px + dx
            if 0 <= ny < chars.shape[0] and 0 <= nx < chars.shape[1]:
                ch = chr(int(chars[ny, nx]))
                if ch in THREAT_CHARS and ch not in {"@", "I"}:
                    count += 1
    return count


def _death_penalty(terminated: bool, hp_after: int) -> float:
    return -100.0 if terminated or hp_after <= 0 else 0.0


def compute_task_rewards(
    task: str,
    obs_before: dict,
    obs_after: dict,
    state_before: dict,
    state_after: dict,
    memory_before,
    memory_after,
    action_name: str,
    reward: float,
    terminated: bool,
    truncated: bool,
    repeated_state: bool = False,
    revisited_recent_tile: bool = False,
    repeated_action: bool = False,
) -> TaskRewardResult:
    """Compute shaped reward for one repo-defined task."""
    del truncated

    hp_before = int(obs_before["blstats"][nethack.NLE_BL_HP])
    hp_after = int(obs_after["blstats"][nethack.NLE_BL_HP])
    depth_before = int(obs_before["blstats"][nethack.NLE_BL_DEPTH])
    depth_after = int(obs_after["blstats"][nethack.NLE_BL_DEPTH])
    gold_before = int(obs_before["blstats"][nethack.NLE_BL_GOLD])
    gold_after = int(obs_after["blstats"][nethack.NLE_BL_GOLD])
    score_before = int(obs_before["blstats"][nethack.NLE_BL_SCORE])
    score_after = int(obs_after["blstats"][nethack.NLE_BL_SCORE])
    xp_before = int(obs_before["blstats"][nethack.NLE_BL_XP])
    xp_after = int(obs_after["blstats"][nethack.NLE_BL_XP])

    new_tiles = max(0, memory_after.total_explored - memory_before.total_explored)
    new_rooms = max(0, len(memory_after.rooms) - len(memory_before.rooms))
    stairs_seen_bonus = 1.0 if _visible_stairs(obs_after) > _visible_stairs(obs_before) else 0.0
    useful_items_seen_bonus = 1.0 if _visible_useful_items(state_after) > _visible_useful_items(state_before) else 0.0
    hp_loss = max(0, hp_before - hp_after)
    hp_gain = max(0, hp_after - hp_before)
    gold_gain = max(0, gold_after - gold_before)
    score_delta = max(0, score_after - score_before)
    xp_gain = max(0, xp_after - xp_before)
    depth_gain = max(0, depth_after - depth_before)
    hostile_before = _adjacent_hostile_count(obs_before, state_before)
    hostile_after = _adjacent_hostile_count(obs_after, state_after)

    invalid_like_action = (
        (action_name == "wait" and hostile_before > 0) or
        (action_name == "pickup" and "see here" not in state_before.get("message", "").lower()) or
        (action_name == "open" and "door" not in set(state_before["adjacent"].values()))
    )

    components = {"env_score_delta": float(reward)}

    if task == "explore":
        components.update({
            "new_tiles": 1.0 * new_tiles,
            "new_rooms": 5.0 * new_rooms,
            "stairs_seen": 10.0 * stairs_seen_bonus,
            "useful_items_seen": 2.0 * useful_items_seen_bonus,
            "repeated_state": -0.25 if repeated_state else 0.0,
            "revisited_tile": -0.10 if revisited_recent_tile else 0.0,
            "repeated_action": -0.35 if repeated_action else 0.0,
            "invalid_action": -0.50 if invalid_like_action else 0.0,
            "death": _death_penalty(terminated, hp_after),
        })
    elif task == "survive":
        low_hp_before = hp_before <= max(1, int(0.5 * max(1, int(obs_before["blstats"][nethack.NLE_BL_HPMAX]))))
        components.update({
            "death": _death_penalty(terminated, hp_after),
            "hp_loss": -1.0 * hp_loss,
            "hp_gain": 0.5 * hp_gain,
            "low_hp_damage": -1.0 * hp_loss if low_hp_before and hp_loss else 0.0,
            "repeated_state": -0.25 if repeated_state else 0.0,
            "revisited_tile": -0.10 if revisited_recent_tile else 0.0,
            "repeated_action": -0.25 if repeated_action else 0.0,
            "repeated_bad_action": -0.5 if repeated_action and invalid_like_action else 0.0,
        })
    elif task == "combat":
        kill_like = 1.0 if hostile_before > 0 and hostile_after < hostile_before else 0.0
        safe_resolution = 1.0 if hostile_before > 0 and hostile_after == 0 else 0.0
        components.update({
            "kill": 15.0 * kill_like,
            "hp_loss": -1.5 * hp_loss,
            "safe_resolution": 5.0 * safe_resolution,
            "threat_noop": -0.5 if hostile_before > 0 and action_name in {"wait", "search"} else 0.0,
            "death": _death_penalty(terminated, hp_after),
        })
    elif task == "descend":
        components.update({
            "reach_stairs": 25.0 * stairs_seen_bonus,
            "descend": 100.0 * depth_gain,
            "score_delta": 0.1 * score_delta,
            "gold_gain": 0.2 * gold_gain,
            "xp_gain": 1.0 * xp_gain,
            "camping": -0.5 if new_tiles == 0 and depth_gain == 0 and score_delta == 0 else 0.0,
            "death": _death_penalty(terminated, hp_after),
        })
    elif task == "resource":
        valid_pickup = 1.0 if action_name == "pickup" and "see here" in state_before.get("message", "").lower() else 0.0
        components.update({
            "pickup": 2.0 * valid_pickup,
            "gold_gain": 1.0 * gold_gain,
            "useful_item_seen": 1.0 * useful_items_seen_bonus,
            "invalid_action": -1.0 if invalid_like_action else 0.0,
            "bad_inventory": -2.0 if action_name in {"eat", "drink", "drop"} and not low_value_inventory_context(state_before) else 0.0,
            "death": _death_penalty(terminated, hp_after),
        })
    else:
        raise ValueError(f"Unknown task: {task}")

    total = float(sum(components.values()))
    return TaskRewardResult(total=total, components=components)


def low_value_inventory_context(state: dict) -> bool:
    """Very conservative heuristic for whether inventory action might be sensible."""
    message = state.get("message", "").lower()
    return "hungry" in message or "faint" in message or "quaff" in message
