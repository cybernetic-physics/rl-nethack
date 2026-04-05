from __future__ import annotations

from src.memory_tracker import MemoryTracker


def build_policy_timestep(
    *,
    state: dict,
    task: str,
    allowed_actions: list[str],
    memory: MemoryTracker,
    step: int,
    recent_positions: list[tuple[int, int]],
    recent_actions: list[str],
    recent_state_hashes: list[str] | None = None,
    obs_hash: str | None = None,
    obs: dict,
) -> dict:
    px, py = state["position"]
    tile_char = chr(int(obs["chars"][py, px])) if px >= 0 and py >= 0 else " "
    prev_action = recent_actions[-1] if recent_actions else None
    return {
        "state": state,
        "active_skill": task,
        "allowed_actions": allowed_actions,
        "memory_total_explored": memory.total_explored,
        "rooms_discovered": len(memory.rooms),
        "steps_in_skill": step,
        "standing_on_down_stairs": tile_char == ">",
        "standing_on_up_stairs": tile_char == "<",
        "recent_positions": list(recent_positions),
        "recent_actions": list(recent_actions),
        "repeated_state_count": (
            sum(1 for state_hash in recent_state_hashes if state_hash == obs_hash)
            if recent_state_hashes is not None and obs_hash is not None
            else 0
        ),
        "revisited_recent_tile_count": sum(1 for pos in recent_positions if tuple(pos) == tuple(state["position"])),
        "repeated_action_count": sum(1 for action in recent_actions if action == prev_action) if prev_action else 0,
    }
