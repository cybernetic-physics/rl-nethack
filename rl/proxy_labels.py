from __future__ import annotations

import re
from statistics import mean


_POS_RE = re.compile(r"Pos:\(([-+]?\d+),\s*([-+]?\d+)\)")


def parse_prompt_adjacent(prompt: str) -> dict[str, str]:
    for line in prompt.splitlines():
        if not line.startswith("Adjacent:"):
            continue
        payload = line.split(":", 1)[1].strip()
        adjacent: dict[str, str] = {}
        for token in payload.split():
            if "=" not in token:
                continue
            direction, tile = token.split("=", 1)
            adjacent[direction.strip()] = tile.strip()
        return adjacent
    return {}


def parse_prompt_counts(prompt: str) -> dict[str, int]:
    monsters = 0
    items = 0
    for line in prompt.splitlines():
        if line.startswith("Monsters:"):
            payload = line.split(":", 1)[1].strip()
            monsters = 0 if payload in {"", "none"} else payload.count("@(")
        elif line.startswith("Items:"):
            payload = line.split(":", 1)[1].strip()
            items = 0 if payload in {"", "none"} else payload.count("@(")
    return {"visible_monsters": monsters, "visible_items": items}


def parse_prompt_position(prompt: str) -> tuple[int, int] | None:
    match = _POS_RE.search(prompt)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def teacher_margin(row: dict) -> float:
    planner_trace = row.get("planner_trace") or []
    if not planner_trace:
        return 0.0
    totals = {candidate["action"]: float(candidate["total"]) for candidate in planner_trace}
    chosen_action = row.get("action")
    chosen_total = totals.get(chosen_action, max(totals.values()))
    best_other = max((total for action, total in totals.items() if action != chosen_action), default=chosen_total)
    return float(chosen_total - best_other)


def search_context_label(row: dict) -> int:
    allowed = set(row.get("allowed_actions", []))
    if "search" not in allowed:
        return 0

    adjacent = parse_prompt_adjacent(row.get("prompt", ""))
    counts = parse_prompt_counts(row.get("prompt", ""))

    wall_count = sum(1 for tile in adjacent.values() if tile == "wall")
    corridor_count = sum(1 for tile in adjacent.values() if tile == "corridor")
    passable_count = sum(1 for tile in adjacent.values() if tile in {"floor", "corridor", "door"})
    unseen_count = sum(1 for tile in adjacent.values() if tile == "unseen")
    monster_adjacent = sum(1 for tile in adjacent.values() if tile.startswith("monster"))
    stairs_adjacent = sum(1 for tile in adjacent.values() if tile in {"stairs_down", "stairs_up"})

    if monster_adjacent > 0 or counts["visible_monsters"] > 0:
        return 0
    if stairs_adjacent > 0:
        return 0

    dead_end_like = wall_count >= 2 and passable_count == 1
    boxed_wall_like = wall_count >= 3 and passable_count >= 1
    corridor_probe_like = wall_count >= 2 and corridor_count >= 1 and unseen_count <= 1
    low_distraction = counts["visible_items"] == 0

    return int(low_distraction and (dead_end_like or boxed_wall_like or corridor_probe_like))


def k_step_progress(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    start = rows[0]
    tiles_gain = sum(len(row.get("delta", {}).get("new_tiles", [])) for row in rows)
    rooms_before = float(start.get("rooms_discovered_before", 0))
    rooms_after = max(float(row.get("rooms_discovered_before", rooms_before)) for row in rows)
    rooms_gain = max(0.0, rooms_after - rooms_before)
    depth_gain = sum(max(0.0, float(row.get("delta", {}).get("depth_delta", 0))) for row in rows)
    stairs_exposure = any("stairs_" in row.get("prompt", "") for row in rows)
    return float(tiles_gain + 5.0 * rooms_gain + (3.0 if stairs_exposure else 0.0) + 10.0 * depth_gain)


def k_step_survival(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    death = any(bool(row.get("done")) and not bool(row.get("delta", {}).get("survived", True)) for row in rows)
    hp_loss = sum(max(0.0, -float(row.get("delta", {}).get("hp_delta", 0))) for row in rows)
    return float((-100.0 if death else 0.0) - hp_loss)


def k_step_loop_risk(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    positions = [parse_prompt_position(row.get("prompt", "")) for row in rows]
    positions = [pos for pos in positions if pos is not None]
    duplicate_positions = 0
    if positions:
        seen = set()
        for pos in positions:
            duplicate_positions += int(pos in seen)
            seen.add(pos)
    obs_hashes = [row.get("obs_hash") for row in rows] + [row.get("next_obs_hash") for row in rows]
    obs_hashes = [h for h in obs_hashes if h]
    duplicate_hashes = len(obs_hashes) - len(set(obs_hashes))
    recent_pressure = mean(
        (
            float(row.get("recent_position_count", 0)) / 8.0
            + float(row.get("recent_action_count", 0)) / 8.0
        )
        for row in rows
    )
    return float(duplicate_positions + duplicate_hashes + recent_pressure)


def k_step_resource_value(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    gold_gain = sum(max(0.0, float(row.get("delta", {}).get("gold_delta", 0))) for row in rows)
    visible_items = [parse_prompt_counts(row.get("prompt", ""))["visible_items"] for row in rows]
    item_exposure = max(visible_items) if visible_items else 0
    pickup_count = sum(1 for row in rows if row.get("action") == "pickup")
    return float(gold_gain / 10.0 + item_exposure + 0.5 * pickup_count)
