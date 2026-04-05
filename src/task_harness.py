"""
Task-directed rollout and evaluation harness for NetHack.

This is the first control-oriented layer for the repo. It does not implement
full PPO-style RL; instead it provides:

1. shaped task rewards for closed-loop trajectories,
2. a one-step counterfactual controller that picks actions by shaped return,
3. trajectory evaluation over fixed seed sets.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections import Counter, deque

import nle.env

from nle_agent.agent_http import _build_action_map
from src.data_generator import wall_avoidance_policy
from src.memory_tracker import MemoryTracker
from src.state_encoder import StateEncoder
from src.task_rewards import (
    compute_task_rewards,
    encode_task_reward_features,
    observation_hash,
    snapshot_memory,
)


SAFE_ACTIONS = ("north", "south", "east", "west", "wait", "search", "pickup", "up", "down")
TASK_DIRECTIVES = {
    "explore": "Prefer unseen frontier, new rooms, stairs discovery, and useful item discovery. Avoid loops and no-op actions.",
    "survive": "Stay alive, minimize HP loss, and avoid repeating bad actions when under threat or low HP.",
    "combat": "Resolve nearby threats safely. Prefer actions that reduce adjacent hostiles while preserving HP.",
    "descend": "Find stairs, reach them, and go deeper. Prefer actions that increase dungeon progress.",
    "resource": "Collect useful resources efficiently and avoid nonsensical inventory actions.",
}

_ACTION_MAP = None


def _get_action_map():
    global _ACTION_MAP
    if _ACTION_MAP is None:
        _ACTION_MAP = _build_action_map()
    return _ACTION_MAP


def _candidate_actions(task: str, obs: dict, state: dict) -> list[str]:
    del task

    candidates = ["north", "south", "east", "west", "wait", "search"]
    message = state.get("message", "").lower()

    if "see here" in message or any(item["pos"] == state["position"] for item in state.get("visible_items", [])):
        candidates.append("pickup")

    px, py = state["position"]
    current_char = chr(int(obs["chars"][py, px]))
    if current_char == ">":
        candidates.append("down")
    elif current_char == "<":
        candidates.append("up")

    # Keep order stable for deterministic tie-breaks.
    deduped = []
    for name in candidates:
        if name not in deduped:
            deduped.append(name)
    return deduped


def _action_visit_count(action_name: str, state: dict, memory: MemoryTracker) -> int:
    px, py = state["position"]
    if action_name == "north":
        py -= 1
    elif action_name == "south":
        py += 1
    elif action_name == "east":
        px += 1
    elif action_name == "west":
        px -= 1
    if 0 <= py < memory.visit_counts.shape[0] and 0 <= px < memory.visit_counts.shape[1]:
        return int(memory.visit_counts[py, px])
    return 9999


def _counterfactual_score_action(
    env,
    obs_before: dict,
    state_before: dict,
    memory_before: MemoryTracker,
    task: str,
    action_name: str,
    prev_action: str | None,
    recent_state_hashes: list[str],
    recent_positions: list[tuple[int, int]],
    encoder: StateEncoder,
) -> dict:
    """Fork the current env state, try one action, and score it with shaped reward."""
    action_map = _get_action_map()
    action_idx = action_map[action_name]
    tmpdir = tempfile.mkdtemp(prefix="task_cf_")
    out_path = os.path.join(tmpdir, f"{action_name}.json")

    pid = os.fork()
    if pid == 0:
        try:
            obs_after, reward, terminated, truncated, _ = env.step(action_idx)
            state_after = encoder.encode_full(obs_after)
            mem_before = snapshot_memory(memory_before)
            memory_before.update(obs_after)
            memory_before.detect_rooms()
            mem_after = snapshot_memory(memory_before)
            next_hash = observation_hash(obs_after)
            repeated_state = next_hash in recent_state_hashes
            revisited_recent_tile = state_after["position"] in recent_positions
            repeated_action = action_name == prev_action

            reward_result = compute_task_rewards(
                task=task,
                obs_before=obs_before,
                obs_after=obs_after,
                state_before=state_before,
                state_after=state_after,
                memory_before=mem_before,
                memory_after=mem_after,
                action_name=action_name,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                repeated_state=repeated_state,
                revisited_recent_tile=revisited_recent_tile,
                repeated_action=repeated_action,
            )
            payload = {
                "action": action_name,
                "total": reward_result.total,
                "components": reward_result.components,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "position": list(state_after["position"]),
                "obs_hash": next_hash,
                "reward_features": encode_task_reward_features(
                    task=task,
                    obs_before=obs_before,
                    obs_after=obs_after,
                    state_before=state_before,
                    state_after=state_after,
                    memory_before=mem_before,
                    memory_after=mem_after,
                    action_name=action_name,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    repeated_state=repeated_state,
                    revisited_recent_tile=revisited_recent_tile,
                    repeated_action=repeated_action,
                ),
            }
            with open(out_path, "w") as f:
                json.dump(payload, f)
        finally:
            env.close()
            os._exit(0)

    os.waitpid(pid, 0)
    with open(out_path, "r") as f:
        result = json.load(f)
    os.unlink(out_path)
    os.rmdir(tmpdir)
    return result


def select_task_action(
    env,
    obs_before: dict,
    state_before: dict,
    memory_before: MemoryTracker,
    task: str,
    prev_action: str | None,
    recent_state_hashes: list[str],
    recent_positions: list[tuple[int, int]],
    encoder: StateEncoder,
) -> tuple[str, list[dict]]:
    """Choose the best safe action by one-step counterfactual shaped reward."""
    candidates = _candidate_actions(task, obs_before, state_before)
    scored = []
    for action_name in candidates:
        scored.append(
            _counterfactual_score_action(
                env=env,
                obs_before=obs_before,
                state_before=state_before,
                memory_before=memory_before,
                task=task,
                action_name=action_name,
                prev_action=prev_action,
                recent_state_hashes=recent_state_hashes,
                recent_positions=recent_positions,
                encoder=encoder,
            )
        )

    # Deterministic tie-break on action name keeps rollouts reproducible.
    best = max(
        scored,
        key=lambda row: (
            row["total"],
            -int(row["terminated"]),
            -_action_visit_count(row["action"], state_before, memory_before),
        ),
    )
    return best["action"], scored


def run_task_episode(
    seed: int,
    task: str,
    max_steps: int,
    policy: str = "task_greedy",
    encoder: StateEncoder | None = None,
) -> dict:
    """Run one episode under a task policy and collect trajectory metrics."""
    if task not in TASK_DIRECTIVES:
        raise ValueError(f"Unknown task: {task}")
    if policy not in {"task_greedy", "wall_avoidance"}:
        raise ValueError(f"Unknown policy: {policy}")

    encoder = encoder or StateEncoder()
    rng = random.Random(seed)
    action_map = _get_action_map()
    env = nle.env.NLE()
    obs, info = env.reset(seed=seed)
    del info

    memory = MemoryTracker()
    memory.update(obs)
    memory.detect_rooms()

    prev_action = None
    recent_state_hashes = deque([observation_hash(obs)], maxlen=8)
    recent_positions = deque([encoder.encode_full(obs)["position"]], maxlen=8)

    total_task_reward = 0.0
    total_env_reward = 0.0
    component_sums = Counter()
    action_counts = Counter()
    planner_rows = []
    repeated_state_steps = 0
    repeated_action_steps = 0
    revisited_recent_tile_steps = 0
    terminated = False
    truncated = False
    step = -1

    for step in range(max_steps):
        state_before = encoder.encode_full(obs)
        if policy == "wall_avoidance":
            action_name = wall_avoidance_policy(state_before["adjacent"], rng)
            planner_trace = []
        else:
            action_name, planner_trace = select_task_action(
                env=env,
                obs_before=obs,
                state_before=state_before,
                memory_before=memory,
                task=task,
                prev_action=prev_action,
                recent_state_hashes=list(recent_state_hashes),
                recent_positions=list(recent_positions),
                encoder=encoder,
            )
        action_idx = action_map.get(action_name, action_map["wait"])

        mem_before = snapshot_memory(memory)
        obs_after, reward, terminated, truncated, _ = env.step(action_idx)
        state_after = encoder.encode_full(obs_after)
        memory.update(obs_after)
        memory.detect_rooms()
        mem_after = snapshot_memory(memory)

        next_hash = observation_hash(obs_after)
        repeated_state = next_hash in recent_state_hashes
        revisited_recent_tile = state_after["position"] in recent_positions
        repeated_action = action_name == prev_action

        reward_result = compute_task_rewards(
            task=task,
            obs_before=obs,
            obs_after=obs_after,
            state_before=state_before,
            state_after=state_after,
            memory_before=mem_before,
            memory_after=mem_after,
            action_name=action_name,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            repeated_state=repeated_state,
            revisited_recent_tile=revisited_recent_tile,
            repeated_action=repeated_action,
        )

        total_task_reward += reward_result.total
        total_env_reward += float(reward)
        component_sums.update(reward_result.components)
        action_counts[action_name] += 1
        repeated_state_steps += int(repeated_state)
        repeated_action_steps += int(repeated_action)
        revisited_recent_tile_steps += int(revisited_recent_tile)
        planner_rows.append(
            {
                "step": step,
                "action": action_name,
                "task_reward": reward_result.total,
                "env_reward": float(reward),
                "hp": state_after["hp"],
                "depth": state_after["depth"],
                "position": list(state_after["position"]),
                "planner_trace": planner_trace,
            }
        )

        recent_state_hashes.append(next_hash)
        recent_positions.append(state_after["position"])
        prev_action = action_name
        obs = obs_after

        if terminated or truncated:
            break

    final_state = encoder.encode_full(obs)
    env.close()
    steps = step + 1 if step >= 0 else 0

    return {
        "seed": seed,
        "task": task,
        "policy": policy,
        "directive": TASK_DIRECTIVES[task],
        "steps": steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "alive": final_state["hp"] > 0 and not terminated,
        "total_task_reward": round(total_task_reward, 4),
        "total_env_reward": round(total_env_reward, 4),
        "unique_tiles": int(memory.total_explored),
        "rooms_discovered": len(memory.rooms),
        "final_hp": final_state["hp"],
        "final_depth": final_state["depth"],
        "final_gold": final_state["gold"],
        "repeated_state_steps": repeated_state_steps,
        "repeated_action_steps": repeated_action_steps,
        "revisited_recent_tile_steps": revisited_recent_tile_steps,
        "action_counts": dict(action_counts),
        "component_sums": {k: round(v, 4) for k, v in sorted(component_sums.items())},
        "trajectory": planner_rows,
    }


def summarize_task_runs(results: list[dict]) -> dict:
    """Aggregate task evaluation metrics across seeds."""
    if not results:
        return {
            "episodes": 0,
            "avg_task_reward": 0.0,
            "avg_env_reward": 0.0,
            "avg_unique_tiles": 0.0,
            "avg_rooms_discovered": 0.0,
            "avg_final_hp": 0.0,
            "avg_final_depth": 0.0,
            "survival_rate": 0.0,
            "repeated_state_rate": 0.0,
            "repeated_action_rate": 0.0,
            "action_counts": {},
            "component_sums": {},
        }

    action_counts = Counter()
    component_sums = Counter()
    total_steps = sum(row["steps"] for row in results) or 1
    for row in results:
        action_counts.update(row["action_counts"])
        component_sums.update(row["component_sums"])

    n = len(results)
    return {
        "episodes": n,
        "avg_task_reward": round(sum(row["total_task_reward"] for row in results) / n, 4),
        "avg_env_reward": round(sum(row["total_env_reward"] for row in results) / n, 4),
        "avg_unique_tiles": round(sum(row["unique_tiles"] for row in results) / n, 2),
        "avg_rooms_discovered": round(sum(row["rooms_discovered"] for row in results) / n, 2),
        "avg_final_hp": round(sum(row["final_hp"] for row in results) / n, 2),
        "avg_final_depth": round(sum(row["final_depth"] for row in results) / n, 2),
        "avg_final_gold": round(sum(row["final_gold"] for row in results) / n, 2),
        "survival_rate": round(sum(1 for row in results if row["alive"]) / n, 4),
        "repeated_state_rate": round(sum(row["repeated_state_steps"] for row in results) / total_steps, 4),
        "repeated_action_rate": round(sum(row["repeated_action_steps"] for row in results) / total_steps, 4),
        "revisited_recent_tile_rate": round(sum(row["revisited_recent_tile_steps"] for row in results) / total_steps, 4),
        "action_counts": dict(action_counts),
        "component_sums": {k: round(v, 4) for k, v in sorted(component_sums.items())},
    }


def evaluate_task_policy(
    task: str,
    seeds: list[int],
    max_steps: int,
    policy: str,
    encoder: StateEncoder | None = None,
) -> dict:
    """Run a task policy on a fixed seed set and aggregate metrics."""
    encoder = encoder or StateEncoder()
    episodes = [
        run_task_episode(seed=seed, task=task, max_steps=max_steps, policy=policy, encoder=encoder)
        for seed in seeds
    ]
    return {
        "task": task,
        "policy": policy,
        "directive": TASK_DIRECTIVES[task],
        "summary": summarize_task_runs(episodes),
        "episodes": episodes,
    }
