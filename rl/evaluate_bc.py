from __future__ import annotations

from collections import Counter

import nle.env

from nle_agent.agent_http import _build_action_map
from rl.bc_model import load_bc_model
from rl.feature_encoder import encode_observation
from rl.options import build_skill_registry
from src.memory_tracker import MemoryTracker
from src.state_encoder import StateEncoder
from src.task_harness import evaluate_task_policy


def evaluate_bc_policy(model_path: str, task: str, seeds: list[int], max_steps: int, compare_baseline: bool = False) -> dict:
    encoder = StateEncoder()
    registry = build_skill_registry()
    input_dim = 106
    policy = load_bc_model(model_path, input_dim=input_dim)
    action_map = _build_action_map()
    episodes = []

    for seed in seeds:
        env = nle.env.NLE()
        obs, _ = env.reset(seed=seed)
        memory = MemoryTracker()
        memory.update(obs)
        memory.detect_rooms()
        action_counts = Counter()
        total_env_reward = 0.0

        for step in range(max_steps):
            state = encoder.encode_full(obs)
            allowed_actions = registry[task].allowed_actions(state, memory)
            timestep = {
                "state": state,
                "active_skill": task,
                "allowed_actions": allowed_actions,
                "memory_total_explored": memory.total_explored,
                "rooms_discovered": len(memory.rooms),
            }
            features = encode_observation(timestep)
            action_name = policy.act(features, allowed_actions=allowed_actions)
            obs, reward, terminated, truncated, _ = env.step(action_map.get(action_name, action_map["wait"]))
            memory.update(obs)
            memory.detect_rooms()
            total_env_reward += float(reward)
            action_counts[action_name] += 1
            if terminated or truncated:
                break

        episodes.append(
            {
                "seed": seed,
                "steps": step + 1,
                "unique_tiles": int(memory.total_explored),
                "rooms_discovered": len(memory.rooms),
                "total_env_reward": round(total_env_reward, 4),
                "action_counts": dict(action_counts),
            }
        )
        env.close()

    summary = {
        "episodes": len(episodes),
        "avg_unique_tiles": round(sum(row["unique_tiles"] for row in episodes) / len(episodes), 2),
        "avg_rooms_discovered": round(sum(row["rooms_discovered"] for row in episodes) / len(episodes), 2),
        "avg_env_reward": round(sum(row["total_env_reward"] for row in episodes) / len(episodes), 4),
        "action_counts": dict(sum((Counter(row["action_counts"]) for row in episodes), Counter())),
    }
    result = {"task": task, "summary": summary, "episodes": episodes}
    if compare_baseline:
        result["baseline"] = evaluate_task_policy(task=task, seeds=seeds, max_steps=max_steps, policy="task_greedy")
    return result
