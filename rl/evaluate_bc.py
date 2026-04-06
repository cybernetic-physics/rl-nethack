from __future__ import annotations

from collections import Counter

import nle.env
import torch

from nle_agent.agent_http import _build_action_map
from rl.bc_model import load_bc_model
from rl.evaluate import LIVE_ENV_WARNING
from rl.feature_encoder import encode_observation
from rl.options import build_skill_registry
from rl.timestep import build_policy_timestep
from rl.world_model import load_world_model
from rl.world_model_features import augment_feature_vector
from src.memory_tracker import MemoryTracker
from src.state_encoder import StateEncoder
from src.task_harness import evaluate_task_policy
from src.task_rewards import observation_hash


def evaluate_bc_policy(model_path: str, task: str, seeds: list[int], max_steps: int, compare_baseline: bool = False) -> dict:
    encoder = StateEncoder()
    registry = build_skill_registry()
    policy = load_bc_model(model_path)
    action_map = _build_action_map()
    payload = torch.load(model_path, map_location="cpu")
    metadata = payload.get("metadata", {})
    observation_version = metadata.get("observation_version", "v1")
    world_model_path = metadata.get("world_model_path")
    world_model_feature_mode = metadata.get("world_model_feature_mode")
    world_model_inference = (
        load_world_model(world_model_path) if world_model_path and world_model_feature_mode else None
    )
    episodes = []

    for seed in seeds:
        env = nle.env.NLE()
        obs, _ = env.reset(seed=seed)
        memory = MemoryTracker()
        memory.update(obs)
        memory.detect_rooms()
        recent_positions = [tuple(encoder.encode_full(obs)["position"])]
        recent_actions: list[str] = []
        recent_state_hashes = [observation_hash(obs)]
        action_counts = Counter()
        total_env_reward = 0.0

        for step in range(max_steps):
            state = encoder.encode_full(obs)
            allowed_actions = registry[task].allowed_actions(state, memory)
            timestep = build_policy_timestep(
                state=state,
                task=task,
                allowed_actions=allowed_actions,
                memory=memory,
                step=step,
                recent_positions=recent_positions,
                recent_actions=recent_actions,
                recent_state_hashes=recent_state_hashes,
                obs_hash=observation_hash(obs),
                obs=obs,
            )
            features = encode_observation(timestep, version=observation_version)
            if world_model_inference is not None and world_model_feature_mode:
                features = augment_feature_vector(features, world_model_inference, mode=world_model_feature_mode)
            action_name = policy.act(features, allowed_actions=allowed_actions)
            obs, reward, terminated, truncated, _ = env.step(action_map.get(action_name, action_map["wait"]))
            memory.update(obs)
            memory.detect_rooms()
            recent_actions.append(action_name)
            recent_positions.append(tuple(encoder.encode_full(obs)["position"]))
            recent_state_hashes.append(observation_hash(obs))
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
        "evaluation_mode": "live_env_seeded",
        "warning": LIVE_ENV_WARNING,
        "avg_unique_tiles": round(sum(row["unique_tiles"] for row in episodes) / len(episodes), 2),
        "avg_rooms_discovered": round(sum(row["rooms_discovered"] for row in episodes) / len(episodes), 2),
        "avg_env_reward": round(sum(row["total_env_reward"] for row in episodes) / len(episodes), 4),
        "action_counts": dict(sum((Counter(row["action_counts"]) for row in episodes), Counter())),
    }
    result = {"task": task, "summary": summary, "episodes": episodes}
    if compare_baseline:
        result["baseline"] = evaluate_task_policy(task=task, seeds=seeds, max_steps=max_steps, policy="task_greedy")
    return result
