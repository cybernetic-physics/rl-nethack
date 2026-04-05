from __future__ import annotations

import json
import os
import random
import urllib.request
from collections import Counter
from typing import Optional

import nle.env
import numpy as np

from nle_agent.agent_http import _build_action_map
from rl.bc_model import load_bc_model
from rl.config import RLConfig
from rl.feature_encoder import ACTION_SET, encode_observation
from rl.options import build_skill_registry
from rl.scheduler import SchedulerContext, build_scheduler
from rl.sf_env import NethackSkillEnv
from src.data_generator import build_messages, wall_avoidance_policy
from src.evaluator import parse_prediction
from src.memory_tracker import MemoryTracker
from src.state_encoder import StateEncoder
from src.task_harness import select_task_action
from src.task_rewards import observation_hash, snapshot_memory


def verify_trace_file(path: str) -> dict:
    episodes = {}
    total_rows = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total_rows += 1
            episodes.setdefault(row["episode_id"], []).append(row["step"])

    episode_lengths = [len(steps) for steps in episodes.values()]
    multi_turn_episodes = sum(1 for length in episode_lengths if length > 1)
    return {
        "episodes": len(episodes),
        "rows": total_rows,
        "max_steps_in_episode": max(episode_lengths) if episode_lengths else 0,
        "avg_steps_in_episode": round(sum(episode_lengths) / len(episode_lengths), 2) if episode_lengths else 0.0,
        "multi_turn_episodes": multi_turn_episodes,
        "all_multi_turn": bool(episodes) and multi_turn_episodes == len(episodes),
    }


def _query_forward_model(server_url: str, model_name: str, prompt_text: str) -> dict:
    messages = build_messages(prompt_text)
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 64,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    return parse_prediction(content)


def _score_forward_prediction(task: str, pred: dict) -> float:
    score = 0.0
    if pred.get("survived", True):
        score += 2.0
    else:
        score -= 20.0
    score += float(pred.get("gold_delta", 0)) * 0.5
    score += float(pred.get("depth_delta", 0)) * 10.0
    score += float(pred.get("hp_delta", 0)) * 0.5
    if task == "explore":
        pos = pred.get("pos", (0, 0))
        score += 1.0 if pos != (0, 0) else -0.25
    elif task == "survive":
        score += max(0.0, float(pred.get("hp_delta", 0)))
    return score


def _forward_model_action(
    encoder: StateEncoder,
    state: dict,
    task: str,
    allowed_actions: list[str],
    server_url: str,
    model_name: str,
) -> str:
    best_action = allowed_actions[0] if allowed_actions else "wait"
    best_score = -1e9
    for action_name in allowed_actions:
        prompt_text = encoder.format_prompt(state, action_name)
        pred = _query_forward_model(server_url, model_name, prompt_text)
        score = _score_forward_prediction(task, pred)
        if score > best_score:
            best_score = score
            best_action = action_name
    return best_action


def generate_multi_turn_traces(
    output_path: str,
    num_episodes: int,
    max_steps: int,
    seed_start: int = 42,
    policy: str = "task_greedy",
    task: str = "explore",
    appo_experiment: Optional[str] = None,
    appo_train_dir: str = "train_dir/rl",
    bc_model_path: Optional[str] = None,
    forward_model_server_url: Optional[str] = None,
    forward_model_name: str = "llama-server",
) -> dict:
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    encoder = StateEncoder()
    action_map = _build_action_map()
    registry = build_skill_registry()
    scheduler = build_scheduler("rule_based")

    appo_eval = None
    if policy == "appo":
        from rl.evaluate import _load_actor_critic
        import torch
        from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
        from sample_factory.algo.utils.action_distributions import argmax_actions
        from sample_factory.model.model_utils import get_rnn_size

        cfg, env_eval, actor_critic, device = _load_actor_critic(appo_experiment, appo_train_dir, device="cpu")
        appo_eval = {
            "cfg": cfg,
            "env": env_eval,
            "actor_critic": actor_critic,
            "device": device,
            "prepare": prepare_and_normalize_obs,
            "argmax": argmax_actions,
            "get_rnn_size": get_rnn_size,
            "torch": torch,
        }

    bc_policy = None
    if policy == "bc":
        bc_policy = load_bc_model(bc_model_path, input_dim=106)

    total_rows = 0
    episode_lengths = []

    with open(output_path, "w") as f:
        for ep_idx in range(num_episodes):
            seed = seed_start + ep_idx
            if policy == "appo":
                env = NethackSkillEnv(RLConfig())
                obs, info = env.reset(seed=seed)
                memory = env.adapter.memory
                state = env.adapter._encode_state(env.adapter.obs)
                rnn_states = appo_eval["torch"].zeros(
                    [1, appo_eval["get_rnn_size"](appo_eval["cfg"])],
                    dtype=appo_eval["torch"].float32,
                    device=appo_eval["device"],
                )
                recent_state_hashes = [observation_hash(env.adapter.obs)]
                recent_positions = [state["position"]]
                prev_action = None
            else:
                env = nle.env.NLE()
                obs, _ = env.reset(seed=seed)
                memory = MemoryTracker()
                memory.update(obs)
                memory.detect_rooms()
                state = encoder.encode_full(obs)
                recent_state_hashes = [observation_hash(obs)]
                recent_positions = [state["position"]]
                prev_action = None

            action_counts = Counter()
            steps = 0
            for step in range(max_steps):
                state = encoder.encode_full(obs if policy != "appo" else env.adapter.obs)
                px, py = state["position"]
                chars = (obs if policy != "appo" else env.adapter.obs)["chars"]
                state["standing_on_down_stairs"] = chr(int(chars[py, px])) == ">"
                available_skills = [name for name, option in registry.items() if option.can_start(state, memory)] or [task]
                active_skill = task if task in available_skills else scheduler.select_skill(
                    SchedulerContext(
                        state=state,
                        memory=memory,
                        active_skill=task,
                        steps_in_skill=step,
                        available_skills=available_skills,
                    )
                )
                allowed_actions = registry[active_skill].allowed_actions(state, memory)

                if policy == "task_greedy":
                    action_name, planner_trace = select_task_action(
                        env=env,
                        obs_before=obs,
                        state_before=state,
                        memory_before=memory,
                        task=active_skill,
                        prev_action=prev_action,
                        recent_state_hashes=list(recent_state_hashes),
                        recent_positions=list(recent_positions),
                        encoder=encoder,
                    )
                elif policy == "wall_avoidance":
                    planner_trace = []
                    action_name = wall_avoidance_policy(state["adjacent"], random.Random(seed + step))
                elif policy == "forward_model":
                    planner_trace = []
                    action_name = _forward_model_action(
                        encoder=encoder,
                        state=state,
                        task=active_skill,
                        allowed_actions=allowed_actions,
                        server_url=forward_model_server_url,
                        model_name=forward_model_name,
                    )
                elif policy == "bc":
                    planner_trace = []
                    timestep = {
                        "state": state,
                        "active_skill": active_skill,
                        "allowed_actions": allowed_actions,
                        "memory_total_explored": memory.total_explored,
                        "rooms_discovered": len(memory.rooms),
                    }
                    features = encode_observation(timestep)
                    action_name = bc_policy.act(features, allowed_actions=allowed_actions)
                elif policy == "appo":
                    planner_trace = []
                    obs_features = appo_eval["torch"].from_numpy(obs).unsqueeze(0).to(appo_eval["device"])
                    normalized_obs = appo_eval["prepare"](appo_eval["actor_critic"], {"obs": obs_features})
                    policy_outputs = appo_eval["actor_critic"](normalized_obs, rnn_states)
                    raw_logits = appo_eval["actor_critic"].action_distribution().raw_logits.clone()
                    for idx, name in enumerate(ACTION_SET):
                        if name not in set(info.get("allowed_actions", allowed_actions)):
                            raw_logits[0, idx] = -1e9
                    action_idx = int(np.argmax(raw_logits.cpu().numpy(), axis=1)[0])
                    action_name = ACTION_SET[action_idx]
                    rnn_states = policy_outputs["new_rnn_states"]
                else:
                    raise ValueError(f"Unknown trace policy: {policy}")

                if action_name not in action_map:
                    action_name = "wait"
                prompt_text = encoder.format_prompt(state, action_name)
                obs_before = obs if policy != "appo" else env.adapter.obs
                if policy == "appo":
                    obs_after, reward, terminated, truncated, info_after = env.step(ACTION_SET.index(action_name))
                    raw_obs_after = env.adapter.obs
                else:
                    obs_after, reward, terminated, truncated, info_after = env.step(action_map[action_name])
                    raw_obs_after = obs_after
                delta = encoder.encode_delta(obs_before, raw_obs_after, action_name)
                if policy != "appo":
                    memory.update(obs_after)
                    memory.detect_rooms()
                else:
                    memory = env.adapter.memory
                next_hash = observation_hash(raw_obs_after)
                record = {
                    "episode_id": f"{policy}:{seed}",
                    "seed": seed,
                    "step": step,
                    "policy": policy,
                    "task": active_skill,
                    "action": action_name,
                    "allowed_actions": allowed_actions,
                    "prompt": prompt_text,
                    "delta": delta,
                    "reward": float(reward),
                    "done": bool(terminated or truncated),
                    "obs_hash": observation_hash(obs_before),
                    "next_obs_hash": next_hash,
                    "feature_vector": encode_observation(
                        {
                            "state": state,
                            "active_skill": active_skill,
                            "allowed_actions": allowed_actions,
                            "memory_total_explored": memory.total_explored,
                            "rooms_discovered": len(memory.rooms),
                        }
                    ).tolist(),
                    "planner_trace": planner_trace,
                }
                f.write(json.dumps(record) + "\n")
                total_rows += 1
                action_counts[action_name] += 1
                steps += 1
                prev_action = action_name
                recent_state_hashes.append(next_hash)
                recent_positions.append(encoder.encode_full(raw_obs_after)["position"])
                obs = obs_after
                if terminated or truncated:
                    break

            episode_lengths.append(steps)
            env.close()

    summary = verify_trace_file(output_path)
    summary["num_episodes_requested"] = num_episodes
    summary["policy"] = policy
    return summary
