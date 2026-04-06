from __future__ import annotations

from collections import Counter

import nle.env
import numpy as np
import torch
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.model_utils import get_rnn_size

from nle_agent.agent_http import _build_action_map
from rl.bc_model import load_bc_model
from rl.config import RLConfig
from rl.evaluate import _load_actor_critic
from rl.feature_encoder import ACTION_SET, encode_observation
from rl.options import build_skill_registry
from rl.sf_env import NethackSkillEnv
from rl.timestep import build_policy_timestep
from src.data_generator import wall_avoidance_policy
from src.memory_tracker import MemoryTracker
from src.state_encoder import StateEncoder
from src.task_harness import evaluate_task_policy, select_task_action
from src.task_rewards import observation_hash


def rollout_policy_episode(
    policy: str,
    task: str,
    seed: int,
    max_steps: int,
    bc_model_path: str | None = None,
    appo_experiment: str | None = None,
    appo_train_dir: str = "train_dir/rl",
    observation_version: str | None = None,
) -> dict:
    encoder = StateEncoder()
    registry = build_skill_registry()
    action_map = _build_action_map()
    recent_actions: list[str] = []
    rows: list[dict] = []
    action_counts: Counter[str] = Counter()

    if policy == "appo":
        cfg, env, actor_critic, device = _load_actor_critic(appo_experiment, appo_train_dir, device="cpu")
        env.config.options.enabled_skills = [task]
        env.config.env.active_skill_bootstrap = task
        env.config.env.max_episode_steps = max_steps
        obs, info = env.reset(seed=seed)
        rnn_states = torch.zeros([1, get_rnn_size(cfg)], dtype=torch.float32, device=device)
        recent_positions = [tuple(env.adapter._encode_state(env.adapter.obs)["position"])]
        for step in range(max_steps):
            raw_obs = env.adapter.obs
            state = env.adapter._encode_state(raw_obs)
            allowed_actions = list(info.get("allowed_actions", []))
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            normalized_obs = prepare_and_normalize_obs(actor_critic, {"obs": obs_tensor})
            policy_outputs = actor_critic(normalized_obs, rnn_states)
            raw_logits = actor_critic.action_distribution().raw_logits.clone()
            for idx, name in enumerate(ACTION_SET):
                if name not in set(allowed_actions):
                    raw_logits[0, idx] = -1e9
            action_idx = int(torch.argmax(raw_logits, dim=1).item())
            action_name = ACTION_SET[action_idx]
            rnn_states = policy_outputs["new_rnn_states"]
            obs, rew, terminated, truncated, info = env.step(action_idx)
            debug = info.get("debug", {})
            rows.append(
                {
                    "step": step,
                    "action": debug.get("action_name", action_name),
                    "requested_action": action_name,
                    "allowed_actions": allowed_actions,
                    "position": list(state["position"]),
                    "obs_hash": observation_hash(raw_obs),
                    "reward": float(rew),
                    "invalid_action_requested": bool(debug.get("invalid_action_requested", False)),
                }
            )
            action_counts[rows[-1]["action"]] += 1
            recent_actions.append(rows[-1]["action"])
            if terminated or truncated:
                break
        unique_tiles = int(env.adapter.memory.total_explored)
        rooms = len(env.adapter.memory.rooms)
        env.close()
        return {
            "policy": policy,
            "task": task,
            "seed": seed,
            "steps": len(rows),
            "unique_tiles": unique_tiles,
            "rooms_discovered": rooms,
            "action_counts": dict(action_counts),
            "rows": rows,
        }

    env = nle.env.NLE()
    obs, _ = env.reset(seed=seed)
    memory = MemoryTracker()
    memory.update(obs)
    memory.detect_rooms()
    recent_positions = [tuple(encoder.encode_full(obs)["position"])]

    bc_policy = load_bc_model(bc_model_path) if policy == "bc" else None

    for step in range(max_steps):
        state = encoder.encode_full(obs)
        allowed_actions = registry[task].allowed_actions(state, memory)
        if policy == "task_greedy":
            action_name, _ = select_task_action(
                env=env,
                obs_before=obs,
                state_before=state,
                memory_before=memory,
                task=task,
                prev_action=recent_actions[-1] if recent_actions else None,
                recent_state_hashes=[observation_hash(obs)],
                recent_positions=list(recent_positions),
                encoder=encoder,
            )
        elif policy == "wall_avoidance":
            import random

            action_name = wall_avoidance_policy(state["adjacent"], random.Random(seed + step))
        elif policy == "bc":
            version = observation_version or "v1"
            timestep = build_policy_timestep(
                state=state,
                task=task,
                allowed_actions=allowed_actions,
                memory=memory,
                step=step,
                recent_positions=recent_positions,
                recent_actions=recent_actions,
                recent_state_hashes=[row["obs_hash"] for row in rows] if rows else [observation_hash(obs)],
                obs_hash=observation_hash(obs),
                obs=obs,
            )
            action_name = bc_policy.act(
                encode_observation(timestep, version=version),
                allowed_actions=allowed_actions,
                prompt_text=encoder.format_state_prompt(state),
            )
        else:
            raise ValueError(f"Unsupported policy for rollout: {policy}")

        obs_before = obs
        obs, rew, terminated, truncated, _ = env.step(action_map.get(action_name, action_map["wait"]))
        memory.update(obs)
        memory.detect_rooms()
        rows.append(
            {
                "step": step,
                "action": action_name,
                "allowed_actions": allowed_actions,
                "position": list(state["position"]),
                "obs_hash": observation_hash(obs_before),
                "reward": float(rew),
                "invalid_action_requested": False,
            }
        )
        action_counts[action_name] += 1
        recent_actions.append(action_name)
        recent_positions.append(tuple(encoder.encode_full(obs)["position"]))
        if terminated or truncated:
            break

    result = {
        "policy": policy,
        "task": task,
        "seed": seed,
        "steps": len(rows),
        "unique_tiles": int(memory.total_explored),
        "rooms_discovered": len(memory.rooms),
        "action_counts": dict(action_counts),
        "rows": rows,
    }
    env.close()
    return result


def check_policy_determinism(
    policy: str,
    task: str,
    seeds: list[int],
    max_steps: int,
    repeats: int,
    bc_model_path: str | None = None,
    appo_experiment: str | None = None,
    appo_train_dir: str = "train_dir/rl",
    observation_version: str | None = None,
) -> dict:
    runs = []
    for run_idx in range(repeats):
        episodes = [
            rollout_policy_episode(
                policy=policy,
                task=task,
                seed=seed,
                max_steps=max_steps,
                bc_model_path=bc_model_path,
                appo_experiment=appo_experiment,
                appo_train_dir=appo_train_dir,
                observation_version=observation_version,
            )
            for seed in seeds
        ]
        signature = [
            {
                "seed": episode["seed"],
                "actions": [row["action"] for row in episode["rows"]],
                "positions": [row["position"] for row in episode["rows"]],
                "obs_hashes": [row["obs_hash"] for row in episode["rows"]],
            }
            for episode in episodes
        ]
        runs.append({"run_index": run_idx, "episodes": episodes, "signature": signature})

    stable = True
    mismatches = []
    baseline = runs[0]["signature"] if runs else []
    for run in runs[1:]:
        if run["signature"] != baseline:
            stable = False
            for base_ep, cur_ep in zip(baseline, run["signature"]):
                if base_ep != cur_ep:
                    first_diff = None
                    for step_idx, (base_action, cur_action) in enumerate(zip(base_ep["actions"], cur_ep["actions"])):
                        if base_action != cur_action:
                            first_diff = {
                                "seed": base_ep["seed"],
                                "step": step_idx,
                                "baseline_action": base_action,
                                "current_action": cur_action,
                            }
                            break
                    mismatches.append(first_diff or {"seed": base_ep["seed"], "step": None})
                    break

    return {
        "policy": policy,
        "task": task,
        "seeds": seeds,
        "max_steps": max_steps,
        "repeats": repeats,
        "stable": stable,
        "mismatches": mismatches,
        "runs": runs,
    }


def compare_actions_on_teacher_states(
    task: str,
    seeds: list[int],
    max_steps: int,
    bc_model_path: str | None = None,
    appo_experiment: str | None = None,
    appo_train_dir: str = "train_dir/rl",
    observation_version: str | None = None,
) -> dict:
    encoder = StateEncoder()
    registry = build_skill_registry()
    action_map = _build_action_map()
    bc_policy = load_bc_model(bc_model_path) if bc_model_path else None
    bc_version = observation_version or "v1"

    appo_bundle = None
    if appo_experiment:
        cfg, _, actor_critic, device = _load_actor_critic(appo_experiment, appo_train_dir, device="cpu")
        appo_bundle = {
            "cfg": cfg,
            "actor_critic": actor_critic,
            "device": device,
            "rnn_states": torch.zeros([1, get_rnn_size(cfg)], dtype=torch.float32, device=device),
        }

    episodes = []
    aggregate = {
        "teacher_vs_bc_matches": 0,
        "teacher_vs_bc_total": 0,
        "teacher_vs_appo_matches": 0,
        "teacher_vs_appo_total": 0,
    }

    for seed in seeds:
        env = nle.env.NLE()
        obs, _ = env.reset(seed=seed)
        memory = MemoryTracker()
        memory.update(obs)
        memory.detect_rooms()
        recent_positions = [tuple(encoder.encode_full(obs)["position"])]
        recent_actions: list[str] = []
        recent_state_hashes = [observation_hash(obs)]
        rows = []

        if appo_bundle:
            appo_bundle["rnn_states"] = torch.zeros_like(appo_bundle["rnn_states"])

        for step in range(max_steps):
            state = encoder.encode_full(obs)
            allowed_actions = registry[task].allowed_actions(state, memory)
            teacher_action, _ = select_task_action(
                env=env,
                obs_before=obs,
                state_before=state,
                memory_before=memory,
                task=task,
                prev_action=recent_actions[-1] if recent_actions else None,
                recent_state_hashes=list(recent_state_hashes),
                recent_positions=list(recent_positions),
                encoder=encoder,
            )
            current_obs_hash = observation_hash(obs)
            row = {
                "seed": seed,
                "step": step,
                "teacher_action": teacher_action,
                "allowed_actions": allowed_actions,
                "position": list(state["position"]),
                "obs_hash": current_obs_hash,
            }

            timestep = build_policy_timestep(
                state=state,
                task=task,
                allowed_actions=allowed_actions,
                memory=memory,
                step=step,
                recent_positions=recent_positions,
                recent_actions=recent_actions,
                recent_state_hashes=recent_state_hashes,
                obs_hash=current_obs_hash,
                obs=obs,
            )
            if bc_policy:
                bc_action = bc_policy.act(
                    encode_observation(timestep, version=bc_version),
                    allowed_actions=allowed_actions,
                    prompt_text=encoder.format_state_prompt(state),
                )
                row["bc_action"] = bc_action
                row["bc_matches"] = bc_action == teacher_action
                aggregate["teacher_vs_bc_total"] += 1
                aggregate["teacher_vs_bc_matches"] += int(row["bc_matches"])

            if appo_bundle:
                obs_vec = encode_observation(timestep, version=getattr(appo_bundle["cfg"], "observation_version", "v1"))
                obs_tensor = torch.from_numpy(obs_vec).unsqueeze(0).to(appo_bundle["device"])
                normalized_obs = prepare_and_normalize_obs(appo_bundle["actor_critic"], {"obs": obs_tensor})
                outputs = appo_bundle["actor_critic"](normalized_obs, appo_bundle["rnn_states"])
                logits = appo_bundle["actor_critic"].action_distribution().raw_logits.clone()
                for idx, name in enumerate(ACTION_SET):
                    if name not in set(allowed_actions):
                        logits[0, idx] = -1e9
                appo_action = ACTION_SET[int(torch.argmax(logits, dim=1).item())]
                appo_bundle["rnn_states"] = outputs["new_rnn_states"]
                row["appo_action"] = appo_action
                row["appo_matches"] = appo_action == teacher_action
                aggregate["teacher_vs_appo_total"] += 1
                aggregate["teacher_vs_appo_matches"] += int(row["appo_matches"])

            rows.append(row)
            obs, _, terminated, truncated, _ = env.step(action_map.get(teacher_action, action_map["wait"]))
            memory.update(obs)
            memory.detect_rooms()
            recent_actions.append(teacher_action)
            recent_positions.append(tuple(encoder.encode_full(obs)["position"]))
            recent_state_hashes.append(observation_hash(obs))
            if terminated or truncated:
                break

        env.close()
        episodes.append({"seed": seed, "rows": rows, "steps": len(rows)})

    summary = {
        "episodes": len(episodes),
        "teacher_vs_bc_match_rate": round(
            aggregate["teacher_vs_bc_matches"] / aggregate["teacher_vs_bc_total"], 4
        ) if aggregate["teacher_vs_bc_total"] else None,
        "teacher_vs_appo_match_rate": round(
            aggregate["teacher_vs_appo_matches"] / aggregate["teacher_vs_appo_total"], 4
        ) if aggregate["teacher_vs_appo_total"] else None,
    }

    return {
        "task": task,
        "seeds": seeds,
        "max_steps": max_steps,
        "summary": summary,
        "episodes": episodes,
    }
