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
from rl.io_utils import atomic_write_text
from rl.options import build_skill_registry
from rl.scheduler import SchedulerContext, build_scheduler
from rl.sf_env import NethackSkillEnv
from rl.timestep import build_policy_timestep
from rl.world_model import load_world_model
from rl.world_model_features import augment_feature_vector
from src.data_generator import build_messages, wall_avoidance_policy
from src.evaluator import parse_prediction
from src.memory_tracker import MemoryTracker
from src.state_encoder import StateEncoder
from src.task_harness import select_task_action
from src.task_rewards import observation_hash, snapshot_memory


def _load_appo_eval_bundle(appo_experiment: str | None, appo_train_dir: str, checkpoint_path: str | None = None) -> dict:
    from rl.evaluate import _load_actor_critic
    import torch
    from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
    from sample_factory.model.model_utils import get_rnn_size

    experiment = appo_experiment
    if not experiment and checkpoint_path:
        checkpoint = os.path.abspath(checkpoint_path)
        checkpoint_dir = os.path.dirname(checkpoint)
        if os.path.basename(checkpoint_dir).startswith("checkpoint_p"):
            experiment = os.path.basename(os.path.dirname(checkpoint_dir))
        else:
            experiment = os.path.basename(checkpoint_dir)
    if not experiment:
        raise ValueError("appo_experiment or checkpoint_path is required to load APPO evaluation bundle")

    cfg, env_eval, actor_critic, device = _load_actor_critic(
        experiment,
        appo_train_dir,
        device="cpu",
        checkpoint_path=checkpoint_path,
    )
    return {
        "cfg": cfg,
        "env": env_eval,
        "actor_critic": actor_critic,
        "device": device,
        "prepare": prepare_and_normalize_obs,
        "get_rnn_size": get_rnn_size,
        "torch": torch,
    }


def _select_trace_policy_action(
    *,
    policy: str,
    task: str,
    obs,
    env,
    info,
    state: dict,
    memory,
    encoder: StateEncoder,
    scheduler,
    registry,
    step: int,
    recent_positions: list,
    recent_actions: list[str],
    recent_state_hashes: list[str],
    prev_action: str | None,
    observation_version: str,
    bc_policy=None,
    bc_observation_version: str = "v1",
    appo_eval: dict | None = None,
    appo_rnn_states=None,
    forward_model_server_url: str | None = None,
    forward_model_name: str = "llama-server",
) -> tuple[str, list[dict], object, dict]:
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
    timestep = build_policy_timestep(
        state=state,
        task=active_skill,
        allowed_actions=allowed_actions,
        memory=memory,
        step=step,
        recent_positions=recent_positions,
        recent_actions=recent_actions,
        recent_state_hashes=recent_state_hashes,
        obs_hash=observation_hash(obs if policy != "appo" else env.adapter.obs),
        obs=obs if policy != "appo" else env.adapter.obs,
    )

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
        return action_name, planner_trace, appo_rnn_states, timestep

    if policy == "wall_avoidance":
        return wall_avoidance_policy(state["adjacent"], random.Random(step)), [], appo_rnn_states, timestep

    if policy == "forward_model":
        action_name = _forward_model_action(
            encoder=encoder,
            state=state,
            task=active_skill,
            allowed_actions=allowed_actions,
            server_url=forward_model_server_url,
            model_name=forward_model_name,
        )
        return action_name, [], appo_rnn_states, timestep

    if policy == "bc":
        features = encode_observation(timestep, version=bc_observation_version)
        action_name = bc_policy.act(features, allowed_actions=allowed_actions)
        return action_name, [], appo_rnn_states, timestep

    if policy == "appo":
        obs_features_np = obs if isinstance(obs, np.ndarray) else encode_observation(timestep, version=observation_version)
        obs_features = appo_eval["torch"].from_numpy(obs_features_np).unsqueeze(0).to(appo_eval["device"])
        normalized_obs = appo_eval["prepare"](appo_eval["actor_critic"], {"obs": obs_features})
        policy_outputs = appo_eval["actor_critic"](normalized_obs, appo_rnn_states)
        raw_logits = appo_eval["actor_critic"].action_distribution().raw_logits.clone()
        for idx, name in enumerate(ACTION_SET):
            if name not in set(info.get("allowed_actions", allowed_actions)):
                raw_logits[0, idx] = -1e9
        action_idx = int(np.argmax(raw_logits.detach().cpu().numpy(), axis=1)[0])
        action_name = ACTION_SET[action_idx]
        return action_name, [], policy_outputs["new_rnn_states"], timestep

    raise ValueError(f"Unknown trace policy: {policy}")


def verify_trace_file(path: str) -> dict:
    episodes = {}
    total_rows = 0
    versions = set()
    feature_dims = set()
    invalid_action_rows = 0
    non_monotonic_episode_count = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total_rows += 1
            episodes.setdefault(row["episode_id"], []).append(row["step"])
            versions.add(row.get("observation_version", "unknown"))
            feature_dims.add(len(row.get("feature_vector", [])))
            if row.get("action") not in row.get("allowed_actions", []):
                invalid_action_rows += 1

    episode_lengths = [len(steps) for steps in episodes.values()]
    multi_turn_episodes = sum(1 for length in episode_lengths if length > 1)
    for steps in episodes.values():
        if steps != list(range(len(steps))):
            non_monotonic_episode_count += 1
    return {
        "episodes": len(episodes),
        "rows": total_rows,
        "max_steps_in_episode": max(episode_lengths) if episode_lengths else 0,
        "avg_steps_in_episode": round(sum(episode_lengths) / len(episode_lengths), 2) if episode_lengths else 0.0,
        "multi_turn_episodes": multi_turn_episodes,
        "all_multi_turn": bool(episodes) and multi_turn_episodes == len(episodes),
        "observation_versions": sorted(versions),
        "feature_dims": sorted(feature_dims),
        "invalid_action_rows": invalid_action_rows,
        "non_monotonic_episode_count": non_monotonic_episode_count,
    }


def shard_trace_file(
    input_path: str,
    output_path: str,
    *,
    max_episodes: int | None = None,
    max_rows: int | None = None,
    seeds: list[int] | None = None,
    teacher_actions: list[str] | None = None,
) -> dict:
    with open(input_path, "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    rows = sorted(rows, key=lambda row: (row["episode_id"], row["step"]))
    seed_filter = set(seeds or [])
    action_filter = set(teacher_actions or [])
    written_rows: list[dict] = []
    seen_episodes: list[str] = []
    current_episode_rows: list[dict] = []
    current_episode_id: str | None = None

    def flush_episode() -> bool:
        if not current_episode_rows:
            return False
        episode_seed = current_episode_rows[0].get("seed")
        if seed_filter and episode_seed not in seed_filter:
            return False
        if action_filter and not any(row.get("action") in action_filter for row in current_episode_rows):
            return False
        if max_episodes is not None and len(seen_episodes) >= max_episodes:
            return True
        if max_rows is not None and written_rows and len(written_rows) + len(current_episode_rows) > max_rows:
            return True
        seen_episodes.append(current_episode_rows[0]["episode_id"])
        written_rows.extend(current_episode_rows)
        return False

    stop = False
    for row in rows:
        episode_id = row["episode_id"]
        if current_episode_id is None:
            current_episode_id = episode_id
        if episode_id != current_episode_id:
            stop = flush_episode()
            current_episode_rows = []
            current_episode_id = episode_id
            if stop:
                break
        current_episode_rows.append(row)
    if not stop:
        flush_episode()

    payload = "".join(json.dumps(row) + "\n" for row in written_rows)
    atomic_write_text(output_path, payload)

    summary = verify_trace_file(output_path)
    summary["input_path"] = input_path
    summary["output_path"] = output_path
    summary["max_episodes"] = max_episodes
    summary["max_rows"] = max_rows
    summary["selected_seeds"] = sorted(seed_filter)
    summary["selected_teacher_actions"] = sorted(action_filter)
    return summary


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
    observation_version: str = "v1",
) -> dict:
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    encoder = StateEncoder()
    action_map = _build_action_map()
    registry = build_skill_registry()
    scheduler = build_scheduler("rule_based")

    appo_eval = None
    if policy == "appo":
        appo_eval = _load_appo_eval_bundle(appo_experiment, appo_train_dir)

    bc_policy = None
    bc_observation_version = "v1"
    if policy == "bc":
        bc_policy = load_bc_model(bc_model_path)
        import torch

        payload = torch.load(bc_model_path, map_location="cpu")
        bc_observation_version = payload.get("metadata", {}).get("observation_version", "v1")

    total_rows = 0
    episode_lengths = []
    records: list[dict] = []

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
                recent_actions: list[str] = []
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
                recent_actions = []
                prev_action = None
                rnn_states = None

            action_counts = Counter()
            steps = 0
            for step in range(max_steps):
                state = encoder.encode_full(obs if policy != "appo" else env.adapter.obs)
                action_name, planner_trace, rnn_states, timestep = _select_trace_policy_action(
                    policy=policy,
                    task=task,
                    obs=obs,
                    env=env,
                    info=info if policy == "appo" else {},
                    state=state,
                    memory=memory,
                    encoder=encoder,
                    scheduler=scheduler,
                    registry=registry,
                    step=step,
                    recent_positions=recent_positions,
                    recent_actions=recent_actions,
                    recent_state_hashes=recent_state_hashes,
                    prev_action=prev_action,
                    observation_version=observation_version,
                    bc_policy=bc_policy,
                    bc_observation_version=bc_observation_version,
                    appo_eval=appo_eval,
                    appo_rnn_states=rnn_states,
                    forward_model_server_url=forward_model_server_url,
                    forward_model_name=forward_model_name,
                )

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
                active_skill = timestep["active_skill"]
                allowed_actions = timestep["allowed_actions"]
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
                    "feature_vector": encode_observation(timestep, version=observation_version).tolist(),
                    "observation_version": observation_version,
                    "memory_total_explored_before": timestep["memory_total_explored"],
                    "rooms_discovered_before": timestep["rooms_discovered"],
                    "steps_in_skill": timestep["steps_in_skill"],
                    "recent_action_count": len(timestep["recent_actions"]),
                    "recent_position_count": len(timestep["recent_positions"]),
                    "repeated_state_count": timestep["repeated_state_count"],
                    "revisited_recent_tile_count": timestep["revisited_recent_tile_count"],
                    "repeated_action_count": timestep["repeated_action_count"],
                    "teacher_action_index": ACTION_SET.index(action_name) if action_name in ACTION_SET else ACTION_SET.index("wait"),
                    "is_weak_action": action_name in {"south", "west", "search"},
                    "is_loop_risk": timestep["repeated_state_count"] > 0 or timestep["repeated_action_count"] > 0,
                    "is_failure_slice": bool((terminated or truncated) and float(reward) < 0.0),
                    "is_disagreement_candidate": False,
                    "planner_trace": planner_trace,
                }
                records.append(record)
                total_rows += 1
                action_counts[action_name] += 1
                steps += 1
                prev_action = action_name
                recent_actions.append(action_name)
                recent_state_hashes.append(next_hash)
                recent_positions.append(encoder.encode_full(raw_obs_after)["position"])
                obs = obs_after
                if terminated or truncated:
                    break

            episode_lengths.append(steps)
            env.close()

    atomic_write_text(output_path, "".join(json.dumps(record) + "\n" for record in records))
    summary = verify_trace_file(output_path)
    summary["num_episodes_requested"] = num_episodes
    summary["policy"] = policy
    summary["observation_version"] = observation_version
    return summary


def generate_dagger_traces(
    output_path: str,
    num_episodes: int,
    max_steps: int,
    *,
    student_policy: str,
    task: str = "explore",
    seed_start: int = 42,
    appo_experiment: Optional[str] = None,
    appo_train_dir: str = "train_dir/rl",
    appo_checkpoint_path: Optional[str] = None,
    bc_model_path: Optional[str] = None,
    observation_version: str = "v1",
) -> dict:
    if student_policy not in {"bc", "appo", "wall_avoidance"}:
        raise ValueError("student_policy must be one of: bc, appo, wall_avoidance")
    if student_policy == "bc" and not bc_model_path:
        raise ValueError("bc_model_path is required for student_policy=bc")
    if student_policy == "appo" and not (appo_experiment or appo_checkpoint_path):
        raise ValueError("appo_experiment or appo_checkpoint_path is required for student_policy=appo")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    encoder = StateEncoder()
    action_map = _build_action_map()
    registry = build_skill_registry()
    scheduler = build_scheduler("rule_based")

    appo_eval = None
    world_model_inference = None
    feature_observation_version = observation_version
    if student_policy == "appo":
        appo_eval = _load_appo_eval_bundle(appo_experiment, appo_train_dir, checkpoint_path=appo_checkpoint_path)
        wm_path = getattr(appo_eval["cfg"], "world_model_path", None)
        wm_mode = getattr(appo_eval["cfg"], "world_model_feature_mode", None)
        if wm_path and wm_mode:
            world_model_inference = load_world_model(wm_path)
            feature_observation_version = f"{observation_version}+wm_{wm_mode}"

    bc_policy = None
    bc_observation_version = observation_version
    if student_policy == "bc":
        bc_policy = load_bc_model(bc_model_path)
        import torch

        payload = torch.load(bc_model_path, map_location="cpu")
        bc_observation_version = payload.get("metadata", {}).get("observation_version", observation_version)

    total_rows = 0
    episode_lengths = []
    relabel_match_count = 0
    records: list[dict] = []

    for ep_idx in range(num_episodes):
            seed = seed_start + ep_idx
            if student_policy == "appo":
                env_config = RLConfig()
                env_config.env.observation_version = str(getattr(appo_eval["cfg"], "observation_version", observation_version))
                env_config.env.max_episode_steps = int(getattr(appo_eval["cfg"], "env_max_episode_steps", env_config.env.max_episode_steps))
                env_config.env.world_model_path = getattr(appo_eval["cfg"], "world_model_path", None)
                env_config.env.world_model_feature_mode = getattr(appo_eval["cfg"], "world_model_feature_mode", None)
                env = NethackSkillEnv(env_config)
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
            else:
                env = nle.env.NLE()
                obs, _ = env.reset(seed=seed)
                memory = MemoryTracker()
                memory.update(obs)
                memory.detect_rooms()
                state = encoder.encode_full(obs)
                recent_state_hashes = [observation_hash(obs)]
                recent_positions = [state["position"]]
                info = {}
                rnn_states = None

            recent_actions: list[str] = []
            prev_action = None
            steps = 0

            for step in range(max_steps):
                raw_obs = env.adapter.obs if student_policy == "appo" else obs
                state = encoder.encode_full(raw_obs)
                student_action, _, rnn_states, timestep = _select_trace_policy_action(
                    policy=student_policy,
                    task=task,
                    obs=obs,
                    env=env,
                    info=info,
                    state=state,
                    memory=memory,
                    encoder=encoder,
                    scheduler=scheduler,
                    registry=registry,
                    step=step,
                    recent_positions=recent_positions,
                    recent_actions=recent_actions,
                    recent_state_hashes=recent_state_hashes,
                    prev_action=prev_action,
                    observation_version=observation_version,
                    bc_policy=bc_policy,
                    bc_observation_version=bc_observation_version,
                    appo_eval=appo_eval,
                    appo_rnn_states=rnn_states,
                )
                teacher_action, teacher_trace = select_task_action(
                    env=env.adapter.env if student_policy == "appo" else env,
                    obs_before=raw_obs,
                    state_before=state,
                    memory_before=memory,
                    task=task,
                    prev_action=prev_action,
                    recent_state_hashes=list(recent_state_hashes),
                    recent_positions=list(recent_positions),
                    encoder=encoder,
                )
                relabel_match_count += int(student_action == teacher_action)

                if student_action not in action_map:
                    student_action = "wait"
                prompt_text = encoder.format_prompt(state, teacher_action)
                obs_before = raw_obs
                if student_policy == "appo":
                    obs_after, reward, terminated, truncated, info = env.step(ACTION_SET.index(student_action))
                    raw_obs_after = env.adapter.obs
                    memory = env.adapter.memory
                else:
                    obs_after, reward, terminated, truncated, info = env.step(action_map[student_action])
                    raw_obs_after = obs_after
                    memory.update(obs_after)
                    memory.detect_rooms()

                delta = encoder.encode_delta(obs_before, raw_obs_after, teacher_action)
                next_hash = observation_hash(raw_obs_after)
                active_skill = timestep["active_skill"]
                allowed_actions = timestep["allowed_actions"]
                record = {
                    "episode_id": f"dagger:{student_policy}:{seed}",
                    "seed": seed,
                    "step": step,
                    "policy": "dagger",
                    "student_policy": student_policy,
                    "task": active_skill,
                    "action": teacher_action,
                    "behavior_action": student_action,
                    "teacher_action": teacher_action,
                    "allowed_actions": allowed_actions,
                    "prompt": prompt_text,
                    "delta": delta,
                    "reward": float(reward),
                    "done": bool(terminated or truncated),
                    "obs_hash": observation_hash(obs_before),
                    "next_obs_hash": next_hash,
                    "feature_vector": (
                        augment_feature_vector(
                            encode_observation(timestep, version=observation_version),
                            world_model_inference,
                            mode=getattr(appo_eval["cfg"], "world_model_feature_mode", None),
                        ).tolist()
                        if student_policy == "appo" and world_model_inference is not None
                        else encode_observation(timestep, version=observation_version).tolist()
                    ),
                    "observation_version": feature_observation_version,
                    "memory_total_explored_before": timestep["memory_total_explored"],
                    "rooms_discovered_before": timestep["rooms_discovered"],
                    "steps_in_skill": timestep["steps_in_skill"],
                    "recent_action_count": len(timestep["recent_actions"]),
                    "recent_position_count": len(timestep["recent_positions"]),
                    "repeated_state_count": timestep["repeated_state_count"],
                    "revisited_recent_tile_count": timestep["revisited_recent_tile_count"],
                    "repeated_action_count": timestep["repeated_action_count"],
                    "teacher_action_index": ACTION_SET.index(teacher_action) if teacher_action in ACTION_SET else ACTION_SET.index("wait"),
                    "is_weak_action": teacher_action in {"south", "west", "search"},
                    "is_loop_risk": timestep["repeated_state_count"] > 0 or timestep["repeated_action_count"] > 0,
                    "is_failure_slice": bool((terminated or truncated) and float(reward) < 0.0),
                    "is_disagreement_candidate": student_action != teacher_action,
                    "planner_trace": teacher_trace,
                }
                records.append(record)
                total_rows += 1
                steps += 1
                prev_action = student_action
                recent_actions.append(student_action)
                recent_state_hashes.append(next_hash)
                recent_positions.append(encoder.encode_full(raw_obs_after)["position"])
                obs = obs_after
                if terminated or truncated:
                    break

            episode_lengths.append(steps)
            env.close()

    atomic_write_text(output_path, "".join(json.dumps(record) + "\n" for record in records))
    summary = verify_trace_file(output_path)
    summary["num_episodes_requested"] = num_episodes
    summary["policy"] = "dagger"
    summary["student_policy"] = student_policy
    summary["observation_version"] = observation_version
    summary["teacher_match_rate"] = round(relabel_match_count / max(1, total_rows), 4)
    return summary
