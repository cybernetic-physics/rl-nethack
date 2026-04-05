from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import torch
from gymnasium import spaces
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.cfg.arguments import load_from_checkpoint, parse_full_cfg, parse_sf_args
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size

from rl.config import RLConfig
from rl.feature_encoder import ACTION_SET
from rl.sf_env import NethackSkillEnv
from src.task_harness import evaluate_task_policy

LIVE_ENV_WARNING = (
    "Live seed-based evaluation is not a stable regression signal in this repo because "
    "NLE reset(seed=...) is not reproducible across runs. Use trace-based evaluation for trusted comparisons."
)


def _checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    name = path.name
    checkpoint_match = re.match(r"^checkpoint_(\d+)_(\d+)\.pth$", name)
    if checkpoint_match:
        return (int(checkpoint_match.group(2)), 0, name)
    best_match = re.match(r"^best_(\d+)_(\d+)_reward_.*\.pth$", name)
    if best_match:
        return (int(best_match.group(2)), 1, name)
    return (int(path.stat().st_mtime), -1, name)


def _find_checkpoint_path(cfg) -> Path:
    checkpoint_dir = Path(Learner.checkpoint_dir(cfg, 0))
    candidates = sorted(
        list(checkpoint_dir.glob("checkpoint_*.pth")) + list(checkpoint_dir.glob("best_*.pth")),
        key=_checkpoint_sort_key,
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return candidates[-1]


def list_checkpoint_paths(experiment: str, train_dir: str) -> list[Path]:
    checkpoint_dir = Path(train_dir) / experiment / "checkpoint_p0"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    candidates = sorted(
        list(checkpoint_dir.glob("checkpoint_*.pth")) + list(checkpoint_dir.glob("best_*.pth")),
        key=_checkpoint_sort_key,
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return candidates


def _load_checkpoint_payload(checkpoint_path: Path, torch_device: torch.device) -> dict:
    try:
        return torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=torch_device)
    except Exception:
        payload = Learner.load_checkpoint([str(checkpoint_path)], torch_device)
        if payload is None or "model" not in payload:
            raise RuntimeError(f"Could not load checkpoint payload from {checkpoint_path}")
        return payload


def _load_actor_critic(experiment: str, train_dir: str, device: str, checkpoint_path: str | None = None):
    argv = ["--algo=APPO", "--env=rl_nethack_skill", f"--experiment={experiment}", f"--train_dir={train_dir}", f"--device={device}"]
    parser, _ = parse_sf_args(argv)
    parser.add_argument("--env_max_episode_steps", type=int, default=200)
    parser.add_argument("--observation_version", type=str, default="v1")
    parser.add_argument("--reward_source", type=str, default="hand_shaped")
    parser.add_argument("--intrinsic_reward_weight", type=float, default=1.0)
    parser.add_argument("--extrinsic_reward_weight", type=float, default=0.0)
    parser.add_argument("--episodic_explore_bonus_enabled", type=str, default="False")
    parser.add_argument("--episodic_explore_bonus_scale", type=float, default=0.0)
    parser.add_argument("--episodic_explore_bonus_mode", type=str, default="state_hash")
    parser.add_argument("--skill_scheduler", type=str, default="rule_based")
    parser.add_argument("--scheduler_model_path", type=str, default=None)
    parser.add_argument("--enabled_skills", type=str, default="explore")
    parser.add_argument("--active_skill_bootstrap", type=str, default="explore")
    parser.add_argument("--learned_reward_path", type=str, default=None)
    parser.add_argument("--teacher_loss_coef", type=float, default=0.0)
    parser.add_argument("--teacher_loss_type", type=str, default="ce")
    parser.add_argument("--teacher_bc_path", type=str, default=None)
    parser.add_argument("--trace_eval_input", type=str, default=None)
    parser.add_argument("--trace_eval_interval_env_steps", type=int, default=0)
    parser.add_argument("--trace_eval_top_k", type=int, default=5)
    parser.add_argument("--enforce_action_mask", type=str, default="True")
    parser.add_argument("--invalid_action_penalty", type=float, default=2.0)
    parser.add_argument("--invalid_action_fallback", type=str, default="wait")
    cfg = parse_full_cfg(parser, argv)
    cfg = load_from_checkpoint(cfg)
    torch_device = torch.device(device)

    env_cfg = RLConfig()
    env_cfg.env.max_episode_steps = cfg.env_max_episode_steps
    env_cfg.env.observation_version = getattr(cfg, "observation_version", "v1")
    env_cfg.options.enabled_skills = [s.strip() for s in str(cfg.enabled_skills).split(",") if s.strip()]
    env_cfg.options.scheduler = cfg.skill_scheduler
    env_cfg.options.scheduler_model_path = getattr(cfg, "scheduler_model_path", None)
    env_cfg.env.active_skill_bootstrap = cfg.active_skill_bootstrap
    env_cfg.reward.source = cfg.reward_source
    env_cfg.reward.learned_reward_path = getattr(cfg, "learned_reward_path", None)
    env_cfg.reward.extrinsic_weight = cfg.extrinsic_reward_weight
    env_cfg.reward.intrinsic_weight = cfg.intrinsic_reward_weight
    env_cfg.reward.invalid_action_penalty = cfg.invalid_action_penalty
    env_cfg.reward.episodic_explore_bonus_enabled = str(getattr(cfg, "episodic_explore_bonus_enabled", "False")).lower() == "true"
    env_cfg.reward.episodic_explore_bonus_scale = getattr(cfg, "episodic_explore_bonus_scale", 0.0)
    env_cfg.reward.episodic_explore_bonus_mode = getattr(cfg, "episodic_explore_bonus_mode", "state_hash")
    env_cfg.env.enforce_action_mask = str(cfg.enforce_action_mask).lower() == "true"
    env_cfg.env.invalid_action_fallback = cfg.invalid_action_fallback
    env = NethackSkillEnv(env_cfg)

    actor_critic = create_actor_critic(cfg, spaces.Dict({"obs": env.observation_space}), env.action_space)
    actor_critic.eval()
    actor_critic.model_to_device(torch_device)
    resolved_checkpoint_path = Path(checkpoint_path) if checkpoint_path else _find_checkpoint_path(cfg)
    checkpoint_dict = _load_checkpoint_payload(resolved_checkpoint_path, torch_device)
    actor_critic.load_state_dict(checkpoint_dict["model"])
    return cfg, env, actor_critic, torch_device


def evaluate_appo_policy(
    experiment: str,
    train_dir: str,
    seeds: list[int],
    max_steps: int,
    deterministic: bool = True,
    mask_actions: bool = True,
    compare_baseline: bool = False,
    checkpoint_path: str | None = None,
):
    cfg, env, actor_critic, device = _load_actor_critic(
        experiment,
        train_dir,
        device="cpu",
        checkpoint_path=checkpoint_path,
    )
    env.config.env.max_episode_steps = max_steps
    rows = []

    for seed in seeds:
        obs, info = env.reset(seed=seed)
        rnn_states = torch.zeros([1, get_rnn_size(cfg)], dtype=torch.float32, device=device)
        total_reward = 0.0
        action_counts = Counter()
        invalid_requested = 0
        repeated_actions = 0
        prev_action = None

        for step in range(max_steps):
            obs_dict = {"obs": torch.from_numpy(obs).unsqueeze(0).to(device)}
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs_dict)
            policy_outputs = actor_critic(normalized_obs, rnn_states)
            raw_logits = actor_critic.action_distribution().raw_logits.clone()
            if mask_actions:
                allowed = set(info.get("allowed_actions", []))
                for idx, name in enumerate(ACTION_SET):
                    if name not in allowed:
                        raw_logits[0, idx] = -1e9
            action = int(torch.argmax(raw_logits, dim=1).item()) if deterministic else int(policy_outputs["actions"].squeeze().item())
            rnn_states = policy_outputs["new_rnn_states"]
            action_name = ACTION_SET[action]
            obs, rew, terminated, truncated, info = env.step(action)
            debug = info.get("debug", {})
            invalid_requested += int(debug.get("invalid_action_requested", False))
            repeated_actions += int(prev_action == debug.get("action_name"))
            prev_action = debug.get("action_name")
            action_counts[debug.get("action_name", action_name)] += 1
            total_reward += float(rew)
            if terminated or truncated:
                break

        rows.append(
            {
                "seed": seed,
                "total_task_reward": round(total_reward, 4),
                "unique_tiles": int(env.adapter.memory.total_explored),
                "rooms_discovered": len(env.adapter.memory.rooms),
                "steps": step + 1,
                "repeated_action_steps": repeated_actions,
                "invalid_action_requests": invalid_requested,
                "action_counts": dict(action_counts),
            }
        )

    env.close()
    total_steps = sum(row["steps"] for row in rows) or 1
    summary = {
        "episodes": len(rows),
        "evaluation_mode": "live_env_seeded",
        "warning": LIVE_ENV_WARNING,
        "avg_task_reward": round(sum(row["total_task_reward"] for row in rows) / len(rows), 4),
        "avg_unique_tiles": round(sum(row["unique_tiles"] for row in rows) / len(rows), 2),
        "avg_rooms_discovered": round(sum(row["rooms_discovered"] for row in rows) / len(rows), 2),
        "repeated_action_rate": round(sum(row["repeated_action_steps"] for row in rows) / total_steps, 4),
        "invalid_action_rate": round(sum(row["invalid_action_requests"] for row in rows) / total_steps, 4),
        "action_counts": dict(sum((Counter(row["action_counts"]) for row in rows), Counter())),
    }
    result = {
        "experiment": experiment,
        "checkpoint_path": checkpoint_path,
        "summary": summary,
        "episodes": rows,
    }
    if compare_baseline and len(env.config.options.enabled_skills) == 1:
        task = env.config.options.enabled_skills[0]
        result["baseline"] = evaluate_task_policy(task=task, seeds=seeds, max_steps=max_steps, policy="task_greedy")
    return result
