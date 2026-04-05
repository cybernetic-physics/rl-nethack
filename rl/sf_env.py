from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from nle_agent.agent_http import _build_action_map
from rl.config import RLConfig
from rl.env_adapter import SkillEnvAdapter
from rl.feature_encoder import ACTION_SET, encode_observation, observation_dim
from rl.rewards import RewardInputs, build_reward_source


@dataclass
class EnvDebugInfo:
    skill_reward: float
    env_reward: float
    episodic_explore_bonus: float
    active_skill: str
    action_name: str
    requested_action_name: str
    invalid_action_requested: bool


class NethackSkillEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: RLConfig):
        super().__init__()
        self.config = config
        self.adapter = SkillEnvAdapter(config)
        self.action_map = _build_action_map()
        self.reward_source = build_reward_source(config.reward.source, config.reward.learned_reward_path)
        self.observation_version = config.env.observation_version
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(observation_dim(self.observation_version),),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(ACTION_SET))
        self._episode_steps = 0

    def reset(self, *, seed=None, options=None):
        del options
        self._episode_steps = 0
        timestep = self.adapter.reset(seed=seed)
        obs = encode_observation(timestep, version=self.observation_version)
        info = {
            "active_skill": timestep["active_skill"],
            "allowed_actions": timestep["allowed_actions"],
        }
        return obs, info

    def step(self, action: int):
        requested_action_name = ACTION_SET[action]
        state_before = self.adapter._encode_state(self.adapter.obs)
        allowed_actions = self.adapter.allowed_actions(state_before)
        invalid_action_requested = requested_action_name not in allowed_actions
        action_name = requested_action_name
        if self.config.env.enforce_action_mask and invalid_action_requested:
            fallback = self.config.env.invalid_action_fallback
            if fallback in allowed_actions:
                action_name = fallback
            elif allowed_actions:
                action_name = allowed_actions[0]
        timestep, env_reward, terminated, truncated, info = self.adapter.step(
            action_idx=self.action_map.get(action_name, self.action_map["wait"]),
            action_name=action_name,
        )
        transition = timestep["transition"]
        skill_reward = self.reward_source.score(
            RewardInputs(
                task=transition["active_skill_before"],
                obs_before=transition["obs_before"],
                obs_after=transition["obs_after"],
                state_before=transition["state_before"],
                state_after=transition["state_after"],
                memory_before=transition["memory_before"],
                memory_after=transition["memory_after"],
                action_name=transition["action_name"],
                env_reward=env_reward,
                terminated=terminated,
                truncated=truncated,
                repeated_state=transition["repeated_state"],
                revisited_recent_tile=transition["revisited_recent_tile"],
                repeated_action=transition["repeated_action"],
                invalid_action_requested=invalid_action_requested,
            )
        )
        episodic_explore_bonus = (
            self.config.reward.episodic_explore_bonus_scale * transition["episodic_explore_bonus_raw"]
        )
        if invalid_action_requested:
            skill_reward -= self.config.reward.invalid_action_penalty
        total_reward = (
            self.config.reward.extrinsic_weight * env_reward
            + self.config.reward.intrinsic_weight * (skill_reward + episodic_explore_bonus)
        )
        self._episode_steps += 1
        if self._episode_steps >= self.config.env.max_episode_steps:
            truncated = True
        obs = encode_observation(timestep, version=self.observation_version)
        info = dict(info)
        info.update(
            {
                "active_skill": timestep["active_skill"],
                "allowed_actions": timestep["allowed_actions"],
                "num_frames": 1,
                "debug": EnvDebugInfo(
                    skill_reward=float(skill_reward),
                    env_reward=float(env_reward),
                    episodic_explore_bonus=float(episodic_explore_bonus),
                    active_skill=timestep["active_skill"],
                    action_name=action_name,
                    requested_action_name=requested_action_name,
                    invalid_action_requested=bool(invalid_action_requested),
                ).__dict__,
            }
        )
        return obs, float(total_reward), bool(terminated), bool(truncated), info

    def close(self):
        self.adapter.close()


def make_nethack_skill_env(full_env_name, cfg, env_config, render_mode=None, **kwargs):
    del full_env_name, env_config, render_mode, kwargs
    rl_config = RLConfig()
    rl_config.env.seed = getattr(cfg, "seed", rl_config.env.seed)
    rl_config.env.max_episode_steps = getattr(cfg, "env_max_episode_steps", rl_config.env.max_episode_steps)
    rl_config.env.active_skill_bootstrap = getattr(
        cfg, "active_skill_bootstrap", rl_config.env.active_skill_bootstrap
    )
    rl_config.env.observation_version = getattr(cfg, "observation_version", rl_config.env.observation_version)
    rl_config.env.enforce_action_mask = str(getattr(cfg, "enforce_action_mask", rl_config.env.enforce_action_mask)).lower() == "true"
    rl_config.env.invalid_action_fallback = getattr(cfg, "invalid_action_fallback", rl_config.env.invalid_action_fallback)
    rl_config.reward.source = getattr(cfg, "reward_source", rl_config.reward.source)
    rl_config.reward.learned_reward_path = getattr(cfg, "learned_reward_path", rl_config.reward.learned_reward_path)
    rl_config.reward.extrinsic_weight = getattr(cfg, "extrinsic_reward_weight", rl_config.reward.extrinsic_weight)
    rl_config.reward.intrinsic_weight = getattr(cfg, "intrinsic_reward_weight", rl_config.reward.intrinsic_weight)
    rl_config.reward.invalid_action_penalty = getattr(cfg, "invalid_action_penalty", rl_config.reward.invalid_action_penalty)
    rl_config.reward.episodic_explore_bonus_enabled = (
        str(getattr(cfg, "episodic_explore_bonus_enabled", rl_config.reward.episodic_explore_bonus_enabled)).lower()
        == "true"
    )
    rl_config.reward.episodic_explore_bonus_scale = getattr(
        cfg, "episodic_explore_bonus_scale", rl_config.reward.episodic_explore_bonus_scale
    )
    rl_config.reward.episodic_explore_bonus_mode = getattr(
        cfg, "episodic_explore_bonus_mode", rl_config.reward.episodic_explore_bonus_mode
    )
    rl_config.options.scheduler = getattr(cfg, "skill_scheduler", rl_config.options.scheduler)
    rl_config.options.scheduler_model_path = getattr(cfg, "scheduler_model_path", rl_config.options.scheduler_model_path)
    enabled_skills = getattr(cfg, "enabled_skills", None)
    if enabled_skills:
        if isinstance(enabled_skills, str):
            rl_config.options.enabled_skills = [s.strip() for s in enabled_skills.split(",") if s.strip()]
        else:
            rl_config.options.enabled_skills = list(enabled_skills)
    return NethackSkillEnv(rl_config)
