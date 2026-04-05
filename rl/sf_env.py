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
    active_skill: str
    action_name: str


class NethackSkillEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: RLConfig):
        super().__init__()
        self.config = config
        self.adapter = SkillEnvAdapter(config)
        self.action_map = _build_action_map()
        self.reward_source = build_reward_source(config.reward.source)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(observation_dim(),),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(ACTION_SET))
        self._episode_steps = 0

    def reset(self, *, seed=None, options=None):
        del options
        self._episode_steps = 0
        timestep = self.adapter.reset(seed=seed)
        obs = encode_observation(timestep)
        info = {
            "active_skill": timestep["active_skill"],
            "allowed_actions": timestep["allowed_actions"],
        }
        return obs, info

    def step(self, action: int):
        action_name = ACTION_SET[action]
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
            )
        )
        total_reward = (
            self.config.reward.extrinsic_weight * env_reward
            + self.config.reward.intrinsic_weight * skill_reward
        )
        self._episode_steps += 1
        if self._episode_steps >= self.config.env.max_episode_steps:
            truncated = True
        obs = encode_observation(timestep)
        info = dict(info)
        info.update(
            {
                "active_skill": timestep["active_skill"],
                "allowed_actions": timestep["allowed_actions"],
                "num_frames": 1,
                "debug": EnvDebugInfo(
                    skill_reward=float(skill_reward),
                    env_reward=float(env_reward),
                    active_skill=timestep["active_skill"],
                    action_name=action_name,
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
    rl_config.reward.source = getattr(cfg, "reward_source", rl_config.reward.source)
    rl_config.reward.extrinsic_weight = getattr(cfg, "extrinsic_reward_weight", rl_config.reward.extrinsic_weight)
    rl_config.reward.intrinsic_weight = getattr(cfg, "intrinsic_reward_weight", rl_config.reward.intrinsic_weight)
    return NethackSkillEnv(rl_config)
