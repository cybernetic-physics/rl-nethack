from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl.reward_model import load_reward_model
from src.task_rewards import compute_task_rewards, encode_task_reward_features


@dataclass
class RewardInputs:
    task: str
    obs_before: dict
    obs_after: dict
    state_before: dict
    state_after: dict
    memory_before: object
    memory_after: object
    action_name: str
    env_reward: float
    terminated: bool
    truncated: bool
    repeated_state: bool = False
    revisited_recent_tile: bool = False
    repeated_action: bool = False
    invalid_action_requested: bool = False


class SkillRewardSource:
    def score(self, inputs: RewardInputs) -> float:
        raise NotImplementedError


class HandShapedSkillReward(SkillRewardSource):
    def score(self, inputs: RewardInputs) -> float:
        result = compute_task_rewards(
            task=inputs.task,
            obs_before=inputs.obs_before,
            obs_after=inputs.obs_after,
            state_before=inputs.state_before,
            state_after=inputs.state_after,
            memory_before=inputs.memory_before,
            memory_after=inputs.memory_after,
            action_name=inputs.action_name,
            reward=inputs.env_reward,
            terminated=inputs.terminated,
            truncated=inputs.truncated,
            repeated_state=inputs.repeated_state,
            revisited_recent_tile=inputs.revisited_recent_tile,
            repeated_action=inputs.repeated_action,
        )
        return result.total


class LearnedSkillReward(SkillRewardSource):
    def __init__(self, model_path: str):
        self.model = load_reward_model(model_path)

    def score(self, inputs: RewardInputs) -> float:
        features = encode_task_reward_features(
            task=inputs.task,
            obs_before=inputs.obs_before,
            obs_after=inputs.obs_after,
            state_before=inputs.state_before,
            state_after=inputs.state_after,
            memory_before=inputs.memory_before,
            memory_after=inputs.memory_after,
            action_name=inputs.action_name,
            reward=inputs.env_reward,
            terminated=inputs.terminated,
            truncated=inputs.truncated,
            repeated_state=inputs.repeated_state,
            revisited_recent_tile=inputs.revisited_recent_tile,
            repeated_action=inputs.repeated_action,
        )
        return self.model.score(np.asarray(features, dtype=np.float32))


def build_reward_source(name: str, learned_reward_path: str | None = None) -> SkillRewardSource:
    if name == "hand_shaped":
        return HandShapedSkillReward()
    if name == "learned":
        if not learned_reward_path:
            raise ValueError("learned reward source requires a model path")
        return LearnedSkillReward(learned_reward_path)
    raise ValueError(f"Unknown reward source: {name}")
