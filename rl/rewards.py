from __future__ import annotations

from dataclasses import dataclass

from src.task_rewards import compute_task_rewards


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


def build_reward_source(name: str) -> SkillRewardSource:
    if name == "hand_shaped":
        return HandShapedSkillReward()
    raise ValueError(f"Unknown reward source: {name}")

