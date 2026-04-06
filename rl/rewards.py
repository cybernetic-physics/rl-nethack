from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl.proxy_reward import ProxyRewardScorer
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
    timestep_before: dict | None = None
    feature_vector_before: np.ndarray | None = None


class SkillRewardSource:
    def score(self, inputs: RewardInputs) -> float:
        raise NotImplementedError

    def details(self, inputs: RewardInputs) -> dict[str, float]:
        return {"total": self.score(inputs)}


class HandShapedSkillReward(SkillRewardSource):
    def _result(self, inputs: RewardInputs):
        return compute_task_rewards(
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

    def score(self, inputs: RewardInputs) -> float:
        return self._result(inputs).total

    def details(self, inputs: RewardInputs) -> dict[str, float]:
        result = self._result(inputs)
        return {"total": result.total, **result.components}


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

    def details(self, inputs: RewardInputs) -> dict[str, float]:
        score = self.score(inputs)
        return {"total": score, "learned_reward": score}


class TeacherProxySkillReward(SkillRewardSource):
    def __init__(self, model_path: str):
        self.proxy = ProxyRewardScorer(model_path)

    def details(self, inputs: RewardInputs) -> dict[str, float]:
        if inputs.feature_vector_before is None:
            raise ValueError("TeacherProxySkillReward requires feature_vector_before")
        scored = self.proxy.score(inputs.feature_vector_before, inputs.action_name)
        return {
            "total": float(scored["total"]),
            "proxy_progress": float(scored["progress"]),
            "proxy_survival": float(scored["survival"]),
            "proxy_loop_risk": float(scored["loop_risk"]),
            "proxy_resource_value": float(scored["resource_value"]),
            "proxy_teacher_margin": float(scored["teacher_margin"]),
            "proxy_search_context_prob": float(scored["search_context_prob"]),
            "proxy_teacher_action_prob": float(scored["teacher_action_prob"]),
        }

    def score(self, inputs: RewardInputs) -> float:
        return float(self.details(inputs)["total"])


class MixedSkillReward(SkillRewardSource):
    def __init__(self, proxy_path: str, proxy_weight: float):
        self.hand = HandShapedSkillReward()
        self.proxy = TeacherProxySkillReward(proxy_path)
        self.proxy_weight = float(proxy_weight)

    def details(self, inputs: RewardInputs) -> dict[str, float]:
        hand = self.hand.details(inputs)
        proxy = self.proxy.details(inputs)
        total = float(hand["total"] + self.proxy_weight * proxy["total"])
        return {
            "total": total,
            "hand_total": float(hand["total"]),
            "proxy_total": float(proxy["total"]),
            "proxy_weight": self.proxy_weight,
            **{f"hand_{k}": v for k, v in hand.items() if k != "total"},
            **{k: v for k, v in proxy.items() if k != "total"},
        }

    def score(self, inputs: RewardInputs) -> float:
        return float(self.details(inputs)["total"])


def build_reward_source(
    name: str,
    learned_reward_path: str | None = None,
    proxy_reward_path: str | None = None,
    proxy_reward_weight: float = 1.0,
) -> SkillRewardSource:
    if name == "hand_shaped":
        return HandShapedSkillReward()
    if name == "learned":
        if not learned_reward_path:
            raise ValueError("learned reward source requires a model path")
        return LearnedSkillReward(learned_reward_path)
    if name == "proxy":
        if not proxy_reward_path:
            raise ValueError("proxy reward source requires a model path")
        return TeacherProxySkillReward(proxy_reward_path)
    if name == "mixed_proxy":
        if not proxy_reward_path:
            raise ValueError("mixed_proxy reward source requires a model path")
        return MixedSkillReward(proxy_reward_path, proxy_reward_weight)
    raise ValueError(f"Unknown reward source: {name}")
