from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from rl.feature_encoder import ACTION_SET, action_name_to_index
from rl.io_utils import atomic_torch_save


DEFAULT_PROXY_REWARD_WEIGHTS = {
    "progress": 1.0,
    "survival": 1.0,
    "resource_value": 0.5,
    "loop_risk": 1.0,
    "teacher_margin": 0.5,
    "search_context": 0.5,
    "teacher_policy": 5.0,
}


class ProxyRewardModel(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 256, action_embed_dim: int = 32):
        super().__init__()
        self.state_trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.action_embedding = nn.Embedding(len(ACTION_SET), action_embed_dim)
        self.joint = nn.Sequential(
            nn.Linear(hidden_size + action_embed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.progress_head = nn.Linear(hidden_size, 1)
        self.survival_head = nn.Linear(hidden_size, 1)
        self.loop_risk_head = nn.Linear(hidden_size, 1)
        self.resource_value_head = nn.Linear(hidden_size, 1)
        self.teacher_margin_head = nn.Linear(hidden_size, 1)
        self.search_context_head = nn.Linear(hidden_size, 1)
        self.action_logits_head = nn.Linear(hidden_size, len(ACTION_SET))

    def forward(self, features: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        state_hidden = self.state_trunk(features)
        action_hidden = self.action_embedding(actions)
        joint_hidden = self.joint(torch.cat([state_hidden, action_hidden], dim=-1))
        return {
            "state_hidden": state_hidden,
            "progress": self.progress_head(joint_hidden).squeeze(-1),
            "survival": self.survival_head(joint_hidden).squeeze(-1),
            "loop_risk": self.loop_risk_head(joint_hidden).squeeze(-1),
            "resource_value": self.resource_value_head(joint_hidden).squeeze(-1),
            "teacher_margin": self.teacher_margin_head(joint_hidden).squeeze(-1),
            "search_context_logit": self.search_context_head(joint_hidden).squeeze(-1),
            "action_logits": self.action_logits_head(state_hidden),
        }


@dataclass
class ProxyRewardInference:
    model: ProxyRewardModel
    device: torch.device
    metadata: dict

    def _target_stats(self, key: str) -> tuple[float, float]:
        stats = self.metadata.get("target_stats", {}).get(key, {})
        return float(stats.get("mean", 0.0)), float(stats.get("std", 1.0))

    def _denorm(self, key: str, value: float) -> float:
        mean, std = self._target_stats(key)
        return float(value * std + mean)

    def _weights(self) -> dict[str, float]:
        return {
            **DEFAULT_PROXY_REWARD_WEIGHTS,
            **self.metadata.get("reward_weights", {}),
        }

    def _score_stats(self) -> tuple[float, float]:
        stats = self.metadata.get("score_stats", {})
        mean = float(stats.get("mean", 0.0))
        std = max(1e-6, float(stats.get("std", 1.0)))
        return mean, std

    def score_action(self, features: np.ndarray, action_name: str) -> dict[str, float]:
        feature_tensor = torch.tensor(np.asarray(features, dtype=np.float32), device=self.device).unsqueeze(0)
        action_tensor = torch.tensor([action_name_to_index(action_name)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model(feature_tensor, action_tensor)
            progress = self._denorm("k_step_progress", float(outputs["progress"].item()))
            survival = self._denorm("k_step_survival", float(outputs["survival"].item()))
            loop_risk = self._denorm("k_step_loop_risk", float(outputs["loop_risk"].item()))
            resource_value = self._denorm("k_step_resource_value", float(outputs["resource_value"].item()))
            teacher_margin = self._denorm("teacher_margin", float(outputs["teacher_margin"].item()))
            search_context_prob = float(torch.sigmoid(outputs["search_context_logit"]).item())
            action_logits = outputs["action_logits"]
            teacher_action_prob = float(torch.softmax(action_logits, dim=-1)[0, action_name_to_index(action_name)].item())

        weights = self._weights()
        total = (
            weights["progress"] * progress
            + weights["survival"] * survival
            + weights["resource_value"] * resource_value
            - weights["loop_risk"] * loop_risk
            + weights["teacher_margin"] * teacher_margin
            + weights["teacher_policy"] * teacher_action_prob
            + (weights["search_context"] * search_context_prob if action_name == "search" else 0.0)
        )
        score_mean, score_std = self._score_stats()
        calibrated_total = float(np.tanh((total - score_mean) / score_std))
        return {
            "action": action_name,
            "total": calibrated_total,
            "raw_total": float(total),
            "progress": progress,
            "survival": survival,
            "loop_risk": loop_risk,
            "resource_value": resource_value,
            "teacher_margin": teacher_margin,
            "search_context_prob": search_context_prob,
            "teacher_action_prob": teacher_action_prob,
        }

    def rank_actions(self, features: np.ndarray, allowed_actions: list[str]) -> list[dict[str, float]]:
        ranked = [self.score_action(features, action_name) for action_name in allowed_actions]
        ranked.sort(key=lambda row: (row["total"], row["teacher_action_prob"], row["action"]), reverse=True)
        return ranked


def save_proxy_model(model: ProxyRewardModel, path: str, metadata: dict | None = None) -> None:
    atomic_torch_save(path, {"state_dict": model.state_dict(), "metadata": metadata or {}})


def load_proxy_model(path: str, device: str = "cpu") -> ProxyRewardInference:
    torch_device = torch.device(device)
    payload = torch.load(path, map_location=torch_device, weights_only=False)
    metadata = payload.get("metadata", {})
    input_dim = int(metadata["input_dim"])
    model = ProxyRewardModel(
        input_dim=input_dim,
        hidden_size=int(metadata.get("hidden_size", 256)),
        action_embed_dim=int(metadata.get("action_embed_dim", 32)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(torch_device)
    model.eval()
    return ProxyRewardInference(model=model, device=torch_device, metadata=metadata)
