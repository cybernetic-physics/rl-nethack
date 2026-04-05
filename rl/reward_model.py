from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from rl.io_utils import atomic_torch_save


def reward_feature_dim() -> int:
    return 37


class RewardMLP(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(reward_feature_dim(), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class RewardInferenceModel:
    model: RewardMLP
    device: torch.device

    def score(self, features: np.ndarray) -> float:
        with torch.no_grad():
            value = self.model(torch.from_numpy(features).to(self.device).unsqueeze(0)).item()
        return float(value)


def save_reward_model(model: RewardMLP, path: str, metadata: dict | None = None) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    atomic_torch_save(path, payload)


def load_reward_model(path: str, device: str = "cpu") -> RewardInferenceModel:
    torch_device = torch.device(device)
    payload = torch.load(path, map_location=torch_device)
    model = RewardMLP()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    model.to(torch_device)
    return RewardInferenceModel(model=model, device=torch_device)


def load_preference_rows(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
