from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from rl.feature_encoder import ACTION_SET


class BCPolicyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 256, output_dim: int = len(ACTION_SET)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class BCPolicyInference:
    model: BCPolicyMLP
    device: torch.device

    def act(self, features: np.ndarray, allowed_actions: list[str] | None = None) -> str:
        features = np.asarray(features, dtype=np.float32)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(features).to(self.device).unsqueeze(0)).squeeze(0)
            if allowed_actions:
                allowed = set(allowed_actions)
                for idx, name in enumerate(ACTION_SET):
                    if name not in allowed:
                        logits[idx] = -1e9
            action_idx = int(torch.argmax(logits).item())
        return ACTION_SET[action_idx]


def save_bc_model(model: BCPolicyMLP, path: str, metadata: dict | None = None) -> None:
    payload = {"state_dict": model.state_dict(), "metadata": metadata or {}}
    torch.save(payload, path)


def load_bc_model(path: str, input_dim: int | None = None, device: str = "cpu") -> BCPolicyInference:
    torch_device = torch.device(device)
    payload = torch.load(path, map_location=torch_device)
    metadata = payload.get("metadata", {})
    resolved_input_dim = input_dim or metadata.get("input_dim")
    if resolved_input_dim is None:
        raise ValueError("BC model input_dim is required and was not found in metadata")
    hidden_size = metadata.get("hidden_size", 256)
    model = BCPolicyMLP(input_dim=resolved_input_dim, hidden_size=hidden_size)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    model.to(torch_device)
    return BCPolicyInference(model=model, device=torch_device)
