from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from rl.feature_encoder import ACTION_SET
from rl.io_utils import atomic_torch_save


class BCPolicyMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        output_dim: int = len(ACTION_SET),
        num_layers: int = 2,
    ):
        super().__init__()
        resolved_layers = max(int(num_layers), 1)
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for _ in range(resolved_layers):
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(nn.ReLU())
            prev_dim = hidden_size
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class BCPolicyInference:
    model: BCPolicyMLP
    device: torch.device

    def _masked_logits(
        self,
        feature_tensor: torch.Tensor,
        *,
        allowed_actions_list: list[list[str] | None] | None = None,
    ) -> torch.Tensor:
        logits = self.model(feature_tensor)
        if allowed_actions_list:
            for row_idx, allowed_actions in enumerate(allowed_actions_list):
                if not allowed_actions:
                    continue
                allowed = set(allowed_actions)
                for idx, name in enumerate(ACTION_SET):
                    if name not in allowed:
                        logits[row_idx, idx] = -1e9
        return logits

    def act(self, features: np.ndarray, allowed_actions: list[str] | None = None) -> str:
        action_idx = self.act_batch([features], allowed_actions_list=[allowed_actions])[0]
        return ACTION_SET[action_idx]

    def logits_batch(
        self,
        features: np.ndarray | list[list[float]],
        *,
        allowed_actions_list: list[list[str] | None] | None = None,
    ) -> np.ndarray:
        feature_array = np.asarray(features, dtype=np.float32)
        if feature_array.ndim == 1:
            feature_array = feature_array.reshape(1, -1)
        with torch.no_grad():
            logits = self._masked_logits(
                torch.from_numpy(feature_array).to(self.device),
                allowed_actions_list=allowed_actions_list,
            )
        return logits.cpu().numpy()

    def act_batch(
        self,
        features: np.ndarray | list[list[float]],
        *,
        allowed_actions_list: list[list[str] | None] | None = None,
    ) -> list[int]:
        feature_array = np.asarray(features, dtype=np.float32)
        if feature_array.ndim == 1:
            feature_array = feature_array.reshape(1, -1)
        with torch.no_grad():
            logits = self._masked_logits(
                torch.from_numpy(feature_array).to(self.device),
                allowed_actions_list=allowed_actions_list,
            )
            action_indices = torch.argmax(logits, dim=1).tolist()
        return [int(idx) for idx in action_indices]

    def act_names_batch(
        self,
        features: np.ndarray | list[list[float]],
        *,
        allowed_actions_list: list[list[str] | None] | None = None,
    ) -> list[str]:
        return [ACTION_SET[idx] for idx in self.act_batch(features, allowed_actions_list=allowed_actions_list)]


def save_bc_model(model: BCPolicyMLP, path: str, metadata: dict | None = None) -> None:
    payload = {"state_dict": model.state_dict(), "metadata": metadata or {}}
    atomic_torch_save(path, payload)


def load_bc_model(path: str, input_dim: int | None = None, device: str = "cpu") -> BCPolicyInference:
    torch_device = torch.device(device)
    payload = torch.load(path, map_location=torch_device)
    metadata = payload.get("metadata", {})
    resolved_input_dim = input_dim or metadata.get("input_dim")
    if resolved_input_dim is None:
        raise ValueError("BC model input_dim is required and was not found in metadata")
    hidden_size = metadata.get("hidden_size", 256)
    num_layers = metadata.get("num_layers", 2)
    model = BCPolicyMLP(input_dim=resolved_input_dim, hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    model.to(torch_device)
    return BCPolicyInference(model=model, device=torch_device)
