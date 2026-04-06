from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from rl.feature_encoder import ACTION_SET, SKILL_SET
from rl.io_utils import atomic_torch_save


class TraceWorldModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_size: int = 256,
        action_dim: int = len(ACTION_SET),
        skill_dim: int = len(SKILL_SET),
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim),
            nn.ReLU(),
        )
        self.action_embedding = nn.Embedding(action_dim, latent_dim)
        self.skill_embedding = nn.Embedding(skill_dim, latent_dim)
        self.transition = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.current_feature_head = nn.Linear(latent_dim, input_dim)
        self.action_head = nn.Linear(latent_dim, action_dim)
        self.future_feature_head = nn.Linear(hidden_size, input_dim)
        self.reward_head = nn.Linear(hidden_size, 1)
        self.done_head = nn.Linear(hidden_size, 1)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def forward(self, features: torch.Tensor, actions: torch.Tensor, tasks: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode(features)
        action_latent = self.action_embedding(actions)
        task_latent = self.skill_embedding(tasks)
        hidden = self.transition(torch.cat([latent, action_latent, task_latent], dim=-1))
        return {
            "latent": latent,
            "hidden": hidden,
            "current_features": self.current_feature_head(latent),
            "action_logits": self.action_head(latent),
            "future_features": self.future_feature_head(hidden),
            "reward": self.reward_head(hidden).squeeze(-1),
            "done_logit": self.done_head(hidden).squeeze(-1),
        }


@dataclass
class WorldModelInference:
    model: TraceWorldModel
    device: torch.device

    def encode(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        with torch.no_grad():
            latent = self._encode_tensor(torch.from_numpy(features).unsqueeze(0).to(self.device))
        return latent.squeeze(0).cpu().numpy()

    def _encode_tensor(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        latent = self.model.encode(feature_tensor)
        metadata = getattr(self.model, "_metadata", {})
        mean = metadata.get("latent_mean")
        std = metadata.get("latent_std")
        if mean is not None and std is not None:
            mean_t = torch.tensor(mean, dtype=torch.float32, device=self.device)
            std_t = torch.tensor(std, dtype=torch.float32, device=self.device).clamp_min(1e-6)
            latent = (latent - mean_t.unsqueeze(0)) / std_t.unsqueeze(0)
        return latent

    def encode_with_aux(self, features: np.ndarray) -> dict[str, np.ndarray]:
        features = np.asarray(features, dtype=np.float32)
        feature_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent = self._encode_tensor(feature_tensor)
            action_logits = self.model.action_head(latent)
            current_features = self.model.current_feature_head(latent)
        return {
            "latent": latent.squeeze(0).cpu().numpy(),
            "action_logits": action_logits.squeeze(0).cpu().numpy(),
            "current_features": current_features.squeeze(0).cpu().numpy(),
        }


def save_world_model(model: TraceWorldModel, path: str, metadata: dict | None = None) -> None:
    payload = {"state_dict": model.state_dict(), "metadata": metadata or {}}
    atomic_torch_save(path, payload)


def load_world_model(path: str, device: str = "cpu") -> WorldModelInference:
    torch_device = torch.device(device)
    payload = torch.load(path, map_location=torch_device)
    metadata = payload.get("metadata", {})
    input_dim = metadata.get("input_dim")
    if input_dim is None:
        raise ValueError("World model input_dim missing from metadata")
    model = TraceWorldModel(
        input_dim=input_dim,
        latent_dim=metadata.get("latent_dim", 128),
        hidden_size=metadata.get("hidden_size", 256),
    )
    model.load_state_dict(payload["state_dict"])
    model._metadata = metadata
    model.to(torch_device)
    model.eval()
    return WorldModelInference(model=model, device=torch_device)
