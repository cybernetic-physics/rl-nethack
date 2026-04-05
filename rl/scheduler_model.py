from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from rl.io_utils import atomic_torch_save, atomic_write_text
from rl.options import build_skill_registry


SKILL_NAMES = list(build_skill_registry().keys())
_SKILL_TO_IDX = {name: i for i, name in enumerate(SKILL_NAMES)}


def scheduler_feature_dim() -> int:
    return 18 + 3 * len(SKILL_NAMES)


def encode_scheduler_features(
    state: dict,
    memory,
    active_skill: str | None,
    steps_in_skill: int,
    available_skills: list[str],
) -> np.ndarray:
    hp = float(state.get("hp", 0))
    hp_max = float(max(1, state.get("hp_max", 1)))
    adjacent = state.get("adjacent", {})
    features = [
        hp / hp_max,
        hp_max / 20.0,
        float(state.get("gold", 0)) / 100.0,
        float(state.get("depth", 1)) / 10.0,
        float(state.get("turn", 0)) / 1000.0,
        float(state.get("ac", 0)) / 20.0,
        float(state.get("strength", 0)) / 25.0,
        float(state.get("dexterity", 0)) / 25.0,
        min(1.0, float(len(state.get("visible_monsters", []))) / 8.0),
        min(1.0, float(len(state.get("visible_items", []))) / 8.0),
        min(1.0, float(getattr(memory, "total_explored", 0)) / 400.0),
        min(1.0, float(len(getattr(memory, "rooms", []))) / 20.0),
        min(1.0, float(steps_in_skill) / 32.0),
        1.0 if state.get("standing_on_down_stairs") else 0.0,
        1.0 if state.get("standing_on_up_stairs") else 0.0,
        1.0 if "see here" in state.get("message", "").lower() else 0.0,
        1.0 if "stairs_down" in set(adjacent.values()) else 0.0,
        1.0 if "stairs_up" in set(adjacent.values()) else 0.0,
    ]

    active_one_hot = np.zeros(len(SKILL_NAMES), dtype=np.float32)
    if active_skill in _SKILL_TO_IDX:
        active_one_hot[_SKILL_TO_IDX[active_skill]] = 1.0

    avail_one_hot = np.zeros(len(SKILL_NAMES), dtype=np.float32)
    for name in available_skills:
        if name in _SKILL_TO_IDX:
            avail_one_hot[_SKILL_TO_IDX[name]] = 1.0

    preferred_by_context = np.zeros(len(SKILL_NAMES), dtype=np.float32)
    if hp <= max(1.0, hp_max * 0.5):
        preferred_by_context[_SKILL_TO_IDX["survive"]] = 1.0
    if state.get("visible_monsters"):
        preferred_by_context[_SKILL_TO_IDX["combat"]] = 1.0
    if "see here" in state.get("message", "").lower() or state.get("visible_items"):
        preferred_by_context[_SKILL_TO_IDX["resource"]] = 1.0
    if "stairs_down" in set(adjacent.values()) or state.get("standing_on_down_stairs"):
        preferred_by_context[_SKILL_TO_IDX["descend"]] = 1.0
    if not preferred_by_context.any():
        preferred_by_context[_SKILL_TO_IDX["explore"]] = 1.0

    return np.concatenate(
        [
            np.asarray(features, dtype=np.float32),
            active_one_hot,
            avail_one_hot,
            preferred_by_context,
        ]
    )


class SchedulerMLP(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(scheduler_feature_dim(), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(SKILL_NAMES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class SchedulerInferenceModel:
    model: SchedulerMLP
    device: torch.device

    def select_skill(self, features: np.ndarray, available_skills: list[str]) -> str:
        allowed = set(available_skills)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(features).to(self.device).unsqueeze(0)).squeeze(0)
            for idx, name in enumerate(SKILL_NAMES):
                if name not in allowed:
                    logits[idx] = -1e9
            chosen = int(torch.argmax(logits).item())
        return SKILL_NAMES[chosen]


def save_scheduler_model(model: SchedulerMLP, path: str, metadata: dict | None = None) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
        "skill_names": SKILL_NAMES,
    }
    atomic_torch_save(path, payload)


def load_scheduler_model(path: str, device: str = "cpu") -> SchedulerInferenceModel:
    torch_device = torch.device(device)
    payload = torch.load(path, map_location=torch_device)
    model = SchedulerMLP()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    model.to(torch_device)
    return SchedulerInferenceModel(model=model, device=torch_device)


def write_scheduler_dataset(rows: list[dict], path: str) -> None:
    atomic_write_text(path, "".join(json.dumps(row) + "\n" for row in rows))
