from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class PolicyInput:
    state: dict
    memory: object
    active_skill: str
    allowed_actions: list[str]


class ActionPolicy:
    def act(self, inputs: PolicyInput) -> str:
        raise NotImplementedError


class RandomAllowedActionPolicy(ActionPolicy):
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def act(self, inputs: PolicyInput) -> str:
        return self.rng.choice(inputs.allowed_actions) if inputs.allowed_actions else "wait"


def build_policy(name: str, seed: int = 42) -> ActionPolicy:
    if name == "random_allowed":
        return RandomAllowedActionPolicy(seed=seed)
    raise ValueError(f"Unknown policy: {name}")

