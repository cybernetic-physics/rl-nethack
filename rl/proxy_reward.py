from __future__ import annotations

import numpy as np

from rl.proxy_model import load_proxy_model


class ProxyRewardScorer:
    def __init__(self, model_path: str):
        self.inference = load_proxy_model(model_path)

    def score(self, feature_vector: np.ndarray | list[float], action_name: str, allowed_actions: list[str] | None = None) -> dict:
        del allowed_actions
        return self.inference.score_action(np.asarray(feature_vector, dtype=np.float32), action_name)
