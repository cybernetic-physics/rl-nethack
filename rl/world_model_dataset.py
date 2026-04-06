from __future__ import annotations

from collections import defaultdict

import numpy as np

from rl.feature_encoder import ACTION_SET, SKILL_SET


_ACTION_TO_IDX = {name: idx for idx, name in enumerate(ACTION_SET)}
_SKILL_TO_IDX = {name: idx for idx, name in enumerate(SKILL_SET)}


def _group_rows_by_episode(rows: list[dict]) -> list[list[dict]]:
    episodes: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        episodes[row["episode_id"]].append(row)
    grouped = []
    for _, episode_rows in episodes.items():
        grouped.append(sorted(episode_rows, key=lambda row: row["step"]))
    return sorted(grouped, key=lambda episode_rows: episode_rows[0]["episode_id"])


def build_world_model_examples(
    rows: list[dict],
    *,
    horizon: int = 8,
    observation_version: str | None = None,
) -> list[dict]:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not rows:
        return []

    examples: list[dict] = []
    for episode_rows in _group_rows_by_episode(rows):
        versions = {row.get("observation_version", observation_version or "unknown") for row in episode_rows}
        if observation_version is not None and versions != {observation_version}:
            raise ValueError(
                f"Requested observation_version={observation_version} but episode contains {sorted(versions)}"
            )

        for start_idx, row in enumerate(episode_rows):
            if start_idx + horizon >= len(episode_rows):
                break
            window = episode_rows[start_idx : start_idx + horizon]
            target_row = episode_rows[start_idx + horizon]
            feature_vector = row.get("feature_vector")
            target_feature_vector = target_row.get("feature_vector")
            if feature_vector is None or target_feature_vector is None:
                raise ValueError("All rows must contain feature_vector")
            examples.append(
                {
                    "episode_id": row["episode_id"],
                    "seed": row.get("seed"),
                    "start_step": row["step"],
                    "horizon": horizon,
                    "observation_version": row.get("observation_version", observation_version or "unknown"),
                    "task": row.get("task", "explore"),
                    "task_index": _SKILL_TO_IDX.get(row.get("task", "explore"), 0),
                    "action": row.get("action", "wait"),
                    "action_index": _ACTION_TO_IDX.get(row.get("action", "wait"), _ACTION_TO_IDX["wait"]),
                    "feature_vector": feature_vector,
                    "target_feature_vector": target_feature_vector,
                    "cumulative_reward": float(sum(step_row.get("reward", 0.0) for step_row in window)),
                    "done_within_horizon": float(any(bool(step_row.get("done", False)) for step_row in window)),
                }
            )
    return examples


def examples_to_arrays(examples: list[dict]) -> dict[str, np.ndarray]:
    if not examples:
        raise ValueError("No examples to convert")
    features = np.asarray([row["feature_vector"] for row in examples], dtype=np.float32)
    target_features = np.asarray([row["target_feature_vector"] for row in examples], dtype=np.float32)
    actions = np.asarray([row["action_index"] for row in examples], dtype=np.int64)
    tasks = np.asarray([row["task_index"] for row in examples], dtype=np.int64)
    rewards = np.asarray([row["cumulative_reward"] for row in examples], dtype=np.float32)
    dones = np.asarray([row["done_within_horizon"] for row in examples], dtype=np.float32)
    return {
        "features": features,
        "target_features": target_features,
        "actions": actions,
        "tasks": tasks,
        "rewards": rewards,
        "dones": dones,
    }
