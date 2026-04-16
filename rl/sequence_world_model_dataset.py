from __future__ import annotations

from collections import defaultdict

import numpy as np

from rl.feature_encoder import ACTION_SET, SKILL_SET
from rl.world_model_features import state_prompt_from_row


_ACTION_TO_IDX = {name: idx for idx, name in enumerate(ACTION_SET)}
_SKILL_TO_IDX = {name: idx for idx, name in enumerate(SKILL_SET)}


def _group_rows_by_episode(rows: list[dict]) -> list[list[dict]]:
    episodes: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        episodes[str(row["episode_id"])].append(row)
    grouped = []
    for episode_rows in episodes.values():
        grouped.append(sorted(episode_rows, key=lambda row: int(row["step"])))
    return sorted(grouped, key=lambda episode_rows: str(episode_rows[0]["episode_id"]))


def build_sequence_world_model_examples(
    rows: list[dict],
    *,
    context_len: int = 4,
    rollout_horizon: int = 8,
    discount: float = 0.99,
    observation_version: str | None = None,
) -> list[dict]:
    if context_len < 1:
        raise ValueError("context_len must be >= 1")
    if rollout_horizon < 1:
        raise ValueError("rollout_horizon must be >= 1")
    if not rows:
        return []

    examples: list[dict] = []
    required_span = context_len + rollout_horizon
    for episode_rows in _group_rows_by_episode(rows):
        versions = {row.get("observation_version", observation_version or "unknown") for row in episode_rows}
        if observation_version is not None and versions != {observation_version}:
            raise ValueError(
                f"Requested observation_version={observation_version} but episode contains {sorted(versions)}"
            )
        if len(episode_rows) < required_span:
            continue

        for start_idx in range(0, len(episode_rows) - required_span + 1):
            window = episode_rows[start_idx : start_idx + required_span]
            obs_rows = window[: context_len + rollout_horizon]
            all_transition_rows = window[:-1]
            rollout_rows = window[context_len - 1 : context_len - 1 + rollout_horizon]
            value_targets = []
            for value_start in range(len(rollout_rows)):
                total = 0.0
                running_discount = 1.0
                for future_row in rollout_rows[value_start:]:
                    total += running_discount * float(future_row.get("reward", 0.0))
                    if bool(future_row.get("done", False)):
                        break
                    running_discount *= float(discount)
                value_targets.append(total)
            planner_action_scores = []
            planner_action_masks = []
            for planner_row in all_transition_rows:
                score_vec = [0.0] * len(ACTION_SET)
                mask_vec = [0.0] * len(ACTION_SET)
                for candidate in planner_row.get("planner_trace") or []:
                    action_name = str(candidate.get("action", ""))
                    if action_name in _ACTION_TO_IDX:
                        score_vec[_ACTION_TO_IDX[action_name]] = float(candidate.get("total", 0.0))
                        mask_vec[_ACTION_TO_IDX[action_name]] = 1.0
                planner_action_scores.append(score_vec)
                planner_action_masks.append(mask_vec)

            feature_vectors = [row.get("feature_vector") for row in obs_rows]
            if any(vector is None for vector in feature_vectors):
                raise ValueError("All rows must contain feature_vector")

            examples.append(
                {
                    "episode_id": obs_rows[0]["episode_id"],
                    "seed": obs_rows[0].get("seed"),
                    "start_step": int(obs_rows[0]["step"]),
                    "context_len": context_len,
                    "rollout_horizon": rollout_horizon,
                    "observation_version": obs_rows[0].get("observation_version", observation_version or "unknown"),
                    "feature_sequence": feature_vectors,
                    "prompt_sequence": [state_prompt_from_row(row) for row in obs_rows],
                    "action_sequence": [
                        _ACTION_TO_IDX.get(row.get("action", "wait"), _ACTION_TO_IDX["wait"])
                        for row in all_transition_rows
                    ],
                    "task_sequence": [_SKILL_TO_IDX.get(row.get("task", "explore"), 0) for row in all_transition_rows],
                    "reward_sequence": [float(row.get("reward", 0.0)) for row in rollout_rows],
                    "done_sequence": [float(bool(row.get("done", False))) for row in rollout_rows],
                    "value_sequence": value_targets,
                    "planner_action_score_sequence": planner_action_scores,
                    "planner_action_mask_sequence": planner_action_masks,
                    "valid_action_sequence": [
                        [
                            1.0 if action_name in row.get("allowed_actions", [row.get("action", "wait")]) else 0.0
                            for action_name in ACTION_SET
                        ]
                        for row in rollout_rows
                    ],
                    "rollout_start_index": context_len - 1,
                }
            )
    return examples


def sequence_examples_to_arrays(examples: list[dict]) -> dict[str, np.ndarray | list[list[str]]]:
    if not examples:
        raise ValueError("No examples to convert")
    features = np.asarray([row["feature_sequence"] for row in examples], dtype=np.float32)
    actions = np.asarray([row["action_sequence"] for row in examples], dtype=np.int64)
    tasks = np.asarray([row["task_sequence"] for row in examples], dtype=np.int64)
    rewards = np.asarray([row["reward_sequence"] for row in examples], dtype=np.float32)
    dones = np.asarray([row["done_sequence"] for row in examples], dtype=np.float32)
    values = np.asarray([row["value_sequence"] for row in examples], dtype=np.float32)
    planner_action_scores = np.asarray([row["planner_action_score_sequence"] for row in examples], dtype=np.float32)
    planner_action_masks = np.asarray([row["planner_action_mask_sequence"] for row in examples], dtype=np.float32)
    valid_actions = np.asarray([row["valid_action_sequence"] for row in examples], dtype=np.float32)
    prompts = [[str(text) for text in row["prompt_sequence"]] for row in examples]
    return {
        "features": features,
        "actions": actions,
        "tasks": tasks,
        "rewards": rewards,
        "dones": dones,
        "values": values,
        "planner_action_scores": planner_action_scores,
        "planner_action_masks": planner_action_masks,
        "valid_actions": valid_actions,
        "prompts": prompts,
        "rollout_start_indices": np.asarray([row["rollout_start_index"] for row in examples], dtype=np.int64),
    }
