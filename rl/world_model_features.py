from __future__ import annotations

import json

import numpy as np
import torch

from rl.feature_encoder import ACTION_SET
from rl.io_utils import atomic_write_text
from rl.train_bc import load_trace_rows
from rl.world_model import load_world_model


def coerce_world_model_feature_vector(
    feature_vector: np.ndarray | list[float],
    expected_dim: int,
) -> np.ndarray:
    features = np.asarray(feature_vector, dtype=np.float32).reshape(-1)
    if features.shape[0] == expected_dim:
        return features
    if features.shape[0] > expected_dim:
        # World-model augmentation appends latent/action features to the base
        # observation. The world model itself always expects the original prefix.
        return features[:expected_dim].astype(np.float32)
    raise ValueError(f"Feature vector dim {features.shape[0]} is smaller than world-model input_dim {expected_dim}")


def world_model_augmented_dim(base_dim: int, model_path: str | None, mode: str | None) -> int:
    if not model_path or not mode:
        return base_dim
    payload = torch.load(model_path, map_location="cpu")
    metadata = payload.get("metadata", {})
    latent_dim = int(metadata.get("latent_dim", 0))
    if mode == "replace":
        return latent_dim
    if mode == "concat":
        return base_dim + latent_dim
    if mode == "concat_aux":
        return base_dim + latent_dim + len(ACTION_SET)
    raise ValueError(f"Unsupported world-model feature mode: {mode}")


def augment_feature_vector(
    feature_vector: np.ndarray | list[float],
    inference,
    *,
    mode: str,
    prompt_text: str | None = None,
) -> np.ndarray:
    expected_dim = int(getattr(inference.model, "_metadata", {}).get("input_dim", 0))
    base_features = coerce_world_model_feature_vector(feature_vector, expected_dim) if expected_dim else np.asarray(
        feature_vector, dtype=np.float32
    )
    encoded = inference.encode_with_aux(base_features, prompt_text=prompt_text)
    latent = encoded["latent"]
    action_logits = encoded["action_logits"]
    features = base_features
    if mode == "replace":
        return latent.astype(np.float32)
    if mode == "concat":
        return np.concatenate([features, latent]).astype(np.float32)
    if mode == "concat_aux":
        return np.concatenate([features, latent, action_logits]).astype(np.float32)
    raise ValueError(f"Unsupported world-model feature mode: {mode}")


def transform_trace_with_world_model(
    input_path: str,
    output_path: str,
    model_path: str,
    *,
    observation_version_suffix: str = "wm",
    mode: str = "replace",
) -> dict:
    if mode not in {"replace", "concat", "concat_aux"}:
        raise ValueError(f"Unsupported world-model transform mode: {mode}")
    rows = load_trace_rows(input_path)
    inference = load_world_model(model_path)
    transformed_rows = []
    latent_dim = None
    feature_dim = None
    action_dim = None
    for row in rows:
        encoded = inference.encode_with_aux(row["feature_vector"], prompt_text=row.get("prompt"))
        latent = encoded["latent"].tolist()
        action_logits = encoded["action_logits"].tolist()
        latent_dim = latent_dim or len(latent)
        feature_dim = feature_dim or len(row["feature_vector"])
        action_dim = action_dim or len(action_logits)
        transformed = dict(row)
        transformed["feature_vector"] = augment_feature_vector(
            row["feature_vector"],
            inference,
            mode=mode,
            prompt_text=row.get("prompt"),
        ).tolist()
        transformed["observation_version"] = f"{row.get('observation_version', 'unknown')}+{observation_version_suffix}_{mode}"
        transformed_rows.append(transformed)

    payload = "".join(json.dumps(row) + "\n" for row in transformed_rows)
    atomic_write_text(output_path, payload)
    return {
        "input_path": input_path,
        "output_path": output_path,
        "rows": len(transformed_rows),
        "latent_dim": latent_dim or 0,
        "action_dim": action_dim or 0,
        "original_feature_dim": feature_dim or 0,
        "mode": mode,
        "observation_version": transformed_rows[0]["observation_version"] if transformed_rows else "unknown",
    }
