from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from rl.bc_model import load_bc_model
from rl.io_utils import atomic_write_text
from rl.train_bc import load_trace_rows
from rl.world_model import load_world_model
from rl.world_model_features import coerce_world_model_feature_vector, state_prompt_from_row


def relabel_trace_actions(
    input_path: str,
    output_path: str,
    *,
    bc_model_path: str,
) -> dict:
    rows = load_trace_rows(input_path)
    bc_policy = load_bc_model(bc_model_path)
    model_payload = torch.load(bc_model_path, map_location="cpu")
    metadata = model_payload.get("metadata", {})
    wm_path = metadata.get("world_model_path")
    wm_mode = metadata.get("world_model_feature_mode")
    wm_inference = load_world_model(wm_path) if wm_path and wm_mode else None

    allowed_actions_list = [row.get("allowed_actions") for row in rows]
    if wm_inference and wm_mode:
        expected_dim = int(getattr(wm_inference.model, "_metadata", {}).get("input_dim", 0))
        base_features = [
            coerce_world_model_feature_vector(row["feature_vector"], expected_dim) if expected_dim else row["feature_vector"]
            for row in rows
        ]
        prompts = [state_prompt_from_row(row) for row in rows]
        encoded = wm_inference.encode_with_aux_batch(base_features, prompt_texts=prompts)
        latents = encoded["latent"]
        action_logits = encoded["action_logits"]
        feature_batch = []
        for base_feature, latent, logits in zip(base_features, latents, action_logits):
            if wm_mode == "replace":
                feature_batch.append(np.asarray(latent, dtype=np.float32))
            elif wm_mode == "concat":
                feature_batch.append(np.concatenate([base_feature, latent]).astype(np.float32))
            elif wm_mode == "concat_aux":
                feature_batch.append(np.concatenate([base_feature, latent, logits]).astype(np.float32))
            else:
                raise ValueError(f"Unsupported world-model feature mode: {wm_mode}")
    else:
        feature_batch = [row["feature_vector"] for row in rows]

    predicted_actions = bc_policy.act_names_batch(feature_batch, allowed_actions_list=allowed_actions_list)

    relabeled_rows = []
    changed = 0
    action_counts: dict[str, int] = {}
    for row, predicted_action in zip(rows, predicted_actions):
        relabeled = dict(row)
        relabeled["original_action"] = row.get("action")
        relabeled["action"] = predicted_action
        relabeled_rows.append(relabeled)
        changed += int(predicted_action != row.get("action"))
        action_counts[predicted_action] = action_counts.get(predicted_action, 0) + 1

    atomic_write_text(output_path, "".join(json.dumps(row) + "\n" for row in relabeled_rows))
    return {
        "input_path": input_path,
        "output_path": output_path,
        "rows": len(relabeled_rows),
        "changed_rows": changed,
        "changed_rate": round(changed / max(1, len(relabeled_rows)), 4),
        "action_counts": action_counts,
        "world_model_path": wm_path,
        "world_model_feature_mode": wm_mode,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Relabel trace actions with a BC teacher")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--bc-model-path", type=str, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = relabel_trace_actions(args.input, args.output, bc_model_path=args.bc_model_path)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
