from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rl.feature_encoder import ACTION_SET
from rl.trace_eval import evaluate_trace_policy
from rl.train_bc import load_trace_rows, train_bc_model
from rl.world_model import load_world_model
from rl.world_model_dataset import build_world_model_examples, examples_to_arrays
from rl.world_model_features import coerce_world_model_feature_vector, transform_trace_with_world_model


def _prepare_eval_arrays(
    model_path: str,
    trace_path: str,
    *,
    horizon: int,
    observation_version: str | None,
) -> tuple[dict[str, np.ndarray], dict]:
    rows = load_trace_rows(trace_path)
    inference = load_world_model(model_path)
    metadata = getattr(inference.model, "_metadata", {})
    expected_input_dim = int(metadata.get("input_dim", 0))
    expected_observation_version = metadata.get("observation_version", observation_version or "unknown")
    examples = build_world_model_examples(rows, horizon=horizon, observation_version=observation_version)
    arrays = examples_to_arrays(examples)

    original_feature_dim = int(arrays["features"].shape[1])
    coerced = 0
    if expected_input_dim and original_feature_dim != expected_input_dim:
        arrays["features"] = np.asarray(
            [coerce_world_model_feature_vector(row, expected_input_dim) for row in arrays["features"]],
            dtype=np.float32,
        )
        arrays["target_features"] = np.asarray(
            [coerce_world_model_feature_vector(row, expected_input_dim) for row in arrays["target_features"]],
            dtype=np.float32,
        )
        coerced = len(examples)

    info = {
        "num_examples": len(examples),
        "trace_observation_version": observation_version or examples[0]["observation_version"],
        "model_observation_version": expected_observation_version,
        "input_feature_dim": original_feature_dim,
        "model_input_dim": expected_input_dim or original_feature_dim,
        "coerced_example_count": coerced,
    }
    return arrays, info


def _per_action_metrics(actions: torch.Tensor, predictions: torch.Tensor, action_names: list[str]) -> dict[str, dict]:
    metrics = {}
    for idx, name in enumerate(action_names):
        mask = actions == idx
        support = int(mask.sum().item())
        if support == 0:
            continue
        accuracy = float((predictions[mask] == actions[mask]).float().mean().item())
        predicted = int((predictions == idx).sum().item())
        metrics[name] = {
            "support": support,
            "predicted": predicted,
            "accuracy": round(accuracy, 4),
        }
    return metrics


def _common_action_mismatches(actions: torch.Tensor, predictions: torch.Tensor, action_names: list[str], top_k: int = 8) -> list[dict]:
    mismatch_counter: Counter[tuple[str, str]] = Counter()
    for truth, pred in zip(actions.tolist(), predictions.tolist()):
        if truth != pred:
            mismatch_counter[(action_names[truth], action_names[pred])] += 1
    return [
        {"teacher_action": truth, "predicted_action": pred, "count": count}
        for (truth, pred), count in mismatch_counter.most_common(top_k)
    ]


def summarize_world_model_outputs(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    *,
    device: torch.device,
    action_names: list[str],
) -> dict:
    x = torch.tensor(arrays["features"], dtype=torch.float32, device=device)
    target_x = torch.tensor(arrays["target_features"], dtype=torch.float32, device=device)
    actions = torch.tensor(arrays["actions"], dtype=torch.long, device=device)
    tasks = torch.tensor(arrays["tasks"], dtype=torch.long, device=device)
    rewards = torch.tensor(arrays["rewards"], dtype=torch.float32, device=device)
    dones = torch.tensor(arrays["dones"], dtype=torch.float32, device=device)

    with torch.no_grad():
        prompt_texts = arrays.get("prompts")
        text_context = None
        if prompt_texts and getattr(model, "text_encoder_backend", "none") != "none" and not getattr(
            model, "text_trainable", False
        ):
            text_context = model.encode_text_context(prompt_texts, device=device)
        outputs = model(x, actions, tasks, prompt_texts=prompt_texts, text_context=text_context)
        action_logits = outputs["action_logits"]
        predictions = torch.argmax(action_logits, dim=-1)
        topk = min(3, action_logits.shape[1])
        topk_hits = torch.topk(action_logits, k=topk, dim=-1).indices.eq(actions.unsqueeze(-1)).any(dim=-1)
        future_cos = F.cosine_similarity(outputs["future_features"], target_x, dim=-1)
        recon_cos = F.cosine_similarity(outputs["current_features"], x, dim=-1)
        latent_std = outputs["latent"].std(dim=0)

        result = {
            "feature_mse": float(F.mse_loss(outputs["future_features"], target_x).item()),
            "feature_mae": float(torch.mean(torch.abs(outputs["future_features"] - target_x)).item()),
            "feature_cosine_mean": float(future_cos.mean().item()),
            "reconstruction_mse": float(F.mse_loss(outputs["current_features"], x).item()),
            "reconstruction_mae": float(torch.mean(torch.abs(outputs["current_features"] - x)).item()),
            "reconstruction_cosine_mean": float(recon_cos.mean().item()),
            "action_accuracy": float((predictions == actions).float().mean().item()),
            "action_top3_accuracy": float(topk_hits.float().mean().item()),
            "reward_mae": float(torch.mean(torch.abs(outputs["reward"] - rewards)).item()),
            "reward_rmse": float(torch.sqrt(F.mse_loss(outputs["reward"], rewards)).item()),
            "done_accuracy": float(
                ((torch.sigmoid(outputs["done_logit"]) >= 0.5) == (dones >= 0.5)).float().mean().item()
            ),
            "done_positive_rate": float(dones.mean().item()),
            "latent_std_mean": float(latent_std.mean().item()),
            "latent_std_min": float(latent_std.min().item()),
            "latent_dead_fraction": float((latent_std < 1e-3).float().mean().item()),
            "text_encoder_backend": getattr(model, "text_encoder_backend", "none"),
            "per_action_metrics": _per_action_metrics(actions, predictions, action_names),
            "common_action_mismatches": _common_action_mismatches(actions, predictions, action_names),
        }
    return result


def evaluate_world_model(
    model_path: str,
    trace_path: str,
    *,
    horizon: int = 8,
    observation_version: str | None = None,
    downstream_train_trace_path: str | None = None,
    downstream_heldout_trace_path: str | None = None,
    downstream_mode: str = "concat_aux",
    downstream_epochs: int = 20,
    downstream_lr: float = 1e-3,
    downstream_hidden_size: int = 256,
) -> dict:
    arrays, input_info = _prepare_eval_arrays(
        model_path,
        trace_path,
        horizon=horizon,
        observation_version=observation_version,
    )
    inference = load_world_model(model_path)
    result = {
        "model_path": model_path,
        "trace_path": trace_path,
        "horizon": horizon,
        "observation_version": input_info["trace_observation_version"],
        "text_encoder_backend": getattr(inference.model, "text_encoder_backend", "none"),
        "text_model_name": getattr(inference.model, "text_model_name", None),
        **input_info,
    }
    result.update(
        summarize_world_model_outputs(
            inference.model,
            arrays,
            device=inference.device,
            action_names=ACTION_SET,
        )
    )

    if downstream_train_trace_path and downstream_heldout_trace_path:
        with tempfile.TemporaryDirectory(prefix="wm_eval_") as tmpdir:
            train_xform = str(Path(tmpdir) / "train_xform.jsonl")
            heldout_xform = str(Path(tmpdir) / "heldout_xform.jsonl")
            bc_output = str(Path(tmpdir) / "bc_from_world_model.pt")
            train_transform = transform_trace_with_world_model(
                downstream_train_trace_path,
                train_xform,
                model_path,
                mode=downstream_mode,
            )
            heldout_transform = transform_trace_with_world_model(
                downstream_heldout_trace_path,
                heldout_xform,
                model_path,
                mode=downstream_mode,
            )
            transformed_rows = load_trace_rows(train_xform)
            train_bc_model(
                transformed_rows,
                bc_output,
                epochs=downstream_epochs,
                lr=downstream_lr,
                hidden_size=downstream_hidden_size,
                observation_version=train_transform["observation_version"],
                world_model_path=model_path,
                world_model_feature_mode=downstream_mode,
            )
            downstream_eval = evaluate_trace_policy(heldout_xform, "bc", bc_model_path=bc_output, summary_only=True)
            result["downstream_bc"] = {
                "mode": downstream_mode,
                "train_transform": train_transform,
                "heldout_transform": heldout_transform,
                "trace_eval_summary": downstream_eval["summary"],
                "bc_epochs": downstream_epochs,
                "bc_learning_rate": downstream_lr,
                "bc_hidden_size": downstream_hidden_size,
            }

    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a short-horizon trace world model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--observation-version", type=str, default=None)
    parser.add_argument("--downstream-train-input", type=str, default=None)
    parser.add_argument("--downstream-heldout-input", type=str, default=None)
    parser.add_argument("--downstream-mode", type=str, default="concat_aux", choices=["replace", "concat", "concat_aux"])
    parser.add_argument("--downstream-epochs", type=int, default=20)
    parser.add_argument("--downstream-lr", type=float, default=1e-3)
    parser.add_argument("--downstream-hidden-size", type=int, default=256)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = evaluate_world_model(
        args.model,
        args.input,
        horizon=args.horizon,
        observation_version=args.observation_version,
        downstream_train_trace_path=args.downstream_train_input,
        downstream_heldout_trace_path=args.downstream_heldout_input,
        downstream_mode=args.downstream_mode,
        downstream_epochs=args.downstream_epochs,
        downstream_lr=args.downstream_lr,
        downstream_hidden_size=args.downstream_hidden_size,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
