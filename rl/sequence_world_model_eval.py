from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from rl.sequence_world_model import load_sequence_world_model
from rl.sequence_world_model_dataset import build_sequence_world_model_examples, sequence_examples_to_arrays
from rl.train_bc import load_trace_rows
from rl.world_model_calibration import expected_calibration_error, fit_temperature_for_binary_logits


def planner_pairwise_metrics(
    pred_scores: torch.Tensor,
    target_scores: torch.Tensor,
    target_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_diff = pred_scores.unsqueeze(-1) - pred_scores.unsqueeze(-2)
    target_diff = target_scores.unsqueeze(-1) - target_scores.unsqueeze(-2)
    pair_mask = (target_masks.unsqueeze(-1) * target_masks.unsqueeze(-2)) > 0.5
    pair_mask = torch.triu(pair_mask, diagonal=1)
    pair_mask = pair_mask & (target_diff.abs() > 1e-6)
    if not torch.any(pair_mask):
        zero = pred_scores.new_zeros(())
        return zero, zero

    target_sign = torch.sign(target_diff)
    pair_weight = target_diff.abs()
    pair_loss = F.softplus(-(target_sign * pred_diff))
    weighted_loss = (pair_loss * pair_weight).masked_select(pair_mask).sum() / pair_weight.masked_select(pair_mask).sum().clamp_min(1e-6)
    pair_acc = ((pred_diff * target_sign) > 0.0).float()
    weighted_acc = (pair_acc * pair_weight).masked_select(pair_mask).sum() / pair_weight.masked_select(pair_mask).sum().clamp_min(1e-6)
    return weighted_loss, weighted_acc


def planner_policy_metrics(
    pred_logits: torch.Tensor,
    target_scores: torch.Tensor,
    target_masks: torch.Tensor,
    *,
    target_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_rows = target_masks.sum(dim=-1) > 0.5
    if not torch.any(valid_rows):
        zero = pred_logits.new_zeros(())
        return zero, zero

    masked_target_scores = target_scores.masked_fill(target_masks <= 0.5, -1e9)
    target_probs = torch.softmax(masked_target_scores / float(target_temperature), dim=-1)
    masked_pred_logits = pred_logits.masked_fill(target_masks <= 0.5, -1e9)
    log_pred_probs = torch.log_softmax(masked_pred_logits, dim=-1)
    ce = -(target_probs * log_pred_probs).sum(dim=-1)

    target_best = masked_target_scores.argmax(dim=-1)
    pred_best = masked_pred_logits.argmax(dim=-1)
    top1 = (pred_best == target_best).float()
    return ce.masked_select(valid_rows).mean(), top1.masked_select(valid_rows).mean()


def _to_tensors(arrays: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "features": torch.tensor(arrays["features"], dtype=torch.float32, device=device),
        "actions": torch.tensor(arrays["actions"], dtype=torch.long, device=device),
        "tasks": torch.tensor(arrays["tasks"], dtype=torch.long, device=device),
        "rewards": torch.tensor(arrays["rewards"], dtype=torch.float32, device=device),
        "dones": torch.tensor(arrays["dones"], dtype=torch.float32, device=device),
        "values": torch.tensor(arrays["values"], dtype=torch.float32, device=device),
        "planner_action_scores": torch.tensor(arrays["planner_action_scores"], dtype=torch.float32, device=device),
        "planner_action_masks": torch.tensor(arrays["planner_action_masks"], dtype=torch.float32, device=device),
        "valid_actions": torch.tensor(arrays["valid_actions"], dtype=torch.float32, device=device),
    }


def summarize_sequence_world_model_outputs(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray | list[list[str]]],
    *,
    device: torch.device,
    planner_policy_target_temperature: float = 1.0,
) -> dict:
    tensors = _to_tensors(arrays, device)
    rollout_start = torch.tensor(arrays["rollout_start_indices"], dtype=torch.long, device=device)
    prompt_sequences = arrays.get("prompts")
    with torch.no_grad():
        outputs = model.forward_sequence(
            tensors["features"],
            tensors["actions"],
            tensors["tasks"],
            prompt_sequences=prompt_sequences,
            deterministic=True,
        )

    target_features = tensors["features"][:, 1:]
    pred_done_probs = torch.sigmoid(outputs["pred_done_logits"])
    max_horizon = int(tensors["rewards"].shape[1])

    rollout_pred_features = []
    rollout_pred_rewards = []
    rollout_pred_done_logits = []
    rollout_pred_done_probs = []
    rollout_pred_valid_logits = []
    rollout_pred_values = []
    rollout_target_features = []
    for batch_idx in range(target_features.shape[0]):
        start = int(rollout_start[batch_idx].item())
        end = start + max_horizon
        rollout_pred_features.append(outputs["pred_features"][batch_idx, start:end])
        rollout_pred_rewards.append(outputs["pred_rewards"][batch_idx, start:end])
        rollout_pred_done_logits.append(outputs["pred_done_logits"][batch_idx, start:end])
        rollout_pred_done_probs.append(pred_done_probs[batch_idx, start:end])
        rollout_pred_valid_logits.append(outputs["pred_action_valid_logits"][batch_idx, start:end])
        rollout_pred_values.append(outputs["pred_values"][batch_idx, start:end])
        rollout_target_features.append(target_features[batch_idx, start:end])
    rollout_pred_features = torch.stack(rollout_pred_features, dim=0)
    rollout_pred_rewards = torch.stack(rollout_pred_rewards, dim=0)
    rollout_pred_done_logits = torch.stack(rollout_pred_done_logits, dim=0)
    rollout_pred_done_probs = torch.stack(rollout_pred_done_probs, dim=0)
    rollout_pred_valid_logits = torch.stack(rollout_pred_valid_logits, dim=0)
    rollout_pred_values = torch.stack(rollout_pred_values, dim=0)
    rollout_target_features = torch.stack(rollout_target_features, dim=0)
    rollout_valid_targets = tensors["valid_actions"]

    horizon_metrics = {}
    for horizon in range(1, max_horizon + 1):
        horizon_metrics[str(horizon)] = {
            "feature_mse": float(F.mse_loss(rollout_pred_features[:, horizon - 1], rollout_target_features[:, horizon - 1]).item()),
            "reward_mae": float(
                torch.mean(torch.abs(rollout_pred_rewards[:, horizon - 1] - tensors["rewards"][:, horizon - 1])).item()
            ),
            "done_brier": float(
                torch.mean((rollout_pred_done_probs[:, horizon - 1] - tensors["dones"][:, horizon - 1]) ** 2).item()
            ),
        }

    flattened_done_labels = tensors["dones"].reshape(-1).cpu().numpy()
    flattened_done_logits = rollout_pred_done_logits.reshape(-1).cpu().numpy()
    flattened_done_probs = rollout_pred_done_probs.reshape(-1).cpu().numpy()
    calibration = fit_temperature_for_binary_logits(flattened_done_labels, flattened_done_logits)
    planner_pairwise_loss, planner_pairwise_accuracy = planner_pairwise_metrics(
        outputs["planner_action_logits"],
        tensors["planner_action_scores"],
        tensors["planner_action_masks"],
    )
    planner_policy_ce, planner_policy_top1 = planner_policy_metrics(
        outputs["planner_action_logits"],
        tensors["planner_action_scores"],
        tensors["planner_action_masks"],
        target_temperature=planner_policy_target_temperature,
    )

    return {
        "num_examples": int(tensors["features"].shape[0]),
        "context_len": int(rollout_start[0].item() + 1),
        "rollout_horizon": int(tensors["rewards"].shape[1]),
        "feature_mse": float(F.mse_loss(rollout_pred_features, rollout_target_features).item()),
        "feature_mae": float(torch.mean(torch.abs(rollout_pred_features - rollout_target_features)).item()),
        "reward_mae": float(torch.mean(torch.abs(rollout_pred_rewards - tensors["rewards"])).item()),
        "value_mae": float(torch.mean(torch.abs(rollout_pred_values - tensors["values"])).item()),
        "planner_action_mae": float(
            (
                torch.abs(outputs["planner_action_logits"] - tensors["planner_action_scores"]) * tensors["planner_action_masks"]
            ).sum().item()
            / max(1.0, float(tensors["planner_action_masks"].sum().item()))
        ),
        "planner_policy_ce": float(planner_policy_ce.item()),
        "planner_policy_top1": float(planner_policy_top1.item()),
        "planner_policy_target_temperature": float(planner_policy_target_temperature),
        "planner_pairwise_loss": float(planner_pairwise_loss.item()),
        "planner_pairwise_accuracy": float(planner_pairwise_accuracy.item()),
        "done_brier": float(torch.mean((rollout_pred_done_probs - tensors["dones"]) ** 2).item()),
        "action_valid_auc_proxy": float(
            torch.mean(torch.sigmoid(rollout_pred_valid_logits) * rollout_valid_targets).item()
        ),
        "kl_mean": float(
            0.5
            * torch.mean(
                outputs["prior_logvars"]
                - outputs["posterior_logvars"][:, 1:]
                + (
                    torch.exp(outputs["posterior_logvars"][:, 1:])
                    + (outputs["posterior_means"][:, 1:] - outputs["prior_means"]) ** 2
                )
                / torch.exp(outputs["prior_logvars"])
                - 1.0
            ).item()
        ),
        "overshooting_kl_mean": float(
            torch.stack(
                [
                    (
                        0.5
                        * torch.mean(
                            outputs["prior_logvars"][:, start + delta - 1]
                            - outputs["posterior_logvars"][:, start + delta]
                            + (
                                torch.exp(outputs["posterior_logvars"][:, start + delta])
                                + (
                                    outputs["posterior_means"][:, start + delta]
                                    - outputs["prior_means"][:, start + delta - 1]
                                )
                                    ** 2
                                )
                                / torch.exp(outputs["prior_logvars"][:, start + delta - 1])
                            - 1.0
                        )
                    )
                    for start in range(max(0, outputs["prior_means"].shape[1] - 1))
                    for delta in range(1, min(3, outputs["prior_means"].shape[1] - start) + 1)
                ]
            ).mean().item()
        ) if outputs["prior_means"].shape[1] > 1 else 0.0,
        "done_ece": float(expected_calibration_error(flattened_done_labels, flattened_done_probs)),
        "done_temperature": calibration["temperature"],
        "done_ece_after_temperature": calibration["ece_after"],
        "horizon_metrics": horizon_metrics,
    }


def evaluate_sequence_world_model(
    model_path: str,
    trace_path: str,
    *,
    context_len: int = 4,
    rollout_horizon: int = 8,
    observation_version: str | None = None,
) -> dict:
    rows = load_trace_rows(trace_path)
    examples = build_sequence_world_model_examples(
        rows,
        context_len=context_len,
        rollout_horizon=rollout_horizon,
        observation_version=observation_version,
    )
    arrays = sequence_examples_to_arrays(examples)
    inference = load_sequence_world_model(model_path)
    planner_policy_target_temperature = float(getattr(inference.model, "_metadata", {}).get("planner_policy_target_temperature", 1.0))
    result = summarize_sequence_world_model_outputs(
        inference.model,
        arrays,
        device=inference.device,
        planner_policy_target_temperature=planner_policy_target_temperature,
    )
    result.update(
        {
            "model_path": model_path,
            "trace_path": trace_path,
            "observation_version": examples[0]["observation_version"] if examples else observation_version or "unknown",
        }
    )
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a sequence world model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--context-len", type=int, default=4)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--observation-version", type=str, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = evaluate_sequence_world_model(
        args.model,
        args.input,
        context_len=args.context_len,
        rollout_horizon=args.rollout_horizon,
        observation_version=args.observation_version,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
