from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from rl.io_utils import atomic_write_json
from rl.sequence_world_model import SequenceWorldModel, save_sequence_world_model
from rl.sequence_world_model_dataset import build_sequence_world_model_examples, sequence_examples_to_arrays
from rl.sequence_world_model_eval import planner_pairwise_metrics, planner_policy_metrics, summarize_sequence_world_model_outputs
from rl.train_bc import load_trace_rows


def _episode_split(rows: list[dict], validation_fraction: float) -> tuple[list[dict], list[dict]]:
    if not rows:
        return [], []
    episode_ids = sorted({str(row["episode_id"]) for row in rows})
    val_count = max(1, int(math.ceil(len(episode_ids) * validation_fraction))) if len(episode_ids) > 1 else 0
    val_ids = set(episode_ids[-val_count:]) if val_count else set()
    train_rows = [row for row in rows if str(row["episode_id"]) not in val_ids]
    val_rows = [row for row in rows if str(row["episode_id"]) in val_ids]
    if not train_rows:
        train_rows = rows
        val_rows = rows
    elif not val_rows:
        val_rows = train_rows
    return train_rows, val_rows


def _tensorize(arrays: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
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


def _kl_normal(post_mean, post_logvar, prior_mean, prior_logvar) -> torch.Tensor:
    return 0.5 * (
        prior_logvar
        - post_logvar
        + (torch.exp(post_logvar) + (post_mean - prior_mean) ** 2) / torch.exp(prior_logvar)
        - 1.0
    )


def _latent_overshooting_loss(
    model: SequenceWorldModel,
    outputs: dict[str, torch.Tensor],
    actions: torch.Tensor,
    tasks: torch.Tensor,
    *,
    overshooting_distance: int,
) -> torch.Tensor:
    if overshooting_distance <= 1:
        return torch.zeros((), dtype=actions.dtype if actions.is_floating_point() else torch.float32, device=actions.device)

    batch_size, transition_len = actions.shape
    losses = []
    for start in range(transition_len - 1):
        hidden = outputs["hidden_states"][:, start].detach()
        latent = outputs["posterior_latents"][:, start].detach()
        max_delta = min(int(overshooting_distance), transition_len - start)
        for delta in range(1, max_delta + 1):
            step = start + delta - 1
            imagined = model.imagine_step(hidden, latent, actions[:, step], tasks[:, step], deterministic=False)
            hidden = imagined["hidden"]
            latent = imagined["prior_latent"]
            target_idx = start + delta
            losses.append(
                _kl_normal(
                    outputs["posterior_means"][:, target_idx].detach(),
                    outputs["posterior_logvars"][:, target_idx].detach(),
                    imagined["prior_mean"],
                    imagined["prior_logvar"],
                ).mean()
            )
    if not losses:
        return torch.zeros((), dtype=torch.float32, device=actions.device)
    return torch.stack(losses).mean()


def _selection_score(summary: dict, metric: str) -> float:
    if metric == "feature_mse":
        return float(summary["feature_mse"])
    if metric == "reward_mae":
        return float(summary["reward_mae"])
    if metric == "value_mae":
        return float(summary["value_mae"])
    if metric == "planner_action_mae":
        return float(summary["planner_action_mae"])
    if metric == "planner_policy_ce":
        return float(summary["planner_policy_ce"])
    if metric == "planner_pairwise_loss":
        return float(summary["planner_pairwise_loss"])
    if metric == "planning_proxy":
        return float(summary["reward_mae"]) + 0.5 * float(summary["value_mae"]) + float(summary["feature_mse"])
    if metric == "planner_trace_proxy":
        return (
            float(summary["planner_action_mae"])
            + 0.5 * float(summary["reward_mae"])
            + 0.25 * float(summary["value_mae"])
            + 0.25 * float(summary["feature_mse"])
        )
    if metric == "planner_rank_proxy":
        return (
            float(summary["planner_pairwise_loss"])
            + 0.5 * float(summary["planner_action_mae"])
            + 0.25 * float(summary["reward_mae"])
            + 0.25 * float(summary["feature_mse"])
        )
    if metric == "planner_policy_proxy":
        return (
            float(summary["planner_policy_ce"])
            + 0.5 * float(summary["planner_action_mae"])
            + 0.25 * float(summary["reward_mae"])
            + 0.25 * float(summary["feature_mse"])
        )
    if metric == "planner_compromise_proxy":
        # Keep planner-policy pressure, but anchor checkpoint choice to predictive quality.
        return (
            0.5 * float(summary["planner_policy_ce"])
            + 0.25 * float(summary["planner_action_mae"])
            + float(summary["reward_mae"])
            + float(summary["feature_mse"])
            + 0.25 * float(summary["value_mae"])
        )
    raise ValueError(f"Unsupported selection metric: {metric}")


def _apply_uncertainty_weight(loss: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    precision = torch.exp(-log_var)
    return precision * loss + log_var


def train_sequence_world_model(
    rows: list[dict],
    output_path: str,
    *,
    context_len: int = 4,
    rollout_horizon: int = 8,
    epochs: int = 20,
    lr: float = 1e-3,
    hidden_size: int = 128,
    latent_dim: int = 64,
    observation_version: str | None = None,
    validation_fraction: float = 0.2,
    reward_loss_coef: float = 1.0,
    value_loss_coef: float = 0.5,
    planner_action_loss_coef: float = 0.25,
    planner_rank_loss_coef: float = 0.0,
    planner_policy_loss_coef: float = 0.0,
    planner_policy_target_temperature: float = 1.0,
    planner_policy_warmup_epochs: int = 0,
    adaptive_loss_balance: bool = False,
    done_loss_coef: float = 0.5,
    feature_loss_coef: float = 1.0,
    kl_loss_coef: float = 0.1,
    overshooting_loss_coef: float = 0.0,
    overshooting_distance: int = 3,
    valid_action_loss_coef: float = 0.25,
    selection_metric: str = "feature_mse",
    text_encoder_backend: str = "none",
    text_embedding_dim: int = 64,
) -> dict:
    train_rows, val_rows = _episode_split(rows, validation_fraction)
    train_examples = build_sequence_world_model_examples(
        train_rows,
        context_len=context_len,
        rollout_horizon=rollout_horizon,
        discount=0.99,
        observation_version=observation_version,
    )
    val_examples = build_sequence_world_model_examples(
        val_rows,
        context_len=context_len,
        rollout_horizon=rollout_horizon,
        discount=0.99,
        observation_version=observation_version,
    )
    train_arrays = sequence_examples_to_arrays(train_examples)
    val_arrays = sequence_examples_to_arrays(val_examples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceWorldModel(
        input_dim=int(train_arrays["features"].shape[-1]),
        latent_dim=latent_dim,
        hidden_size=hidden_size,
        text_encoder_backend=text_encoder_backend,
        text_embedding_dim=text_embedding_dim,
    ).to(device)
    loss_balance_params = None
    if adaptive_loss_balance:
        loss_balance_params = nn.ParameterDict(
            {
                "feature": nn.Parameter(torch.zeros((), device=device)),
                "reward": nn.Parameter(torch.zeros((), device=device)),
                "value": nn.Parameter(torch.zeros((), device=device)),
                "planner_action": nn.Parameter(torch.zeros((), device=device)),
                "planner_policy": nn.Parameter(torch.zeros((), device=device)),
                "done": nn.Parameter(torch.zeros((), device=device)),
                "valid_action": nn.Parameter(torch.zeros((), device=device)),
            }
        )
        optimizer = torch.optim.Adam(list(model.parameters()) + list(loss_balance_params.parameters()), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_tensors = _tensorize(train_arrays, device)
    rollout_start_indices = torch.tensor(train_arrays["rollout_start_indices"], dtype=torch.long, device=device)
    epoch_summaries = []
    best_state = None
    best_selection_score = float("inf")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model.forward_sequence(
            train_tensors["features"],
            train_tensors["actions"],
            train_tensors["tasks"],
            prompt_sequences=train_arrays.get("prompts"),
            deterministic=False,
        )
        target_features = train_tensors["features"][:, 1:]
        horizon = train_tensors["rewards"].shape[1]
        rollout_pred_features = []
        rollout_target_features = []
        rollout_pred_rewards = []
        rollout_pred_done_logits = []
        rollout_pred_valid_logits = []
        rollout_pred_values = []
        for batch_idx in range(target_features.shape[0]):
            start = int(rollout_start_indices[batch_idx].item())
            end = start + int(horizon)
            rollout_pred_features.append(outputs["pred_features"][batch_idx, start:end])
            rollout_target_features.append(target_features[batch_idx, start:end])
            rollout_pred_rewards.append(outputs["pred_rewards"][batch_idx, start:end])
            rollout_pred_done_logits.append(outputs["pred_done_logits"][batch_idx, start:end])
            rollout_pred_valid_logits.append(outputs["pred_action_valid_logits"][batch_idx, start:end])
            rollout_pred_values.append(outputs["pred_values"][batch_idx, start:end])
        rollout_pred_features = torch.stack(rollout_pred_features, dim=0)
        rollout_target_features = torch.stack(rollout_target_features, dim=0)
        rollout_pred_rewards = torch.stack(rollout_pred_rewards, dim=0)
        rollout_pred_done_logits = torch.stack(rollout_pred_done_logits, dim=0)
        rollout_pred_valid_logits = torch.stack(rollout_pred_valid_logits, dim=0)
        rollout_pred_values = torch.stack(rollout_pred_values, dim=0)
        rollout_valid_targets = train_tensors["valid_actions"]

        feature_loss = F.mse_loss(rollout_pred_features, rollout_target_features)
        reward_loss = F.mse_loss(rollout_pred_rewards, train_tensors["rewards"])
        value_loss = F.mse_loss(rollout_pred_values, train_tensors["values"])
        planner_action_mask = train_tensors["planner_action_masks"]
        planner_action_error = (outputs["planner_action_logits"] - train_tensors["planner_action_scores"]) ** 2
        planner_action_loss = planner_action_error.mul(planner_action_mask).sum() / planner_action_mask.sum().clamp_min(1.0)
        planner_rank_loss, planner_pairwise_accuracy = planner_pairwise_metrics(
            outputs["planner_action_logits"],
            train_tensors["planner_action_scores"],
            train_tensors["planner_action_masks"],
        )
        planner_policy_loss, planner_policy_top1 = planner_policy_metrics(
            outputs["planner_action_logits"],
            train_tensors["planner_action_scores"],
            train_tensors["planner_action_masks"],
            target_temperature=planner_policy_target_temperature,
        )
        if planner_policy_warmup_epochs > 0:
            policy_loss_scale = min(1.0, float(epoch + 1) / float(planner_policy_warmup_epochs))
        else:
            policy_loss_scale = 1.0
        done_loss = F.binary_cross_entropy_with_logits(rollout_pred_done_logits, train_tensors["dones"])
        valid_action_loss = F.binary_cross_entropy_with_logits(rollout_pred_valid_logits, rollout_valid_targets)
        kl_loss = _kl_normal(
            outputs["posterior_means"][:, 1:],
            outputs["posterior_logvars"][:, 1:],
            outputs["prior_means"],
            outputs["prior_logvars"],
        ).mean()
        overshooting_loss = _latent_overshooting_loss(
            model,
            outputs,
            train_tensors["actions"],
            train_tensors["tasks"],
            overshooting_distance=overshooting_distance,
        )
        if adaptive_loss_balance and loss_balance_params is not None:
            loss = (
                _apply_uncertainty_weight(feature_loss_coef * feature_loss, loss_balance_params["feature"])
                + _apply_uncertainty_weight(reward_loss_coef * reward_loss, loss_balance_params["reward"])
                + _apply_uncertainty_weight(value_loss_coef * value_loss, loss_balance_params["value"])
                + _apply_uncertainty_weight(planner_action_loss_coef * planner_action_loss, loss_balance_params["planner_action"])
                + _apply_uncertainty_weight(
                    (planner_policy_loss_coef * policy_loss_scale) * planner_policy_loss,
                    loss_balance_params["planner_policy"],
                )
                + _apply_uncertainty_weight(done_loss_coef * done_loss, loss_balance_params["done"])
                + _apply_uncertainty_weight(valid_action_loss_coef * valid_action_loss, loss_balance_params["valid_action"])
                + planner_rank_loss_coef * planner_rank_loss
                + kl_loss_coef * kl_loss
                + overshooting_loss_coef * overshooting_loss
            )
        else:
            loss = (
                feature_loss_coef * feature_loss
                + reward_loss_coef * reward_loss
                + value_loss_coef * value_loss
                + planner_action_loss_coef * planner_action_loss
                + planner_rank_loss_coef * planner_rank_loss
                + (planner_policy_loss_coef * policy_loss_scale) * planner_policy_loss
                + done_loss_coef * done_loss
                + kl_loss_coef * kl_loss
                + overshooting_loss_coef * overshooting_loss
                + valid_action_loss_coef * valid_action_loss
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        val_summary = summarize_sequence_world_model_outputs(
            model,
            val_arrays,
            device=device,
            planner_policy_target_temperature=planner_policy_target_temperature,
        )
        current_selection_score = _selection_score(val_summary, selection_metric)
        if current_selection_score < best_selection_score:
            best_selection_score = current_selection_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        epoch_summaries.append(
            {
                "epoch": epoch + 1,
                "loss": float(loss.item()),
                "feature_loss": float(feature_loss.item()),
                "reward_loss": float(reward_loss.item()),
                "value_loss": float(value_loss.item()),
                "planner_action_loss": float(planner_action_loss.item()),
                "planner_rank_loss": float(planner_rank_loss.item()),
                "planner_pairwise_accuracy": float(planner_pairwise_accuracy.item()),
                "planner_policy_loss": float(planner_policy_loss.item()),
                "planner_policy_top1": float(planner_policy_top1.item()),
                "planner_policy_loss_scale": float(policy_loss_scale),
                "adaptive_loss_balance": bool(adaptive_loss_balance),
                "loss_balance_log_vars": (
                    {name: float(param.detach().item()) for name, param in loss_balance_params.items()}
                    if loss_balance_params is not None
                    else {}
                ),
                "done_loss": float(done_loss.item()),
                "valid_action_loss": float(valid_action_loss.item()),
                "kl_loss": float(kl_loss.item()),
                "overshooting_loss": float(overshooting_loss.item()),
                "val_feature_mse": float(val_summary["feature_mse"]),
                "val_reward_mae": float(val_summary["reward_mae"]),
                "val_value_mae": float(val_summary["value_mae"]),
                "val_planner_policy_ce": float(val_summary["planner_policy_ce"]),
                "val_planner_policy_top1": float(val_summary["planner_policy_top1"]),
                "val_planner_pairwise_loss": float(val_summary["planner_pairwise_loss"]),
                "val_planner_pairwise_accuracy": float(val_summary["planner_pairwise_accuracy"]),
                "val_done_brier": float(val_summary["done_brier"]),
                "selection_score": float(current_selection_score),
            }
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_summary = summarize_sequence_world_model_outputs(
        model,
        train_arrays,
        device=device,
        planner_policy_target_temperature=planner_policy_target_temperature,
    )
    val_summary = summarize_sequence_world_model_outputs(
        model,
        val_arrays,
        device=device,
        planner_policy_target_temperature=planner_policy_target_temperature,
    )
    metadata = {
        "model_type": "sequence_world_model",
        "input_dim": int(train_arrays["features"].shape[-1]),
        "latent_dim": latent_dim,
        "hidden_size": hidden_size,
        "context_len": context_len,
        "rollout_horizon": rollout_horizon,
        "observation_version": train_examples[0]["observation_version"],
        "num_train_examples": len(train_examples),
        "num_val_examples": len(val_examples),
        "epochs": epochs,
        "learning_rate": lr,
        "text_encoder_backend": text_encoder_backend,
        "text_embedding_dim": text_embedding_dim,
        "reward_loss_coef": reward_loss_coef,
        "value_loss_coef": value_loss_coef,
        "planner_action_loss_coef": planner_action_loss_coef,
        "planner_rank_loss_coef": planner_rank_loss_coef,
        "planner_policy_loss_coef": planner_policy_loss_coef,
        "planner_policy_target_temperature": planner_policy_target_temperature,
        "planner_policy_warmup_epochs": planner_policy_warmup_epochs,
        "adaptive_loss_balance": adaptive_loss_balance,
        "loss_balance_log_vars": (
            {name: float(param.detach().item()) for name, param in loss_balance_params.items()}
            if loss_balance_params is not None
            else {}
        ),
        "done_loss_coef": done_loss_coef,
        "feature_loss_coef": feature_loss_coef,
        "kl_loss_coef": kl_loss_coef,
        "overshooting_loss_coef": overshooting_loss_coef,
        "overshooting_distance": overshooting_distance,
        "valid_action_loss_coef": valid_action_loss_coef,
        "selection_metric": selection_metric,
        "train_summary": train_summary,
        "val_summary": val_summary,
        "epoch_summaries": epoch_summaries,
    }
    save_sequence_world_model(model, output_path, metadata=metadata)
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a sequence world model")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-output", default=None)
    parser.add_argument("--context-len", type=int, default=4)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--observation-version", type=str, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--reward-loss-coef", type=float, default=1.0)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--planner-action-loss-coef", type=float, default=0.25)
    parser.add_argument("--planner-rank-loss-coef", type=float, default=0.0)
    parser.add_argument("--planner-policy-loss-coef", type=float, default=0.0)
    parser.add_argument("--planner-policy-target-temperature", type=float, default=1.0)
    parser.add_argument("--planner-policy-warmup-epochs", type=int, default=0)
    parser.add_argument("--adaptive-loss-balance", action="store_true")
    parser.add_argument("--done-loss-coef", type=float, default=0.5)
    parser.add_argument("--feature-loss-coef", type=float, default=1.0)
    parser.add_argument("--kl-loss-coef", type=float, default=0.1)
    parser.add_argument("--overshooting-loss-coef", type=float, default=0.0)
    parser.add_argument("--overshooting-distance", type=int, default=3)
    parser.add_argument("--valid-action-loss-coef", type=float, default=0.25)
    parser.add_argument("--selection-metric", choices=["feature_mse", "reward_mae", "value_mae", "planner_action_mae", "planner_policy_ce", "planner_pairwise_loss", "planning_proxy", "planner_trace_proxy", "planner_rank_proxy", "planner_policy_proxy", "planner_compromise_proxy"], default="feature_mse")
    parser.add_argument("--text-encoder-backend", choices=["none", "hash"], default="none")
    parser.add_argument("--text-embedding-dim", type=int, default=64)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = load_trace_rows(args.input)
    result = train_sequence_world_model(
        rows,
        args.output,
        context_len=args.context_len,
        rollout_horizon=args.rollout_horizon,
        epochs=args.epochs,
        lr=args.lr,
        hidden_size=args.hidden_size,
        latent_dim=args.latent_dim,
        observation_version=args.observation_version,
        validation_fraction=args.validation_fraction,
        reward_loss_coef=args.reward_loss_coef,
        value_loss_coef=args.value_loss_coef,
        planner_action_loss_coef=args.planner_action_loss_coef,
        planner_rank_loss_coef=args.planner_rank_loss_coef,
        planner_policy_loss_coef=args.planner_policy_loss_coef,
        planner_policy_target_temperature=args.planner_policy_target_temperature,
        planner_policy_warmup_epochs=args.planner_policy_warmup_epochs,
        adaptive_loss_balance=args.adaptive_loss_balance,
        done_loss_coef=args.done_loss_coef,
        feature_loss_coef=args.feature_loss_coef,
        kl_loss_coef=args.kl_loss_coef,
        overshooting_loss_coef=args.overshooting_loss_coef,
        overshooting_distance=args.overshooting_distance,
        valid_action_loss_coef=args.valid_action_loss_coef,
        selection_metric=args.selection_metric,
        text_encoder_backend=args.text_encoder_backend,
        text_embedding_dim=args.text_embedding_dim,
    )
    if args.report_output:
        atomic_write_json(args.report_output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
