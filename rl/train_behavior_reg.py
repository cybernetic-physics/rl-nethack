from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from rl.bc_model import BCPolicyMLP, save_bc_model
from rl.feature_encoder import ACTION_SET
from rl.io_utils import atomic_torch_save
from rl.train_bc import load_trace_rows
from rl.trace_eval import evaluate_trace_policy


def _masked_behavior_targets(
    action_masks: torch.Tensor,
    behavior_prior: torch.Tensor,
) -> torch.Tensor:
    masked_prior = behavior_prior.unsqueeze(0) * action_masks
    row_sums = masked_prior.sum(dim=1, keepdim=True)
    fallback = action_masks / action_masks.sum(dim=1, keepdim=True).clamp_min(1.0)
    return torch.where(row_sums > 0, masked_prior / row_sums.clamp_min(1e-8), fallback)


def train_behavior_regularized_policy(
    rows: list[dict],
    output_path: str,
    *,
    heldout_trace_path: str | None = None,
    epochs: int = 20,
    lr: float = 1e-3,
    hidden_size: int = 256,
    observation_version: str = "v1",
    behavior_coef: float = 0.1,
    temperature: float = 1.0,
    class_balance_power: float = 0.0,
    teacher_action_boost: list[str] | None = None,
    teacher_action_boost_scale: float = 1.0,
) -> dict:
    if not rows:
        raise ValueError("No trace rows to train on")

    versions = {row.get("observation_version", observation_version) for row in rows}
    if len(versions) != 1:
        raise ValueError(f"Mixed observation versions in trace rows: {sorted(versions)}")
    trace_version = next(iter(versions))
    if trace_version != observation_version:
        raise ValueError(
            f"Requested observation_version={observation_version} but trace rows use {trace_version}"
        )

    input_dims = {len(row["feature_vector"]) for row in rows}
    if len(input_dims) != 1:
        raise ValueError(f"Mixed feature dimensions in trace rows: {sorted(input_dims)}")
    input_dim = len(rows[0]["feature_vector"])
    device = torch.device("cpu")
    model = BCPolicyMLP(input_dim=input_dim, hidden_size=hidden_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x = torch.tensor([row["feature_vector"] for row in rows], dtype=torch.float32, device=device)
    y = torch.tensor([ACTION_SET.index(row["action"]) for row in rows], dtype=torch.long, device=device)
    action_masks = torch.tensor(
        [[1.0 if name in row.get("allowed_actions", ACTION_SET) else 0.0 for name in ACTION_SET] for row in rows],
        dtype=torch.float32,
        device=device,
    )

    action_counts = torch.bincount(y, minlength=len(ACTION_SET)).float()
    behavior_prior = (action_counts / max(1.0, float(action_counts.sum()))).clamp_min(1e-8)
    behavior_targets = _masked_behavior_targets(action_masks, behavior_prior)
    class_weights = torch.ones(len(ACTION_SET), dtype=torch.float32, device=device)
    if class_balance_power > 0.0:
        normalized_counts = behavior_prior.clamp_min(1e-6)
        inverse = torch.pow(1.0 / normalized_counts, class_balance_power)
        class_weights = inverse / inverse.mean()
    boosted_actions = set(teacher_action_boost or [])
    for idx, name in enumerate(ACTION_SET):
        if name in boosted_actions:
            class_weights[idx] *= teacher_action_boost_scale

    best_payload = None
    best_score = float("-inf")
    epoch_summaries = []

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(x) / max(1e-6, temperature)
        masked_logits = logits.masked_fill(action_masks <= 0, -1e9)
        imitation_loss = F.cross_entropy(masked_logits, y, weight=class_weights)
        behavior_reg = F.kl_div(
            F.log_softmax(masked_logits, dim=-1),
            behavior_targets,
            reduction="batchmean",
            log_target=False,
        )
        loss = imitation_loss + behavior_coef * behavior_reg
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        train_preds = torch.argmax(model(x).masked_fill(action_masks <= 0, -1e9), dim=1)
        train_accuracy = float((train_preds == y).float().mean().item())
        epoch_summary = {
            "epoch": epoch + 1,
            "loss": float(loss.item()),
            "train_accuracy": train_accuracy,
        }

        candidate_path = output_path + ".tmp_eval"
        save_bc_model(
            model,
            candidate_path,
            metadata={
                "epochs": epoch + 1,
                "learning_rate": lr,
                "num_examples": len(rows),
                "input_dim": input_dim,
                "hidden_size": hidden_size,
                "observation_version": observation_version,
                "behavior_coef": behavior_coef,
                "temperature": temperature,
                "class_balance_power": class_balance_power,
                "teacher_action_boost": sorted(boosted_actions),
                "teacher_action_boost_scale": teacher_action_boost_scale,
            },
        )
        if heldout_trace_path:
            heldout_eval = evaluate_trace_policy(
                heldout_trace_path,
                "bc",
                bc_model_path=candidate_path,
                summary_only=True,
            )["summary"]
            epoch_summary["heldout_match_rate"] = heldout_eval["match_rate"]
            score = heldout_eval["match_rate"]
        else:
            score = train_accuracy

        epoch_summaries.append(epoch_summary)
        if score >= best_score:
            best_score = score
            best_payload = torch.load(candidate_path, map_location="cpu")
            best_payload["metadata"] = {
                **best_payload.get("metadata", {}),
                "selected_epoch": epoch + 1,
                "selection_metric": "heldout_match_rate" if heldout_trace_path else "train_accuracy",
                "selection_score": score,
            }
        if os.path.exists(candidate_path):
            os.remove(candidate_path)

    preds = torch.argmax(model(x).masked_fill(action_masks <= 0, -1e9), dim=1)
    accuracy = float((preds == y).float().mean().item())
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    metadata = {
        "epochs": epochs,
        "learning_rate": lr,
        "num_examples": len(rows),
        "input_dim": input_dim,
        "hidden_size": hidden_size,
        "observation_version": observation_version,
        "final_loss": losses[-1],
        "train_accuracy": accuracy,
        "behavior_coef": behavior_coef,
        "temperature": temperature,
        "behavior_action_prior": behavior_prior.tolist(),
        "class_balance_power": class_balance_power,
        "teacher_action_boost": sorted(boosted_actions),
        "teacher_action_boost_scale": teacher_action_boost_scale,
        "epoch_summaries": epoch_summaries,
    }
    if best_payload is not None:
        best_payload["metadata"] = {
            **metadata,
            **best_payload.get("metadata", {}),
        }
        atomic_torch_save(output_path, best_payload)
    else:
        save_bc_model(model, output_path, metadata=metadata)
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train an experimental behavior-regularized improver on trace data")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--heldout-input", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--observation-version", type=str, default="v1")
    parser.add_argument("--behavior-coef", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--class-balance-power", type=float, default=0.0)
    parser.add_argument("--teacher-action-boost", type=str, default="")
    parser.add_argument("--teacher-action-boost-scale", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = load_trace_rows(args.input)
    train_result = train_behavior_regularized_policy(
        rows,
        args.output,
        heldout_trace_path=args.heldout_input,
        epochs=args.epochs,
        lr=args.lr,
        hidden_size=args.hidden_size,
        observation_version=args.observation_version,
        behavior_coef=args.behavior_coef,
        temperature=args.temperature,
        class_balance_power=args.class_balance_power,
        teacher_action_boost=[x.strip() for x in args.teacher_action_boost.split(",") if x.strip()],
        teacher_action_boost_scale=args.teacher_action_boost_scale,
    )
    result = {"train": train_result}
    result["base_trace_eval"] = evaluate_trace_policy(args.input, "bc", bc_model_path=args.output, summary_only=True)["summary"]
    if args.heldout_input:
        result["heldout_trace_eval"] = evaluate_trace_policy(
            args.heldout_input,
            "bc",
            bc_model_path=args.output,
            summary_only=True,
        )["summary"]
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
