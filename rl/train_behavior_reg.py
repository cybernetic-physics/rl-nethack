from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from rl.bc_model import BCPolicyMLP, save_bc_model
from rl.feature_encoder import ACTION_SET
from rl.train_bc import load_trace_rows
from rl.trace_eval import evaluate_trace_policy


def train_behavior_regularized_policy(
    rows: list[dict],
    output_path: str,
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    hidden_size: int = 256,
    observation_version: str = "v1",
    behavior_coef: float = 0.1,
    temperature: float = 1.0,
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
    behavior_log_prior = torch.log(behavior_prior).to(device)

    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x) / max(1e-6, temperature)
        masked_logits = logits.masked_fill(action_masks <= 0, -1e9)
        imitation_loss = F.cross_entropy(masked_logits, y)
        behavior_reg = F.kl_div(
            F.log_softmax(masked_logits, dim=-1),
            behavior_prior.unsqueeze(0).expand_as(masked_logits),
            reduction="batchmean",
            log_target=False,
        )
        loss = imitation_loss + behavior_coef * behavior_reg
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

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
    }
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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = load_trace_rows(args.input)
    train_result = train_behavior_regularized_policy(
        rows,
        args.output,
        epochs=args.epochs,
        lr=args.lr,
        hidden_size=args.hidden_size,
        observation_version=args.observation_version,
        behavior_coef=args.behavior_coef,
        temperature=args.temperature,
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
