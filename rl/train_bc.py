from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from rl.bc_model import BCPolicyMLP, save_bc_model
from rl.feature_encoder import ACTION_SET


def load_trace_rows(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def train_bc_model(
    rows: list[dict],
    output_path: str,
    epochs: int = 20,
    lr: float = 1e-3,
    hidden_size: int = 256,
    observation_version: str = "v1",
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

    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x)
        logits = logits.masked_fill(action_masks <= 0, -1e9)
        loss = F.cross_entropy(logits, y)
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
    }
    save_bc_model(model, output_path, metadata=metadata)
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a behavior cloning policy from multi-turn traces")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--observation-version", type=str, default="v1")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = load_trace_rows(args.input)
    result = train_bc_model(
        rows,
        args.output,
        epochs=args.epochs,
        lr=args.lr,
        hidden_size=args.hidden_size,
        observation_version=args.observation_version,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
