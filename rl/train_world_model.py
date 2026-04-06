from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from rl.io_utils import atomic_write_json
from rl.train_bc import load_trace_rows
from rl.world_model import TraceWorldModel, save_world_model
from rl.world_model_dataset import build_world_model_examples, examples_to_arrays


def train_world_model(
    rows: list[dict],
    output_path: str,
    *,
    horizon: int = 8,
    epochs: int = 20,
    lr: float = 1e-3,
    hidden_size: int = 256,
    latent_dim: int = 128,
    observation_version: str | None = None,
    reward_loss_coef: float = 1.0,
    done_loss_coef: float = 0.5,
    reconstruction_loss_coef: float = 1.0,
    action_loss_coef: float = 0.25,
) -> dict:
    examples = build_world_model_examples(rows, horizon=horizon, observation_version=observation_version)
    arrays = examples_to_arrays(examples)
    device = torch.device("cpu")
    model = TraceWorldModel(
        input_dim=arrays["features"].shape[1],
        latent_dim=latent_dim,
        hidden_size=hidden_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x = torch.tensor(arrays["features"], dtype=torch.float32, device=device)
    target_x = torch.tensor(arrays["target_features"], dtype=torch.float32, device=device)
    actions = torch.tensor(arrays["actions"], dtype=torch.long, device=device)
    tasks = torch.tensor(arrays["tasks"], dtype=torch.long, device=device)
    rewards = torch.tensor(arrays["rewards"], dtype=torch.float32, device=device)
    dones = torch.tensor(arrays["dones"], dtype=torch.float32, device=device)

    epoch_summaries = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x, actions, tasks)
        reconstruction_loss = F.mse_loss(outputs["current_features"], x)
        action_loss = F.cross_entropy(outputs["action_logits"], actions)
        feature_loss = F.mse_loss(outputs["future_features"], target_x)
        reward_loss = F.mse_loss(outputs["reward"], rewards)
        done_loss = F.binary_cross_entropy_with_logits(outputs["done_logit"], dones)
        loss = (
            feature_loss
            + reconstruction_loss_coef * reconstruction_loss
            + action_loss_coef * action_loss
            + reward_loss_coef * reward_loss
            + done_loss_coef * done_loss
        )
        loss.backward()
        optimizer.step()
        epoch_summaries.append(
            {
                "epoch": epoch + 1,
                "loss": float(loss.item()),
                "reconstruction_loss": float(reconstruction_loss.item()),
                "action_loss": float(action_loss.item()),
                "feature_loss": float(feature_loss.item()),
                "reward_loss": float(reward_loss.item()),
                "done_loss": float(done_loss.item()),
            }
        )

    with torch.no_grad():
        outputs = model(x, actions, tasks)
        latent_mean = outputs["latent"].mean(dim=0).cpu().tolist()
        latent_std = outputs["latent"].std(dim=0).clamp_min(1e-6).cpu().tolist()
        feature_mse = float(F.mse_loss(outputs["future_features"], target_x).item())
        reconstruction_mse = float(F.mse_loss(outputs["current_features"], x).item())
        action_accuracy = float((torch.argmax(outputs["action_logits"], dim=-1) == actions).float().mean().item())
        reward_mae = float(torch.mean(torch.abs(outputs["reward"] - rewards)).item())
        done_acc = float(
            ((torch.sigmoid(outputs["done_logit"]) >= 0.5) == (dones >= 0.5)).float().mean().item()
        )

    metadata = {
        "input_dim": arrays["features"].shape[1],
        "latent_dim": latent_dim,
        "hidden_size": hidden_size,
        "num_examples": len(examples),
        "epochs": epochs,
        "learning_rate": lr,
        "horizon": horizon,
        "observation_version": observation_version or examples[0]["observation_version"],
        "reward_loss_coef": reward_loss_coef,
        "done_loss_coef": done_loss_coef,
        "reconstruction_loss_coef": reconstruction_loss_coef,
        "action_loss_coef": action_loss_coef,
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "feature_mse": feature_mse,
        "reconstruction_mse": reconstruction_mse,
        "action_accuracy": action_accuracy,
        "reward_mae": reward_mae,
        "done_accuracy": done_acc,
        "epoch_summaries": epoch_summaries,
    }
    save_world_model(model, output_path, metadata=metadata)
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a short-horizon trace world model")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--report-output", type=str, default=None)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--observation-version", type=str, default=None)
    parser.add_argument("--reward-loss-coef", type=float, default=1.0)
    parser.add_argument("--done-loss-coef", type=float, default=0.5)
    parser.add_argument("--reconstruction-loss-coef", type=float, default=1.0)
    parser.add_argument("--action-loss-coef", type=float, default=0.25)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = load_trace_rows(args.input)
    result = train_world_model(
        rows,
        args.output,
        horizon=args.horizon,
        epochs=args.epochs,
        lr=args.lr,
        hidden_size=args.hidden_size,
        latent_dim=args.latent_dim,
        observation_version=args.observation_version,
        reward_loss_coef=args.reward_loss_coef,
        done_loss_coef=args.done_loss_coef,
        reconstruction_loss_coef=args.reconstruction_loss_coef,
        action_loss_coef=args.action_loss_coef,
    )
    if args.report_output:
        atomic_write_json(args.report_output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
