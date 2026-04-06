from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from rl.train_bc import load_trace_rows
from rl.world_model import load_world_model
from rl.world_model_dataset import build_world_model_examples, examples_to_arrays


def evaluate_world_model(model_path: str, trace_path: str, *, horizon: int = 8, observation_version: str | None = None) -> dict:
    rows = load_trace_rows(trace_path)
    examples = build_world_model_examples(rows, horizon=horizon, observation_version=observation_version)
    arrays = examples_to_arrays(examples)
    inference = load_world_model(model_path)
    model = inference.model
    device = inference.device

    x = torch.tensor(arrays["features"], dtype=torch.float32, device=device)
    target_x = torch.tensor(arrays["target_features"], dtype=torch.float32, device=device)
    actions = torch.tensor(arrays["actions"], dtype=torch.long, device=device)
    tasks = torch.tensor(arrays["tasks"], dtype=torch.long, device=device)
    rewards = torch.tensor(arrays["rewards"], dtype=torch.float32, device=device)
    dones = torch.tensor(arrays["dones"], dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = model(x, actions, tasks)
        feature_mse = float(F.mse_loss(outputs["future_features"], target_x).item())
        reward_mae = float(torch.mean(torch.abs(outputs["reward"] - rewards)).item())
        done_acc = float(
            ((torch.sigmoid(outputs["done_logit"]) >= 0.5) == (dones >= 0.5)).float().mean().item()
        )

    return {
        "model_path": model_path,
        "trace_path": trace_path,
        "horizon": horizon,
        "observation_version": observation_version or examples[0]["observation_version"],
        "num_examples": len(examples),
        "feature_mse": feature_mse,
        "reward_mae": reward_mae,
        "done_accuracy": done_acc,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a short-horizon trace world model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--observation-version", type=str, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = evaluate_world_model(
        args.model,
        args.input,
        horizon=args.horizon,
        observation_version=args.observation_version,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
