from __future__ import annotations

import argparse
import json
import os
from statistics import mean

import torch
import torch.nn.functional as F

from rl.reward_model import RewardMLP, load_preference_rows, save_reward_model
from src.task_harness import run_task_episode


def generate_preference_rows(task: str, seeds: list[int], max_steps: int) -> list[dict]:
    rows = []
    for seed in seeds:
        episode = run_task_episode(seed=seed, task=task, max_steps=max_steps, policy="task_greedy")
        for step_row in episode["trajectory"]:
            planner_trace = step_row.get("planner_trace") or []
            if len(planner_trace) < 2:
                continue
            scored = sorted(planner_trace, key=lambda row: row["total"])
            chosen = scored[-1]
            rejected = scored[0]
            if chosen["total"] <= rejected["total"]:
                continue
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "step": step_row["step"],
                    "chosen_action": chosen["action"],
                    "rejected_action": rejected["action"],
                    "chosen_score": chosen["total"],
                    "rejected_score": rejected["total"],
                    "chosen_features": chosen["reward_features"],
                    "rejected_features": rejected["reward_features"],
                }
            )
    return rows


def write_preference_rows(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def train_reward_model(rows: list[dict], output_path: str, epochs: int = 20, lr: float = 1e-3) -> dict:
    if not rows:
        raise ValueError("No reward preference rows to train on")

    device = torch.device("cpu")
    model = RewardMLP()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    chosen = torch.tensor([row["chosen_features"] for row in rows], dtype=torch.float32, device=device)
    rejected = torch.tensor([row["rejected_features"] for row in rows], dtype=torch.float32, device=device)

    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        chosen_scores = model(chosen)
        rejected_scores = model(rejected)
        loss = F.softplus(-(chosen_scores - rejected_scores)).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    metadata = {
        "epochs": epochs,
        "learning_rate": lr,
        "num_pairs": len(rows),
        "final_loss": losses[-1],
    }
    save_reward_model(model, output_path, metadata=metadata)
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a learned reward model from task-harness preferences")
    parser.add_argument("--task", type=str, default="explore")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46,47")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--dataset-output", type=str, default=None)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.input:
        rows = load_preference_rows(args.input)
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        rows = generate_preference_rows(args.task, seeds, args.max_steps)
        if args.dataset_output:
            write_preference_rows(rows, args.dataset_output)

    result = train_reward_model(rows, args.output, epochs=args.epochs, lr=args.lr)
    result["num_pairs"] = len(rows)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
