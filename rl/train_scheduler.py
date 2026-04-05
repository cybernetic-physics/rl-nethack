from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from rl.options import build_skill_registry
from rl.scheduler import RuleBasedScheduler, SchedulerContext
from rl.scheduler_model import (
    SKILL_NAMES,
    SchedulerMLP,
    encode_scheduler_features,
    save_scheduler_model,
    write_scheduler_dataset,
)
from src.memory_tracker import MemoryTracker
from src.state_encoder import StateEncoder
import nle.env


def generate_scheduler_rows(seeds: list[int], max_steps: int) -> list[dict]:
    registry = build_skill_registry()
    teacher = RuleBasedScheduler()
    encoder = StateEncoder()
    rows = []

    for seed in seeds:
        env = nle.env.NLE()
        obs, info = env.reset(seed=seed)
        del info
        memory = MemoryTracker()
        memory.update(obs)
        memory.detect_rooms()
        active_skill = "explore"
        steps_in_skill = 0
        for step in range(max_steps):
            state = encoder.encode_full(obs)
            px, py = state["position"]
            tile_char = chr(int(obs["chars"][py, px])) if px >= 0 and py >= 0 else " "
            state["standing_on_down_stairs"] = tile_char == ">"
            state["standing_on_up_stairs"] = tile_char == "<"
            available = [
                name for name, option in registry.items()
                if option.can_start(state, memory)
            ] or ["explore"]
            label = teacher.select_skill(
                SchedulerContext(
                    state=state,
                    memory=memory,
                    active_skill=active_skill,
                    steps_in_skill=steps_in_skill,
                    available_skills=available,
                )
            )
            rows.append(
                {
                    "seed": seed,
                    "step": step,
                    "features": encode_scheduler_features(
                        state=state,
                        memory=memory,
                        active_skill=active_skill,
                        steps_in_skill=steps_in_skill,
                        available_skills=available,
                    ).tolist(),
                    "label": label,
                    "available_skills": available,
                }
            )
            action_name = registry[label].allowed_actions(state, memory)[0]
            from nle_agent.agent_http import _build_action_map

            action_map = _build_action_map()
            obs, reward, terminated, truncated, _ = env.step(action_map.get(action_name, action_map["wait"]))
            memory.update(obs)
            memory.detect_rooms()
            if label == active_skill:
                steps_in_skill += 1
            else:
                active_skill = label
                steps_in_skill = 1
            if terminated or truncated:
                break
        env.close()
    return rows


def train_scheduler_model(rows: list[dict], output_path: str, epochs: int = 20, lr: float = 1e-3) -> dict:
    if not rows:
        raise ValueError("No scheduler rows to train on")

    device = torch.device("cpu")
    model = SchedulerMLP()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x = torch.tensor([row["features"] for row in rows], dtype=torch.float32, device=device)
    y = torch.tensor([SKILL_NAMES.index(row["label"]) for row in rows], dtype=torch.long, device=device)

    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    preds = torch.argmax(model(x), dim=1)
    accuracy = float((preds == y).float().mean().item())
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    metadata = {
        "epochs": epochs,
        "learning_rate": lr,
        "num_examples": len(rows),
        "final_loss": losses[-1],
        "train_accuracy": accuracy,
    }
    save_scheduler_model(model, output_path, metadata=metadata)
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a learned scheduler from rule-based labels")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46,47")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--dataset-output", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = generate_scheduler_rows(seeds, args.max_steps)
    if args.dataset_output:
        write_scheduler_dataset(rows, args.dataset_output)
    result = train_scheduler_model(rows, args.output, epochs=args.epochs, lr=args.lr)
    result["num_examples"] = len(rows)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
