"""
Closed-loop style debugging helpers for tiny golden episodes.

These helpers do not assume that `env.reset(seed=...)` reproduces identical
NetHack dungeons across processes. Instead, they save the exact prompts,
actions, targets, and observation hashes from one tiny reference episode so
model behavior can be checked step-by-step against the saved distribution.
"""

import hashlib
import json
from pathlib import Path
from typing import Callable, Optional

import nle.env

from nle_agent.agent_http import _build_action_map
from src.data_generator import build_messages, wall_avoidance_policy
from src.evaluator import compute_accuracy, evaluate_model, hash_messages, parse_prediction
from src.state_encoder import StateEncoder


def _hash_obs(obs: dict) -> str:
    """Hash the stable observation fields we rely on for debugging."""
    digest = hashlib.sha256()
    for key in ("chars", "blstats", "message"):
        digest.update(key.encode())
        digest.update(obs[key].tobytes())
    return digest.hexdigest()


def build_golden_episode(
    seed: int,
    max_steps: int,
    encoder: StateEncoder,
    output_path: str,
    policy: Optional[Callable] = None,
) -> dict:
    """Record a tiny reference episode with prompts, actions, targets, and hashes."""
    if policy is None:
        policy = wall_avoidance_policy

    env = nle.env.NLE()
    obs, _ = env.reset(seed=seed)
    action_map = _build_action_map()
    import random
    rng = random.Random(seed)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output.open("w") as f:
        for step in range(max_steps):
            state = encoder.encode_full(obs)
            action_name = policy(state["adjacent"], rng)
            action_idx = action_map.get(action_name, action_map.get("wait", 18))
            if action_name not in action_map:
                action_name = "wait"

            prompt_text = encoder.format_prompt(state, action_name)
            messages = build_messages(prompt_text)
            obs_after, _, terminated, truncated, _ = env.step(action_idx)
            delta = encoder.encode_delta(obs, obs_after, action_name)
            target_text = encoder.format_target(delta)

            record = {
                "seed": seed,
                "step": step,
                "action": action_name,
                "messages": messages,
                "prompt": prompt_text,
                "target": target_text,
                "ground_truth_delta": delta,
                "obs_hash": _hash_obs(obs),
                "next_obs_hash": _hash_obs(obs_after),
                "message_hash": hash_messages(messages),
            }
            f.write(json.dumps(record) + "\n")
            count += 1
            obs = obs_after

            if terminated or truncated:
                break

    env.close()
    return {"path": str(output), "examples": count, "seed": seed}


def load_golden_episode(path: str) -> list[dict]:
    """Load a golden episode JSONL file."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_golden_episode(
    path: str,
    server_url: str,
    model_name_or_path: str = "llama-server",
    max_samples: Optional[int] = None,
) -> dict:
    """Run the model against a saved golden episode and compare step-by-step."""
    rows = load_golden_episode(path)
    if max_samples is not None:
        rows = rows[:max_samples]

    eval_result = evaluate_model(
        model_name_or_path=model_name_or_path,
        test_data=rows,
        server_url=server_url,
        max_samples=max_samples,
    )

    comparisons = []
    ground_truth = []
    for idx, row in enumerate(rows):
        gt = row["ground_truth_delta"]
        ground_truth.append(
            {
                "pos_delta": tuple(gt["pos_delta"]),
                "hp_delta": gt["hp_delta"],
                "gold_delta": gt["gold_delta"],
                "depth_delta": gt["depth_delta"],
                "survived": gt["survived"],
            }
        )

        pred = eval_result["predictions"][idx] if idx < len(eval_result["predictions"]) else parse_prediction("")
        comparisons.append(
            {
                "step": row["step"],
                "action": row["action"],
                "message_hash": row["message_hash"],
                "target": row["target"],
                "prediction": pred,
                "exact_match": (
                    pred.get("pos", (0, 0)) == tuple(gt["pos_delta"])
                    and pred.get("hp_delta", 0) == gt["hp_delta"]
                    and pred.get("gold_delta", 0) == gt["gold_delta"]
                    and pred.get("depth_delta", 0) == gt["depth_delta"]
                    and pred.get("survived", True) == gt["survived"]
                ),
            }
        )

    return {
        "golden_path": path,
        "server_available": eval_result["server_available"],
        "accuracy": compute_accuracy(eval_result["predictions"], ground_truth),
        "comparisons": comparisons,
        "errors": eval_result["errors"],
    }
