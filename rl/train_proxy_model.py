from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from rl.feature_encoder import action_name_to_index
from rl.io_utils import atomic_write_json
from rl.proxy_dataset import load_proxy_rows
from rl.proxy_eval import evaluate_proxy_rows
from rl.proxy_model import (
    DEFAULT_PROXY_REWARD_WEIGHTS,
    ProxyRewardModel,
    load_proxy_model,
    save_proxy_model,
)
from rl.proxy_report import build_proxy_report


REGRESSION_KEYS = [
    "k_step_progress",
    "k_step_survival",
    "k_step_loop_risk",
    "k_step_resource_value",
    "teacher_margin",
]


def _proxy_total_from_score(score: dict[str, float]) -> float:
    return float(score.get("raw_total", score["total"]))


def _rows_to_tensors(rows: list[dict], target_stats: dict | None = None) -> dict[str, torch.Tensor]:
    x = torch.tensor([row["feature_vector"] for row in rows], dtype=torch.float32)
    actions = torch.tensor([action_name_to_index(row["action"]) for row in rows], dtype=torch.long)
    search_targets = torch.tensor([float(row["search_context_label"]) for row in rows], dtype=torch.float32)
    teacher_action_targets = torch.tensor([action_name_to_index(row["action"]) for row in rows], dtype=torch.long)

    computed_stats = {}
    targets = {}
    for key in REGRESSION_KEYS:
        values = torch.tensor([float(row[key]) for row in rows], dtype=torch.float32)
        if target_stats is None:
            mean = float(values.mean().item())
            std = float(values.std(unbiased=False).item())
            std = std if std > 1e-6 else 1.0
            computed_stats[key] = {"mean": mean, "std": std}
        else:
            computed_stats[key] = {
                "mean": float(target_stats[key]["mean"]),
                "std": max(1e-6, float(target_stats[key]["std"])),
            }
        targets[key] = (values - computed_stats[key]["mean"]) / computed_stats[key]["std"]
    return {
        "features": x,
        "actions": actions,
        "search_targets": search_targets,
        "teacher_action_targets": teacher_action_targets,
        "target_stats": computed_stats,
        **targets,
    }


def train_proxy_model(
    train_rows: list[dict],
    output_path: str,
    *,
    heldout_rows: list[dict] | None = None,
    epochs: int = 40,
    lr: float = 1e-3,
    hidden_size: int = 256,
    action_embed_dim: int = 32,
) -> dict:
    if not train_rows:
        raise ValueError("No proxy rows to train on")
    input_dims = {len(row["feature_vector"]) for row in train_rows}
    if len(input_dims) != 1:
        raise ValueError(f"Mixed proxy feature dimensions: {sorted(input_dims)}")

    train_tensors = _rows_to_tensors(train_rows)
    heldout_tensors = _rows_to_tensors(heldout_rows, target_stats=train_tensors["target_stats"]) if heldout_rows else None
    device = torch.device("cpu")
    model = ProxyRewardModel(
        input_dim=len(train_rows[0]["feature_vector"]),
        hidden_size=hidden_size,
        action_embed_dim=action_embed_dim,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = None
    best_score = float("-inf")
    epoch_summaries = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_tensors["features"], train_tensors["actions"])
        loss = (
            F.mse_loss(outputs["progress"], train_tensors["k_step_progress"])
            + F.mse_loss(outputs["survival"], train_tensors["k_step_survival"])
            + F.mse_loss(outputs["loop_risk"], train_tensors["k_step_loop_risk"])
            + F.mse_loss(outputs["resource_value"], train_tensors["k_step_resource_value"])
            + F.mse_loss(outputs["teacher_margin"], train_tensors["teacher_margin"])
            + F.binary_cross_entropy_with_logits(outputs["search_context_logit"], train_tensors["search_targets"])
            + F.cross_entropy(outputs["action_logits"], train_tensors["teacher_action_targets"])
        )
        loss.backward()
        optimizer.step()

        summary = {"epoch": epoch + 1, "train_loss": round(float(loss.item()), 6)}
        score = -float(loss.item())

        if heldout_rows:
            metadata = {
                "input_dim": len(train_rows[0]["feature_vector"]),
                "hidden_size": hidden_size,
                "action_embed_dim": action_embed_dim,
                "target_stats": train_tensors["target_stats"],
                "reward_weights": DEFAULT_PROXY_REWARD_WEIGHTS,
            }
            tmp_path = output_path + ".tmp_eval"
            save_proxy_model(model, tmp_path, metadata=metadata)
            inference = load_proxy_model(tmp_path)
            heldout_eval = evaluate_proxy_rows(heldout_rows, inference)
            summary["heldout_action_top1_accuracy"] = heldout_eval["action_top1_accuracy"]
            summary["heldout_search_accuracy"] = heldout_eval["search_context"]["accuracy"]
            score = heldout_eval["action_top1_accuracy"]
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        epoch_summaries.append(summary)
        if score >= best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    calibration_metadata = {
        "input_dim": len(train_rows[0]["feature_vector"]),
        "hidden_size": hidden_size,
        "action_embed_dim": action_embed_dim,
        "target_stats": train_tensors["target_stats"],
        "reward_weights": DEFAULT_PROXY_REWARD_WEIGHTS,
    }
    calibration_path = output_path + ".tmp_calibration"
    save_proxy_model(model, calibration_path, metadata=calibration_metadata)
    calibration_inference = load_proxy_model(calibration_path)
    raw_totals = [
        _proxy_total_from_score(calibration_inference.score_action(row["feature_vector"], row["action"]))
        for row in train_rows
    ]
    if os.path.exists(calibration_path):
        os.remove(calibration_path)
    score_mean = float(sum(raw_totals) / max(1, len(raw_totals)))
    score_var = float(sum((value - score_mean) ** 2 for value in raw_totals) / max(1, len(raw_totals)))
    score_std = max(1e-6, score_var ** 0.5)

    metadata = {
        "input_dim": len(train_rows[0]["feature_vector"]),
        "hidden_size": hidden_size,
        "action_embed_dim": action_embed_dim,
        "epochs": epochs,
        "learning_rate": lr,
        "target_stats": train_tensors["target_stats"],
        "reward_weights": DEFAULT_PROXY_REWARD_WEIGHTS,
        "score_stats": {"mean": score_mean, "std": score_std},
        "num_rows": len(train_rows),
        "observation_version": train_rows[0].get("observation_version"),
        "selection_metric": "heldout_action_top1_accuracy" if heldout_rows else "train_loss",
        "selection_score": best_score,
        "epoch_summaries": epoch_summaries,
    }
    save_proxy_model(model, output_path, metadata=metadata)

    result = {"train": metadata}
    if heldout_rows:
        result["heldout_eval"] = evaluate_proxy_rows(heldout_rows, load_proxy_model(output_path))
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a teacher-derived proxy model from trace-derived proxy rows")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--heldout-input", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--action-embed-dim", type=int, default=32)
    parser.add_argument("--report-output", type=str, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train_rows = load_proxy_rows(args.input)
    heldout_rows = load_proxy_rows(args.heldout_input) if args.heldout_input else None
    result = train_proxy_model(
        train_rows,
        args.output,
        heldout_rows=heldout_rows,
        epochs=args.epochs,
        lr=args.lr,
        hidden_size=args.hidden_size,
        action_embed_dim=args.action_embed_dim,
    )
    if args.report_output:
        report_input = args.heldout_input or args.input
        report = build_proxy_report(args.output, report_input)
        atomic_write_json(args.report_output, report)
        result["report_path"] = args.report_output
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
