from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from rl.bc_model import BCPolicyMLP, load_bc_model, save_bc_model
from rl.feature_encoder import ACTION_SET
from rl.teacher_report import build_teacher_report, write_teacher_report
from rl.world_model import load_world_model


def load_trace_rows(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _teacher_logits_for_rows(rows: list[dict], teacher_bc_path: str) -> np.ndarray:
    from rl.world_model_features import coerce_world_model_feature_vector, state_prompt_from_row

    teacher_policy = load_bc_model(teacher_bc_path)
    teacher_payload = torch.load(teacher_bc_path, map_location="cpu")
    metadata = teacher_payload.get("metadata", {})
    teacher_wm_path = metadata.get("world_model_path")
    teacher_wm_mode = metadata.get("world_model_feature_mode")
    allowed_actions_list = [row.get("allowed_actions") for row in rows]

    if teacher_wm_path and teacher_wm_mode:
        wm_inference = load_world_model(teacher_wm_path)
        expected_dim = int(getattr(wm_inference.model, "_metadata", {}).get("input_dim", 0))
        base_features = [
            coerce_world_model_feature_vector(row["feature_vector"], expected_dim) if expected_dim else np.asarray(row["feature_vector"], dtype=np.float32)
            for row in rows
        ]
        prompts = [state_prompt_from_row(row) for row in rows]
        encoded = wm_inference.encode_with_aux_batch(base_features, prompt_texts=prompts)
        latents = encoded["latent"]
        action_logits = encoded["action_logits"]
        teacher_features = []
        for base_feature, latent, logits in zip(base_features, latents, action_logits):
            if teacher_wm_mode == "replace":
                teacher_features.append(np.asarray(latent, dtype=np.float32))
            elif teacher_wm_mode == "concat":
                teacher_features.append(np.concatenate([base_feature, latent]).astype(np.float32))
            elif teacher_wm_mode == "concat_aux":
                teacher_features.append(np.concatenate([base_feature, latent, logits]).astype(np.float32))
            else:
                raise ValueError(f"Unsupported teacher world-model mode: {teacher_wm_mode}")
    else:
        teacher_features = [np.asarray(row["feature_vector"], dtype=np.float32) for row in rows]

    return teacher_policy.logits_batch(teacher_features, allowed_actions_list=allowed_actions_list)


def train_bc_model(
    rows: list[dict],
    output_path: str,
    epochs: int = 20,
    lr: float = 1e-3,
    hidden_size: int = 256,
    observation_version: str = "v1",
    world_model_path: str | None = None,
    world_model_feature_mode: str | None = None,
    distill_teacher_bc_path: str | None = None,
    distill_loss_coef: float = 0.0,
    distill_temperature: float = 1.0,
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
    teacher_logits = None
    if distill_teacher_bc_path:
        teacher_logits_np = _teacher_logits_for_rows(rows, distill_teacher_bc_path)
        teacher_logits = torch.tensor(teacher_logits_np, dtype=torch.float32, device=device)

    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x)
        logits = logits.masked_fill(action_masks <= 0, -1e9)
        loss = F.cross_entropy(logits, y)
        if teacher_logits is not None and distill_loss_coef > 0:
            temperature = max(float(distill_temperature), 1e-6)
            student_log_probs = F.log_softmax(logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
            distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
            loss = loss + distill_loss_coef * distill_loss
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
        "world_model_path": world_model_path,
        "world_model_feature_mode": world_model_feature_mode,
        "distill_teacher_bc_path": distill_teacher_bc_path,
        "distill_loss_coef": distill_loss_coef,
        "distill_temperature": distill_temperature,
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
    parser.add_argument("--world-model-path", type=str, default=None)
    parser.add_argument("--world-model-feature-mode", type=str, default=None)
    parser.add_argument("--distill-teacher-bc-path", type=str, default=None)
    parser.add_argument("--distill-loss-coef", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--heldout-input", type=str, default=None)
    parser.add_argument("--teacher-report-output", type=str, default=None)
    parser.add_argument("--weak-action-input", type=str, default=None)
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
        world_model_path=args.world_model_path,
        world_model_feature_mode=args.world_model_feature_mode,
        distill_teacher_bc_path=args.distill_teacher_bc_path,
        distill_loss_coef=args.distill_loss_coef,
        distill_temperature=args.distill_temperature,
    )
    output = {"train": result}
    if args.heldout_input:
        report = build_teacher_report(
            model_path=args.output,
            heldout_trace_path=args.heldout_input,
            train_result=result,
            teacher_kind="bc",
            weak_action_trace_path=args.weak_action_input,
            source_trace_path=args.input,
            observation_version=args.observation_version,
            world_model_path=args.world_model_path,
            world_model_feature_mode=args.world_model_feature_mode,
        )
        output["teacher_report"] = report
        output["teacher_report_path"] = write_teacher_report(report, args.teacher_report_output)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
