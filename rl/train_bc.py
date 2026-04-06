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


def _normalize_teacher_paths(
    distill_teacher_bc_path: str | None,
    distill_teacher_bc_paths: list[str] | None,
) -> list[str]:
    paths: list[str] = []
    if distill_teacher_bc_path:
        paths.extend(part.strip() for part in str(distill_teacher_bc_path).split(",") if part.strip())
    if distill_teacher_bc_paths:
        for path in distill_teacher_bc_paths:
            paths.extend(part.strip() for part in str(path).split(",") if part.strip())
    deduped: list[str] = []
    seen = set()
    for path in paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def _teacher_logits_for_rows(rows: list[dict], teacher_bc_paths: list[str]) -> np.ndarray:
    from rl.world_model_features import coerce_world_model_feature_vector, state_prompt_from_row

    allowed_actions_list = [row.get("allowed_actions") for row in rows]
    prompt_texts = [state_prompt_from_row(row) for row in rows]
    ensembled_logits = None
    for teacher_bc_path in teacher_bc_paths:
        teacher_policy = load_bc_model(teacher_bc_path)
        teacher_payload = torch.load(teacher_bc_path, map_location="cpu")
        metadata = teacher_payload.get("metadata", {})
        teacher_wm_path = metadata.get("world_model_path")
        teacher_wm_mode = metadata.get("world_model_feature_mode")

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

        teacher_logits = teacher_policy.logits_batch(
            teacher_features,
            allowed_actions_list=allowed_actions_list,
            prompt_texts=prompt_texts,
        )
        ensembled_logits = teacher_logits if ensembled_logits is None else (ensembled_logits + teacher_logits)

    if ensembled_logits is None:
        raise ValueError("At least one distillation teacher path is required")
    return ensembled_logits / float(len(teacher_bc_paths))


def _parse_action_weight_boosts(raw: str | None) -> dict[int, float]:
    boosts: dict[int, float] = {}
    if not raw:
        return boosts
    for part in str(raw).split(","):
        piece = part.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"Invalid action weight boost '{piece}', expected action=value")
        name, value = piece.split("=", 1)
        action_name = name.strip()
        if action_name not in ACTION_SET:
            raise ValueError(f"Unknown action '{action_name}' in action weight boosts")
        weight = float(value)
        if weight <= 0:
            raise ValueError(f"Action weight boost for '{action_name}' must be positive")
        boosts[ACTION_SET.index(action_name)] = weight
    return boosts


def train_bc_model(
    rows: list[dict],
    output_path: str,
    epochs: int = 20,
    lr: float = 1e-3,
    hidden_size: int = 256,
    num_layers: int = 2,
    observation_version: str = "v1",
    world_model_path: str | None = None,
    world_model_feature_mode: str | None = None,
    distill_teacher_bc_path: str | None = None,
    distill_teacher_bc_paths: list[str] | None = None,
    distill_loss_coef: float = 0.0,
    distill_temperature: float = 1.0,
    supervised_loss_coef: float = 1.0,
    action_weight_boosts: str | None = None,
    text_encoder_backend: str = "none",
    text_vocab_size: int = 4096,
    text_embedding_dim: int = 128,
    text_model_name: str | None = None,
    text_max_length: int = 128,
    text_trainable: bool = False,
) -> dict:
    if not rows:
        raise ValueError("No trace rows to train on")
    from rl.world_model_features import state_prompt_from_row

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
    model = BCPolicyMLP(
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        text_encoder_backend=text_encoder_backend,
        text_vocab_size=text_vocab_size,
        text_embedding_dim=text_embedding_dim,
        text_model_name=text_model_name,
        text_max_length=text_max_length,
        text_trainable=text_trainable,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x = torch.tensor([row["feature_vector"] for row in rows], dtype=torch.float32, device=device)
    y = torch.tensor([ACTION_SET.index(row["action"]) for row in rows], dtype=torch.long, device=device)
    prompt_texts = [state_prompt_from_row(row) for row in rows]
    action_masks = torch.tensor(
        [[1.0 if name in row.get("allowed_actions", ACTION_SET) else 0.0 for name in ACTION_SET] for row in rows],
        dtype=torch.float32,
        device=device,
    )
    teacher_paths = _normalize_teacher_paths(distill_teacher_bc_path, distill_teacher_bc_paths)
    teacher_logits = None
    if teacher_paths:
        teacher_logits_np = _teacher_logits_for_rows(rows, teacher_paths)
        teacher_logits = torch.tensor(teacher_logits_np, dtype=torch.float32, device=device)
    example_weights = torch.ones(len(rows), dtype=torch.float32, device=device)
    for action_idx, weight in _parse_action_weight_boosts(action_weight_boosts).items():
        example_weights = torch.where(y == action_idx, torch.full_like(example_weights, weight), example_weights)
    cached_text_context = None
    if text_encoder_backend == "transformer" and not text_trainable:
        with torch.no_grad():
            cached_text_context = model.encode_text_context(prompt_texts, device=device)

    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x, prompt_texts=prompt_texts, text_context=cached_text_context)
        logits = logits.masked_fill(action_masks <= 0, -1e9)
        supervised_loss = F.cross_entropy(logits, y, reduction="none")
        supervised_loss = (supervised_loss * example_weights).mean()
        loss = supervised_loss * float(supervised_loss_coef)
        if teacher_logits is not None and distill_loss_coef > 0:
            temperature = max(float(distill_temperature), 1e-6)
            student_log_probs = F.log_softmax(logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
            distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
            loss = loss + distill_loss_coef * distill_loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    preds = torch.argmax(
        model(x, prompt_texts=prompt_texts, text_context=cached_text_context).masked_fill(action_masks <= 0, -1e9),
        dim=1,
    )
    accuracy = float((preds == y).float().mean().item())
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    metadata = {
        "epochs": epochs,
        "learning_rate": lr,
        "num_examples": len(rows),
        "input_dim": input_dim,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "observation_version": observation_version,
        "world_model_path": world_model_path,
        "world_model_feature_mode": world_model_feature_mode,
        "distill_teacher_bc_path": distill_teacher_bc_path,
        "distill_teacher_bc_paths": teacher_paths or None,
        "distill_loss_coef": distill_loss_coef,
        "distill_temperature": distill_temperature,
        "supervised_loss_coef": supervised_loss_coef,
        "action_weight_boosts": action_weight_boosts,
        "text_encoder_backend": text_encoder_backend,
        "text_vocab_size": text_vocab_size,
        "text_embedding_dim": text_embedding_dim,
        "text_model_name": text_model_name,
        "text_max_length": text_max_length,
        "text_trainable": text_trainable,
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
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--observation-version", type=str, default="v1")
    parser.add_argument("--world-model-path", type=str, default=None)
    parser.add_argument("--world-model-feature-mode", type=str, default=None)
    parser.add_argument("--distill-teacher-bc-path", type=str, default=None)
    parser.add_argument("--distill-teacher-bc-paths", type=str, nargs="*", default=None)
    parser.add_argument("--distill-loss-coef", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--supervised-loss-coef", type=float, default=1.0)
    parser.add_argument("--action-weight-boosts", type=str, default=None)
    parser.add_argument("--text-encoder-backend", type=str, default="none", choices=["none", "hash", "transformer"])
    parser.add_argument("--text-vocab-size", type=int, default=4096)
    parser.add_argument("--text-embedding-dim", type=int, default=128)
    parser.add_argument("--text-model-name", type=str, default=None)
    parser.add_argument("--text-max-length", type=int, default=128)
    parser.add_argument("--text-trainable", action="store_true")
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
        num_layers=args.num_layers,
        observation_version=args.observation_version,
        world_model_path=args.world_model_path,
        world_model_feature_mode=args.world_model_feature_mode,
        distill_teacher_bc_path=args.distill_teacher_bc_path,
        distill_teacher_bc_paths=args.distill_teacher_bc_paths,
        distill_loss_coef=args.distill_loss_coef,
        distill_temperature=args.distill_temperature,
        supervised_loss_coef=args.supervised_loss_coef,
        action_weight_boosts=args.action_weight_boosts,
        text_encoder_backend=args.text_encoder_backend,
        text_vocab_size=args.text_vocab_size,
        text_embedding_dim=args.text_embedding_dim,
        text_model_name=args.text_model_name,
        text_max_length=args.text_max_length,
        text_trainable=args.text_trainable,
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
