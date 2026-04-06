from __future__ import annotations

import json
from collections import Counter

import numpy as np
import torch

from rl.bc_model import load_bc_model
from rl.feature_encoder import ACTION_SET


def load_trace_rows(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _group_trace_rows(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    rows = sorted(rows, key=lambda row: (row["episode_id"], row["step"]))
    episodes: list[tuple[str, list[dict]]] = []
    current_episode_id = None
    current_rows: list[dict] = []
    for row in rows:
        episode_id = row["episode_id"]
        if current_episode_id is None:
            current_episode_id = episode_id
        if episode_id != current_episode_id:
            episodes.append((current_episode_id, current_rows))
            current_episode_id = episode_id
            current_rows = []
        current_rows.append(row)
    if current_rows:
        episodes.append((current_episode_id, current_rows))
    return episodes


def _mask_logits(logits: torch.Tensor, allowed_actions: list[str]) -> torch.Tensor:
    masked = logits.clone()
    allowed = set(allowed_actions)
    for idx, name in enumerate(ACTION_SET):
        if name not in allowed:
            masked[0, idx] = -1e9
    return masked


def _validate_trace_rows(rows: list[dict]) -> None:
    versions = {row.get("observation_version", "unknown") for row in rows}
    feature_dims = {len(row.get("feature_vector", [])) for row in rows}
    if len(versions) != 1:
        raise ValueError(f"Mixed observation versions in trace file: {sorted(versions)}")
    if len(feature_dims) != 1:
        raise ValueError(f"Mixed feature dimensions in trace file: {sorted(feature_dims)}")
    if not rows:
        raise ValueError("No trace rows provided")


def _state_prompt_from_row(row: dict) -> str:
    if row.get("state_prompt"):
        return str(row["state_prompt"])
    prompt = row.get("prompt")
    if not prompt:
        return ""
    lines = [line for line in str(prompt).splitlines() if not line.startswith("Action:")]
    return "\n".join(lines)


def _evaluate_trace_rows(
    rows: list[dict],
    *,
    bc_policy=None,
    appo_bundle: dict | None = None,
    deterministic: bool = True,
    summary_only: bool = False,
) -> dict:
    _validate_trace_rows(rows)
    versions = {row.get("observation_version", "unknown") for row in rows}
    episodes = _group_trace_rows(rows)
    if not episodes:
        raise ValueError("No trace rows provided")

    episode_results = []
    total_matches = 0
    total_rows = 0
    invalid_actions = 0
    action_counts: Counter[str] = Counter()
    observation_versions = sorted(versions)

    for episode_id, episode_rows in episodes:
        per_episode_rows = []
        rnn_states = None
        if appo_bundle:
            rnn_states = torch.zeros(
                [1, appo_bundle["get_rnn_size"](appo_bundle["cfg"])],
                dtype=torch.float32,
                device=appo_bundle["device"],
            )

        for row in episode_rows:
            features = np.asarray(row["feature_vector"], dtype=np.float32)
            allowed_actions = list(row.get("allowed_actions", []))
            teacher_action = row["action"]

            if bc_policy:
                predicted_action = bc_policy.act(
                    features,
                    allowed_actions=allowed_actions,
                    prompt_text=_state_prompt_from_row(row),
                )
            else:
                obs_tensor = torch.from_numpy(features).unsqueeze(0).to(appo_bundle["device"])
                normalized_obs = appo_bundle["prepare_and_normalize_obs"](
                    appo_bundle["actor_critic"],
                    {"obs": obs_tensor},
                )
                outputs = appo_bundle["actor_critic"](normalized_obs, rnn_states)
                logits = appo_bundle["actor_critic"].action_distribution().raw_logits
                logits = _mask_logits(logits, allowed_actions)
                if deterministic:
                    action_idx = int(torch.argmax(logits, dim=1).item())
                else:
                    action_idx = int(outputs["actions"].squeeze().item())
                predicted_action = ACTION_SET[action_idx]
                rnn_states = outputs["new_rnn_states"]

            matched = predicted_action == teacher_action
            invalid_action = predicted_action not in set(allowed_actions)
            total_rows += 1
            total_matches += int(matched)
            invalid_actions += int(invalid_action)
            action_counts[predicted_action] += 1
            per_episode_rows.append(
                {
                    "episode_id": episode_id,
                    "seed": row.get("seed"),
                    "step": row["step"],
                    "obs_hash": row.get("obs_hash"),
                    "teacher_action": teacher_action,
                    "predicted_action": predicted_action,
                    "matched": matched,
                    "invalid_action": invalid_action,
                    "allowed_actions": allowed_actions,
                }
            )

        episode_results.append(
            {
                "episode_id": episode_id,
                "seed": episode_rows[0].get("seed"),
                "steps": len(per_episode_rows),
                "match_rate": round(
                    sum(1 for row in per_episode_rows if row["matched"]) / max(1, len(per_episode_rows)),
                    4,
                ),
                "rows": per_episode_rows,
            }
        )

    result = {
        "summary": {
            "episodes": len(episode_results),
            "rows": total_rows,
            "match_rate": round(total_matches / max(1, total_rows), 4),
            "invalid_action_rate": round(invalid_actions / max(1, total_rows), 4),
            "action_counts": dict(action_counts),
            "observation_versions": observation_versions,
        },
    }
    if not summary_only:
        result["episodes"] = episode_results
    return result


def evaluate_trace_appo_bundle(
    trace_path: str,
    *,
    cfg,
    actor_critic,
    device: str = "cpu",
    deterministic: bool = True,
    summary_only: bool = False,
) -> dict:
    from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
    from sample_factory.model.model_utils import get_rnn_size

    rows = load_trace_rows(trace_path)
    bundle = {
        "cfg": cfg,
        "actor_critic": actor_critic,
        "device": torch.device(device),
        "prepare_and_normalize_obs": prepare_and_normalize_obs,
        "get_rnn_size": get_rnn_size,
    }
    result = _evaluate_trace_rows(
        rows,
        appo_bundle=bundle,
        deterministic=deterministic,
        summary_only=summary_only,
    )
    result.update(
        {
            "trace_path": trace_path,
            "policy": "appo",
            "evaluation_mode": "trace_dataset",
            "checkpoint_path": None,
        }
    )
    return result


def evaluate_trace_policy(
    trace_path: str,
    policy: str,
    *,
    bc_model_path: str | None = None,
    appo_experiment: str | None = None,
    appo_train_dir: str = "train_dir/rl",
    appo_checkpoint_path: str | None = None,
    deterministic: bool = True,
    summary_only: bool = False,
) -> dict:
    rows = load_trace_rows(trace_path)
    _validate_trace_rows(rows)
    bc_policy = None
    appo_bundle = None
    if policy == "bc":
        if not bc_model_path:
            raise ValueError("bc_model_path is required for policy=bc")
        bc_policy = load_bc_model(bc_model_path)
    elif policy == "appo":
        if not appo_experiment:
            raise ValueError("appo_experiment is required for policy=appo")
        from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
        from sample_factory.model.model_utils import get_rnn_size

        from rl.evaluate import _load_actor_critic

        cfg, _, actor_critic, device = _load_actor_critic(
            appo_experiment,
            appo_train_dir,
            device="cpu",
            checkpoint_path=appo_checkpoint_path,
        )
        appo_bundle = {
            "cfg": cfg,
            "actor_critic": actor_critic,
            "device": device,
            "prepare_and_normalize_obs": prepare_and_normalize_obs,
            "get_rnn_size": get_rnn_size,
        }
    else:
        raise ValueError(f"Unsupported trace policy: {policy}")
    result = _evaluate_trace_rows(
        rows,
        bc_policy=bc_policy,
        appo_bundle=appo_bundle,
        deterministic=deterministic,
        summary_only=summary_only,
    )
    result.update(
        {
            "trace_path": trace_path,
            "policy": policy,
            "evaluation_mode": "trace_dataset",
            "checkpoint_path": appo_checkpoint_path if policy == "appo" else bc_model_path,
        }
    )
    return result


def compare_trace_policies(
    trace_path: str,
    *,
    bc_model_path: str | None = None,
    appo_experiment: str | None = None,
    appo_train_dir: str = "train_dir/rl",
    appo_checkpoint_path: str | None = None,
    summary_only: bool = False,
    ) -> dict:
    result = {
        "trace_path": trace_path,
        "evaluation_mode": "trace_dataset",
    }
    if bc_model_path:
        result["bc"] = evaluate_trace_policy(trace_path, "bc", bc_model_path=bc_model_path, summary_only=summary_only)
    if appo_experiment:
        result["appo"] = evaluate_trace_policy(
            trace_path,
            "appo",
            appo_experiment=appo_experiment,
            appo_train_dir=appo_train_dir,
            appo_checkpoint_path=appo_checkpoint_path,
            summary_only=summary_only,
        )
    return result


def _summarize_disagreements(result: dict, top_k: int = 10) -> dict:
    episodes = result.get("episodes", [])
    mismatch_counter: Counter[tuple[str, str]] = Counter()
    teacher_counter: Counter[str] = Counter()
    predicted_counter: Counter[str] = Counter()
    per_teacher_action: dict[str, Counter[str]] = {}
    per_teacher_totals: Counter[str] = Counter()
    per_teacher_matches: Counter[str] = Counter()
    predicted_totals: Counter[str] = Counter()
    predicted_correct: Counter[str] = Counter()

    for episode in episodes:
        for row in episode.get("rows", []):
            teacher = row["teacher_action"]
            predicted = row["predicted_action"]
            teacher_counter[teacher] += 1
            predicted_counter[predicted] += 1
            predicted_totals[predicted] += 1
            per_teacher_action.setdefault(teacher, Counter())[predicted] += 1
            per_teacher_totals[teacher] += 1
            if row["matched"]:
                per_teacher_matches[teacher] += 1
                predicted_correct[predicted] += 1
            else:
                mismatch_counter[(teacher, predicted)] += 1

    by_teacher_action = {}
    for teacher, total in per_teacher_totals.items():
        preds = per_teacher_action[teacher]
        by_teacher_action[teacher] = {
            "rows": total,
            "match_rate": round(per_teacher_matches[teacher] / max(1, total), 4),
            "top_predictions": [
                {"predicted_action": pred, "count": count}
                for pred, count in preds.most_common(top_k)
            ],
        }

    per_action_metrics = {}
    for action in sorted(set(teacher_counter) | set(predicted_counter)):
        support = per_teacher_totals[action]
        predicted = predicted_totals[action]
        true_positive = per_teacher_matches[action]
        recall = true_positive / max(1, support)
        precision = predicted_correct[action] / max(1, predicted)
        if precision + recall:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        per_action_metrics[action] = {
            "support": support,
            "predicted": predicted,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return {
        "rows": result["summary"]["rows"],
        "match_rate": result["summary"]["match_rate"],
        "invalid_action_rate": result["summary"]["invalid_action_rate"],
        "teacher_action_counts": dict(teacher_counter),
        "predicted_action_counts": dict(predicted_counter),
        "per_action_metrics": per_action_metrics,
        "common_mismatches": [
            {"teacher_action": teacher, "predicted_action": predicted, "count": count}
            for (teacher, predicted), count in mismatch_counter.most_common(top_k)
        ],
        "by_teacher_action": by_teacher_action,
    }


def trace_disagreement_report(
    trace_path: str,
    *,
    bc_model_path: str | None = None,
    appo_experiment: str | None = None,
    appo_train_dir: str = "train_dir/rl",
    appo_checkpoint_path: str | None = None,
    top_k: int = 10,
) -> dict:
    result = {
        "trace_path": trace_path,
        "evaluation_mode": "trace_dataset",
    }
    if bc_model_path:
        bc_result = evaluate_trace_policy(
            trace_path,
            "bc",
            bc_model_path=bc_model_path,
            summary_only=False,
        )
        result["bc"] = _summarize_disagreements(bc_result, top_k=top_k)
    if appo_experiment:
        appo_result = evaluate_trace_policy(
            trace_path,
            "appo",
            appo_experiment=appo_experiment,
            appo_train_dir=appo_train_dir,
            appo_checkpoint_path=appo_checkpoint_path,
            summary_only=False,
        )
        result["appo"] = _summarize_disagreements(appo_result, top_k=top_k)
    return result
