from __future__ import annotations

import json
from pathlib import Path

from rl.io_utils import atomic_write_json
from rl.trace_eval import evaluate_trace_policy, trace_disagreement_report


def build_teacher_report(
    *,
    model_path: str,
    heldout_trace_path: str | None,
    train_result: dict,
    teacher_kind: str,
    weak_action_trace_path: str | None = None,
    source_trace_path: str | None = None,
    observation_version: str | None = None,
    world_model_path: str | None = None,
    world_model_feature_mode: str | None = None,
) -> dict:
    report = {
        "teacher_kind": teacher_kind,
        "model_path": model_path,
        "source_trace_path": source_trace_path,
        "heldout_trace_path": heldout_trace_path,
        "observation_version": observation_version,
        "world_model_path": world_model_path,
        "world_model_feature_mode": world_model_feature_mode,
        "train": train_result,
    }
    if heldout_trace_path:
        report["heldout_trace_eval"] = evaluate_trace_policy(
            heldout_trace_path,
            "bc",
            bc_model_path=model_path,
            summary_only=True,
        )["summary"]
        report["heldout_disagreements"] = trace_disagreement_report(
            heldout_trace_path,
            bc_model_path=model_path,
            top_k=5,
        )["bc"]
    if weak_action_trace_path:
        report["weak_action_trace_path"] = weak_action_trace_path
        report["weak_action_trace_eval"] = evaluate_trace_policy(
            weak_action_trace_path,
            "bc",
            bc_model_path=model_path,
            summary_only=True,
        )["summary"]
    return report


def default_teacher_report_path(model_path: str) -> str:
    path = Path(model_path)
    return str(path.with_suffix(path.suffix + ".teacher_report.json"))


def write_teacher_report(report: dict, output_path: str | None = None) -> str:
    target = output_path or default_teacher_report_path(report["model_path"])
    atomic_write_json(target, report)
    return target
