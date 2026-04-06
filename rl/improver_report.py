from __future__ import annotations

import json
from pathlib import Path

from rl.config import RLConfig
from rl.io_utils import atomic_write_json


def default_improver_report_path(train_dir: str, experiment: str) -> str:
    return str(Path(train_dir) / experiment / "improver_report.json")


def _load_json_if_exists(path: str | None) -> dict | None:
    if not path:
        return None
    target = Path(path)
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text())
    except Exception:
        return None


def build_improver_report(
    *,
    config: RLConfig,
    plan: dict,
    argv: list[str],
    warmstart_checkpoint: str | None,
    status: str,
) -> dict:
    experiment_dir = Path(config.train_dir) / config.experiment
    teacher_report = _load_json_if_exists(config.model.teacher_report_path)
    best_trace_metadata = _load_json_if_exists(str(experiment_dir / "checkpoint_p0" / "best_trace_match.json"))
    warmstart_trace_metadata = _load_json_if_exists(str(experiment_dir / "checkpoint_p0" / "warmstart_trace_match.json"))

    teacher_checkpoint = (
        config.appo.teacher_bc_path
        or config.model.bc_init_path
        or config.model.appo_init_checkpoint_path
    )
    best_learned_checkpoint = None
    if best_trace_metadata:
        best_learned_checkpoint = best_trace_metadata.get("best_checkpoint_path")

    return {
        "improver_kind": "appo_teacher_replay",
        "status": status,
        "experiment": config.experiment,
        "train_dir": config.train_dir,
        "experiment_dir": str(experiment_dir),
        "teacher_checkpoint_path": teacher_checkpoint,
        "teacher_report_path": config.model.teacher_report_path,
        "teacher_report_summary": {
            "teacher_kind": teacher_report.get("teacher_kind"),
            "heldout_match_rate": teacher_report.get("heldout_trace_eval", {}).get("match_rate"),
            "weak_action_match_rate": teacher_report.get("weak_action_trace_eval", {}).get("match_rate"),
        }
        if teacher_report
        else None,
        "warmstart_checkpoint": warmstart_checkpoint,
        "warmstart_trace_metadata": warmstart_trace_metadata,
        "replay_source": {
            "trace_input": config.appo.teacher_replay_trace_input,
            "source_mode": config.appo.teacher_replay_source_mode,
            "priority_power": config.appo.teacher_replay_priority_power,
            "batch_size": config.appo.teacher_replay_batch_size,
            "coef": config.appo.teacher_replay_coef,
            "final_coef": config.appo.teacher_replay_final_coef,
        },
        "reward_objective": {
            "source": config.reward.source,
            "learned_reward_path": config.reward.learned_reward_path,
            "intrinsic_weight": config.reward.intrinsic_weight,
            "extrinsic_weight": config.reward.extrinsic_weight,
            "reward_scale": config.appo.reward_scale,
        },
        "trace_gate": {
            "trace_eval_input": config.appo.trace_eval_input,
            "trace_eval_interval_env_steps": config.appo.trace_eval_interval_env_steps,
            "best_checkpoint_path": best_learned_checkpoint,
            "best_trace_metadata": best_trace_metadata,
        },
        "observation": {
            "version": config.env.observation_version,
            "world_model_path": config.env.world_model_path,
            "world_model_feature_mode": config.env.world_model_feature_mode,
        },
        "argv": argv,
        "plan": plan,
    }


def write_improver_report(report: dict, output_path: str | None = None) -> str:
    target = output_path or default_improver_report_path(report["train_dir"], report["experiment"])
    return atomic_write_json(target, report)
