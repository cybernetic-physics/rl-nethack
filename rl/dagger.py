from __future__ import annotations

import json

from rl.io_utils import atomic_write_text
from rl.train_bc import load_trace_rows, train_bc_model
from rl.traces import generate_dagger_traces, verify_trace_file


def run_dagger_iteration(
    *,
    base_trace_input: str,
    dagger_trace_output: str,
    bc_output: str,
    student_policy: str,
    task: str = "explore",
    num_episodes: int = 8,
    max_steps: int = 20,
    seed_start: int = 42,
    appo_experiment: str | None = None,
    appo_train_dir: str = "train_dir/rl",
    appo_checkpoint_path: str | None = None,
    bc_model_path: str | None = None,
    observation_version: str = "v1",
    merge_ratio: float = 0.5,
    epochs: int = 20,
    lr: float = 1e-3,
    hidden_size: int = 256,
    merged_trace_output: str | None = None,
) -> dict:
    if student_policy == "bc" and not bc_model_path:
        raise ValueError("bc_model_path is required for student_policy=bc")
    if student_policy == "appo" and not appo_experiment:
        raise ValueError("appo_experiment is required for student_policy=appo")

    dagger_summary = generate_dagger_traces(
        output_path=dagger_trace_output,
        num_episodes=num_episodes,
        max_steps=max_steps,
        student_policy=student_policy,
        task=task,
        seed_start=seed_start,
        appo_experiment=appo_experiment,
        appo_train_dir=appo_train_dir,
        appo_checkpoint_path=appo_checkpoint_path,
        bc_model_path=bc_model_path,
        observation_version=observation_version,
    )

    base_rows = load_trace_rows(base_trace_input)
    dagger_rows = load_trace_rows(dagger_trace_output)
    keep_base = int(round(len(base_rows) * max(0.0, min(1.0, merge_ratio))))
    merged_rows = base_rows[:keep_base] + dagger_rows

    if merged_trace_output:
        atomic_write_text(merged_trace_output, "".join(json.dumps(row) + "\n" for row in merged_rows))

    train_result = train_bc_model(
        merged_rows,
        bc_output,
        epochs=epochs,
        lr=lr,
        hidden_size=hidden_size,
        observation_version=observation_version,
    )

    return {
        "base_trace_input": base_trace_input,
        "dagger_trace_output": dagger_trace_output,
        "merged_trace_output": merged_trace_output,
        "bc_output": bc_output,
        "student_policy": student_policy,
        "task": task,
        "merge_ratio": merge_ratio,
        "base_rows": len(base_rows),
        "dagger_rows": len(dagger_rows),
        "merged_rows": len(merged_rows),
        "dagger_trace_verify": verify_trace_file(dagger_trace_output),
        "dagger_trace_summary": dagger_summary,
        "bc_train": train_result,
    }
