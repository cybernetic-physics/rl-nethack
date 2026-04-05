from __future__ import annotations

import json
import random
from pathlib import Path

from rl.io_utils import atomic_write_json, atomic_write_text
from rl.trace_eval import evaluate_trace_policy
from rl.train_bc import load_trace_rows, train_bc_model
from rl.traces import generate_dagger_traces, verify_trace_file


MERGE_POLICIES = {"base_only", "uniform_merge", "weighted_recent"}


def _normalize_merge_ratio(merge_ratio: float) -> float:
    return max(0.0, min(1.0, merge_ratio))


def build_merged_trace_rows(
    *,
    base_rows: list[dict],
    relabeled_rows: list[dict],
    merge_policy: str,
    merge_ratio: float,
    rng: random.Random,
) -> list[dict]:
    if merge_policy not in MERGE_POLICIES:
        raise ValueError(f"Unknown merge_policy: {merge_policy}")

    merge_ratio = _normalize_merge_ratio(merge_ratio)
    if merge_policy == "base_only":
        keep_base = int(round(len(base_rows) * merge_ratio))
        return list(base_rows[:keep_base]) + list(relabeled_rows)

    if merge_policy == "uniform_merge":
        keep_base = int(round(len(base_rows) * merge_ratio))
        if keep_base >= len(base_rows):
            sampled_base = list(base_rows)
        else:
            sampled_base = rng.sample(base_rows, keep_base)
        sampled_base.sort(key=lambda row: (row["episode_id"], row["step"]))
        return sampled_base + list(relabeled_rows)

    recent_weight = []
    total = len(base_rows)
    for idx, row in enumerate(base_rows):
        weight = 1.0 + (idx / max(1, total - 1))
        recent_weight.append((row, weight))
    keep_base = int(round(len(base_rows) * merge_ratio))
    if keep_base <= 0:
        sampled_base = []
    elif keep_base >= len(base_rows):
        sampled_base = list(base_rows)
    else:
        population = [row for row, _ in recent_weight]
        weights = [weight for _, weight in recent_weight]
        chosen = []
        pool = list(zip(population, weights))
        for _ in range(keep_base):
            total_weight = sum(weight for _, weight in pool)
            pick = rng.random() * total_weight
            upto = 0.0
            for index, (row, weight) in enumerate(pool):
                upto += weight
                if upto >= pick:
                    chosen.append(row)
                    del pool[index]
                    break
        sampled_base = chosen
    sampled_base.sort(key=lambda row: (row["episode_id"], row["step"]))
    return sampled_base + list(relabeled_rows)


def _write_rows(path: str | None, rows: list[dict]) -> str | None:
    if not path:
        return None
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    return atomic_write_text(path, payload)


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
    merge_policy: str = "uniform_merge",
    epochs: int = 20,
    lr: float = 1e-3,
    hidden_size: int = 256,
    merged_trace_output: str | None = None,
    heldout_trace_input: str | None = None,
    random_seed: int = 123,
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
    merged_rows = build_merged_trace_rows(
        base_rows=base_rows,
        relabeled_rows=dagger_rows,
        merge_policy=merge_policy,
        merge_ratio=merge_ratio,
        rng=random.Random(random_seed),
    )

    _write_rows(merged_trace_output, merged_rows)

    train_result = train_bc_model(
        merged_rows,
        bc_output,
        epochs=epochs,
        lr=lr,
        hidden_size=hidden_size,
        observation_version=observation_version,
    )
    base_eval = evaluate_trace_policy(base_trace_input, "bc", bc_model_path=bc_output, summary_only=True)
    heldout_eval = (
        evaluate_trace_policy(heldout_trace_input, "bc", bc_model_path=bc_output, summary_only=True)
        if heldout_trace_input
        else None
    )

    return {
        "base_trace_input": base_trace_input,
        "heldout_trace_input": heldout_trace_input,
        "dagger_trace_output": dagger_trace_output,
        "merged_trace_output": merged_trace_output,
        "bc_output": bc_output,
        "student_policy": student_policy,
        "task": task,
        "merge_ratio": merge_ratio,
        "merge_policy": merge_policy,
        "base_rows": len(base_rows),
        "dagger_rows": len(dagger_rows),
        "merged_rows": len(merged_rows),
        "dagger_trace_verify": verify_trace_file(dagger_trace_output),
        "dagger_trace_summary": dagger_summary,
        "bc_train": train_result,
        "base_trace_eval": base_eval["summary"],
        "heldout_trace_eval": heldout_eval["summary"] if heldout_eval else None,
    }


def run_dagger_schedule(
    *,
    base_trace_input: str,
    output_dir: str,
    student_policy: str,
    task: str = "explore",
    iterations: int = 3,
    num_episodes: int = 8,
    max_steps: int = 20,
    seed_start: int = 42,
    appo_experiment: str | None = None,
    appo_train_dir: str = "train_dir/rl",
    appo_checkpoint_path: str | None = None,
    bc_model_path: str | None = None,
    observation_version: str = "v1",
    merge_ratio: float = 0.5,
    merge_policy: str = "uniform_merge",
    epochs: int = 20,
    lr: float = 1e-3,
    hidden_size: int = 256,
    heldout_trace_input: str | None = None,
    random_seed: int = 123,
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    current_base = base_trace_input
    current_bc_model = bc_model_path
    reports = []

    for iteration in range(iterations):
        iteration_seed = seed_start + (iteration * max(1, num_episodes))
        dagger_trace_output = str(Path(output_dir) / f"dagger_iter_{iteration:02d}.jsonl")
        merged_trace_output = str(Path(output_dir) / f"merged_iter_{iteration:02d}.jsonl")
        bc_output = str(Path(output_dir) / f"bc_iter_{iteration:02d}.pt")
        report = run_dagger_iteration(
            base_trace_input=current_base,
            dagger_trace_output=dagger_trace_output,
            bc_output=bc_output,
            student_policy=student_policy if iteration == 0 else "bc",
            task=task,
            num_episodes=num_episodes,
            max_steps=max_steps,
            seed_start=iteration_seed,
            appo_experiment=appo_experiment,
            appo_train_dir=appo_train_dir,
            appo_checkpoint_path=appo_checkpoint_path,
            bc_model_path=current_bc_model,
            observation_version=observation_version,
            merge_ratio=merge_ratio,
            merge_policy=merge_policy,
            epochs=epochs,
            lr=lr,
            hidden_size=hidden_size,
            merged_trace_output=merged_trace_output,
            heldout_trace_input=heldout_trace_input,
            random_seed=random_seed + iteration,
        )
        report["iteration"] = iteration
        report["seed_start"] = iteration_seed
        reports.append(report)
        current_base = merged_trace_output
        current_bc_model = bc_output

    summary = {
        "base_trace_input": base_trace_input,
        "heldout_trace_input": heldout_trace_input,
        "output_dir": output_dir,
        "student_policy": student_policy,
        "task": task,
        "iterations": iterations,
        "merge_policy": merge_policy,
        "merge_ratio": merge_ratio,
        "observation_version": observation_version,
        "final_bc_model": current_bc_model,
        "final_trace_input": current_base,
        "reports": reports,
    }
    atomic_write_json(Path(output_dir) / "dagger_schedule_report.json", summary)
    return summary
