from __future__ import annotations

import json
import random
from pathlib import Path

from rl.io_utils import atomic_write_json, atomic_write_text
from rl.trace_eval import evaluate_trace_policy
from rl.train_bc import load_trace_rows, train_bc_model
from rl.traces import generate_dagger_traces, verify_trace_file


MERGE_POLICIES = {"base_only", "uniform_merge", "weighted_recent"}
ROW_SELECTION_POLICIES = {"all", "disagreement", "loop_risk", "failure_slice", "weak_action", "hard_only"}


def _normalize_merge_ratio(merge_ratio: float) -> float:
    return max(0.0, min(1.0, merge_ratio))


def _normalize_keep_ratio(keep_ratio: float) -> float:
    return max(0.0, min(1.0, keep_ratio))


def _parse_confusion_pairs(spec: str | None) -> set[tuple[str, str]]:
    if not spec:
        return set()
    pairs = set()
    for raw_token in str(spec).split(","):
        token = raw_token.strip()
        if not token:
            continue
        if "->" not in token:
            raise ValueError(f"Invalid confusion pair '{token}'; expected behavior->teacher")
        behavior_action, teacher_action = token.split("->", 1)
        behavior_action = behavior_action.strip()
        teacher_action = teacher_action.strip()
        if not behavior_action or not teacher_action:
            raise ValueError(f"Invalid confusion pair '{token}'; expected non-empty behavior->teacher")
        pairs.add((behavior_action, teacher_action))
    return pairs


def _matches_confusion_pairs(row: dict, confusion_pairs: set[tuple[str, str]]) -> bool:
    if not confusion_pairs:
        return True
    return (str(row.get("behavior_action")), str(row.get("teacher_action"))) in confusion_pairs


def _matches_row_selection_policy(row: dict, row_selection_policy: str) -> bool:
    if row_selection_policy not in ROW_SELECTION_POLICIES:
        raise ValueError(f"Unknown row_selection_policy: {row_selection_policy}")
    if row_selection_policy == "all":
        return True
    if row_selection_policy == "disagreement":
        return bool(row.get("is_disagreement_candidate", False))
    if row_selection_policy == "loop_risk":
        return bool(row.get("is_loop_risk", False))
    if row_selection_policy == "failure_slice":
        return bool(row.get("is_failure_slice", False))
    if row_selection_policy == "weak_action":
        return bool(row.get("is_weak_action", False))
    return any(
        bool(row.get(key, False))
        for key in ("is_disagreement_candidate", "is_loop_risk", "is_failure_slice", "is_weak_action")
    )


def summarize_dagger_rows(rows: list[dict], confusion_pairs: set[tuple[str, str]] | None = None) -> dict:
    summary = {
        "rows": len(rows),
        "disagreement_rows": sum(int(bool(row.get("is_disagreement_candidate", False))) for row in rows),
        "loop_risk_rows": sum(int(bool(row.get("is_loop_risk", False))) for row in rows),
        "failure_slice_rows": sum(int(bool(row.get("is_failure_slice", False))) for row in rows),
        "weak_action_rows": sum(int(bool(row.get("is_weak_action", False))) for row in rows),
        "match_rows": sum(int(row.get("behavior_action") == row.get("teacher_action")) for row in rows),
    }
    if confusion_pairs:
        summary["confusion_pair_rows"] = sum(int(_matches_confusion_pairs(row, confusion_pairs)) for row in rows)
    return summary


def select_dagger_rows(
    *,
    rows: list[dict],
    row_selection_policy: str,
    keep_match_ratio: float,
    confusion_pairs: set[tuple[str, str]] | None,
    rng: random.Random,
) -> list[dict]:
    keep_match_ratio = _normalize_keep_ratio(keep_match_ratio)
    confusion_pairs = confusion_pairs or set()
    if row_selection_policy == "all" and keep_match_ratio >= 1.0 and not confusion_pairs:
        return list(rows)

    selected = []
    anchor_rows = []
    for row in rows:
        if _matches_row_selection_policy(row, row_selection_policy) and _matches_confusion_pairs(row, confusion_pairs):
            selected.append(row)
        else:
            anchor_rows.append(row)

    if keep_match_ratio <= 0.0 or not anchor_rows:
        return selected

    keep_anchor = max(1, int(round(len(anchor_rows) * keep_match_ratio)))
    if keep_anchor <= 0:
        return selected
    if keep_anchor >= len(anchor_rows):
        return selected + anchor_rows

    sampled = rng.sample(anchor_rows, keep_anchor)
    sampled.sort(key=lambda row: (row.get("episode_id", ""), row.get("step", 0)))
    return selected + sampled


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
    teacher_bc_model_path: str | None = None,
    observation_version: str = "v1",
    merge_ratio: float = 0.5,
    merge_policy: str = "uniform_merge",
    dagger_row_policy: str = "all",
    dagger_keep_match_ratio: float = 0.0,
    dagger_confusion_pairs: str = "",
    epochs: int = 20,
    lr: float = 1e-3,
    hidden_size: int = 256,
    distill_teacher_bc_path: str | None = None,
    distill_loss_coef: float = 0.0,
    distill_temperature: float = 1.0,
    merged_trace_output: str | None = None,
    heldout_trace_input: str | None = None,
    random_seed: int = 123,
) -> dict:
    if student_policy == "bc" and not bc_model_path:
        raise ValueError("bc_model_path is required for student_policy=bc")
    if student_policy == "appo" and not (appo_experiment or appo_checkpoint_path):
        raise ValueError("appo_experiment or appo_checkpoint_path is required for student_policy=appo")

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
        teacher_bc_model_path=teacher_bc_model_path,
        observation_version=observation_version,
    )

    base_rows = load_trace_rows(base_trace_input)
    dagger_rows = load_trace_rows(dagger_trace_output)
    confusion_pairs = _parse_confusion_pairs(dagger_confusion_pairs)
    selected_dagger_rows = select_dagger_rows(
        rows=dagger_rows,
        row_selection_policy=dagger_row_policy,
        keep_match_ratio=dagger_keep_match_ratio,
        confusion_pairs=confusion_pairs,
        rng=random.Random(random_seed + 100_003),
    )
    merged_rows = build_merged_trace_rows(
        base_rows=base_rows,
        relabeled_rows=selected_dagger_rows,
        merge_policy=merge_policy,
        merge_ratio=merge_ratio,
        rng=random.Random(random_seed),
    )

    _write_rows(merged_trace_output, merged_rows)

    effective_distill_teacher = distill_teacher_bc_path or teacher_bc_model_path
    train_observation_version = merged_rows[0].get("observation_version", observation_version) if merged_rows else observation_version
    train_result = train_bc_model(
        merged_rows,
        bc_output,
        epochs=epochs,
        lr=lr,
        hidden_size=hidden_size,
        observation_version=train_observation_version,
        distill_teacher_bc_path=effective_distill_teacher,
        distill_loss_coef=distill_loss_coef,
        distill_temperature=distill_temperature,
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
        "teacher_bc_model_path": teacher_bc_model_path,
        "task": task,
        "merge_ratio": merge_ratio,
        "merge_policy": merge_policy,
        "distill_teacher_bc_path": effective_distill_teacher,
        "distill_loss_coef": distill_loss_coef,
        "distill_temperature": distill_temperature,
        "base_rows": len(base_rows),
        "dagger_rows": len(dagger_rows),
        "selected_dagger_rows": len(selected_dagger_rows),
        "dagger_row_policy": dagger_row_policy,
        "dagger_keep_match_ratio": dagger_keep_match_ratio,
        "dagger_confusion_pairs": sorted(f"{behavior}->{teacher}" for behavior, teacher in confusion_pairs),
        "dagger_row_summary": summarize_dagger_rows(dagger_rows, confusion_pairs=confusion_pairs),
        "selected_dagger_row_summary": summarize_dagger_rows(selected_dagger_rows, confusion_pairs=confusion_pairs),
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
    teacher_bc_model_path: str | None = None,
    observation_version: str = "v1",
    merge_ratio: float = 0.5,
    merge_policy: str = "uniform_merge",
    dagger_row_policy: str = "all",
    dagger_keep_match_ratio: float = 0.0,
    dagger_confusion_pairs: str = "",
    epochs: int = 20,
    lr: float = 1e-3,
    hidden_size: int = 256,
    distill_teacher_bc_path: str | None = None,
    distill_loss_coef: float = 0.0,
    distill_temperature: float = 1.0,
    heldout_trace_input: str | None = None,
    random_seed: int = 123,
    stop_on_heldout_regression: bool = False,
    min_improvement: float = 0.0,
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    current_base = base_trace_input
    current_bc_model = bc_model_path
    reports = []
    best_heldout = float("-inf")
    best_iteration = None
    best_bc_model = None
    best_trace_input = None

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
            teacher_bc_model_path=teacher_bc_model_path,
            observation_version=observation_version,
            merge_ratio=merge_ratio,
            merge_policy=merge_policy,
            dagger_row_policy=dagger_row_policy,
            dagger_keep_match_ratio=dagger_keep_match_ratio,
            dagger_confusion_pairs=dagger_confusion_pairs,
            epochs=epochs,
            lr=lr,
            hidden_size=hidden_size,
            distill_teacher_bc_path=distill_teacher_bc_path,
            distill_loss_coef=distill_loss_coef,
            distill_temperature=distill_temperature,
            merged_trace_output=merged_trace_output,
            heldout_trace_input=heldout_trace_input,
            random_seed=random_seed + iteration,
        )
        report["iteration"] = iteration
        report["seed_start"] = iteration_seed
        heldout_match = None
        if report.get("heldout_trace_eval") is not None:
            heldout_match = report["heldout_trace_eval"]["match_rate"]
            report["heldout_match_delta"] = None if best_heldout == float("-inf") else round(heldout_match - best_heldout, 4)
            if heldout_match > best_heldout + min_improvement:
                best_heldout = heldout_match
                best_iteration = iteration
                best_bc_model = bc_output
                best_trace_input = merged_trace_output
        elif best_iteration is None:
            best_iteration = iteration
            best_bc_model = bc_output
            best_trace_input = merged_trace_output
        reports.append(report)
        if stop_on_heldout_regression and heldout_match is not None and best_heldout != float("-inf") and heldout_match + min_improvement < best_heldout:
            report["early_stop_triggered"] = True
            break
        current_base = merged_trace_output
        current_bc_model = bc_output

    summary = {
        "base_trace_input": base_trace_input,
        "heldout_trace_input": heldout_trace_input,
        "output_dir": output_dir,
        "student_policy": student_policy,
        "teacher_bc_model_path": teacher_bc_model_path,
        "task": task,
        "iterations": iterations,
        "merge_policy": merge_policy,
        "merge_ratio": merge_ratio,
        "dagger_row_policy": dagger_row_policy,
        "dagger_keep_match_ratio": dagger_keep_match_ratio,
        "dagger_confusion_pairs": sorted(
            f"{behavior}->{teacher}" for behavior, teacher in _parse_confusion_pairs(dagger_confusion_pairs)
        ),
        "distill_teacher_bc_path": distill_teacher_bc_path or teacher_bc_model_path,
        "distill_loss_coef": distill_loss_coef,
        "distill_temperature": distill_temperature,
        "observation_version": observation_version,
        "final_bc_model": current_bc_model,
        "final_trace_input": current_base,
        "best_iteration": best_iteration,
        "best_bc_model": best_bc_model or current_bc_model,
        "best_trace_input": best_trace_input or current_base,
        "best_heldout_match_rate": None if best_heldout == float("-inf") else round(best_heldout, 4),
        "stop_on_heldout_regression": stop_on_heldout_regression,
        "min_improvement": min_improvement,
        "reports": reports,
    }
    atomic_write_json(Path(output_dir) / "dagger_schedule_report.json", summary)
    return summary
