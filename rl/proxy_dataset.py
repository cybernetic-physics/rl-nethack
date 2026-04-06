from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from rl.proxy_labels import (
    k_step_loop_risk,
    k_step_progress,
    k_step_resource_value,
    k_step_survival,
    search_context_label,
    teacher_margin,
)
from rl.trace_eval import load_trace_rows


def build_proxy_rows(
    trace_rows: list[dict],
    *,
    horizon: int = 8,
    task_filter: str | None = None,
    max_rows: int | None = None,
) -> list[dict]:
    rows = sorted(trace_rows, key=lambda row: (row["episode_id"], row["step"]))
    proxy_rows: list[dict] = []
    current_episode: list[dict] = []
    current_episode_id: str | None = None

    def flush_episode() -> None:
        if not current_episode:
            return
        for idx, row in enumerate(current_episode):
            if task_filter and row.get("task") != task_filter:
                continue
            horizon_rows = current_episode[idx : min(len(current_episode), idx + horizon)]
            proxy_row = {
                "episode_id": row["episode_id"],
                "step": row["step"],
                "seed": row.get("seed"),
                "task": row.get("task"),
                "action": row["action"],
                "allowed_actions": row.get("allowed_actions", []),
                "feature_vector": row["feature_vector"],
                "observation_version": row.get("observation_version"),
                "obs_hash": row.get("obs_hash"),
                "next_obs_hash": row.get("next_obs_hash"),
                "prompt": row.get("prompt", ""),
                "planner_trace": row.get("planner_trace", []),
                "horizon": len(horizon_rows),
                "k_step_progress": k_step_progress(horizon_rows),
                "k_step_survival": k_step_survival(horizon_rows),
                "k_step_loop_risk": k_step_loop_risk(horizon_rows),
                "k_step_resource_value": k_step_resource_value(horizon_rows),
                "search_context_label": search_context_label(row),
                "teacher_margin": teacher_margin(row),
            }
            proxy_rows.append(proxy_row)
            if max_rows is not None and len(proxy_rows) >= max_rows:
                return

    for row in rows:
        episode_id = row["episode_id"]
        if current_episode_id is None:
            current_episode_id = episode_id
        if episode_id != current_episode_id:
            flush_episode()
            if max_rows is not None and len(proxy_rows) >= max_rows:
                break
            current_episode = []
            current_episode_id = episode_id
        current_episode.append(row)
    if max_rows is None or len(proxy_rows) < max_rows:
        flush_episode()
    if max_rows is not None:
        proxy_rows = proxy_rows[:max_rows]
    return proxy_rows


def summarize_proxy_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"rows": 0}
    action_counts: dict[str, int] = {}
    for row in rows:
        action_counts[row["action"]] = action_counts.get(row["action"], 0) + 1
    return {
        "rows": len(rows),
        "observation_versions": sorted({row.get("observation_version", "unknown") for row in rows}),
        "feature_dims": sorted({len(row.get("feature_vector", [])) for row in rows}),
        "action_counts": action_counts,
        "search_context_positive_rate": round(
            sum(int(row["search_context_label"]) for row in rows) / max(1, len(rows)),
            4,
        ),
        "avg_k_step_progress": round(mean(row["k_step_progress"] for row in rows), 4),
        "avg_k_step_survival": round(mean(row["k_step_survival"] for row in rows), 4),
        "avg_k_step_loop_risk": round(mean(row["k_step_loop_risk"] for row in rows), 4),
        "avg_k_step_resource_value": round(mean(row["k_step_resource_value"] for row in rows), 4),
        "avg_teacher_margin": round(mean(row["teacher_margin"] for row in rows), 4),
        "top_progress_examples": [
            {
                "episode_id": row["episode_id"],
                "step": row["step"],
                "action": row["action"],
                "k_step_progress": row["k_step_progress"],
            }
            for row in sorted(rows, key=lambda r: r["k_step_progress"], reverse=True)[:5]
        ],
        "top_loop_risk_examples": [
            {
                "episode_id": row["episode_id"],
                "step": row["step"],
                "action": row["action"],
                "k_step_loop_risk": row["k_step_loop_risk"],
            }
            for row in sorted(rows, key=lambda r: r["k_step_loop_risk"], reverse=True)[:5]
        ],
    }


def load_proxy_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_proxy_rows(rows: list[dict], path: str) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return str(target)


def build_proxy_dataset_from_trace_file(
    *,
    input_path: str,
    output_path: str,
    horizon: int = 8,
    task_filter: str | None = None,
    max_rows: int | None = None,
) -> dict:
    trace_rows = load_trace_rows(input_path)
    proxy_rows = build_proxy_rows(trace_rows, horizon=horizon, task_filter=task_filter, max_rows=max_rows)
    write_proxy_rows(proxy_rows, output_path)
    summary = summarize_proxy_rows(proxy_rows)
    summary.update(
        {
            "input_path": input_path,
            "output_path": output_path,
            "horizon": horizon,
            "task_filter": task_filter,
            "max_rows": max_rows,
        }
    )
    return summary
