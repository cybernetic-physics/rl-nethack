from __future__ import annotations

from pathlib import Path

from rl.evaluate import list_checkpoint_paths
from rl.io_utils import atomic_copy_file, atomic_write_json
from rl.trace_eval import evaluate_trace_policy


def rank_appo_checkpoints_by_trace(
    *,
    experiment: str,
    train_dir: str,
    trace_input: str,
    top_k: int = 5,
) -> dict:
    ranked = []
    skipped = []
    for checkpoint_path in list_checkpoint_paths(experiment, train_dir):
        try:
            result = evaluate_trace_policy(
                trace_path=trace_input,
                policy="appo",
                appo_experiment=experiment,
                appo_train_dir=train_dir,
                appo_checkpoint_path=str(checkpoint_path),
                summary_only=True,
            )
            summary = result["summary"]
            ranked.append(
                {
                    "checkpoint_path": str(checkpoint_path),
                    "match_rate": summary["match_rate"],
                    "invalid_action_rate": summary["invalid_action_rate"],
                    "action_counts": summary["action_counts"],
                }
            )
        except Exception as exc:
            skipped.append({"checkpoint_path": str(checkpoint_path), "error": f"{type(exc).__name__}: {exc}"})

    ranked.sort(key=lambda row: (row["match_rate"], -row["invalid_action_rate"], row["checkpoint_path"]), reverse=True)
    return {
        "experiment": experiment,
        "train_dir": train_dir,
        "trace_input": trace_input,
        "num_checkpoints": len(ranked),
        "num_skipped": len(skipped),
        "best_checkpoint_path": ranked[0]["checkpoint_path"] if ranked else None,
        "ranked": ranked[:top_k],
        "all_ranked": ranked,
        "skipped": skipped,
    }


def write_trace_best_alias(result: dict, output_path: str) -> str:
    payload = {
        "experiment": result["experiment"],
        "trace_input": result["trace_input"],
        "best_checkpoint_path": result["best_checkpoint_path"],
        "num_checkpoints": result["num_checkpoints"],
    }
    return atomic_write_json(output_path, payload)


def materialize_trace_best_checkpoint(result: dict) -> str | None:
    best_checkpoint_path = result.get("best_checkpoint_path")
    if not best_checkpoint_path:
        return None

    best_source = Path(best_checkpoint_path)
    if not best_source.exists():
        return None

    alias_path = best_source.parent / "best_trace_match.pth"
    atomic_copy_file(best_source, alias_path)
    metadata_path = best_source.parent / "best_trace_match.json"
    atomic_write_json(
        metadata_path,
        {
            "experiment": result["experiment"],
            "trace_input": result["trace_input"],
            "best_checkpoint_path": str(best_source),
            "alias_checkpoint_path": str(alias_path),
            "num_checkpoints": result["num_checkpoints"],
        },
    )
    return str(alias_path)
