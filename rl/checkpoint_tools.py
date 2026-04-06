from __future__ import annotations

import re
import threading
import time
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


def checkpoint_env_steps(checkpoint_path: str | Path) -> int:
    path = Path(checkpoint_path)
    match = re.match(r"^(?:checkpoint|best)_(\d+)_(\d+)", path.name)
    if not match:
        return -1
    return int(match.group(2))


def evaluate_checkpoint_trace_match(
    *,
    experiment: str,
    train_dir: str,
    trace_input: str,
    checkpoint_path: str,
) -> dict:
    result = evaluate_trace_policy(
        trace_path=trace_input,
        policy="appo",
        appo_experiment=experiment,
        appo_train_dir=train_dir,
        appo_checkpoint_path=checkpoint_path,
        summary_only=True,
    )
    summary = result["summary"]
    return {
        "checkpoint_path": checkpoint_path,
        "env_steps": checkpoint_env_steps(checkpoint_path),
        "match_rate": summary["match_rate"],
        "invalid_action_rate": summary["invalid_action_rate"],
        "action_counts": summary["action_counts"],
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


def materialize_trace_checkpoint_alias(checkpoint_path: str, alias_name: str) -> str | None:
    source = Path(checkpoint_path)
    if not source.exists():
        return None
    alias_path = source.parent / alias_name
    atomic_copy_file(source, alias_path)
    return str(alias_path)


def record_warmstart_trace_match(
    *,
    experiment: str,
    train_dir: str,
    trace_input: str,
    checkpoint_path: str,
    evaluation: dict,
) -> dict:
    checkpoint_dir = Path(train_dir) / experiment / "checkpoint_p0"
    metadata = {
        "experiment": experiment,
        "trace_input": trace_input,
        "warmstart_checkpoint_path": checkpoint_path,
        "env_steps": evaluation["env_steps"],
        "match_rate": evaluation["match_rate"],
        "invalid_action_rate": evaluation["invalid_action_rate"],
        "action_counts": evaluation["action_counts"],
    }
    alias_path = materialize_trace_checkpoint_alias(checkpoint_path, "warmstart_trace_match.pth")
    if alias_path:
        metadata["alias_checkpoint_path"] = alias_path
    atomic_write_json(checkpoint_dir / "warmstart_trace_match.json", metadata)
    return metadata


class TraceCheckpointMonitor:
    def __init__(
        self,
        *,
        experiment: str,
        train_dir: str,
        trace_input: str,
        interval_env_steps: int,
        poll_seconds: float = 5.0,
    ):
        self.experiment = experiment
        self.train_dir = train_dir
        self.trace_input = trace_input
        self.interval_env_steps = max(1, interval_env_steps)
        self.poll_seconds = poll_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._best_match_rate = float("-inf")
        self._seen_paths: set[str] = set()
        self._next_eval_at = self.interval_env_steps

    def _evaluate_due_checkpoints(self) -> None:
        try:
            checkpoint_paths = list_checkpoint_paths(self.experiment, self.train_dir)
        except FileNotFoundError:
            return
        checkpoint_dir = Path(self.train_dir) / self.experiment / "checkpoint_p0"
        for checkpoint_path in checkpoint_paths:
            checkpoint_path_str = str(checkpoint_path)
            if checkpoint_path_str in self._seen_paths:
                continue
            env_steps = checkpoint_env_steps(checkpoint_path)
            if env_steps == 0:
                self._seen_paths.add(checkpoint_path_str)
                continue
            if env_steps < self._next_eval_at:
                continue
            evaluation = evaluate_checkpoint_trace_match(
                experiment=self.experiment,
                train_dir=self.train_dir,
                trace_input=self.trace_input,
                checkpoint_path=checkpoint_path_str,
            )
            self._seen_paths.add(checkpoint_path_str)
            if evaluation["match_rate"] > self._best_match_rate:
                self._best_match_rate = evaluation["match_rate"]
                metadata = {
                    "experiment": self.experiment,
                    "trace_input": self.trace_input,
                    "best_checkpoint_path": checkpoint_path_str,
                    "env_steps": evaluation["env_steps"],
                    "match_rate": evaluation["match_rate"],
                    "invalid_action_rate": evaluation["invalid_action_rate"],
                    "action_counts": evaluation["action_counts"],
                }
                materialize_trace_best_checkpoint(
                    {
                        "experiment": self.experiment,
                        "trace_input": self.trace_input,
                        "best_checkpoint_path": checkpoint_path_str,
                        "num_checkpoints": len(checkpoint_paths),
                    }
                )
                atomic_write_json(checkpoint_dir / "best_trace_match.json", metadata)
            self._next_eval_at = max(self._next_eval_at + self.interval_env_steps, env_steps + self.interval_env_steps)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._evaluate_due_checkpoints()
            except Exception:
                pass
            self._stop_event.wait(self.poll_seconds)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name=f"trace-monitor-{self.experiment}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_seconds * 2)
        self._evaluate_due_checkpoints()
