from __future__ import annotations

import json
import os
from dataclasses import asdict

from rl.config import RLConfig
from rl.model import build_model_spec


class APPOTrainerScaffold:
    """Thin trainer bootstrap around a future Sample Factory integration."""

    def __init__(self, config: RLConfig):
        self.config = config
        self.model_spec = build_model_spec(config.model)

    def dependency_status(self) -> dict:
        try:
            import sample_factory  # noqa: F401
            available = True
        except Exception:
            available = False
        return {
            "sample_factory_available": available,
            "backend": "sample_factory_appo" if available else "scaffold_only",
        }

    def render_training_plan(self) -> dict:
        total_envs = self.config.rollout.num_workers * self.config.rollout.num_envs_per_worker
        return {
            "experiment": self.config.experiment,
            "train_dir": self.config.train_dir,
            "total_parallel_envs": total_envs,
            "rollout_length": self.config.rollout.rollout_length,
            "recurrence": self.config.rollout.recurrence,
            "train_for_env_steps": self.config.appo.train_for_env_steps,
            "enabled_skills": list(self.config.options.enabled_skills),
            "scheduler": self.config.options.scheduler,
            "reward_source": self.config.reward.source,
            "model": asdict(self.model_spec),
            "dependency_status": self.dependency_status(),
        }

    def write_plan(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.render_training_plan(), f, indent=2)
            f.write("\n")
        return path

    def launch(self, dry_run: bool = True) -> dict:
        plan = self.render_training_plan()
        if dry_run or not plan["dependency_status"]["sample_factory_available"]:
            return {
                "status": "dry_run",
                "plan": plan,
                "message": "Sample Factory is not wired yet; scaffold only.",
            }
        raise NotImplementedError("Real APPO launch wiring is the next implementation step.")

