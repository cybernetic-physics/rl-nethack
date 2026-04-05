from __future__ import annotations

import json
import os
from dataclasses import asdict

from rl.config import RLConfig
from rl.model import build_model_spec
from rl.sf_env import make_nethack_skill_env


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
            "scheduler_model_path": self.config.options.scheduler_model_path,
            "reward_source": self.config.reward.source,
            "learned_reward_path": self.config.reward.learned_reward_path,
            "model": asdict(self.model_spec),
            "dependency_status": self.dependency_status(),
        }

    def write_plan(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.render_training_plan(), f, indent=2)
            f.write("\n")
        return path

    def build_sf_argv(self) -> list[str]:
        cfg = self.config
        worker_num_splits = 1 if cfg.serial_mode or cfg.rollout.num_envs_per_worker % 2 != 0 else 2
        return [
            f"--algo=APPO",
            f"--env={cfg.env.env_id}",
            f"--experiment={cfg.experiment}",
            f"--train_dir={cfg.train_dir}",
            f"--restart_behavior=overwrite",
            f"--device={'cpu' if cfg.serial_mode else 'gpu'}",
            f"--seed={cfg.env.seed}",
            f"--serial_mode={str(cfg.serial_mode)}",
            f"--async_rl={str(cfg.async_rl)}",
            f"--num_workers={cfg.rollout.num_workers}",
            f"--num_envs_per_worker={cfg.rollout.num_envs_per_worker}",
            f"--worker_num_splits={worker_num_splits}",
            f"--rollout={cfg.rollout.rollout_length}",
            f"--recurrence={cfg.rollout.recurrence}",
            f"--batch_size={cfg.appo.batch_size}",
            f"--num_batches_per_epoch={cfg.appo.num_batches_per_epoch}",
            f"--num_epochs={cfg.appo.ppo_epochs}",
            f"--ppo_clip_ratio={cfg.appo.ppo_clip_ratio}",
            f"--value_loss_coeff={cfg.appo.value_loss_coeff}",
            f"--exploration_loss_coeff={cfg.appo.entropy_coeff}",
            f"--gamma={cfg.appo.gamma}",
            f"--gae_lambda={cfg.appo.gae_lambda}",
            f"--learning_rate={cfg.appo.learning_rate}",
            f"--max_grad_norm={cfg.appo.max_grad_norm}",
            f"--reward_scale={cfg.appo.reward_scale}",
            f"--train_for_env_steps={cfg.appo.train_for_env_steps}",
            f"--use_rnn={str(cfg.model.use_lstm)}",
            f"--env_max_episode_steps={cfg.env.max_episode_steps}",
            f"--reward_source={cfg.reward.source}",
            f"--intrinsic_reward_weight={cfg.reward.intrinsic_weight}",
            f"--extrinsic_reward_weight={cfg.reward.extrinsic_weight}",
            f"--skill_scheduler={cfg.options.scheduler}",
            f"--scheduler_model_path={cfg.options.scheduler_model_path or ''}",
            f"--enabled_skills={','.join(cfg.options.enabled_skills)}",
            f"--active_skill_bootstrap={cfg.env.active_skill_bootstrap}",
            f"--learned_reward_path={cfg.reward.learned_reward_path or ''}",
            f"--enforce_action_mask={str(cfg.env.enforce_action_mask)}",
            f"--invalid_action_penalty={cfg.reward.invalid_action_penalty}",
            f"--invalid_action_fallback={cfg.env.invalid_action_fallback}",
        ]

    def launch(self, dry_run: bool = True) -> dict:
        plan = self.render_training_plan()
        if dry_run or not plan["dependency_status"]["sample_factory_available"]:
            return {
                "status": "dry_run",
                "plan": plan,
                "message": (
                    "Dry run only."
                    if plan["dependency_status"]["sample_factory_available"]
                    else "Sample Factory is not available in the active environment."
                ),
            }

        from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
        from sample_factory.envs.env_utils import register_env
        from sample_factory.train import run_rl

        argv = self.build_sf_argv()
        register_env(self.config.env.env_id, make_nethack_skill_env)
        parser, _ = parse_sf_args(argv)
        parser.add_argument("--env_max_episode_steps", type=int, default=self.config.env.max_episode_steps)
        parser.add_argument("--reward_source", type=str, default=self.config.reward.source)
        parser.add_argument("--intrinsic_reward_weight", type=float, default=self.config.reward.intrinsic_weight)
        parser.add_argument("--extrinsic_reward_weight", type=float, default=self.config.reward.extrinsic_weight)
        parser.add_argument("--skill_scheduler", type=str, default=self.config.options.scheduler)
        parser.add_argument("--scheduler_model_path", type=str, default=self.config.options.scheduler_model_path)
        parser.add_argument("--enabled_skills", type=str, default=",".join(self.config.options.enabled_skills))
        parser.add_argument("--active_skill_bootstrap", type=str, default=self.config.env.active_skill_bootstrap)
        parser.add_argument("--learned_reward_path", type=str, default=self.config.reward.learned_reward_path)
        parser.add_argument("--enforce_action_mask", type=str, default=str(self.config.env.enforce_action_mask))
        parser.add_argument("--invalid_action_penalty", type=float, default=self.config.reward.invalid_action_penalty)
        parser.add_argument("--invalid_action_fallback", type=str, default=self.config.env.invalid_action_fallback)
        sf_cfg = parse_full_cfg(parser, argv)
        status = run_rl(sf_cfg)
        return {
            "status": str(status),
            "plan": plan,
            "argv": argv,
        }
