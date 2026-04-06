from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict

import torch

from rl.config import RLConfig
from rl.checkpoint_tools import TraceCheckpointMonitor, record_warmstart_trace_match
from rl.io_utils import atomic_torch_save, atomic_write_text, experiment_lock
from rl.improver_report import write_improver_report, build_improver_report
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
            "proxy_reward_path": self.config.reward.proxy_reward_path,
            "proxy_reward_weight": self.config.reward.proxy_reward_weight,
            "teacher_bc_path": self.config.appo.teacher_bc_path,
            "teacher_prior_bc_path": self.config.appo.teacher_prior_bc_path,
            "teacher_report_path": self.config.model.teacher_report_path,
            "teacher_loss_coef": self.config.appo.teacher_loss_coef,
            "teacher_loss_type": self.config.appo.teacher_loss_type,
            "teacher_action_boosts": self.config.appo.teacher_action_boosts,
            "teacher_loss_final_coef": self.config.appo.teacher_loss_final_coef,
            "teacher_loss_warmup_env_steps": self.config.appo.teacher_loss_warmup_env_steps,
            "teacher_loss_decay_env_steps": self.config.appo.teacher_loss_decay_env_steps,
            "teacher_replay_trace_input": self.config.appo.teacher_replay_trace_input,
            "teacher_replay_coef": self.config.appo.teacher_replay_coef,
            "teacher_replay_final_coef": self.config.appo.teacher_replay_final_coef,
            "teacher_replay_warmup_env_steps": self.config.appo.teacher_replay_warmup_env_steps,
            "teacher_replay_decay_env_steps": self.config.appo.teacher_replay_decay_env_steps,
            "teacher_replay_batch_size": self.config.appo.teacher_replay_batch_size,
            "teacher_replay_priority_power": self.config.appo.teacher_replay_priority_power,
            "teacher_replay_source_mode": self.config.appo.teacher_replay_source_mode,
            "teacher_policy_logit_residual_scale": self.config.appo.teacher_policy_logit_residual_scale,
            "teacher_policy_blend_coef": self.config.appo.teacher_policy_blend_coef,
            "teacher_policy_fallback_confidence": self.config.appo.teacher_policy_fallback_confidence,
            "teacher_policy_disagreement_margin": self.config.appo.teacher_policy_disagreement_margin,
            "param_anchor_coef": self.config.appo.param_anchor_coef,
            "actor_loss_scale": self.config.appo.actor_loss_scale,
            "actor_loss_final_scale": self.config.appo.actor_loss_final_scale,
            "actor_loss_warmup_env_steps": self.config.appo.actor_loss_warmup_env_steps,
            "actor_loss_decay_env_steps": self.config.appo.actor_loss_decay_env_steps,
            "trace_eval_input": self.config.appo.trace_eval_input,
            "trace_eval_interval_env_steps": self.config.appo.trace_eval_interval_env_steps,
            "trace_eval_top_k": self.config.appo.trace_eval_top_k,
            "save_every_sec": self.config.appo.save_every_sec,
            "save_best_every_sec": self.config.appo.save_best_every_sec,
            "episodic_explore_bonus_enabled": self.config.reward.episodic_explore_bonus_enabled,
            "episodic_explore_bonus_scale": self.config.reward.episodic_explore_bonus_scale,
            "episodic_explore_bonus_mode": self.config.reward.episodic_explore_bonus_mode,
            "observation_version": self.config.env.observation_version,
            "world_model_path": self.config.env.world_model_path,
            "world_model_feature_mode": self.config.env.world_model_feature_mode,
            "appo_init_checkpoint_path": self.config.model.appo_init_checkpoint_path,
            "improver_report_output": self.config.appo.improver_report_output,
            "model": asdict(self.model_spec),
            "dependency_status": self.dependency_status(),
        }

    def write_plan(self, path: str) -> str:
        return atomic_write_text(path, json.dumps(self.render_training_plan(), indent=2) + "\n")

    def build_sf_argv(self) -> list[str]:
        cfg = self.config
        worker_num_splits = 1 if cfg.serial_mode or cfg.rollout.num_envs_per_worker % 2 != 0 else 2
        restart_behavior = "resume" if (cfg.model.bc_init_path or cfg.model.appo_init_checkpoint_path) else "overwrite"
        encoder_layers = [str(cfg.model.hidden_size)] * max(int(cfg.model.num_layers), 1)
        return [
            f"--algo=APPO",
            f"--env={cfg.env.env_id}",
            f"--experiment={cfg.experiment}",
            f"--train_dir={cfg.train_dir}",
            f"--restart_behavior={restart_behavior}",
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
            "--encoder_mlp_layers",
            *encoder_layers,
            f"--normalize_input={str(cfg.model.normalize_input)}",
            f"--nonlinearity={cfg.model.nonlinearity}",
            f"--actor_critic_share_weights={str(cfg.model.actor_critic_share_weights)}",
            f"--gamma={cfg.appo.gamma}",
            f"--gae_lambda={cfg.appo.gae_lambda}",
            f"--learning_rate={cfg.appo.learning_rate}",
            f"--max_grad_norm={cfg.appo.max_grad_norm}",
            f"--reward_scale={cfg.appo.reward_scale}",
            f"--train_for_env_steps={cfg.appo.train_for_env_steps}",
            f"--teacher_loss_coef={cfg.appo.teacher_loss_coef}",
            f"--teacher_loss_type={cfg.appo.teacher_loss_type}",
            f"--teacher_bc_path={cfg.appo.teacher_bc_path or ''}",
            f"--teacher_prior_bc_path={cfg.appo.teacher_prior_bc_path or ''}",
            f"--teacher_report_path={cfg.model.teacher_report_path or ''}",
            f"--teacher_action_boosts={cfg.appo.teacher_action_boosts}",
            f"--teacher_loss_final_coef={cfg.appo.teacher_loss_final_coef}",
            f"--teacher_loss_warmup_env_steps={cfg.appo.teacher_loss_warmup_env_steps}",
            f"--teacher_loss_decay_env_steps={cfg.appo.teacher_loss_decay_env_steps}",
            f"--teacher_replay_trace_input={cfg.appo.teacher_replay_trace_input or ''}",
            f"--teacher_replay_coef={cfg.appo.teacher_replay_coef}",
            f"--teacher_replay_final_coef={cfg.appo.teacher_replay_final_coef}",
            f"--teacher_replay_warmup_env_steps={cfg.appo.teacher_replay_warmup_env_steps}",
            f"--teacher_replay_decay_env_steps={cfg.appo.teacher_replay_decay_env_steps}",
            f"--teacher_replay_batch_size={cfg.appo.teacher_replay_batch_size}",
            f"--teacher_replay_priority_power={cfg.appo.teacher_replay_priority_power}",
            f"--teacher_replay_source_mode={cfg.appo.teacher_replay_source_mode}",
            f"--teacher_policy_logit_residual_scale={cfg.appo.teacher_policy_logit_residual_scale}",
            f"--teacher_policy_blend_coef={cfg.appo.teacher_policy_blend_coef}",
            f"--teacher_policy_fallback_confidence={cfg.appo.teacher_policy_fallback_confidence}",
            f"--teacher_policy_disagreement_margin={cfg.appo.teacher_policy_disagreement_margin}",
            f"--param_anchor_coef={cfg.appo.param_anchor_coef}",
            f"--actor_loss_scale={cfg.appo.actor_loss_scale}",
            f"--actor_loss_final_scale={cfg.appo.actor_loss_final_scale}",
            f"--actor_loss_warmup_env_steps={cfg.appo.actor_loss_warmup_env_steps}",
            f"--actor_loss_decay_env_steps={cfg.appo.actor_loss_decay_env_steps}",
            f"--trace_eval_input={cfg.appo.trace_eval_input or ''}",
            f"--trace_eval_interval_env_steps={cfg.appo.trace_eval_interval_env_steps}",
            f"--trace_eval_top_k={cfg.appo.trace_eval_top_k}",
            f"--save_every_sec={cfg.appo.save_every_sec}",
            f"--save_best_every_sec={cfg.appo.save_best_every_sec}",
            f"--improver_report_output={cfg.appo.improver_report_output or ''}",
            f"--use_rnn={str(cfg.model.use_lstm)}",
            f"--env_max_episode_steps={cfg.env.max_episode_steps}",
            f"--observation_version={cfg.env.observation_version}",
            f"--world_model_path={cfg.env.world_model_path or ''}",
            f"--world_model_feature_mode={cfg.env.world_model_feature_mode or ''}",
            f"--reward_source={cfg.reward.source}",
            f"--intrinsic_reward_weight={cfg.reward.intrinsic_weight}",
            f"--extrinsic_reward_weight={cfg.reward.extrinsic_weight}",
            f"--proxy_reward_path={cfg.reward.proxy_reward_path or ''}",
            f"--proxy_reward_weight={cfg.reward.proxy_reward_weight}",
            f"--episodic_explore_bonus_enabled={str(cfg.reward.episodic_explore_bonus_enabled)}",
            f"--episodic_explore_bonus_scale={cfg.reward.episodic_explore_bonus_scale}",
            f"--episodic_explore_bonus_mode={cfg.reward.episodic_explore_bonus_mode}",
            f"--skill_scheduler={cfg.options.scheduler}",
            f"--scheduler_model_path={cfg.options.scheduler_model_path or ''}",
            f"--enabled_skills={','.join(cfg.options.enabled_skills)}",
            f"--active_skill_bootstrap={cfg.env.active_skill_bootstrap}",
            f"--learned_reward_path={cfg.reward.learned_reward_path or ''}",
            f"--bc_init_path={cfg.model.bc_init_path or ''}",
            f"--appo_init_checkpoint_path={cfg.model.appo_init_checkpoint_path or ''}",
            f"--enforce_action_mask={str(cfg.env.enforce_action_mask)}",
            f"--invalid_action_penalty={cfg.reward.invalid_action_penalty}",
            f"--invalid_action_fallback={cfg.env.invalid_action_fallback}",
        ]

    @staticmethod
    def _copy_prefix(dst: torch.Tensor, src: torch.Tensor) -> None:
        rows = min(dst.shape[0], src.shape[0])
        if dst.ndim == 1:
            dst[:rows].copy_(src[:rows])
            return
        cols = min(dst.shape[1], src.shape[1])
        dst[:rows, :cols].copy_(src[:rows, :cols])

    @staticmethod
    def _linear_state_keys(state_dict: dict[str, torch.Tensor], module_prefix: str) -> list[tuple[str, str]]:
        weight_keys = sorted(
            [key for key in state_dict if key.startswith(module_prefix) and key.endswith(".weight")],
            key=lambda key: int(key.rsplit(".", 2)[1]),
        )
        bias_keys = {
            key.rsplit(".", 1)[0]: key
            for key in state_dict
            if key.startswith(module_prefix) and key.endswith(".bias")
        }
        linear_layers: list[tuple[str, str]] = []
        for weight_key in weight_keys:
            prefix = weight_key.rsplit(".", 1)[0]
            bias_key = bias_keys.get(prefix)
            if bias_key:
                linear_layers.append((weight_key, bias_key))
        return linear_layers

    @staticmethod
    def _model_encoder_prefixes(state_dict: dict[str, torch.Tensor]) -> tuple[str, str | None]:
        actor_prefix = "actor_encoder.encoders.obs.mlp_head."
        critic_prefix = "critic_encoder.encoders.obs.mlp_head."
        if any(key.startswith(actor_prefix) for key in state_dict):
            return actor_prefix, critic_prefix
        shared_prefix = "encoder.encoders.obs.mlp_head."
        return shared_prefix, None

    def maybe_write_bc_warmstart_checkpoint(self, sf_cfg, actor_critic) -> str | None:
        if not self.config.model.bc_init_path:
            return None

        payload = torch.load(self.config.model.bc_init_path, map_location="cpu")
        bc_state = payload["state_dict"]
        model_state = actor_critic.state_dict()
        linear_layers = self._linear_state_keys(bc_state, "net.")
        if len(linear_layers) < 2:
            raise ValueError("BC warmstart requires at least one hidden linear layer and one output linear layer")
        hidden_layers = linear_layers[:-1]
        output_weight_key, output_bias_key = linear_layers[-1]
        actor_prefix, critic_prefix = self._model_encoder_prefixes(model_state)
        actor_hidden_layers = self._linear_state_keys(model_state, actor_prefix)
        for (dst_weight_key, dst_bias_key), (src_weight_key, src_bias_key) in zip(actor_hidden_layers, hidden_layers):
            self._copy_prefix(model_state[dst_weight_key], bc_state[src_weight_key])
            self._copy_prefix(model_state[dst_bias_key], bc_state[src_bias_key])
        if critic_prefix is not None:
            critic_hidden_layers = self._linear_state_keys(model_state, critic_prefix)
            for (dst_weight_key, dst_bias_key), (src_weight_key, src_bias_key) in zip(critic_hidden_layers, hidden_layers):
                self._copy_prefix(model_state[dst_weight_key], bc_state[src_weight_key])
                self._copy_prefix(model_state[dst_bias_key], bc_state[src_bias_key])
        self._copy_prefix(
            model_state["action_parameterization.distribution_linear.weight"],
            bc_state[output_weight_key],
        )
        self._copy_prefix(
            model_state["action_parameterization.distribution_linear.bias"],
            bc_state[output_bias_key],
        )
        actor_critic.load_state_dict(model_state, strict=False)

        checkpoint_dir = os.path.join(self.config.train_dir, self.config.experiment, "checkpoint_p0")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_000000000_0.pth")
        optimizer = torch.optim.Adam(actor_critic.parameters(), lr=sf_cfg.learning_rate, eps=sf_cfg.adam_eps)
        atomic_torch_save(
            checkpoint_path,
            {
                "best_performance": float("-inf"),
                "curr_lr": float(sf_cfg.learning_rate),
                "env_steps": 0,
                "model": actor_critic.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_step": 0,
            },
        )
        return checkpoint_path

    def maybe_write_appo_warmstart_checkpoint(self, sf_cfg, actor_critic) -> str | None:
        if not self.config.model.appo_init_checkpoint_path:
            return None

        payload = torch.load(self.config.model.appo_init_checkpoint_path, map_location="cpu", weights_only=False)
        actor_critic.load_state_dict(payload["model"], strict=False)

        checkpoint_dir = os.path.join(self.config.train_dir, self.config.experiment, "checkpoint_p0")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_000000000_0.pth")
        optimizer = torch.optim.Adam(actor_critic.parameters(), lr=sf_cfg.learning_rate, eps=sf_cfg.adam_eps)
        atomic_torch_save(
            checkpoint_path,
            {
                "best_performance": float("-inf"),
                "curr_lr": float(sf_cfg.learning_rate),
                "env_steps": 0,
                "model": actor_critic.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_step": 0,
            },
        )
        return checkpoint_path

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
        from sample_factory.model.actor_critic import create_actor_critic
        from sample_factory.train import run_rl
        from gymnasium import spaces
        from rl.trace_eval import evaluate_trace_appo_bundle
        from rl.teacher_reg import patch_sample_factory_teacher_reg

        argv = self.build_sf_argv()
        patch_sample_factory_teacher_reg()
        register_env(self.config.env.env_id, make_nethack_skill_env)
        parser, _ = parse_sf_args(argv)
        parser.add_argument("--env_max_episode_steps", type=int, default=self.config.env.max_episode_steps)
        parser.add_argument("--observation_version", type=str, default=self.config.env.observation_version)
        parser.add_argument("--world_model_path", type=str, default=self.config.env.world_model_path)
        parser.add_argument("--world_model_feature_mode", type=str, default=self.config.env.world_model_feature_mode)
        parser.add_argument("--reward_source", type=str, default=self.config.reward.source)
        parser.add_argument("--intrinsic_reward_weight", type=float, default=self.config.reward.intrinsic_weight)
        parser.add_argument("--extrinsic_reward_weight", type=float, default=self.config.reward.extrinsic_weight)
        parser.add_argument("--proxy_reward_path", type=str, default=self.config.reward.proxy_reward_path)
        parser.add_argument("--proxy_reward_weight", type=float, default=self.config.reward.proxy_reward_weight)
        parser.add_argument("--episodic_explore_bonus_enabled", type=str, default=str(self.config.reward.episodic_explore_bonus_enabled))
        parser.add_argument("--episodic_explore_bonus_scale", type=float, default=self.config.reward.episodic_explore_bonus_scale)
        parser.add_argument("--episodic_explore_bonus_mode", type=str, default=self.config.reward.episodic_explore_bonus_mode)
        parser.add_argument("--skill_scheduler", type=str, default=self.config.options.scheduler)
        parser.add_argument("--scheduler_model_path", type=str, default=self.config.options.scheduler_model_path)
        parser.add_argument("--enabled_skills", type=str, default=",".join(self.config.options.enabled_skills))
        parser.add_argument("--active_skill_bootstrap", type=str, default=self.config.env.active_skill_bootstrap)
        parser.add_argument("--learned_reward_path", type=str, default=self.config.reward.learned_reward_path)
        parser.add_argument("--bc_init_path", type=str, default=self.config.model.bc_init_path)
        parser.add_argument("--appo_init_checkpoint_path", type=str, default=self.config.model.appo_init_checkpoint_path)
        parser.add_argument("--teacher_bc_path", type=str, default=self.config.appo.teacher_bc_path)
        parser.add_argument("--teacher_prior_bc_path", type=str, default=self.config.appo.teacher_prior_bc_path)
        parser.add_argument("--teacher_report_path", type=str, default=self.config.model.teacher_report_path)
        parser.add_argument("--teacher_loss_coef", type=float, default=self.config.appo.teacher_loss_coef)
        parser.add_argument("--teacher_loss_type", type=str, default=self.config.appo.teacher_loss_type)
        parser.add_argument("--teacher_action_boosts", type=str, default=self.config.appo.teacher_action_boosts)
        parser.add_argument("--teacher_loss_final_coef", type=float, default=self.config.appo.teacher_loss_final_coef)
        parser.add_argument("--teacher_loss_warmup_env_steps", type=int, default=self.config.appo.teacher_loss_warmup_env_steps)
        parser.add_argument("--teacher_loss_decay_env_steps", type=int, default=self.config.appo.teacher_loss_decay_env_steps)
        parser.add_argument("--teacher_replay_trace_input", type=str, default=self.config.appo.teacher_replay_trace_input)
        parser.add_argument("--teacher_replay_coef", type=float, default=self.config.appo.teacher_replay_coef)
        parser.add_argument("--teacher_replay_final_coef", type=float, default=self.config.appo.teacher_replay_final_coef)
        parser.add_argument("--teacher_replay_warmup_env_steps", type=int, default=self.config.appo.teacher_replay_warmup_env_steps)
        parser.add_argument("--teacher_replay_decay_env_steps", type=int, default=self.config.appo.teacher_replay_decay_env_steps)
        parser.add_argument("--teacher_replay_batch_size", type=int, default=self.config.appo.teacher_replay_batch_size)
        parser.add_argument("--teacher_replay_priority_power", type=float, default=self.config.appo.teacher_replay_priority_power)
        parser.add_argument("--teacher_replay_source_mode", type=str, default=self.config.appo.teacher_replay_source_mode)
        parser.add_argument(
            "--teacher_policy_logit_residual_scale",
            type=float,
            default=self.config.appo.teacher_policy_logit_residual_scale,
        )
        parser.add_argument("--teacher_policy_blend_coef", type=float, default=self.config.appo.teacher_policy_blend_coef)
        parser.add_argument(
            "--teacher_policy_fallback_confidence",
            type=float,
            default=self.config.appo.teacher_policy_fallback_confidence,
        )
        parser.add_argument(
            "--teacher_policy_disagreement_margin",
            type=float,
            default=self.config.appo.teacher_policy_disagreement_margin,
        )
        parser.add_argument("--param_anchor_coef", type=float, default=self.config.appo.param_anchor_coef)
        parser.add_argument("--actor_loss_scale", type=float, default=self.config.appo.actor_loss_scale)
        parser.add_argument("--actor_loss_final_scale", type=float, default=self.config.appo.actor_loss_final_scale)
        parser.add_argument("--actor_loss_warmup_env_steps", type=int, default=self.config.appo.actor_loss_warmup_env_steps)
        parser.add_argument("--actor_loss_decay_env_steps", type=int, default=self.config.appo.actor_loss_decay_env_steps)
        parser.add_argument("--trace_eval_input", type=str, default=self.config.appo.trace_eval_input)
        parser.add_argument("--trace_eval_interval_env_steps", type=int, default=self.config.appo.trace_eval_interval_env_steps)
        parser.add_argument("--trace_eval_top_k", type=int, default=self.config.appo.trace_eval_top_k)
        parser.add_argument("--improver_report_output", type=str, default=self.config.appo.improver_report_output)
        parser.add_argument("--enforce_action_mask", type=str, default=str(self.config.env.enforce_action_mask))
        parser.add_argument("--invalid_action_penalty", type=float, default=self.config.reward.invalid_action_penalty)
        parser.add_argument("--invalid_action_fallback", type=str, default=self.config.env.invalid_action_fallback)
        sf_cfg = parse_full_cfg(parser, argv)
        warmstart_checkpoint = None
        experiment_dir = os.path.join(self.config.train_dir, self.config.experiment)
        lock_path = os.path.join(experiment_dir, ".launch.lock")
        with experiment_lock(lock_path):
            if (self.config.model.bc_init_path or self.config.model.appo_init_checkpoint_path) and os.path.isdir(experiment_dir):
                shutil.rmtree(experiment_dir)
            if self.config.model.bc_init_path or self.config.model.appo_init_checkpoint_path:
                env = make_nethack_skill_env(None, sf_cfg, None)
                actor_critic = create_actor_critic(sf_cfg, spaces.Dict({"obs": env.observation_space}), env.action_space)
                if self.config.model.appo_init_checkpoint_path:
                    warmstart_checkpoint = self.maybe_write_appo_warmstart_checkpoint(sf_cfg, actor_critic)
                else:
                    warmstart_checkpoint = self.maybe_write_bc_warmstart_checkpoint(sf_cfg, actor_critic)
                if warmstart_checkpoint and self.config.appo.trace_eval_input:
                    warmstart_result = evaluate_trace_appo_bundle(
                        self.config.appo.trace_eval_input,
                        cfg=sf_cfg,
                        actor_critic=actor_critic,
                        device="cpu",
                        summary_only=True,
                    )
                    record_warmstart_trace_match(
                        experiment=self.config.experiment,
                        train_dir=self.config.train_dir,
                        trace_input=self.config.appo.trace_eval_input,
                        checkpoint_path=warmstart_checkpoint,
                        evaluation={
                            "env_steps": 0,
                            "match_rate": warmstart_result["summary"]["match_rate"],
                            "invalid_action_rate": warmstart_result["summary"]["invalid_action_rate"],
                            "action_counts": warmstart_result["summary"]["action_counts"],
                        },
                    )
                env.close()
            monitor = None
            try:
                if self.config.appo.trace_eval_input and self.config.appo.trace_eval_interval_env_steps > 0:
                    monitor = TraceCheckpointMonitor(
                        experiment=self.config.experiment,
                        train_dir=self.config.train_dir,
                        trace_input=self.config.appo.trace_eval_input,
                        interval_env_steps=self.config.appo.trace_eval_interval_env_steps,
                    )
                    monitor.start()
                status = run_rl(sf_cfg)
            finally:
                if monitor is not None:
                    monitor.stop()
            improver_report_path = write_improver_report(
                build_improver_report(
                    config=self.config,
                    plan=plan,
                    argv=argv,
                    warmstart_checkpoint=warmstart_checkpoint,
                    status=str(status),
                ),
                self.config.appo.improver_report_output,
            )
            return {
                "status": str(status),
                "plan": plan,
                "argv": argv,
                "warmstart_checkpoint": warmstart_checkpoint,
                "experiment_lock": lock_path,
                "improver_report_path": improver_report_path,
            }
