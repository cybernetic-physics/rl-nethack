from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from rl.config import RLConfig
from rl.trainer import APPOTrainerScaffold


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="APPO + options RL scaffold")
    parser.add_argument("--experiment", type=str, default="appo_options_scaffold")
    parser.add_argument("--train-dir", type=str, default="train_dir/rl")
    parser.add_argument("--serial-mode", action="store_true")
    parser.add_argument("--async-rl", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-envs-per-worker", type=int, default=16)
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument("--recurrence", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-batches-per-epoch", type=int, default=1)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--value-loss-coeff", type=float, default=0.5)
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--ppo-clip-ratio", type=float, default=0.1)
    parser.add_argument("--train-for-env-steps", type=int, default=50_000_000)
    parser.add_argument("--scheduler", type=str, default="rule_based")
    parser.add_argument("--reward-source", type=str, default="hand_shaped")
    parser.add_argument("--learned-reward-path", type=str, default=None)
    parser.add_argument("--proxy-reward-path", type=str, default=None)
    parser.add_argument("--proxy-reward-weight", type=float, default=1.0)
    parser.add_argument("--episodic-explore-bonus-enabled", action="store_true")
    parser.add_argument("--episodic-explore-bonus-scale", type=float, default=0.0)
    parser.add_argument("--episodic-explore-bonus-mode", type=str, default="state_hash", choices=["state_hash", "tile"])
    parser.add_argument("--scheduler-model-path", type=str, default=None)
    parser.add_argument("--enabled-skills", type=str, default="explore,survive,combat,descend,resource")
    parser.add_argument("--observation-version", type=str, default="v2")
    parser.add_argument("--world-model-path", type=str, default=None)
    parser.add_argument("--world-model-feature-mode", type=str, default=None, choices=["replace", "concat", "concat_aux"])
    parser.add_argument("--env-max-episode-steps", type=int, default=5000)
    parser.add_argument("--model-hidden-size", type=int, default=None)
    parser.add_argument("--model-num-layers", type=int, default=None)
    parser.add_argument("--separate-actor-critic", action="store_true")
    parser.add_argument("--disable-input-normalization", action="store_true")
    parser.add_argument("--nonlinearity", type=str, default=None, choices=["elu", "relu", "tanh"])
    parser.add_argument("--bc-init-path", type=str, default=None)
    parser.add_argument("--appo-init-checkpoint-path", type=str, default=None)
    parser.add_argument("--teacher-bc-path", type=str, default=None)
    parser.add_argument("--teacher-prior-bc-path", type=str, default=None)
    parser.add_argument("--teacher-report-path", type=str, default=None)
    parser.add_argument("--teacher-loss-coef", type=float, default=0.01)
    parser.add_argument("--teacher-loss-type", type=str, default="ce", choices=["ce", "kl"])
    parser.add_argument("--teacher-action-boosts", type=str, default="")
    parser.add_argument("--teacher-loss-final-coef", type=float, default=0.0)
    parser.add_argument("--teacher-loss-warmup-env-steps", type=int, default=0)
    parser.add_argument("--teacher-loss-decay-env-steps", type=int, default=0)
    parser.add_argument("--teacher-replay-trace-input", type=str, default=None)
    parser.add_argument("--teacher-replay-coef", type=float, default=0.0)
    parser.add_argument("--teacher-replay-final-coef", type=float, default=0.0)
    parser.add_argument("--teacher-replay-warmup-env-steps", type=int, default=0)
    parser.add_argument("--teacher-replay-decay-env-steps", type=int, default=0)
    parser.add_argument("--teacher-replay-batch-size", type=int, default=128)
    parser.add_argument("--teacher-replay-priority-power", type=float, default=1.0)
    parser.add_argument("--teacher-replay-source-mode", type=str, default="uniform")
    parser.add_argument("--teacher-policy-logit-residual-scale", type=float, default=1.0)
    parser.add_argument("--teacher-policy-blend-coef", type=float, default=0.0)
    parser.add_argument("--teacher-policy-fallback-confidence", type=float, default=0.0)
    parser.add_argument("--teacher-policy-disagreement-margin", type=float, default=0.0)
    parser.add_argument("--param-anchor-coef", type=float, default=0.0)
    parser.add_argument("--actor-loss-scale", type=float, default=1.0)
    parser.add_argument("--actor-loss-final-scale", type=float, default=1.0)
    parser.add_argument("--actor-loss-warmup-env-steps", type=int, default=0)
    parser.add_argument("--actor-loss-decay-env-steps", type=int, default=0)
    parser.add_argument("--trace-eval-input", type=str, default=None)
    parser.add_argument("--trace-eval-interval-env-steps", type=int, default=0)
    parser.add_argument("--trace-eval-top-k", type=int, default=5)
    parser.add_argument("--save-every-sec", type=int, default=120)
    parser.add_argument("--save-best-every-sec", type=int, default=5)
    parser.add_argument("--no-rnn", action="store_true")
    parser.add_argument("--use-rnn", action="store_true")
    parser.add_argument("--disable-action-mask", action="store_true")
    parser.add_argument("--write-plan", type=str, default=None)
    parser.add_argument("--improver-report-output", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_config(args) -> RLConfig:
    config = RLConfig()
    config.experiment = args.experiment
    config.train_dir = args.train_dir
    config.serial_mode = args.serial_mode
    config.async_rl = args.async_rl or not args.serial_mode
    config.rollout.num_workers = args.num_workers
    config.rollout.num_envs_per_worker = args.num_envs_per_worker
    config.rollout.rollout_length = args.rollout_length
    config.rollout.recurrence = args.recurrence
    config.appo.batch_size = args.batch_size
    config.appo.num_batches_per_epoch = args.num_batches_per_epoch
    config.appo.ppo_epochs = args.ppo_epochs
    config.appo.learning_rate = float(getattr(args, "learning_rate", config.appo.learning_rate))
    config.appo.gamma = float(getattr(args, "gamma", config.appo.gamma))
    config.appo.gae_lambda = float(getattr(args, "gae_lambda", config.appo.gae_lambda))
    config.appo.value_loss_coeff = float(getattr(args, "value_loss_coeff", config.appo.value_loss_coeff))
    config.appo.reward_scale = float(getattr(args, "reward_scale", config.appo.reward_scale))
    config.appo.entropy_coeff = float(getattr(args, "entropy_coeff", config.appo.entropy_coeff))
    config.appo.ppo_clip_ratio = float(getattr(args, "ppo_clip_ratio", config.appo.ppo_clip_ratio))
    config.appo.train_for_env_steps = args.train_for_env_steps
    config.options.scheduler = args.scheduler
    config.options.scheduler_model_path = args.scheduler_model_path
    config.reward.source = args.reward_source
    config.reward.learned_reward_path = args.learned_reward_path
    config.reward.proxy_reward_path = getattr(args, "proxy_reward_path", None)
    config.reward.proxy_reward_weight = float(getattr(args, "proxy_reward_weight", config.reward.proxy_reward_weight))
    config.reward.episodic_explore_bonus_enabled = args.episodic_explore_bonus_enabled
    config.reward.episodic_explore_bonus_scale = args.episodic_explore_bonus_scale
    config.reward.episodic_explore_bonus_mode = args.episodic_explore_bonus_mode
    config.options.enabled_skills = [s.strip() for s in args.enabled_skills.split(",") if s.strip()]
    config.env.observation_version = args.observation_version
    config.env.world_model_path = getattr(args, "world_model_path", None)
    config.env.world_model_feature_mode = getattr(args, "world_model_feature_mode", None)
    config.env.max_episode_steps = getattr(args, "env_max_episode_steps", config.env.max_episode_steps)
    if args.model_hidden_size is not None:
        config.model.hidden_size = args.model_hidden_size
    config.model.actor_critic_share_weights = not bool(getattr(args, "separate_actor_critic", False))
    if getattr(args, "model_num_layers", None) is not None:
        config.model.num_layers = int(args.model_num_layers)
    elif getattr(args, "appo_init_checkpoint_path", None):
        try:
            payload = torch.load(args.appo_init_checkpoint_path, map_location="cpu", weights_only=False)
            model_state = payload.get("model", {})
            first_layer = model_state.get("encoder.encoders.obs.mlp_head.0.weight")
            if first_layer is not None:
                config.model.hidden_size = int(first_layer.shape[0])
        except Exception:
            pass
    elif args.bc_init_path:
        try:
            payload = torch.load(args.bc_init_path, map_location="cpu")
            metadata = payload.get("metadata", {})
            config.model.hidden_size = int(metadata.get("hidden_size", config.model.hidden_size))
            config.model.num_layers = int(metadata.get("num_layers", config.model.num_layers))
        except Exception:
            pass
    config.model.bc_init_path = args.bc_init_path
    config.model.appo_init_checkpoint_path = getattr(args, "appo_init_checkpoint_path", None)
    config.model.teacher_report_path = getattr(args, "teacher_report_path", None)
    appo_init_config_path = None
    if config.model.appo_init_checkpoint_path:
        checkpoint_path = Path(config.model.appo_init_checkpoint_path)
        try:
            appo_init_config_path = checkpoint_path.parent.parent / "config.json"
        except Exception:
            appo_init_config_path = None
    nonlinearity = getattr(args, "nonlinearity", None)
    if nonlinearity is not None:
        config.model.nonlinearity = nonlinearity
    elif appo_init_config_path and appo_init_config_path.exists():
        try:
            source_cfg = json.loads(appo_init_config_path.read_text())
            config.model.nonlinearity = str(source_cfg.get("nonlinearity", config.model.nonlinearity))
        except Exception:
            pass
    elif args.bc_init_path:
        config.model.nonlinearity = "relu"
    disable_input_normalization = bool(getattr(args, "disable_input_normalization", False))
    config.model.normalize_input = not disable_input_normalization
    if appo_init_config_path and appo_init_config_path.exists() and getattr(args, "disable_input_normalization", False) is False:
        try:
            source_cfg = json.loads(appo_init_config_path.read_text())
            normalize_input = source_cfg.get("normalize_input", None)
            if normalize_input is not None:
                if isinstance(normalize_input, str):
                    config.model.normalize_input = normalize_input.lower() == "true"
                else:
                    config.model.normalize_input = bool(normalize_input)
        except Exception:
            pass
    if args.bc_init_path and not disable_input_normalization:
        config.model.normalize_input = False
    config.appo.teacher_bc_path = args.teacher_bc_path or args.bc_init_path
    config.appo.teacher_prior_bc_path = getattr(args, "teacher_prior_bc_path", None)
    config.appo.teacher_loss_coef = args.teacher_loss_coef
    config.appo.teacher_loss_type = args.teacher_loss_type
    config.appo.teacher_action_boosts = args.teacher_action_boosts
    config.appo.teacher_loss_final_coef = args.teacher_loss_final_coef
    config.appo.teacher_loss_warmup_env_steps = args.teacher_loss_warmup_env_steps
    config.appo.teacher_loss_decay_env_steps = args.teacher_loss_decay_env_steps
    config.appo.teacher_replay_trace_input = getattr(args, "teacher_replay_trace_input", None)
    config.appo.teacher_replay_coef = float(getattr(args, "teacher_replay_coef", config.appo.teacher_replay_coef))
    config.appo.teacher_replay_final_coef = float(
        getattr(args, "teacher_replay_final_coef", config.appo.teacher_replay_final_coef)
    )
    config.appo.teacher_replay_warmup_env_steps = int(
        getattr(args, "teacher_replay_warmup_env_steps", config.appo.teacher_replay_warmup_env_steps)
    )
    config.appo.teacher_replay_decay_env_steps = int(
        getattr(args, "teacher_replay_decay_env_steps", config.appo.teacher_replay_decay_env_steps)
    )
    config.appo.teacher_replay_batch_size = int(
        getattr(args, "teacher_replay_batch_size", config.appo.teacher_replay_batch_size)
    )
    config.appo.teacher_replay_priority_power = float(
        getattr(args, "teacher_replay_priority_power", config.appo.teacher_replay_priority_power)
    )
    config.appo.teacher_replay_source_mode = str(
        getattr(args, "teacher_replay_source_mode", config.appo.teacher_replay_source_mode)
    )
    config.appo.teacher_policy_logit_residual_scale = float(
        getattr(args, "teacher_policy_logit_residual_scale", config.appo.teacher_policy_logit_residual_scale)
    )
    config.appo.teacher_policy_blend_coef = float(
        getattr(args, "teacher_policy_blend_coef", config.appo.teacher_policy_blend_coef)
    )
    config.appo.teacher_policy_fallback_confidence = float(
        getattr(args, "teacher_policy_fallback_confidence", config.appo.teacher_policy_fallback_confidence)
    )
    config.appo.teacher_policy_disagreement_margin = float(
        getattr(args, "teacher_policy_disagreement_margin", config.appo.teacher_policy_disagreement_margin)
    )
    config.appo.param_anchor_coef = args.param_anchor_coef
    config.appo.actor_loss_scale = float(getattr(args, "actor_loss_scale", config.appo.actor_loss_scale))
    config.appo.actor_loss_final_scale = float(
        getattr(args, "actor_loss_final_scale", config.appo.actor_loss_final_scale)
    )
    config.appo.actor_loss_warmup_env_steps = int(
        getattr(args, "actor_loss_warmup_env_steps", config.appo.actor_loss_warmup_env_steps)
    )
    config.appo.actor_loss_decay_env_steps = int(
        getattr(args, "actor_loss_decay_env_steps", config.appo.actor_loss_decay_env_steps)
    )
    config.appo.trace_eval_input = args.trace_eval_input
    config.appo.trace_eval_interval_env_steps = args.trace_eval_interval_env_steps
    config.appo.trace_eval_top_k = args.trace_eval_top_k
    config.appo.save_every_sec = int(getattr(args, "save_every_sec", config.appo.save_every_sec))
    config.appo.save_best_every_sec = int(getattr(args, "save_best_every_sec", config.appo.save_best_every_sec))
    config.appo.improver_report_output = getattr(args, "improver_report_output", None)
    config.model.use_lstm = bool(args.use_rnn and not args.no_rnn)
    if not config.model.use_lstm:
        config.rollout.recurrence = 1
    config.env.enforce_action_mask = not args.disable_action_mask
    return config


def main(argv=None):
    args = parse_args(argv)
    config = build_config(args)
    trainer = APPOTrainerScaffold(config)

    if args.write_plan:
        path = trainer.write_plan(args.write_plan)
        print(f"Wrote APPO plan to: {path}")

    result = trainer.launch(dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
