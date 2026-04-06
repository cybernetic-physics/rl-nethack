from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class EnvConfig:
    env_id: str = "rl_nethack_skill"
    seed: int = 42
    max_episode_steps: int = 5000
    use_memory_tracker: bool = True
    active_skill_bootstrap: str = "explore"
    observation_version: str = "v1"
    world_model_path: str | None = None
    world_model_feature_mode: str | None = None
    enforce_action_mask: bool = True
    invalid_action_fallback: str = "wait"


@dataclass
class RolloutConfig:
    num_workers: int = 8
    num_envs_per_worker: int = 16
    rollout_length: int = 64
    recurrence: int = 32


@dataclass
class APPOConfig:
    train_for_env_steps: int = 50_000_000
    batch_size: int = 4096
    num_batches_per_epoch: int = 1
    ppo_clip_ratio: float = 0.1
    ppo_epochs: int = 1
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    gamma: float = 0.999
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    max_grad_norm: float = 4.0
    reward_scale: float = 0.1
    teacher_loss_coef: float = 0.0
    teacher_loss_type: str = "ce"
    teacher_bc_path: str | None = None
    teacher_action_boosts: str = ""
    teacher_loss_final_coef: float = 0.0
    teacher_loss_warmup_env_steps: int = 0
    teacher_loss_decay_env_steps: int = 0
    teacher_replay_trace_input: str | None = None
    teacher_replay_coef: float = 0.0
    teacher_replay_final_coef: float = 0.0
    teacher_replay_warmup_env_steps: int = 0
    teacher_replay_decay_env_steps: int = 0
    teacher_replay_batch_size: int = 128
    teacher_replay_priority_power: float = 1.0
    teacher_replay_source_mode: str = "uniform"
    param_anchor_coef: float = 0.0
    actor_loss_scale: float = 1.0
    actor_loss_final_scale: float = 1.0
    actor_loss_warmup_env_steps: int = 0
    actor_loss_decay_env_steps: int = 0
    trace_eval_input: str | None = None
    trace_eval_interval_env_steps: int = 0
    trace_eval_top_k: int = 5
    save_every_sec: int = 120
    save_best_every_sec: int = 5
    improver_report_output: str | None = None


@dataclass
class ModelConfig:
    backbone: str = "skill_conditioned_mlp"
    hidden_size: int = 512
    num_layers: int = 2
    normalize_input: bool = True
    nonlinearity: str = "elu"
    use_lstm: bool = True
    lstm_size: int = 512
    skill_embedding_dim: int = 32
    share_backbone_across_skills: bool = True
    bc_init_path: str | None = None
    appo_init_checkpoint_path: str | None = None
    teacher_report_path: str | None = None


@dataclass
class RewardConfig:
    source: str = "hand_shaped"
    learned_reward_path: str | None = None
    proxy_reward_path: str | None = None
    proxy_reward_weight: float = 1.0
    extrinsic_weight: float = 0.0
    intrinsic_weight: float = 1.0
    repeated_state_penalty: float = 0.25
    repeated_action_penalty: float = 0.25
    invalid_action_penalty: float = 2.0
    episodic_explore_bonus_enabled: bool = False
    episodic_explore_bonus_scale: float = 0.0
    episodic_explore_bonus_mode: str = "state_hash"


@dataclass
class OptionConfig:
    enabled_skills: list[str] = field(
        default_factory=lambda: ["explore", "survive", "combat", "descend", "resource"]
    )
    scheduler: str = "rule_based"
    scheduler_model_path: str | None = None
    max_steps_per_skill: int = 32
    allow_forced_skill_switch: bool = True


@dataclass
class RLConfig:
    experiment: str = "appo_options_scaffold"
    train_dir: str = "train_dir/rl"
    serial_mode: bool = False
    async_rl: bool = True
    env: EnvConfig = field(default_factory=EnvConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    appo: APPOConfig = field(default_factory=APPOConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    options: OptionConfig = field(default_factory=OptionConfig)

    def to_dict(self) -> dict:
        return asdict(self)
