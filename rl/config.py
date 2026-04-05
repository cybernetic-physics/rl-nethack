from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class EnvConfig:
    env_id: str = "rl_nethack_skill"
    seed: int = 42
    max_episode_steps: int = 200
    use_memory_tracker: bool = True
    active_skill_bootstrap: str = "explore"


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


@dataclass
class ModelConfig:
    backbone: str = "skill_conditioned_mlp"
    hidden_size: int = 512
    use_lstm: bool = True
    lstm_size: int = 512
    skill_embedding_dim: int = 32
    share_backbone_across_skills: bool = True


@dataclass
class RewardConfig:
    source: str = "hand_shaped"
    learned_reward_path: str | None = None
    extrinsic_weight: float = 0.0
    intrinsic_weight: float = 1.0
    repeated_state_penalty: float = 0.25
    repeated_action_penalty: float = 0.25


@dataclass
class OptionConfig:
    enabled_skills: list[str] = field(
        default_factory=lambda: ["explore", "survive", "combat", "descend", "resource"]
    )
    scheduler: str = "rule_based"
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
