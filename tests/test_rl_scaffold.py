import os
import sys
import json
import tempfile
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rl.config import RLConfig
from rl.feature_encoder import ACTION_SET, action_mask_slice, observation_dim, encode_observation
from rl.reward_model import reward_feature_dim
from rl.scheduler_model import scheduler_feature_dim
from rl.sf_env import NethackSkillEnv
from rl.options import build_skill_registry
from rl.scheduler import SchedulerContext, build_scheduler
from rl.trainer import APPOTrainerScaffold
from rl.train_bc import load_trace_rows, train_bc_model, _parse_action_weight_boosts
from rl.bc_model import load_bc_model
from rl.debug_tools import check_policy_determinism, compare_actions_on_teacher_states
from rl.trace_eval import evaluate_trace_policy, trace_disagreement_report
from rl.evaluate import _load_checkpoint_payload
from rl.traces import (
    shard_trace_file,
    generate_dagger_traces,
    generate_multi_turn_traces,
    verify_trace_file,
    parse_adjacent_from_prompt_text,
    matches_adjacent_signature,
)
from rl.checkpoint_tools import TraceCheckpointMonitor, write_trace_best_alias, rank_appo_checkpoints_by_trace
import rl.checkpoint_tools as checkpoint_tools
import rl.dagger as dagger_module
import cli as cli_module
from pathlib import Path
import torch
from rl.timestep import build_policy_timestep
from rl.io_utils import experiment_lock
from rl.teacher_reg import (
    patch_sample_factory_teacher_reg,
    _action_mask_from_raw_obs,
    _forward_replay_action_logits,
    _invalid_preference_fraction,
    _mask_logits_with_action_mask,
    _parse_teacher_bc_paths,
    _parse_teacher_action_boosts,
    _parse_teacher_confusion_pair_boosts,
    _resolve_teacher_prior_bc_paths,
    _teacher_policy_blend,
    _teacher_policy_logit_residual,
    _teacher_policy_fallback_details,
    _teacher_policy_fallback_mask,
    _scheduled_actor_loss_scale,
    _scheduled_teacher_replay_coef,
    _row_replay_flags,
    _replay_priority_weights,
    _weight_replay_losses_by_confusion_pairs,
    _weight_replay_losses_by_current_disagreement,
)
from rl.env_adapter import SkillEnvAdapter, EpisodeContext
from rl.dagger import build_merged_trace_rows, run_dagger_iteration, run_dagger_schedule, select_dagger_rows
from rl.train_behavior_reg import train_behavior_regularized_policy, _masked_behavior_targets
from rl.train_appo import build_config as build_appo_config
from rl.teacher_report import build_teacher_report
from rl.improver_report import build_improver_report
from rl.train_world_model import train_world_model
from rl.world_model import load_world_model
from rl.world_model_dataset import build_world_model_examples, examples_to_arrays
from rl.world_model_eval import evaluate_world_model
from rl.world_model_features import (
    strip_action_from_prompt,
    state_prompt_from_row,
    transform_trace_with_world_model,
    world_model_augmented_dim,
)
from rl.proxy_dataset import build_proxy_rows, summarize_proxy_rows, write_proxy_rows
from rl.proxy_eval import evaluate_proxy_model
from rl.proxy_labels import search_context_label, teacher_margin
from rl.proxy_model import load_proxy_model
from rl.relabel_traces import relabel_trace_actions
from rl.rewards import RewardInputs, build_reward_source
from rl.train_proxy_model import train_proxy_model
from src.state_encoder import StateEncoder


def test_skill_registry_contains_expected_skills():
    registry = build_skill_registry()
    for name in ("explore", "survive", "combat", "descend", "resource"):
        assert name in registry
        assert registry[name].directive


def test_rule_based_scheduler_prefers_survive():
    scheduler = build_scheduler("rule_based")
    skill = scheduler.select_skill(
        SchedulerContext(
            state={"hp": 3, "hp_max": 10, "visible_monsters": [], "message": "", "adjacent": {}},
            memory=None,
            active_skill="explore",
            steps_in_skill=5,
            available_skills=["explore", "survive", "combat"],
        )
    )
    assert skill == "survive"


def test_parse_teacher_bc_paths_supports_csv():
    assert _parse_teacher_bc_paths(None) == []
    assert _parse_teacher_bc_paths("") == []
    assert _parse_teacher_bc_paths("/tmp/a.pt") == ["/tmp/a.pt"]
    assert _parse_teacher_bc_paths("/tmp/a.pt, /tmp/b.pt") == ["/tmp/a.pt", "/tmp/b.pt"]


def test_resolve_teacher_prior_bc_paths_prefers_explicit_prior():
    cfg = argparse.Namespace(teacher_bc_path="/tmp/a.pt,/tmp/b.pt", teacher_prior_bc_path="/tmp/c.pt")
    assert _resolve_teacher_prior_bc_paths(cfg) == ["/tmp/c.pt"]
    cfg = argparse.Namespace(teacher_bc_path="/tmp/a.pt,/tmp/b.pt", teacher_prior_bc_path=None)
    assert _resolve_teacher_prior_bc_paths(cfg) == ["/tmp/a.pt", "/tmp/b.pt"]


def test_parse_action_weight_boosts_supports_csv():
    assert _parse_action_weight_boosts(None) == {}
    assert _parse_action_weight_boosts("") == {}
    assert _parse_action_weight_boosts("east=2.0,west=1.5") == {
        ACTION_SET.index("east"): 2.0,
        ACTION_SET.index("west"): 1.5,
    }


def test_action_mask_slice_extracts_allowed_actions_prefix():
    timestep = {
        "state": {
            "hp": 10,
            "hp_max": 10,
            "gold": 0,
            "depth": 1,
            "turn": 1,
            "ac": 4,
            "strength": 10,
            "dexterity": 10,
            "visible_monsters": [],
            "visible_items": [],
            "adjacent": {"north": "floor", "south": "wall", "east": "floor", "west": "wall"},
            "position": (10, 10),
            "message": "",
        },
        "active_skill": "explore",
        "allowed_actions": ["north", "east", "search"],
        "memory_total_explored": 0,
        "rooms_discovered": 0,
        "steps_in_skill": 0,
        "repeated_state_count": 0,
        "revisited_recent_tile_count": 0,
        "repeated_action_count": 0,
        "standing_on_down_stairs": False,
        "standing_on_up_stairs": False,
        "recent_actions": [],
        "recent_positions": [],
        "obs": {"chars": np.zeros((21, 79), dtype=np.int16)},
    }
    features = encode_observation(timestep, version="v4")
    mask = features[action_mask_slice()]
    assert mask.shape == (len(ACTION_SET),)
    assert mask[ACTION_SET.index("north")] == 1.0
    assert mask[ACTION_SET.index("east")] == 1.0
    assert mask[ACTION_SET.index("search")] == 1.0
    assert mask[ACTION_SET.index("south")] == 0.0
    assert mask[ACTION_SET.index("west")] == 0.0


def test_teacher_reg_masks_invalid_teacher_and_student_preferences():
    raw_obs = torch.zeros((2, observation_dim("v4")), dtype=torch.float32)
    mask = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0] + [0.0] * (len(ACTION_SET) - 4),
            [0.0, 1.0, 1.0, 0.0] + [0.0] * (len(ACTION_SET) - 4),
        ],
        dtype=torch.float32,
    )
    raw_obs[:, action_mask_slice()] = mask
    extracted = _action_mask_from_raw_obs(raw_obs)
    assert torch.allclose(extracted, mask)

    logits = torch.tensor(
        [
            [0.0, 5.0, 1.0, 4.0] + [0.0] * (len(ACTION_SET) - 4),
            [6.0, 2.0, 1.0, 0.0] + [0.0] * (len(ACTION_SET) - 4),
        ],
        dtype=torch.float32,
    )
    assert round(float(_invalid_preference_fraction(logits, extracted).item()), 6) == 1.0

    masked_logits = _mask_logits_with_action_mask(logits, extracted)
    masked_actions = torch.argmax(masked_logits, dim=-1)
    assert ACTION_SET[int(masked_actions[0].item())] == "west"
    assert ACTION_SET[int(masked_actions[1].item())] == "south"


def test_teacher_policy_blend_and_fallback_helpers():
    student_logits = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)
    teacher_logits = torch.tensor([[0.0, 2.0, 1.0]], dtype=torch.float32)
    residual_logits = _teacher_policy_logit_residual(student_logits, teacher_logits, 0.25)
    assert torch.allclose(residual_logits, torch.tensor([[0.5, 1.75, 0.75]], dtype=torch.float32))

    student_probs = torch.tensor(
        [
            [0.70, 0.20, 0.10],
            [0.40, 0.35, 0.25],
        ],
        dtype=torch.float32,
    )
    teacher_probs = torch.tensor(
        [
            [0.10, 0.80, 0.10],
            [0.05, 0.90, 0.05],
        ],
        dtype=torch.float32,
    )
    blended = _teacher_policy_blend(student_probs, teacher_probs, 0.25)
    assert torch.allclose(
        blended,
        torch.tensor(
            [
                [0.55, 0.35, 0.10],
                [0.3125, 0.4875, 0.20],
            ],
            dtype=torch.float32,
        ),
    )
    fallback_mask = _teacher_policy_fallback_mask(student_probs, 0.5)
    assert fallback_mask.tolist() == [False, True]
    disagreement_fallback_mask, disagreement_mask, weak_override_mask = _teacher_policy_fallback_details(
        student_probs,
        0.0,
        teacher_probs=teacher_probs,
        disagreement_margin=0.15,
    )
    assert disagreement_mask.tolist() == [True, True]
    assert weak_override_mask.tolist() == [False, True]
    assert disagreement_fallback_mask.tolist() == [False, True]


def test_cli_mine_reset_slice_forwards_signature(monkeypatch):
    captured = {}

    def fake_mine(**kwargs):
        captured["kwargs"] = kwargs
        return {"rows": 0}

    monkeypatch.setattr("rl.traces.mine_reset_teacher_slice", fake_mine)
    args = argparse.Namespace(
        output="/tmp/mine.jsonl",
        seed_start=300,
        num_seeds=100,
        task="explore",
        observation_version="v4",
        adjacent_signature="north=monster_*,south=floor,east=monster_*,west=floor",
        recreate_every=123,
        max_rows=7,
    )
    rc = cli_module.cmd_rl_mine_reset_slice(args)
    assert rc == 0
    assert captured["kwargs"]["adjacent_signature"] == {
        "north": "monster_*",
        "south": "floor",
        "east": "monster_*",
        "west": "floor",
    }
    assert captured["kwargs"]["recreate_every"] == 123
    assert captured["kwargs"]["max_rows"] == 7


def test_parse_adjacent_from_prompt_text_and_match_signature():
    prompt = "\n".join(
        [
            "HP:14/14 AC:4",
            "Adjacent: north=monster_f south=floor east=monster_F west=floor",
            "Action: east",
        ]
    )
    adjacent = parse_adjacent_from_prompt_text(prompt)
    assert adjacent == {
        "north": "monster_f",
        "south": "floor",
        "east": "monster_F",
        "west": "floor",
    }
    assert matches_adjacent_signature(
        adjacent,
        {
            "north": "monster_*",
            "south": "floor",
            "east": "monster_*",
            "west": "floor",
        },
    )
    assert not matches_adjacent_signature(adjacent, {"east": "wall"})


def test_cli_train_bc_forwards_explicit_device(monkeypatch):
    captured = {}

    def fake_train(argv):
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr("rl.train_bc.main", fake_train)
    args = argparse.Namespace(
        input="/tmp/train.jsonl",
        output="/tmp/bc.pt",
        epochs=5,
        lr=1e-3,
        hidden_size=128,
        num_layers=2,
        observation_version="v4",
        world_model_path=None,
        world_model_feature_mode=None,
        distill_teacher_bc_path=None,
        distill_teacher_bc_paths=None,
        distill_loss_coef=0.0,
        distill_temperature=1.0,
        supervised_loss_coef=1.0,
        action_weight_boosts=None,
        text_encoder_backend="none",
        text_vocab_size=4096,
        text_embedding_dim=128,
        text_model_name=None,
        text_max_length=128,
        text_trainable=False,
        device="cuda",
        select_by_heldout=True,
        heldout_input=None,
        teacher_report_output=None,
        weak_action_input=None,
    )
    rc = cli_module.cmd_rl_train_bc(args)
    assert rc == 0
    argv = captured["argv"]
    assert "--device" in argv
    assert argv[argv.index("--device") + 1] == "cuda"
    assert "--select-by-heldout" in argv


def test_cli_rl_train_appo_forwards_teacher_prior_controls(monkeypatch):
    captured = {}

    def fake_train(argv):
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr("rl.train_appo.main", fake_train)
    args = argparse.Namespace(
        experiment="exp",
        train_dir="train_dir/rl",
        serial_mode=False,
        async_rl=False,
        num_workers=1,
        num_envs_per_worker=1,
        rollout_length=8,
        recurrence=8,
        batch_size=8,
        num_batches_per_epoch=1,
        ppo_epochs=1,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.9,
        value_loss_coeff=0.1,
        reward_scale=0.005,
        entropy_coeff=0.01,
        ppo_clip_ratio=0.1,
        train_for_env_steps=32,
        scheduler="rule_based",
        reward_source="hand_shaped",
        learned_reward_path=None,
        proxy_reward_path=None,
        proxy_reward_weight=1.0,
        episodic_explore_bonus_enabled=False,
        episodic_explore_bonus_scale=0.0,
        episodic_explore_bonus_mode="state_hash",
        scheduler_model_path=None,
        enabled_skills="explore",
        observation_version="v4",
        world_model_path=None,
        world_model_feature_mode=None,
        env_max_episode_steps=500,
        model_hidden_size=None,
        model_num_layers=None,
        separate_actor_critic=False,
        disable_input_normalization=False,
        nonlinearity=None,
        bc_init_path="/tmp/teacher.pt",
        appo_init_checkpoint_path=None,
        teacher_bc_path="/tmp/teacher.pt",
        teacher_prior_bc_path="/tmp/prior.pt",
        teacher_report_path=None,
        teacher_loss_coef=0.01,
        teacher_loss_type="ce",
        teacher_action_boosts="",
        teacher_loss_final_coef=0.0,
        teacher_loss_warmup_env_steps=0,
        teacher_loss_decay_env_steps=0,
        teacher_replay_trace_input=None,
        teacher_replay_coef=0.0,
        teacher_replay_final_coef=0.0,
        teacher_replay_warmup_env_steps=0,
        teacher_replay_decay_env_steps=0,
        teacher_replay_batch_size=128,
        teacher_replay_priority_power=1.0,
        teacher_replay_source_mode="uniform",
        teacher_replay_action_boosts="east=2.0,south=2.0",
        teacher_replay_current_disagreement_boost=2.5,
        teacher_replay_confusion_pair_boosts="east->south=3.0,south->east=3.0",
        teacher_policy_logit_residual_scale=0.3,
        teacher_policy_blend_coef=0.2,
        teacher_policy_fallback_confidence=0.6,
        teacher_policy_disagreement_margin=0.1,
        param_anchor_coef=0.0,
        actor_loss_scale=1.0,
        actor_loss_final_scale=1.0,
        actor_loss_warmup_env_steps=0,
        actor_loss_decay_env_steps=0,
        trace_eval_input=None,
        trace_eval_interval_env_steps=0,
        trace_eval_top_k=5,
        save_every_sec=120,
        save_best_every_sec=5,
        no_rnn=True,
        use_rnn=False,
        disable_action_mask=False,
        write_plan=None,
        improver_report_output=None,
        dry_run=False,
    )
    rc = cli_module.cmd_rl_train_appo(args)
    assert rc == 0
    argv = captured["argv"]
    assert "--teacher-policy-logit-residual-scale" in argv
    assert argv[argv.index("--teacher-policy-logit-residual-scale") + 1] == "0.3"
    assert "--teacher-replay-action-boosts" in argv
    assert argv[argv.index("--teacher-replay-action-boosts") + 1] == "east=2.0,south=2.0"
    assert "--teacher-replay-current-disagreement-boost" in argv
    assert argv[argv.index("--teacher-replay-current-disagreement-boost") + 1] == "2.5"
    assert "--teacher-replay-confusion-pair-boosts" in argv
    assert argv[argv.index("--teacher-replay-confusion-pair-boosts") + 1] == "east->south=3.0,south->east=3.0"
    assert "--teacher-policy-blend-coef" in argv
    assert argv[argv.index("--teacher-policy-blend-coef") + 1] == "0.2"
    assert "--teacher-policy-fallback-confidence" in argv
    assert argv[argv.index("--teacher-policy-fallback-confidence") + 1] == "0.6"
    assert "--teacher-policy-disagreement-margin" in argv
    assert argv[argv.index("--teacher-policy-disagreement-margin") + 1] == "0.1"
    assert "--teacher-prior-bc-path" in argv
    assert argv[argv.index("--teacher-prior-bc-path") + 1] == "/tmp/prior.pt"


def test_trainer_scaffold_renders_plan():
    config = RLConfig()
    trainer = APPOTrainerScaffold(config)
    plan = trainer.render_training_plan()
    assert plan["experiment"] == config.experiment
    assert plan["total_parallel_envs"] == config.rollout.num_workers * config.rollout.num_envs_per_worker
    assert "dependency_status" in plan


def test_trainer_scaffold_includes_teacher_reg_args():
    config = RLConfig()
    config.appo.teacher_bc_path = "/tmp/teacher.pt"
    config.appo.teacher_prior_bc_path = "/tmp/prior.pt"
    config.appo.teacher_loss_coef = 0.25
    config.appo.teacher_loss_type = "ce"
    config.appo.teacher_action_boosts = "west=2.0,south=1.5"
    config.appo.teacher_loss_final_coef = 0.002
    config.appo.teacher_loss_warmup_env_steps = 1024
    config.appo.teacher_loss_decay_env_steps = 4096
    config.appo.teacher_replay_action_boosts = "east=2.0,south=2.0"
    config.appo.teacher_replay_current_disagreement_boost = 2.0
    config.appo.teacher_replay_confusion_pair_boosts = "east->south=3.0,south->east=3.0"
    config.appo.teacher_policy_logit_residual_scale = 0.3
    config.appo.teacher_policy_blend_coef = 0.15
    config.appo.teacher_policy_fallback_confidence = 0.55
    config.appo.teacher_policy_disagreement_margin = 0.1
    config.appo.param_anchor_coef = 0.005
    config.appo.learning_rate = 1e-4
    config.appo.entropy_coeff = 0.0
    config.appo.ppo_clip_ratio = 0.05
    config.model.hidden_size = 256
    config.model.num_layers = 3
    config.model.actor_critic_share_weights = False
    config.model.normalize_input = False
    config.model.nonlinearity = "relu"
    trainer = APPOTrainerScaffold(config)
    argv = trainer.build_sf_argv()
    assert "--teacher_loss_coef=0.25" in argv
    assert "--teacher_loss_type=ce" in argv
    assert "--teacher_bc_path=/tmp/teacher.pt" in argv
    assert "--teacher_prior_bc_path=/tmp/prior.pt" in argv
    assert "--teacher_action_boosts=west=2.0,south=1.5" in argv
    assert "--teacher_loss_final_coef=0.002" in argv
    assert "--teacher_loss_warmup_env_steps=1024" in argv
    assert "--teacher_loss_decay_env_steps=4096" in argv
    assert "--teacher_replay_action_boosts=east=2.0,south=2.0" in argv
    assert "--teacher_replay_current_disagreement_boost=2.0" in argv
    assert "--teacher_replay_confusion_pair_boosts=east->south=3.0,south->east=3.0" in argv
    assert "--teacher_policy_logit_residual_scale=0.3" in argv
    assert "--teacher_policy_blend_coef=0.15" in argv
    assert "--teacher_policy_fallback_confidence=0.55" in argv
    assert "--teacher_policy_disagreement_margin=0.1" in argv
    assert "--param_anchor_coef=0.005" in argv
    assert "--learning_rate=0.0001" in argv
    assert "--exploration_loss_coeff=0.0" in argv
    assert "--ppo_clip_ratio=0.05" in argv
    assert "--normalize_input=False" in argv
    assert "--nonlinearity=relu" in argv
    assert "--actor_critic_share_weights=False" in argv
    idx = argv.index("--encoder_mlp_layers")
    assert argv[idx + 1:idx + 4] == ["256", "256", "256"]


def test_build_appo_config_respects_value_stability_args():
    args = type(
        "Args",
        (),
        {
            "experiment": "exp",
            "train_dir": "train_dir/rl",
            "serial_mode": False,
            "async_rl": False,
            "num_workers": 1,
            "num_envs_per_worker": 1,
            "rollout_length": 8,
            "recurrence": 8,
            "batch_size": 8,
            "num_batches_per_epoch": 1,
            "ppo_epochs": 1,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.9,
            "value_loss_coeff": 0.1,
            "reward_scale": 0.005,
            "entropy_coeff": 0.01,
            "ppo_clip_ratio": 0.1,
            "train_for_env_steps": 32,
            "scheduler": "rule_based",
            "reward_source": "hand_shaped",
            "learned_reward_path": None,
            "episodic_explore_bonus_enabled": False,
            "episodic_explore_bonus_scale": 0.0,
            "episodic_explore_bonus_mode": "state_hash",
            "scheduler_model_path": None,
            "enabled_skills": "explore",
            "observation_version": "v4",
            "world_model_path": None,
            "world_model_feature_mode": None,
            "env_max_episode_steps": 500,
            "model_hidden_size": None,
            "disable_input_normalization": False,
            "nonlinearity": None,
            "bc_init_path": None,
            "teacher_bc_path": None,
            "teacher_prior_bc_path": "/tmp/prior.pt",
            "teacher_report_path": "/tmp/teacher_report.json",
            "teacher_loss_coef": 0.01,
            "teacher_loss_type": "ce",
            "teacher_action_boosts": "",
            "teacher_loss_final_coef": 0.0,
            "teacher_loss_warmup_env_steps": 0,
            "teacher_loss_decay_env_steps": 0,
            "teacher_replay_trace_input": None,
            "teacher_replay_coef": 0.0,
            "teacher_replay_final_coef": 0.002,
            "teacher_replay_warmup_env_steps": 64,
            "teacher_replay_decay_env_steps": 256,
            "teacher_replay_batch_size": 128,
            "teacher_replay_priority_power": 1.7,
            "teacher_replay_source_mode": "disagreement",
            "teacher_replay_action_boosts": "east=2.0,south=2.0",
            "teacher_replay_current_disagreement_boost": 2.5,
            "teacher_replay_confusion_pair_boosts": "east->south=3.0,south->east=3.0",
            "teacher_policy_logit_residual_scale": 0.3,
            "teacher_policy_blend_coef": 0.2,
            "teacher_policy_fallback_confidence": 0.6,
            "teacher_policy_disagreement_margin": 0.1,
            "param_anchor_coef": 0.0,
            "actor_loss_scale": 1.0,
            "actor_loss_final_scale": 1.0,
            "actor_loss_warmup_env_steps": 0,
            "actor_loss_decay_env_steps": 0,
            "trace_eval_input": None,
            "trace_eval_interval_env_steps": 0,
            "trace_eval_top_k": 5,
            "improver_report_output": "/tmp/improver_report.json",
            "use_rnn": False,
            "no_rnn": True,
            "disable_action_mask": False,
        },
    )()
    config = build_appo_config(args)
    assert config.appo.gamma == 0.99
    assert config.appo.gae_lambda == 0.9
    assert config.appo.value_loss_coeff == 0.1
    assert config.appo.reward_scale == 0.005
    assert config.appo.teacher_replay_final_coef == 0.002
    assert config.appo.teacher_replay_warmup_env_steps == 64
    assert config.appo.teacher_replay_decay_env_steps == 256
    assert config.appo.teacher_replay_priority_power == 1.7
    assert config.appo.teacher_replay_source_mode == "disagreement"
    assert config.appo.teacher_replay_action_boosts == "east=2.0,south=2.0"
    assert config.appo.teacher_replay_current_disagreement_boost == 2.5
    assert config.appo.teacher_replay_confusion_pair_boosts == "east->south=3.0,south->east=3.0"
    assert config.appo.teacher_prior_bc_path == "/tmp/prior.pt"
    assert config.appo.teacher_policy_logit_residual_scale == 0.3
    assert config.appo.teacher_policy_blend_coef == 0.2
    assert config.appo.teacher_policy_fallback_confidence == 0.6
    assert config.appo.teacher_policy_disagreement_margin == 0.1
    assert config.model.teacher_report_path == "/tmp/teacher_report.json"
    assert config.appo.improver_report_output == "/tmp/improver_report.json"


def test_build_appo_config_infers_hidden_size_from_bc_checkpoint():
    rows = [
        {"feature_vector": [0.0] * 160, "action": "east", "allowed_actions": ["east", "west"]},
        {"feature_vector": [0.1] * 160, "action": "west", "allowed_actions": ["east", "west"]},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "bc.pt")
        train_bc_model(rows, out, epochs=1, lr=1e-3, hidden_size=64, num_layers=3, observation_version="v2")
        args = type(
            "Args",
            (),
            {
                "experiment": "exp",
                "train_dir": "train_dir/rl",
                "serial_mode": False,
                "async_rl": False,
                "num_workers": 1,
                "num_envs_per_worker": 1,
                "rollout_length": 8,
                "recurrence": 8,
                "batch_size": 8,
                "num_batches_per_epoch": 1,
                    "ppo_epochs": 1,
                    "learning_rate": 3e-4,
                    "entropy_coeff": 0.01,
                    "ppo_clip_ratio": 0.1,
                    "train_for_env_steps": 32,
                "scheduler": "rule_based",
                "reward_source": "hand_shaped",
                "learned_reward_path": None,
                "episodic_explore_bonus_enabled": False,
                "episodic_explore_bonus_scale": 0.0,
                "episodic_explore_bonus_mode": "state_hash",
                "scheduler_model_path": None,
                "enabled_skills": "explore",
                    "observation_version": "v2",
                    "model_hidden_size": None,
                    "model_num_layers": None,
                    "separate_actor_critic": True,
                    "disable_input_normalization": False,
                    "nonlinearity": None,
                    "bc_init_path": out,
                "teacher_bc_path": None,
                "teacher_loss_coef": 0.01,
                "teacher_loss_type": "ce",
                "teacher_action_boosts": "",
                "teacher_loss_final_coef": 0.0,
                "teacher_loss_warmup_env_steps": 0,
                "teacher_loss_decay_env_steps": 0,
                "param_anchor_coef": 0.0,
                "trace_eval_input": None,
                "trace_eval_interval_env_steps": 0,
                "trace_eval_top_k": 5,
                "use_rnn": False,
                "no_rnn": True,
                "disable_action_mask": False,
            },
        )()
        config = build_appo_config(args)
        assert config.model.actor_critic_share_weights is False
        assert config.model.hidden_size == 64
        assert config.model.num_layers == 3


def test_build_appo_config_keeps_bc_warmstart_normalization_off_with_explicit_model_shape():
    rows = [
        {"feature_vector": [0.0] * 160, "action": "east", "allowed_actions": ["east", "west"]},
        {"feature_vector": [0.1] * 160, "action": "west", "allowed_actions": ["east", "west"]},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "bc.pt")
        train_bc_model(rows, out, epochs=1, lr=1e-3, hidden_size=64, num_layers=3, observation_version="v2")
        args = type(
            "Args",
            (),
            {
                "experiment": "exp",
                "train_dir": "train_dir/rl",
                "serial_mode": False,
                "async_rl": False,
                "num_workers": 1,
                "num_envs_per_worker": 1,
                "rollout_length": 8,
                "recurrence": 8,
                "batch_size": 8,
                "num_batches_per_epoch": 1,
                "ppo_epochs": 1,
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.9,
                "value_loss_coeff": 0.1,
                "reward_scale": 0.005,
                "entropy_coeff": 0.01,
                "ppo_clip_ratio": 0.1,
                "train_for_env_steps": 32,
                "scheduler": "rule_based",
                "reward_source": "hand_shaped",
                "learned_reward_path": None,
                "proxy_reward_path": None,
                "proxy_reward_weight": 1.0,
                "episodic_explore_bonus_enabled": False,
                "episodic_explore_bonus_scale": 0.0,
                "episodic_explore_bonus_mode": "state_hash",
                "scheduler_model_path": None,
                "enabled_skills": "explore",
                "observation_version": "v2",
                "world_model_path": None,
                "world_model_feature_mode": None,
                "env_max_episode_steps": 500,
                "model_hidden_size": 64,
                "model_num_layers": 3,
                "disable_input_normalization": False,
                "nonlinearity": None,
                "bc_init_path": out,
                "appo_init_checkpoint_path": None,
                "teacher_bc_path": None,
                "teacher_report_path": None,
                "teacher_loss_coef": 0.0,
                "teacher_loss_type": "ce",
                "teacher_action_boosts": "",
                "teacher_loss_final_coef": 0.0,
                "teacher_loss_warmup_env_steps": 0,
                "teacher_loss_decay_env_steps": 0,
                "teacher_replay_trace_input": None,
                "teacher_replay_coef": 0.0,
                "teacher_replay_final_coef": 0.0,
                "teacher_replay_warmup_env_steps": 0,
                "teacher_replay_decay_env_steps": 0,
                "teacher_replay_batch_size": 128,
                "teacher_replay_priority_power": 1.0,
                "teacher_replay_source_mode": "uniform",
                "param_anchor_coef": 0.0,
                "actor_loss_scale": 1.0,
                "actor_loss_final_scale": 1.0,
                "actor_loss_warmup_env_steps": 0,
                "actor_loss_decay_env_steps": 0,
                "trace_eval_input": None,
                "trace_eval_interval_env_steps": 0,
                "trace_eval_top_k": 5,
                "save_every_sec": 120,
                "save_best_every_sec": 5,
                "improver_report_output": None,
                "use_rnn": False,
                "no_rnn": True,
                "disable_action_mask": False,
            },
        )()
        config = build_appo_config(args)
        assert config.model.hidden_size == 64
        assert config.model.num_layers == 3
        assert config.model.normalize_input is False
        assert config.model.normalize_input is False
        assert config.model.nonlinearity == "relu"


def test_build_appo_config_records_appo_init_checkpoint():
    args = type(
        "Args",
        (),
        {
            "experiment": "exp",
            "train_dir": "train_dir/rl",
            "serial_mode": False,
            "async_rl": False,
            "num_workers": 1,
            "num_envs_per_worker": 1,
            "rollout_length": 8,
            "recurrence": 8,
            "batch_size": 8,
            "num_batches_per_epoch": 1,
            "ppo_epochs": 1,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.9,
            "value_loss_coeff": 0.1,
            "reward_scale": 0.005,
            "entropy_coeff": 0.01,
            "ppo_clip_ratio": 0.1,
            "train_for_env_steps": 32,
            "scheduler": "rule_based",
            "reward_source": "hand_shaped",
            "learned_reward_path": None,
            "episodic_explore_bonus_enabled": False,
            "episodic_explore_bonus_scale": 0.0,
            "episodic_explore_bonus_mode": "state_hash",
            "scheduler_model_path": None,
            "enabled_skills": "explore",
            "observation_version": "v4",
            "world_model_path": None,
            "world_model_feature_mode": None,
            "env_max_episode_steps": 500,
            "model_hidden_size": None,
            "disable_input_normalization": False,
            "nonlinearity": None,
            "bc_init_path": None,
            "appo_init_checkpoint_path": "/tmp/best_trace_match.pth",
            "teacher_bc_path": None,
            "teacher_loss_coef": 0.01,
            "teacher_loss_type": "ce",
            "teacher_action_boosts": "",
            "teacher_loss_final_coef": 0.0,
            "teacher_loss_warmup_env_steps": 0,
            "teacher_loss_decay_env_steps": 0,
            "teacher_replay_trace_input": None,
            "teacher_replay_coef": 0.0,
            "teacher_replay_final_coef": 0.0,
            "teacher_replay_warmup_env_steps": 0,
            "teacher_replay_decay_env_steps": 0,
            "teacher_replay_batch_size": 128,
            "teacher_replay_priority_power": 1.0,
            "teacher_replay_source_mode": "uniform",
            "param_anchor_coef": 0.0,
            "trace_eval_input": None,
            "trace_eval_interval_env_steps": 0,
            "trace_eval_top_k": 5,
            "use_rnn": False,
            "no_rnn": True,
            "disable_action_mask": False,
        },
    )()
    config = build_appo_config(args)
    assert config.model.appo_init_checkpoint_path == "/tmp/best_trace_match.pth"


def test_cli_train_appo_forwards_zero_valued_teacher_controls(monkeypatch):
    captured = {}

    def fake_backend():
        return False

    def fake_train(argv):
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr("rl.bootstrap.ensure_sample_factory_backend", fake_backend)
    monkeypatch.setattr("rl.train_appo.main", fake_train)

    args = argparse.Namespace(
        experiment="exp",
        train_dir="train_dir/rl",
        serial_mode=False,
        async_rl=True,
        num_workers=1,
        num_envs_per_worker=1,
        rollout_length=8,
        recurrence=8,
        batch_size=8,
        num_batches_per_epoch=1,
        ppo_epochs=1,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.9,
        value_loss_coeff=0.1,
        reward_scale=0.005,
        entropy_coeff=0.01,
        ppo_clip_ratio=0.1,
        train_for_env_steps=32,
        scheduler="rule_based",
        reward_source="hand_shaped",
        learned_reward_path=None,
        proxy_reward_path=None,
        proxy_reward_weight=1.0,
        episodic_explore_bonus_enabled=False,
        episodic_explore_bonus_scale=0.0,
        episodic_explore_bonus_mode="state_hash",
        scheduler_model_path=None,
        enabled_skills="explore",
        observation_version="v4",
        world_model_path=None,
        world_model_feature_mode=None,
        env_max_episode_steps=200,
        model_hidden_size=None,
        model_num_layers=None,
        separate_actor_critic=False,
        disable_input_normalization=False,
        nonlinearity=None,
        bc_init_path="/tmp/teacher.pt",
        appo_init_checkpoint_path=None,
        teacher_bc_path=None,
        teacher_report_path=None,
        teacher_loss_coef=0.0,
        teacher_loss_type="ce",
        teacher_action_boosts="",
        teacher_loss_final_coef=0.0,
        teacher_loss_warmup_env_steps=0,
        teacher_loss_decay_env_steps=0,
        teacher_replay_trace_input=None,
        teacher_replay_coef=0.0,
        teacher_replay_final_coef=0.0,
        teacher_replay_warmup_env_steps=0,
        teacher_replay_decay_env_steps=0,
        teacher_replay_batch_size=128,
        teacher_replay_priority_power=1.0,
        teacher_replay_source_mode="uniform",
        param_anchor_coef=0.0,
        actor_loss_scale=0.0,
        actor_loss_final_scale=0.0,
        actor_loss_warmup_env_steps=0,
        actor_loss_decay_env_steps=0,
        trace_eval_input="/tmp/heldout.jsonl",
        trace_eval_interval_env_steps=128,
        trace_eval_top_k=5,
        save_every_sec=5,
        save_best_every_sec=5,
        no_rnn=True,
        use_rnn=False,
        disable_action_mask=False,
        write_plan=None,
        improver_report_output=None,
        dry_run=False,
    )

    rc = cli_module.cmd_rl_train_appo(args)
    assert rc == 0
    argv = captured["argv"]
    assert "--teacher-loss-coef" in argv
    assert argv[argv.index("--teacher-loss-coef") + 1] == "0.0"
    assert "--teacher-replay-coef" in argv
    assert argv[argv.index("--teacher-replay-coef") + 1] == "0.0"
    assert "--actor-loss-scale" in argv
    assert argv[argv.index("--actor-loss-scale") + 1] == "0.0"


def test_parse_teacher_action_boosts():
    boosts = _parse_teacher_action_boosts("west=2.0,south=1.5")
    assert boosts == {3: 2.0, 1: 1.5}


def test_parse_teacher_confusion_pair_boosts():
    boosts = _parse_teacher_confusion_pair_boosts("east->south=3.0,south->east=2.5")
    assert boosts == {
        (ACTION_SET.index("east"), ACTION_SET.index("south")): 3.0,
        (ACTION_SET.index("south"), ACTION_SET.index("east")): 2.5,
    }


def test_trainer_scaffold_includes_episodic_bonus_args():
    config = RLConfig()
    config.reward.episodic_explore_bonus_enabled = True
    config.reward.episodic_explore_bonus_scale = 0.2
    config.reward.episodic_explore_bonus_mode = "tile"
    trainer = APPOTrainerScaffold(config)
    argv = trainer.build_sf_argv()
    assert "--episodic_explore_bonus_enabled=True" in argv
    assert "--episodic_explore_bonus_scale=0.2" in argv
    assert "--episodic_explore_bonus_mode=tile" in argv


def test_trainer_scaffold_includes_trace_eval_args():
    config = RLConfig()
    config.appo.trace_eval_input = "data/trace.jsonl"
    config.appo.trace_eval_interval_env_steps = 2048
    config.appo.trace_eval_top_k = 7
    config.appo.teacher_replay_trace_input = "data/replay.jsonl"
    config.appo.teacher_replay_coef = 0.02
    config.appo.teacher_replay_final_coef = 0.005
    config.appo.teacher_replay_warmup_env_steps = 128
    config.appo.teacher_replay_decay_env_steps = 512
    config.appo.teacher_replay_priority_power = 1.4
    config.appo.teacher_replay_source_mode = "mixed"
    config.appo.teacher_replay_action_boosts = "east=2.0,south=2.0"
    config.appo.teacher_replay_current_disagreement_boost = 2.0
    config.appo.teacher_replay_confusion_pair_boosts = "east->south=3.0,south->east=3.0"
    config.appo.actor_loss_scale = 0.0
    config.appo.actor_loss_final_scale = 1.0
    config.appo.actor_loss_warmup_env_steps = 64
    config.appo.actor_loss_decay_env_steps = 256
    config.appo.save_every_sec = 9
    config.appo.save_best_every_sec = 3
    config.model.appo_init_checkpoint_path = "/tmp/seed_checkpoint.pth"
    config.model.teacher_report_path = "/tmp/teacher_report.json"
    config.appo.improver_report_output = "/tmp/improver_report.json"
    trainer = APPOTrainerScaffold(config)
    argv = trainer.build_sf_argv()
    assert "--trace_eval_input=data/trace.jsonl" in argv
    assert "--trace_eval_interval_env_steps=2048" in argv
    assert "--trace_eval_top_k=7" in argv
    assert "--teacher_replay_trace_input=data/replay.jsonl" in argv
    assert "--teacher_replay_coef=0.02" in argv
    assert "--teacher_replay_final_coef=0.005" in argv
    assert "--teacher_replay_warmup_env_steps=128" in argv
    assert "--teacher_replay_decay_env_steps=512" in argv
    assert "--teacher_replay_priority_power=1.4" in argv
    assert "--teacher_replay_source_mode=mixed" in argv
    assert "--teacher_replay_action_boosts=east=2.0,south=2.0" in argv
    assert "--teacher_replay_current_disagreement_boost=2.0" in argv
    assert "--teacher_replay_confusion_pair_boosts=east->south=3.0,south->east=3.0" in argv
    assert "--actor_loss_scale=0.0" in argv
    assert "--actor_loss_warmup_env_steps=64" in argv
    assert "--actor_loss_decay_env_steps=256" in argv
    assert "--appo_init_checkpoint_path=/tmp/seed_checkpoint.pth" in argv
    assert "--teacher_report_path=/tmp/teacher_report.json" in argv
    assert "--improver_report_output=/tmp/improver_report.json" in argv
    assert "--save_every_sec=9" in argv
    assert "--save_best_every_sec=3" in argv


def test_trainer_scaffold_includes_proxy_reward_args():
    config = RLConfig()
    config.reward.source = "mixed_proxy"
    config.reward.proxy_reward_path = "/tmp/proxy.pt"
    config.reward.proxy_reward_weight = 0.25
    trainer = APPOTrainerScaffold(config)
    argv = trainer.build_sf_argv()
    assert "--reward_source=mixed_proxy" in argv
    assert "--proxy_reward_path=/tmp/proxy.pt" in argv
    assert "--proxy_reward_weight=0.25" in argv


def test_proxy_labels_and_dataset_summary():
    row0 = {
        "episode_id": "ep-1",
        "step": 0,
        "seed": 42,
        "task": "explore",
        "action": "search",
        "allowed_actions": ["search", "east", "west"],
        "feature_vector": [0.0, 1.0, 2.0, 3.0],
        "observation_version": "v4+wm_concat_aux",
        "obs_hash": "a",
        "next_obs_hash": "b",
        "prompt": "Pos:(10, 20)\nAdjacent: north=wall south=floor east=wall west=wall\nMonsters: none\nItems: none\n",
        "planner_trace": [
            {"action": "search", "total": 1.5},
            {"action": "east", "total": 0.2},
        ],
        "delta": {"new_tiles": ["x", "y"], "hp_delta": 0, "gold_delta": 0, "depth_delta": 0, "survived": True},
        "done": False,
        "rooms_discovered_before": 1,
        "recent_position_count": 0,
        "recent_action_count": 0,
    }
    row1 = dict(row0)
    row1.update(
        {
            "step": 1,
            "action": "east",
            "obs_hash": "b",
            "next_obs_hash": "c",
            "prompt": "Pos:(11, 20)\nAdjacent: north=floor south=floor east=floor west=wall\nMonsters: none\nItems: none\n",
            "delta": {"new_tiles": ["z"], "hp_delta": -1, "gold_delta": 5, "depth_delta": 1, "survived": True},
        }
    )
    assert search_context_label(row0) == 1
    assert teacher_margin(row0) == 1.3
    proxy_rows = build_proxy_rows([row1, row0], horizon=2, task_filter="explore")
    assert len(proxy_rows) == 2
    summary = summarize_proxy_rows(proxy_rows)
    assert summary["rows"] == 2
    assert summary["feature_dims"] == [4]
    assert summary["search_context_positive_rate"] == 0.5


def test_proxy_train_eval_and_reward_source():
    rows = [
        {
            "episode_id": "ep-1",
            "step": 0,
            "action": "east",
            "allowed_actions": ["east", "west", "search"],
            "feature_vector": [1.0, 0.0, 0.0, 0.0],
            "observation_version": "v4+wm_concat_aux",
            "k_step_progress": 4.0,
            "k_step_survival": 0.0,
            "k_step_loop_risk": 0.1,
            "k_step_resource_value": 0.0,
            "teacher_margin": 1.0,
            "search_context_label": 0,
            "prompt": "Pos:(1, 1)\nAdjacent: north=floor south=floor east=floor west=floor\nMonsters: none\nItems: none\n",
        },
        {
            "episode_id": "ep-1",
            "step": 1,
            "action": "west",
            "allowed_actions": ["east", "west", "search"],
            "feature_vector": [0.0, 1.0, 0.0, 0.0],
            "observation_version": "v4+wm_concat_aux",
            "k_step_progress": 1.0,
            "k_step_survival": 0.0,
            "k_step_loop_risk": 0.2,
            "k_step_resource_value": 0.0,
            "teacher_margin": 0.8,
            "search_context_label": 0,
            "prompt": "Pos:(2, 1)\nAdjacent: north=floor south=floor east=floor west=floor\nMonsters: none\nItems: none\n",
        },
        {
            "episode_id": "ep-1",
            "step": 2,
            "action": "search",
            "allowed_actions": ["east", "west", "search"],
            "feature_vector": [0.0, 0.0, 1.0, 0.0],
            "observation_version": "v4+wm_concat_aux",
            "k_step_progress": 0.5,
            "k_step_survival": 0.0,
            "k_step_loop_risk": 0.0,
            "k_step_resource_value": 0.0,
            "teacher_margin": 0.6,
            "search_context_label": 1,
            "prompt": "Pos:(3, 1)\nAdjacent: north=wall south=floor east=wall west=wall\nMonsters: none\nItems: none\n",
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        heldout_path = os.path.join(tmpdir, "heldout.jsonl")
        model_path = os.path.join(tmpdir, "proxy.pt")
        report_path = os.path.join(tmpdir, "proxy_report.json")
        write_proxy_rows(rows, heldout_path)
        result = train_proxy_model(
            rows,
            model_path,
            heldout_rows=rows,
            epochs=20,
            lr=1e-2,
            hidden_size=32,
            action_embed_dim=8,
        )
        assert os.path.exists(model_path)
        assert "heldout_eval" in result
        inference = load_proxy_model(model_path)
        ranking = inference.rank_actions(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), ["east", "west"])
        assert ranking[0]["action"] in {"east", "west"}
        eval_result = evaluate_proxy_model(model_path, heldout_path)
        assert eval_result["rows"] == 3
        from rl.proxy_report import build_proxy_report
        report = build_proxy_report(model_path, heldout_path)
        with open(report_path, "w") as f:
            json.dump(report, f)
        assert os.path.exists(report_path)
        proxy_reward = build_reward_source("proxy", proxy_reward_path=model_path)
        reward_inputs = RewardInputs(
            task="explore",
            obs_before={},
            obs_after={},
            state_before={},
            state_after={},
            memory_before=None,
            memory_after=None,
            action_name="east",
            env_reward=0.0,
            terminated=False,
            truncated=False,
            feature_vector_before=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        proxy_details = proxy_reward.details(reward_inputs)
        assert "proxy_teacher_margin" in proxy_details
        mixed_reward = build_reward_source("mixed_proxy", proxy_reward_path=model_path, proxy_reward_weight=0.5)
        assert mixed_reward.proxy_weight == 0.5


def test_teacher_report_includes_source_and_feature_metadata():
    report = build_teacher_report(
        model_path="/tmp/model.pt",
        heldout_trace_path=None,
        train_result={"train_accuracy": 1.0},
        teacher_kind="bc",
        source_trace_path="/tmp/train.jsonl",
        observation_version="v4",
        world_model_path="/tmp/world_model.pt",
        world_model_feature_mode="concat_aux",
    )
    assert report["source_trace_path"] == "/tmp/train.jsonl"
    assert report["observation_version"] == "v4"
    assert report["world_model_path"] == "/tmp/world_model.pt"
    assert report["world_model_feature_mode"] == "concat_aux"


def test_improver_report_links_teacher_and_best_trace_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir) / "exp"
        checkpoint_dir = experiment_dir / "checkpoint_p0"
        checkpoint_dir.mkdir(parents=True)
        final_ckpt = checkpoint_dir / "checkpoint_000000012_96.pth"
        torch.save({"model": {"weight": torch.tensor([1.0])}}, final_ckpt)
        teacher_report_path = Path(tmpdir) / "teacher_report.json"
        teacher_report_path.write_text(
            json.dumps(
                {
                    "teacher_kind": "bc",
                    "heldout_trace_eval": {"match_rate": 0.95},
                    "weak_action_trace_eval": {"match_rate": 0.875},
                }
            )
        )
        (checkpoint_dir / "best_trace_match.json").write_text(
            json.dumps(
                {
                    "best_checkpoint_path": str(checkpoint_dir / "checkpoint_000000010_80.pth"),
                    "match_rate": 0.9375,
                }
            )
        )
        (checkpoint_dir / "warmstart_trace_match.json").write_text(
            json.dumps(
                {
                    "warmstart_checkpoint_path": str(checkpoint_dir / "checkpoint_000000000_0.pth"),
                    "match_rate": 0.975,
                }
            )
        )
        cfg = RLConfig()
        cfg.train_dir = tmpdir
        cfg.experiment = "exp"
        cfg.model.teacher_report_path = str(teacher_report_path)
        cfg.model.bc_init_path = "/tmp/teacher.pt"
        cfg.appo.teacher_replay_trace_input = "/tmp/replay.jsonl"
        cfg.appo.trace_eval_input = "/tmp/heldout.jsonl"
        cfg.appo.teacher_prior_bc_path = "/tmp/prior.pt"
        cfg.appo.teacher_replay_action_boosts = "east=2.0,south=2.0"
        cfg.appo.teacher_replay_current_disagreement_boost = 2.0
        cfg.appo.teacher_replay_confusion_pair_boosts = "east->south=3.0,south->east=3.0"
        cfg.appo.teacher_policy_logit_residual_scale = 0.3
        cfg.appo.teacher_policy_blend_coef = 0.25
        cfg.appo.teacher_policy_fallback_confidence = 0.55

        original_eval = build_improver_report.__globals__["evaluate_checkpoint_trace_match"]
        original_list = build_improver_report.__globals__["list_checkpoint_paths"]
        build_improver_report.__globals__["list_checkpoint_paths"] = lambda experiment, train_dir: [final_ckpt]
        build_improver_report.__globals__["evaluate_checkpoint_trace_match"] = lambda **kwargs: {
            "checkpoint_path": str(final_ckpt),
            "env_steps": 96,
            "match_rate": 0.9,
            "invalid_action_rate": 0.0,
            "action_counts": {"east": 4},
        }
        report = build_improver_report(
            config=cfg,
            plan={"experiment": "exp"},
            argv=["--foo"],
            warmstart_checkpoint="/tmp/warmstart.pth",
            status="SUCCESS",
        )
        build_improver_report.__globals__["evaluate_checkpoint_trace_match"] = original_eval
        build_improver_report.__globals__["list_checkpoint_paths"] = original_list
        assert report["teacher_checkpoint_path"] == "/tmp/teacher.pt"
        assert report["teacher_report_summary"]["heldout_match_rate"] == 0.95
        assert report["warmstart_trace_metadata"]["match_rate"] == 0.975
        assert report["best_trace_metadata"]["match_rate"] == 0.9375
        assert report["final_trace_metadata"]["match_rate"] == 0.9
        assert report["teacher_policy"]["prior_checkpoint_path"] == "/tmp/prior.pt"
        assert report["replay_source"]["current_disagreement_boost"] == 2.0
        assert report["replay_source"]["confusion_pair_boosts"] == "east->south=3.0,south->east=3.0"
        assert report["teacher_policy"]["logit_residual_scale"] == 0.3
        assert report["replay_source"]["action_boosts"] == "east=2.0,south=2.0"
        assert report["teacher_policy"]["blend_coef"] == 0.25
        assert report["teacher_policy"]["fallback_confidence"] == 0.55
        assert report["trace_gate"]["best_checkpoint_path"].endswith("checkpoint_000000010_80.pth")
        assert report["trace_gate"]["final_checkpoint_path"].endswith("checkpoint_000000012_96.pth")


def test_teacher_patch_is_idempotent():
    patch_sample_factory_teacher_reg()
    patch_sample_factory_teacher_reg()


def test_scheduled_teacher_replay_coef_decays_linearly():
    learner = type(
        "DummyLearner",
        (),
        {
            "teacher_replay_coef": 0.02,
            "teacher_replay_final_coef": 0.005,
            "teacher_replay_warmup_env_steps": 100,
            "teacher_replay_decay_env_steps": 300,
            "env_steps": 0,
        },
    )()
    assert _scheduled_teacher_replay_coef(learner) == 0.02
    learner.env_steps = 100
    assert _scheduled_teacher_replay_coef(learner) == 0.02
    learner.env_steps = 250
    assert round(_scheduled_teacher_replay_coef(learner), 6) == 0.0125
    learner.env_steps = 500
    assert round(_scheduled_teacher_replay_coef(learner), 6) == 0.005


def test_scheduled_actor_loss_scale_decays_linearly():
    learner = type(
        "DummyLearner",
        (),
        {
            "actor_loss_scale": 0.0,
            "actor_loss_final_scale": 1.0,
            "actor_loss_warmup_env_steps": 100,
            "actor_loss_decay_env_steps": 300,
            "env_steps": 0,
        },
    )()
    assert _scheduled_actor_loss_scale(learner) == 0.0
    learner.env_steps = 100
    assert _scheduled_actor_loss_scale(learner) == 0.0
    learner.env_steps = 250
    assert round(_scheduled_actor_loss_scale(learner), 6) == 0.5
    learner.env_steps = 500
    assert round(_scheduled_actor_loss_scale(learner), 6) == 1.0


def test_row_replay_flags_detect_disagreement_and_weak_action():
    flags = _row_replay_flags(
        {
            "action": "south",
            "teacher_action": "south",
            "behavior_action": "east",
            "repeated_state_count": 1,
            "repeated_action_count": 0,
            "reward": -1.0,
            "done": True,
        }
    )
    assert flags["is_disagreement_candidate"] == 1.0
    assert flags["is_weak_action"] == 1.0
    assert flags["is_loop_risk"] == 1.0
    assert flags["is_failure_slice"] == 1.0


def test_replay_priority_weights_target_requested_rows():
    rows = [
        {"action": "east", "teacher_action": "east", "behavior_action": "east"},
        {"action": "south", "teacher_action": "south", "behavior_action": "east"},
        {"action": "west", "teacher_action": "west", "behavior_action": "west", "repeated_state_count": 1},
    ]
    disagreement = _replay_priority_weights(rows, source_mode="disagreement", priority_power=1.0)
    weak = _replay_priority_weights(rows, source_mode="weak_action", priority_power=1.0)
    mixed = _replay_priority_weights(rows, source_mode="mixed", priority_power=1.5)
    assert disagreement[1] > disagreement[0]
    assert weak[1] > weak[0]
    assert weak[2] > weak[0]
    assert mixed[2] > mixed[0]
    boosted = _replay_priority_weights(
        rows,
        source_mode="uniform",
        priority_power=1.0,
        action_boosts={
            ACTION_SET.index("east"): 2.0,
            ACTION_SET.index("south"): 3.0,
        },
    )
    assert boosted[0] > boosted[2]
    assert boosted[1] > boosted[0]


def test_weight_replay_losses_by_current_disagreement_boosts_mismatched_rows():
    replay_loss_all = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    student_replay_logits = torch.tensor(
        [
            [5.0, 1.0, 0.0],
            [0.0, 1.0, 5.0],
            [0.0, 4.0, 1.0],
        ],
        dtype=torch.float32,
    )
    replay_actions = torch.tensor([0, 1, 1], dtype=torch.long)
    weighted, disagreement_fraction = _weight_replay_losses_by_current_disagreement(
        replay_loss_all,
        student_replay_logits,
        replay_actions,
        disagreement_boost=2.5,
    )
    assert torch.allclose(weighted, torch.tensor([1.0, 5.0, 3.0]))
    assert round(disagreement_fraction.item(), 6) == round(1.0 / 3.0, 6)


def test_weight_replay_losses_by_confusion_pairs_boosts_exact_pairs():
    replay_loss_all = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    student_replay_logits = torch.tensor(
        [
            [0.0, 1.0, 4.0, 0.0],
            [0.0, 4.0, 1.0, 0.0],
            [5.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    replay_actions = torch.tensor(
        [
            ACTION_SET.index("south"),
            ACTION_SET.index("east"),
            ACTION_SET.index("south"),
        ],
        dtype=torch.long,
    )
    weighted, confusion_pair_fraction = _weight_replay_losses_by_confusion_pairs(
        replay_loss_all,
        student_replay_logits,
        replay_actions,
        confusion_pair_boosts={
            (ACTION_SET.index("east"), ACTION_SET.index("south")): 3.0,
            (ACTION_SET.index("south"), ACTION_SET.index("east")): 2.0,
        },
    )
    assert torch.allclose(weighted, torch.tensor([3.0, 4.0, 3.0]))
    assert round(confusion_pair_fraction.item(), 6) == round(2.0 / 3.0, 6)


def test_forward_replay_action_logits_bypasses_teacher_prior_path_for_shared_weights():
    class FakeActorCritic:
        def __init__(self):
            self._teacher_prior_raw_obs = torch.full((2, len(ACTION_SET)), 50.0)
            self.decoder = lambda x: x + 1.0

        def forward_head(self, obs):
            return {"x": obs["obs"] + 1.0}

        def forward_core(self, x, *_args, **_kwargs):
            return {"x": x + 1.0}

        def action_parameterization(self, x):
            return x + 1.0, object()

        def forward_tail(self, *_args, **_kwargs):
            raise AssertionError("Replay path should bypass teacher-prior forward_tail")

    actor_critic = FakeActorCritic()
    stale = actor_critic._teacher_prior_raw_obs.clone()
    features = torch.tensor(
        [
            [0.0] * len(ACTION_SET),
            [1.0] * len(ACTION_SET),
        ],
        dtype=torch.float32,
    )
    allowed_masks = torch.tensor(
        [
            [1.0] * len(ACTION_SET),
            [0.0] + [1.0] * (len(ACTION_SET) - 1),
        ],
        dtype=torch.float32,
    )
    logits = _forward_replay_action_logits(actor_critic, features, allowed_masks)
    expected = (features + 1.0 + 1.0 + 1.0 + 1.0).masked_fill(allowed_masks <= 0, -1e9)
    assert torch.allclose(logits, expected)
    assert torch.allclose(actor_critic._teacher_prior_raw_obs, stale)


def test_forward_replay_action_logits_bypasses_teacher_prior_path_for_separate_weights():
    class FakeActorCritic:
        def __init__(self):
            self._teacher_prior_raw_obs = torch.full((2, len(ACTION_SET)), 50.0)
            self.cores = [object(), object()]
            self.actor_decoder = lambda x: x + 2.0

        def forward_head(self, obs):
            return {"x": torch.cat((obs["obs"] + 1.0, obs["obs"] + 3.0), dim=1)}

        def forward_core(self, x, *_args, **_kwargs):
            return {"x": x + 1.0}

        def action_parameterization(self, x):
            return x + 4.0, object()

        def forward_tail(self, *_args, **_kwargs):
            raise AssertionError("Replay path should bypass teacher-prior forward_tail")

    actor_critic = FakeActorCritic()
    stale = actor_critic._teacher_prior_raw_obs.clone()
    features = torch.zeros((2, len(ACTION_SET)), dtype=torch.float32)
    allowed_masks = torch.ones_like(features)
    logits = _forward_replay_action_logits(actor_critic, features, allowed_masks)
    expected = torch.full_like(features, 8.0)
    assert torch.allclose(logits, expected)
    assert torch.allclose(actor_critic._teacher_prior_raw_obs, stale)


def test_skill_env_reset_and_step():
    config = RLConfig()
    env = NethackSkillEnv(config)
    obs, info = env.reset(seed=42)
    assert obs.shape == (observation_dim(),)
    assert info["active_skill"] in config.options.enabled_skills
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (observation_dim(),)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "debug" in info
    env.close()


def test_skill_env_v3_reset_and_step():
    config = RLConfig()
    config.env.observation_version = "v3"
    env = NethackSkillEnv(config)
    obs, info = env.reset(seed=42)
    assert obs.shape == (observation_dim("v3"),)
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (observation_dim("v3"),)
    env.close()


def test_feature_dims_are_stable():
    assert observation_dim("v1") == 106
    assert observation_dim("v2") == 160
    assert observation_dim("v3") == 244
    assert observation_dim("v4") == 302
    assert reward_feature_dim() == 37
    assert scheduler_feature_dim() > 0


def test_feature_encoder_v2_shape():
    timestep = {
        "state": {
            "hp": 10,
            "hp_max": 12,
            "gold": 5,
            "depth": 1,
            "turn": 20,
            "ac": 4,
            "strength": 10,
            "dexterity": 8,
            "visible_monsters": [{"char": "k", "pos": (4, 4)}],
            "visible_items": [{"type": "gold", "pos": (5, 4)}],
            "adjacent": {"north": "wall", "south": "floor", "east": "unseen", "west": "door"},
            "message": "You see here a gold piece.",
            "position": (4, 5),
        },
        "active_skill": "explore",
        "allowed_actions": ["north", "south", "east", "wait", "pickup"],
        "memory_total_explored": 20,
        "rooms_discovered": 2,
        "steps_in_skill": 3,
        "standing_on_down_stairs": False,
        "standing_on_up_stairs": False,
        "recent_positions": [(4, 5), (4, 4)],
        "recent_actions": ["east", "east", "search"],
        "repeated_state_count": 1,
        "revisited_recent_tile_count": 1,
        "repeated_action_count": 2,
    }
    assert encode_observation(timestep, version="v2").shape == (160,)


def test_feature_encoder_v3_shape():
    obs = {"chars": np.full((10, 10), ord("."), dtype=np.int32)}
    obs["chars"][4, 5] = ord("@")
    obs["chars"][4, 6] = ord("#")
    timestep = {
        "state": {
            "hp": 10,
            "hp_max": 12,
            "gold": 5,
            "depth": 1,
            "turn": 20,
            "ac": 4,
            "strength": 10,
            "dexterity": 8,
            "visible_monsters": [{"char": "k", "pos": (4, 4)}],
            "visible_items": [{"type": "gold", "pos": (5, 4)}],
            "adjacent": {"north": "wall", "south": "floor", "east": "unseen", "west": "door"},
            "message": "You see here a gold piece.",
            "position": (5, 4),
        },
        "active_skill": "explore",
        "allowed_actions": ["north", "south", "east", "wait", "pickup"],
        "memory_total_explored": 20,
        "rooms_discovered": 2,
        "steps_in_skill": 3,
        "standing_on_down_stairs": False,
        "standing_on_up_stairs": False,
        "recent_positions": [(5, 4), (5, 3)],
        "recent_actions": ["east", "east", "search"],
        "repeated_state_count": 1,
        "revisited_recent_tile_count": 1,
        "repeated_action_count": 2,
        "obs": obs,
    }
    assert encode_observation(timestep, version="v3").shape == (244,)


def test_feature_encoder_v4_shape():
    obs = {"chars": np.full((10, 10), ord("."), dtype=np.int32)}
    obs["chars"][4, 5] = ord("@")
    obs["chars"][4, 6] = ord('#')
    obs["chars"][5, 5] = ord('|')
    timestep = {
        "state": {
            "hp": 10,
            "hp_max": 12,
            "gold": 5,
            "depth": 1,
            "turn": 20,
            "ac": 4,
            "strength": 10,
            "dexterity": 8,
            "visible_monsters": [{"char": "k", "pos": (4, 4)}],
            "visible_items": [{"type": "gold", "pos": (5, 4)}],
            "adjacent": {"north": "wall", "south": "floor", "east": "unseen", "west": "door"},
            "message": "You see here a gold piece.",
            "position": (5, 4),
        },
        "active_skill": "explore",
        "allowed_actions": ["north", "south", "east", "wait", "pickup"],
        "memory_total_explored": 20,
        "rooms_discovered": 2,
        "steps_in_skill": 3,
        "standing_on_down_stairs": False,
        "standing_on_up_stairs": False,
        "recent_positions": [(5, 4), (5, 3)],
        "recent_actions": ["east", "east", "search"],
        "repeated_state_count": 1,
        "revisited_recent_tile_count": 1,
        "repeated_action_count": 2,
        "obs": obs,
    }
    assert encode_observation(timestep, version="v4").shape == (302,)


def test_world_model_examples_build_k_step_windows():
    rows = [
        {
            "episode_id": "ep0",
            "step": 0,
            "seed": 1,
            "task": "explore",
            "action": "north",
            "reward": 0.1,
            "done": False,
            "observation_version": "v4",
            "prompt": "obs north wall east floor",
            "feature_vector": [0.0, 1.0, 2.0],
        },
        {
            "episode_id": "ep0",
            "step": 1,
            "seed": 1,
            "task": "explore",
            "action": "east",
            "reward": 0.2,
            "done": False,
            "observation_version": "v4",
            "prompt": "obs east corridor",
            "feature_vector": [1.0, 2.0, 3.0],
        },
        {
            "episode_id": "ep0",
            "step": 2,
            "seed": 1,
            "task": "explore",
            "action": "south",
            "reward": 0.3,
            "done": False,
            "observation_version": "v4",
            "prompt": "obs south room",
            "feature_vector": [2.0, 3.0, 4.0],
        },
    ]
    examples = build_world_model_examples(rows, horizon=1, observation_version="v4")
    assert len(examples) == 2
    assert examples[0]["action_index"] == 0
    assert examples[0]["task_index"] >= 0
    assert examples[0]["cumulative_reward"] == 0.1
    arrays = examples_to_arrays(examples)
    assert arrays["features"].shape == (2, 3)
    assert arrays["target_features"].shape == (2, 3)
    assert arrays["prompts"][0].startswith("obs")


def test_world_model_uses_state_only_prompt_text():
    encoder = StateEncoder()
    state = {
        "hp": 10,
        "hp_max": 12,
        "ac": 4,
        "strength": 12,
        "dexterity": 9,
        "position": (4, 5),
        "gold": 0,
        "depth": 1,
        "turn": 7,
        "adjacent": {"north": "wall", "south": "floor", "east": "door", "west": "corridor"},
        "visible_monsters": [],
        "visible_items": [],
    }
    full_prompt = encoder.format_prompt(state, "west")
    assert "Action: west" in full_prompt
    stripped = strip_action_from_prompt(full_prompt)
    assert "Action:" not in stripped
    row = {"prompt": full_prompt}
    assert state_prompt_from_row(row) == stripped


def test_world_model_train_and_eval_smoke():
    rows = []
    for step in range(6):
        rows.append(
            {
                "episode_id": "ep0",
                "step": step,
                "seed": 42,
                "task": "explore",
                "action": "east" if step % 2 == 0 else "west",
                "reward": float(step) / 10.0,
                "done": step == 5,
                "observation_version": "v4",
                "prompt": f"turn {step} action {'east' if step % 2 == 0 else 'west'}",
                "feature_vector": [float(step), float(step + 1), float(step + 2), float(step % 2)],
            }
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "world_model.pt")
        trace_path = os.path.join(tmpdir, "trace.jsonl")
        with open(trace_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        train_result = train_world_model(
            rows,
            model_path,
            horizon=2,
            epochs=2,
            hidden_size=32,
            latent_dim=16,
            observation_version="v4",
            action_class_balance=True,
            text_encoder_backend="hash",
            text_embedding_dim=16,
        )
        assert train_result["num_examples"] > 0
        assert "action_accuracy" in train_result
        assert "feature_cosine_mean" in train_result
        assert train_result["text_encoder_backend"] == "hash"
        assert train_result["action_class_balance"] is True
        assert train_result["action_loss_weights"] is not None
        inference = load_world_model(model_path)
        single = inference.encode_with_aux(rows[0]["feature_vector"], prompt_text=rows[0]["prompt"])
        batch = inference.encode_with_aux_batch(
            [rows[0]["feature_vector"], rows[1]["feature_vector"]],
            prompt_texts=[rows[0]["prompt"], rows[1]["prompt"]],
        )
        assert batch["latent"].shape == (2, len(single["latent"]))
        assert np.allclose(batch["latent"][0], single["latent"], atol=5e-5, rtol=5e-5)
        eval_result = evaluate_world_model(model_path, trace_path, horizon=2, observation_version="v4")
        assert eval_result["num_examples"] == train_result["num_examples"]
        assert eval_result["feature_mse"] >= 0.0
        assert "action_accuracy" in eval_result
        assert "action_top3_accuracy" in eval_result
        assert "latent_dead_fraction" in eval_result
        assert eval_result["text_encoder_backend"] == "hash"
        latent_trace_path = os.path.join(tmpdir, "trace_latent.jsonl")
        transform_result = transform_trace_with_world_model(trace_path, latent_trace_path, model_path, mode="concat")
        assert transform_result["rows"] == len(rows)
        transformed_rows = load_trace_rows(latent_trace_path)
        assert len(transformed_rows[0]["feature_vector"]) == transform_result["original_feature_dim"] + transform_result["latent_dim"]
        aux_trace_path = os.path.join(tmpdir, "trace_latent_aux.jsonl")
        aux_result = transform_trace_with_world_model(trace_path, aux_trace_path, model_path, mode="concat_aux")
        aux_rows = load_trace_rows(aux_trace_path)
        assert len(aux_rows[0]["feature_vector"]) == (
            aux_result["original_feature_dim"] + aux_result["latent_dim"] + aux_result["action_dim"]
        )
        assert world_model_augmented_dim(aux_result["original_feature_dim"], model_path, "concat_aux") == len(
            aux_rows[0]["feature_vector"]
        )
        augmented_eval = evaluate_world_model(model_path, aux_trace_path, horizon=2)
        assert augmented_eval["model_input_dim"] == 4
        assert augmented_eval["input_feature_dim"] == len(aux_rows[0]["feature_vector"])
        assert augmented_eval["coerced_example_count"] > 0
        downstream_eval = evaluate_world_model(
            model_path,
            trace_path,
            horizon=2,
            observation_version="v4",
            downstream_train_trace_path=trace_path,
            downstream_heldout_trace_path=trace_path,
            downstream_mode="concat_aux",
            downstream_epochs=2,
            downstream_hidden_size=32,
        )
        assert "downstream_bc" in downstream_eval
        assert downstream_eval["downstream_bc"]["trace_eval_summary"]["match_rate"] >= 0.0


def test_relabel_traces_with_bc_teacher():
    rows = [
        {
            "episode_id": "ep0",
            "step": 0,
            "seed": 1,
            "task": "explore",
            "action": "west",
            "allowed_actions": ["east", "west"],
            "observation_version": "v2",
            "feature_vector": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "episode_id": "ep0",
            "step": 1,
            "seed": 1,
            "task": "explore",
            "action": "west",
            "allowed_actions": ["east", "west"],
            "observation_version": "v2",
            "feature_vector": [0.0, 1.0, 0.0, 0.0],
        },
    ]
    teacher_rows = [
        dict(rows[0], action="east"),
        dict(rows[1], action="east"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        teacher_path = os.path.join(tmpdir, "teacher.pt")
        trace_path = os.path.join(tmpdir, "trace.jsonl")
        out_path = os.path.join(tmpdir, "relabeled.jsonl")
        train_bc_model(teacher_rows, teacher_path, epochs=20, lr=1e-3, hidden_size=32, observation_version="v2")
        with open(trace_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        result = relabel_trace_actions(trace_path, out_path, bc_model_path=teacher_path)
        relabeled_rows = load_trace_rows(out_path)
        assert result["rows"] == 2
        assert result["teacher_bc_model_paths"] == [teacher_path]
        assert result["changed_rows"] >= 1
        assert all(row["action"] in {"east", "west"} for row in relabeled_rows)
        assert all(row["original_action"] == "west" for row in relabeled_rows)


def test_relabel_traces_with_bc_teacher_ensemble(monkeypatch):
    rows = [
        {
            "episode_id": "ep0",
            "step": 0,
            "seed": 1,
            "task": "explore",
            "action": "west",
            "allowed_actions": ["east", "west"],
            "observation_version": "v2",
            "feature_vector": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "episode_id": "ep0",
            "step": 1,
            "seed": 1,
            "task": "explore",
            "action": "north",
            "allowed_actions": ["north", "south"],
            "observation_version": "v2",
            "feature_vector": [0.0, 1.0, 0.0, 0.0],
        },
    ]

    def fake_teacher_logits(relabeled_rows, teacher_paths):
        assert teacher_paths == ["/tmp/a.pt", "/tmp/b.pt"]
        assert len(relabeled_rows) == 2
        logits = np.full((2, len(ACTION_SET)), -1e9, dtype=np.float32)
        logits[0, ACTION_SET.index("east")] = 5.0
        logits[1, ACTION_SET.index("south")] = 5.0
        return logits

    monkeypatch.setattr("rl.relabel_traces._teacher_logits_for_rows", fake_teacher_logits)

    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = os.path.join(tmpdir, "trace.jsonl")
        out_path = os.path.join(tmpdir, "relabeled.jsonl")
        with open(trace_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        result = relabel_trace_actions(trace_path, out_path, bc_model_path="/tmp/a.pt, /tmp/b.pt")
        relabeled_rows = load_trace_rows(out_path)
        assert result["teacher_bc_model_paths"] == ["/tmp/a.pt", "/tmp/b.pt"]
        assert [row["action"] for row in relabeled_rows] == ["east", "south"]
        assert [row["original_action"] for row in relabeled_rows] == ["west", "north"]


def test_train_bc_with_teacher_distillation_metadata():
    base_rows = [
        {
            "episode_id": "ep0",
            "step": 0,
            "seed": 1,
            "task": "explore",
            "action": "east",
            "allowed_actions": ["east", "west"],
            "observation_version": "v2",
            "feature_vector": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "episode_id": "ep0",
            "step": 1,
            "seed": 1,
            "task": "explore",
            "action": "west",
            "allowed_actions": ["east", "west"],
            "observation_version": "v2",
            "feature_vector": [0.0, 1.0, 0.0, 0.0],
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        teacher_path = os.path.join(tmpdir, "teacher.pt")
        student_path = os.path.join(tmpdir, "student.pt")
        train_bc_model(base_rows, teacher_path, epochs=10, lr=1e-3, hidden_size=32, observation_version="v2")
        result = train_bc_model(
            base_rows,
            student_path,
            epochs=5,
            lr=1e-3,
            hidden_size=32,
            observation_version="v2",
            distill_teacher_bc_path=teacher_path,
            distill_loss_coef=0.5,
            distill_temperature=2.0,
        )
        assert result["distill_teacher_bc_path"] == teacher_path
        assert result["distill_loss_coef"] == 0.5
        assert result["distill_temperature"] == 2.0


def test_train_bc_with_teacher_distillation_ensemble_metadata():
    base_rows = [
        {
            "episode_id": "ep0",
            "step": 0,
            "seed": 1,
            "task": "explore",
            "action": "east",
            "allowed_actions": ["east", "west"],
            "observation_version": "v2",
            "feature_vector": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "episode_id": "ep0",
            "step": 1,
            "seed": 1,
            "task": "explore",
            "action": "west",
            "allowed_actions": ["east", "west"],
            "observation_version": "v2",
            "feature_vector": [0.0, 1.0, 0.0, 0.0],
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        teacher_a = os.path.join(tmpdir, "teacher_a.pt")
        teacher_b = os.path.join(tmpdir, "teacher_b.pt")
        student_path = os.path.join(tmpdir, "student.pt")
        train_bc_model(base_rows, teacher_a, epochs=10, lr=1e-3, hidden_size=32, observation_version="v2")
        train_bc_model(base_rows, teacher_b, epochs=10, lr=1e-3, hidden_size=32, observation_version="v2")
        result = train_bc_model(
            base_rows,
            student_path,
            epochs=5,
            lr=1e-3,
            hidden_size=32,
            observation_version="v2",
            distill_teacher_bc_paths=[teacher_a, teacher_b],
            distill_loss_coef=0.25,
            distill_temperature=2.0,
            supervised_loss_coef=0.25,
        )
        assert result["distill_teacher_bc_path"] is None
        assert result["distill_teacher_bc_paths"] == [teacher_a, teacher_b]
        assert result["distill_loss_coef"] == 0.25
        assert result["distill_temperature"] == 2.0
        assert result["supervised_loss_coef"] == 0.25


def test_train_bc_supports_deeper_student_metadata():
    rows = [
        {
            "episode_id": "ep0",
            "step": 0,
            "seed": 1,
            "task": "explore",
            "action": "east",
            "allowed_actions": ["east", "west"],
            "observation_version": "v2",
            "feature_vector": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "episode_id": "ep0",
            "step": 1,
            "seed": 1,
            "task": "explore",
            "action": "west",
            "allowed_actions": ["east", "west"],
            "observation_version": "v2",
            "feature_vector": [0.0, 1.0, 0.0, 0.0],
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "deep.pt")
        result = train_bc_model(
            rows,
            path,
            epochs=5,
            lr=1e-3,
            hidden_size=32,
            num_layers=3,
            observation_version="v2",
        )
        policy = load_bc_model(path)
        linear_layers = sum(1 for module in policy.model.net if isinstance(module, torch.nn.Linear))
        assert result["num_layers"] == 3
        assert linear_layers == 4


def test_bc_warmstart_uses_final_linear_for_deeper_teacher():
    rows = [
        {
            "episode_id": "ep0",
            "step": 0,
            "seed": 1,
            "task": "explore",
            "action": "east",
            "allowed_actions": ["east", "west"],
            "observation_version": "v2",
            "feature_vector": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "episode_id": "ep0",
            "step": 1,
            "seed": 1,
            "task": "explore",
            "action": "west",
            "allowed_actions": ["east", "west"],
            "observation_version": "v2",
            "feature_vector": [0.0, 1.0, 0.0, 0.0],
        },
    ]

    class DummyActorCritic:
        def __init__(self):
            self._param = torch.nn.Parameter(torch.zeros(1))
            self.loaded_state = {
                "encoder.encoders.obs.mlp_head.0.weight": torch.zeros(32, 4),
                "encoder.encoders.obs.mlp_head.0.bias": torch.zeros(32),
                "encoder.encoders.obs.mlp_head.2.weight": torch.zeros(32, 32),
                "encoder.encoders.obs.mlp_head.2.bias": torch.zeros(32),
                "encoder.encoders.obs.mlp_head.4.weight": torch.zeros(32, 32),
                "encoder.encoders.obs.mlp_head.4.bias": torch.zeros(32),
                "action_parameterization.distribution_linear.weight": torch.zeros(len(ACTION_SET), 32),
                "action_parameterization.distribution_linear.bias": torch.zeros(len(ACTION_SET)),
            }

        def state_dict(self):
            return {key: value.clone() for key, value in self.loaded_state.items()}

        def load_state_dict(self, state_dict, strict=False):
            self.loaded_state = {key: value.clone() for key, value in state_dict.items()}

        def parameters(self):
            return [self._param]

    with tempfile.TemporaryDirectory() as tmpdir:
        teacher_path = os.path.join(tmpdir, "teacher.pt")
        train_bc_model(
            rows,
            teacher_path,
            epochs=5,
            lr=1e-3,
            hidden_size=32,
            num_layers=3,
            observation_version="v2",
        )
        config = RLConfig()
        config.train_dir = tmpdir
        config.experiment = "warmstart"
        config.model.bc_init_path = teacher_path
        config.model.num_layers = 3
        trainer = APPOTrainerScaffold(config)
        dummy = DummyActorCritic()
        sf_cfg = type("DummyCfg", (), {"learning_rate": 1e-4, "adam_eps": 1e-6})()
        trainer.maybe_write_bc_warmstart_checkpoint(sf_cfg, dummy)
        bc_state = torch.load(teacher_path, map_location="cpu")["state_dict"]
        assert torch.allclose(dummy.loaded_state["encoder.encoders.obs.mlp_head.0.weight"], bc_state["net.0.weight"])
        assert torch.allclose(dummy.loaded_state["encoder.encoders.obs.mlp_head.2.weight"], bc_state["net.2.weight"])
        assert torch.allclose(dummy.loaded_state["encoder.encoders.obs.mlp_head.4.weight"], bc_state["net.4.weight"])
        assert torch.allclose(
            dummy.loaded_state["action_parameterization.distribution_linear.weight"],
            bc_state["net.6.weight"],
        )


def test_bc_warmstart_copies_teacher_to_separate_actor_and_critic_encoders():
    rows = [
        {
            "episode_id": "ep0",
            "step": 0,
            "seed": 1,
            "task": "explore",
            "action": "east",
            "allowed_actions": ["east", "west"],
            "observation_version": "v2",
            "feature_vector": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "episode_id": "ep0",
            "step": 1,
            "seed": 1,
            "task": "explore",
            "action": "west",
            "allowed_actions": ["east", "west"],
            "observation_version": "v2",
            "feature_vector": [0.0, 1.0, 0.0, 0.0],
        },
    ]

    class DummyActorCritic:
        def __init__(self):
            self._param = torch.nn.Parameter(torch.zeros(1))
            self.loaded_state = {
                "actor_encoder.encoders.obs.mlp_head.0.weight": torch.zeros(32, 4),
                "actor_encoder.encoders.obs.mlp_head.0.bias": torch.zeros(32),
                "actor_encoder.encoders.obs.mlp_head.2.weight": torch.zeros(32, 32),
                "actor_encoder.encoders.obs.mlp_head.2.bias": torch.zeros(32),
                "actor_encoder.encoders.obs.mlp_head.4.weight": torch.zeros(32, 32),
                "actor_encoder.encoders.obs.mlp_head.4.bias": torch.zeros(32),
                "critic_encoder.encoders.obs.mlp_head.0.weight": torch.zeros(32, 4),
                "critic_encoder.encoders.obs.mlp_head.0.bias": torch.zeros(32),
                "critic_encoder.encoders.obs.mlp_head.2.weight": torch.zeros(32, 32),
                "critic_encoder.encoders.obs.mlp_head.2.bias": torch.zeros(32),
                "critic_encoder.encoders.obs.mlp_head.4.weight": torch.zeros(32, 32),
                "critic_encoder.encoders.obs.mlp_head.4.bias": torch.zeros(32),
                "action_parameterization.distribution_linear.weight": torch.zeros(len(ACTION_SET), 32),
                "action_parameterization.distribution_linear.bias": torch.zeros(len(ACTION_SET)),
            }

        def state_dict(self):
            return {key: value.clone() for key, value in self.loaded_state.items()}

        def load_state_dict(self, state_dict, strict=False):
            self.loaded_state = {key: value.clone() for key, value in state_dict.items()}

        def parameters(self):
            return [self._param]

    with tempfile.TemporaryDirectory() as tmpdir:
        teacher_path = os.path.join(tmpdir, "teacher.pt")
        train_bc_model(
            rows,
            teacher_path,
            epochs=5,
            lr=1e-3,
            hidden_size=32,
            num_layers=3,
            observation_version="v2",
        )
        config = RLConfig()
        config.train_dir = tmpdir
        config.experiment = "warmstart_separate"
        config.model.bc_init_path = teacher_path
        config.model.num_layers = 3
        config.model.actor_critic_share_weights = False
        trainer = APPOTrainerScaffold(config)
        dummy = DummyActorCritic()
        sf_cfg = type("DummyCfg", (), {"learning_rate": 1e-4, "adam_eps": 1e-6})()
        trainer.maybe_write_bc_warmstart_checkpoint(sf_cfg, dummy)
        bc_state = torch.load(teacher_path, map_location="cpu")["state_dict"]
        for prefix in ["actor_encoder", "critic_encoder"]:
            assert torch.allclose(dummy.loaded_state[f"{prefix}.encoders.obs.mlp_head.0.weight"], bc_state["net.0.weight"])
            assert torch.allclose(dummy.loaded_state[f"{prefix}.encoders.obs.mlp_head.2.weight"], bc_state["net.2.weight"])
            assert torch.allclose(dummy.loaded_state[f"{prefix}.encoders.obs.mlp_head.4.weight"], bc_state["net.4.weight"])
        assert torch.allclose(
            dummy.loaded_state["action_parameterization.distribution_linear.weight"],
            bc_state["net.6.weight"],
        )


def test_skill_env_masks_invalid_action_requests():
    config = RLConfig()
    config.options.enabled_skills = ["explore"]
    env = NethackSkillEnv(config)
    obs, info = env.reset(seed=42)
    assert "drink" not in info["allowed_actions"]
    drink_idx = 11
    obs, reward, terminated, truncated, info = env.step(drink_idx)
    assert info["debug"]["invalid_action_requested"] is True
    assert info["debug"]["requested_action_name"] == "drink"
    assert info["debug"]["action_name"] != "drink"
    env.close()


def test_explore_bonus_helper_uses_visit_counts():
    config = RLConfig()
    config.reward.episodic_explore_bonus_enabled = True
    config.reward.episodic_explore_bonus_mode = "tile"
    adapter = SkillEnvAdapter(config)
    adapter.ctx = EpisodeContext(active_skill="explore")
    adapter.ctx.tile_visit_counts[(3, 4)] = 0
    first = adapter._compute_episodic_explore_bonus("explore", "hash-a", (3, 4))
    adapter.ctx.tile_visit_counts[(3, 4)] = 3
    later = adapter._compute_episodic_explore_bonus("explore", "hash-a", (3, 4))
    assert first > later
    assert adapter._compute_episodic_explore_bonus("survive", "hash-a", (3, 4)) == 0.0
    adapter.close()


def test_skill_env_reports_episodic_bonus_debug():
    config = RLConfig()
    config.options.enabled_skills = ["explore"]
    config.reward.episodic_explore_bonus_enabled = True
    config.reward.episodic_explore_bonus_scale = 0.1
    env = NethackSkillEnv(config)
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(0)
    assert "episodic_explore_bonus" in info["debug"]
    assert info["debug"]["episodic_explore_bonus"] >= 0.0
    env.close()


def test_bc_model_round_trips_with_metadata():
    rows = [
        {
            "feature_vector": [0.0] * 160,
            "action": "east",
            "allowed_actions": ["east", "west"],
        },
        {
            "feature_vector": [0.1] * 160,
            "action": "west",
            "allowed_actions": ["east", "west"],
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "bc.pt")
        meta = train_bc_model(rows, out, epochs=1, lr=1e-3, hidden_size=64, observation_version="v2", device="cpu")
        assert meta["observation_version"] == "v2"
        assert meta["device_requested"] == "cpu"
        assert meta["training_device"] == "cpu"
        policy = load_bc_model(out)
        action = policy.act(rows[0]["feature_vector"], allowed_actions=["east", "west"])
        assert action in {"east", "west"}


def test_bc_can_select_by_heldout_metric(monkeypatch):
    rows = [
        {
            "feature_vector": [0.0] * 160,
            "action": "east",
            "allowed_actions": ["east", "west"],
            "observation_version": "v2",
        },
        {
            "feature_vector": [0.1] * 160,
            "action": "west",
            "allowed_actions": ["east", "west"],
            "observation_version": "v2",
        },
    ]
    scores = iter([0.9, 0.95, 0.92])

    def fake_eval(*args, **kwargs):
        return {"summary": {"match_rate": next(scores)}}

    monkeypatch.setattr("rl.trace_eval.evaluate_trace_policy", fake_eval)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "bc.pt")
        meta = train_bc_model(
            rows,
            out,
            epochs=3,
            lr=1e-3,
            hidden_size=64,
            observation_version="v2",
            device="cpu",
            heldout_trace_path="/tmp/heldout.jsonl",
            select_by_heldout=True,
        )
        payload = torch.load(out, map_location="cpu")
        assert payload["metadata"]["selection_metric"] == "heldout_match_rate"
        assert payload["metadata"]["selected_epoch"] == 2
        assert payload["metadata"]["selection_score"] == 0.95
        assert len(meta["epoch_summaries"]) == 3


def test_text_conditioned_bc_uses_prompt_text_in_trace_eval():
    train_rows = [
        {
            "episode_id": "ep0",
            "seed": 10,
            "step": 0,
            "action": "east",
            "allowed_actions": ["east", "west"],
            "feature_vector": [0.0] * 160,
            "observation_version": "v2",
            "state_prompt": "Adjacent: east=floor west=wall\nGoal: move east",
        },
        {
            "episode_id": "ep1",
            "seed": 11,
            "step": 0,
            "action": "west",
            "allowed_actions": ["east", "west"],
            "feature_vector": [0.0] * 160,
            "observation_version": "v2",
            "state_prompt": "Adjacent: east=wall west=floor\nGoal: move west",
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = os.path.join(tmpdir, "trace.jsonl")
        out = os.path.join(tmpdir, "bc_text.pt")
        with open(trace_path, "w") as f:
            for row in train_rows:
                f.write(json.dumps(row) + "\n")
        meta = train_bc_model(
            train_rows,
            out,
            epochs=80,
            lr=1e-3,
            hidden_size=64,
            observation_version="v2",
            text_encoder_backend="hash",
            text_embedding_dim=32,
        )
        assert meta["text_encoder_backend"] == "hash"
        policy = load_bc_model(out)
        assert policy.act(
            train_rows[0]["feature_vector"],
            allowed_actions=["east", "west"],
            prompt_text=train_rows[0]["state_prompt"],
        ) == "east"
        assert policy.act(
            train_rows[1]["feature_vector"],
            allowed_actions=["east", "west"],
            prompt_text=train_rows[1]["state_prompt"],
        ) == "west"
        result = evaluate_trace_policy(trace_path=trace_path, policy="bc", bc_model_path=out)
        assert result["summary"]["match_rate"] == 1.0


def test_determinism_check_runs_for_wall_avoidance():
    result = check_policy_determinism(
        policy="wall_avoidance",
        task="explore",
        seeds=[42],
        max_steps=3,
        repeats=2,
    )
    assert result["policy"] == "wall_avoidance"
    assert len(result["runs"]) == 2
    assert result["runs"][0]["episodes"][0]["steps"] >= 1


def test_compare_actions_on_teacher_states_with_bc():
    rows = [
        {
            "feature_vector": [0.0] * 160,
            "action": "east",
            "allowed_actions": ["east", "west"],
        },
        {
            "feature_vector": [0.1] * 160,
            "action": "west",
            "allowed_actions": ["east", "west"],
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "bc.pt")
        train_bc_model(rows, out, epochs=1, lr=1e-3, hidden_size=64, observation_version="v2")
        result = compare_actions_on_teacher_states(
            task="explore",
            seeds=[42],
            max_steps=2,
            bc_model_path=out,
            observation_version="v2",
        )
        assert result["summary"]["teacher_vs_bc_match_rate"] is not None
        assert len(result["episodes"]) == 1


def test_trace_eval_is_deterministic_for_bc():
    rows = [
        {
            "episode_id": "ep0",
            "seed": 42,
            "step": 0,
            "action": "east",
            "allowed_actions": ["east", "west"],
            "feature_vector": [0.0] * 160,
            "observation_version": "v2",
        },
        {
            "episode_id": "ep0",
            "seed": 42,
            "step": 1,
            "action": "west",
            "allowed_actions": ["east", "west"],
            "feature_vector": [0.1] * 160,
            "observation_version": "v2",
        },
        {
            "episode_id": "ep1",
            "seed": 43,
            "step": 0,
            "action": "east",
            "allowed_actions": ["east", "west"],
            "feature_vector": [0.2] * 160,
            "observation_version": "v2",
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = os.path.join(tmpdir, "trace.jsonl")
        with open(trace_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        out = os.path.join(tmpdir, "bc.pt")
        train_bc_model(rows, out, epochs=50, lr=1e-3, hidden_size=64, observation_version="v2")
        result_a = evaluate_trace_policy(trace_path, "bc", bc_model_path=out)
        result_b = evaluate_trace_policy(trace_path, "bc", bc_model_path=out)
        assert result_a["summary"]["rows"] == 3
        assert result_a["summary"]["match_rate"] == result_b["summary"]["match_rate"]
        assert result_a["episodes"] == result_b["episodes"]


def test_checkpoint_payload_loader_accepts_plain_torch_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "checkpoint_000000001_100.pth"
        torch.save({"model": {"weight": torch.tensor([1.0])}}, ckpt)
        payload = _load_checkpoint_payload(ckpt, torch.device("cpu"))
        assert "model" in payload


def test_timestep_repeated_state_count_uses_hash_history():
    timestep = build_policy_timestep(
        state={"position": (4, 5)},
        task="explore",
        allowed_actions=["east"],
        memory=type("Mem", (), {"total_explored": 10, "rooms": {1}})(),
        step=3,
        recent_positions=[(4, 5), (4, 6)],
        recent_actions=["east", "east"],
        recent_state_hashes=["a", "b", "a"],
        obs_hash="a",
        obs={"chars": np.full((10, 10), ord("."), dtype=np.int32)},
    )
    assert timestep["repeated_state_count"] == 2


def test_trace_eval_rejects_mixed_versions():
    rows = [
        {
            "episode_id": "ep0",
            "seed": 1,
            "step": 0,
            "action": "east",
            "allowed_actions": ["east"],
            "feature_vector": [0.0] * 160,
            "observation_version": "v2",
        },
        {
            "episode_id": "ep0",
            "seed": 1,
            "step": 1,
            "action": "east",
            "allowed_actions": ["east"],
            "feature_vector": [0.0] * 106,
            "observation_version": "v1",
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = os.path.join(tmpdir, "trace.jsonl")
        with open(trace_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        try:
            evaluate_trace_policy(trace_path, "bc", bc_model_path=os.path.join(tmpdir, "missing.pt"))
            assert False, "expected ValueError for mixed observation versions"
        except ValueError as exc:
            assert "Mixed observation versions" in str(exc)


def test_trace_disagreement_report_counts_confusions():
    rows = [
        {
            "episode_id": "ep0",
            "seed": 42,
            "step": 0,
            "action": "east",
            "allowed_actions": ["east", "west"],
            "feature_vector": [0.0] * 160,
            "observation_version": "v2",
        },
        {
            "episode_id": "ep0",
            "seed": 42,
            "step": 1,
            "action": "west",
            "allowed_actions": ["east", "west"],
            "feature_vector": [0.1] * 160,
            "observation_version": "v2",
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = os.path.join(tmpdir, "trace.jsonl")
        with open(trace_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        out = os.path.join(tmpdir, "bc.pt")
        train_bc_model(rows, out, epochs=50, lr=1e-3, hidden_size=64, observation_version="v2")
        report = trace_disagreement_report(trace_path, bc_model_path=out, top_k=3)
        assert report["bc"]["rows"] == 2
        assert "teacher_action_counts" in report["bc"]
        assert "predicted_action_counts" in report["bc"]
        assert "by_teacher_action" in report["bc"]


def test_shard_trace_file_preserves_full_episodes():
    rows = [
        {"episode_id": "ep0", "seed": 1, "step": 0, "action": "east", "allowed_actions": ["east"], "feature_vector": [0.0] * 160, "observation_version": "v2"},
        {"episode_id": "ep0", "seed": 1, "step": 1, "action": "east", "allowed_actions": ["east"], "feature_vector": [0.0] * 160, "observation_version": "v2"},
        {"episode_id": "ep1", "seed": 2, "step": 0, "action": "west", "allowed_actions": ["west"], "feature_vector": [0.0] * 160, "observation_version": "v2"},
        {"episode_id": "ep1", "seed": 2, "step": 1, "action": "west", "allowed_actions": ["west"], "feature_vector": [0.0] * 160, "observation_version": "v2"},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = os.path.join(tmpdir, "trace.jsonl")
        shard_path = os.path.join(tmpdir, "shard.jsonl")
        with open(trace_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        summary = shard_trace_file(trace_path, shard_path, max_episodes=1)
        assert summary["episodes"] == 1
        assert summary["rows"] == 2
        assert summary["all_multi_turn"] is True


def test_shard_trace_file_can_filter_by_teacher_action():
    rows = [
        {"episode_id": "ep0", "seed": 1, "step": 0, "action": "east", "allowed_actions": ["east"], "feature_vector": [0.0] * 160, "observation_version": "v2"},
        {"episode_id": "ep0", "seed": 1, "step": 1, "action": "east", "allowed_actions": ["east"], "feature_vector": [0.0] * 160, "observation_version": "v2"},
        {"episode_id": "ep1", "seed": 2, "step": 0, "action": "north", "allowed_actions": ["north"], "feature_vector": [0.0] * 160, "observation_version": "v2"},
        {"episode_id": "ep1", "seed": 2, "step": 1, "action": "north", "allowed_actions": ["north"], "feature_vector": [0.0] * 160, "observation_version": "v2"},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = os.path.join(tmpdir, "trace.jsonl")
        shard_path = os.path.join(tmpdir, "shard.jsonl")
        with open(trace_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        summary = shard_trace_file(trace_path, shard_path, teacher_actions=["east"])
        assert summary["episodes"] == 1
        assert summary["rows"] == 2
        assert summary["selected_teacher_actions"] == ["east"]


def test_shard_trace_file_can_filter_by_adjacent_signature():
    rows = [
        {
            "episode_id": "ep0",
            "seed": 1,
            "step": 0,
            "action": "east",
            "allowed_actions": ["east"],
            "feature_vector": [0.0] * 160,
            "observation_version": "v2",
            "state_prompt": "Adjacent: north=monster_d south=floor east=monster_f west=floor",
        },
        {
            "episode_id": "ep0",
            "seed": 1,
            "step": 1,
            "action": "east",
            "allowed_actions": ["east"],
            "feature_vector": [0.0] * 160,
            "observation_version": "v2",
            "state_prompt": "Adjacent: north=floor south=floor east=floor west=floor",
        },
        {
            "episode_id": "ep1",
            "seed": 2,
            "step": 0,
            "action": "north",
            "allowed_actions": ["north"],
            "feature_vector": [0.0] * 160,
            "observation_version": "v2",
            "state_prompt": "Adjacent: north=monster_d south=wall east=monster_f west=wall",
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = os.path.join(tmpdir, "trace.jsonl")
        shard_path = os.path.join(tmpdir, "shard.jsonl")
        with open(trace_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        summary = shard_trace_file(
            trace_path,
            shard_path,
            adjacent_signature={
                "north": "monster_*",
                "south": "floor",
                "east": "monster_*",
                "west": "floor",
            },
        )
        assert summary["episodes"] == 1
        assert summary["rows"] == 2
        assert summary["selected_adjacent_signature"] == {
            "north": "monster_*",
            "south": "floor",
            "east": "monster_*",
            "west": "floor",
        }


def test_behavior_reg_can_select_by_heldout_metric():
    train_rows = [
        {"episode_id": "ep0", "seed": 1, "step": 0, "action": "east", "allowed_actions": ["east", "west"], "feature_vector": [0.0] * 160, "observation_version": "v2"},
        {"episode_id": "ep0", "seed": 1, "step": 1, "action": "east", "allowed_actions": ["east", "west"], "feature_vector": [0.1] * 160, "observation_version": "v2"},
        {"episode_id": "ep1", "seed": 2, "step": 0, "action": "west", "allowed_actions": ["east", "west"], "feature_vector": [0.2] * 160, "observation_version": "v2"},
        {"episode_id": "ep1", "seed": 2, "step": 1, "action": "west", "allowed_actions": ["east", "west"], "feature_vector": [0.3] * 160, "observation_version": "v2"},
    ]
    heldout_rows = [
        {"episode_id": "ep2", "seed": 3, "step": 0, "action": "east", "allowed_actions": ["east", "west"], "feature_vector": [0.0] * 160, "observation_version": "v2"},
        {"episode_id": "ep2", "seed": 3, "step": 1, "action": "west", "allowed_actions": ["east", "west"], "feature_vector": [0.2] * 160, "observation_version": "v2"},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        heldout_path = os.path.join(tmpdir, "heldout.jsonl")
        with open(heldout_path, "w") as f:
            for row in heldout_rows:
                f.write(json.dumps(row) + "\n")
        out = os.path.join(tmpdir, "breg.pt")
        meta = train_behavior_regularized_policy(
            train_rows,
            out,
            heldout_trace_path=heldout_path,
            epochs=3,
            hidden_size=64,
            observation_version="v2",
        )
        payload = torch.load(out, map_location="cpu")
        assert payload["metadata"]["selection_metric"] == "heldout_match_rate"
        assert payload["metadata"]["selected_epoch"] >= 1
        assert meta["epoch_summaries"]


def test_dagger_schedule_can_stop_on_regression():
    rows = [
        {"episode_id": "ep0", "seed": 1, "step": 0, "action": "east", "allowed_actions": ["east", "west"], "feature_vector": [0.0] * 160, "observation_version": "v2"},
        {"episode_id": "ep0", "seed": 1, "step": 1, "action": "east", "allowed_actions": ["east", "west"], "feature_vector": [0.1] * 160, "observation_version": "v2"},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = os.path.join(tmpdir, "trace.jsonl")
        with open(trace_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        bc_path = os.path.join(tmpdir, "bc.pt")
        train_bc_model(rows, bc_path, epochs=1, lr=1e-3, hidden_size=64, observation_version="v2")
        summary = run_dagger_schedule(
            base_trace_input=trace_path,
            output_dir=os.path.join(tmpdir, "dagger"),
            student_policy="bc",
            task="explore",
            iterations=2,
            num_episodes=1,
            max_steps=1,
            bc_model_path=bc_path,
            observation_version="v2",
            merge_ratio=0.5,
            merge_policy="uniform_merge",
            epochs=1,
            hidden_size=64,
            heldout_trace_input=trace_path,
            stop_on_heldout_regression=True,
        )
        assert "best_bc_model" in summary
        assert summary["best_bc_model"] is not None


def test_trace_disagreement_report_includes_per_action_metrics():
    rows = [
        {
            "episode_id": "ep0",
            "seed": 42,
            "step": 0,
            "action": "east",
            "allowed_actions": ["east", "west"],
            "feature_vector": [0.0] * 160,
            "observation_version": "v2",
        },
        {
            "episode_id": "ep0",
            "seed": 42,
            "step": 1,
            "action": "west",
            "allowed_actions": ["east", "west"],
            "feature_vector": [0.1] * 160,
            "observation_version": "v2",
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = os.path.join(tmpdir, "trace.jsonl")
        with open(trace_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        out = os.path.join(tmpdir, "bc.pt")
        train_bc_model(rows, out, epochs=50, lr=1e-3, hidden_size=64, observation_version="v2")
        report = trace_disagreement_report(trace_path, bc_model_path=out, top_k=3)
        assert "per_action_metrics" in report["bc"]
        assert "east" in report["bc"]["per_action_metrics"]


def test_generate_multi_turn_traces_task_greedy_v3():
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = os.path.join(tmpdir, "trace_v3.jsonl")
        summary = generate_multi_turn_traces(
            output_path=trace_path,
            num_episodes=1,
            max_steps=2,
            seed_start=42,
            policy="task_greedy",
            task="explore",
            observation_version="v3",
        )
        assert summary["observation_versions"] == ["v3"]
        verify = verify_trace_file(trace_path)
        assert verify["episodes"] == 1
        assert verify["feature_dims"] == [244]


def test_write_trace_best_alias_writes_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "best_trace.json")
        result = write_trace_best_alias(
            {
                "experiment": "exp",
                "trace_input": "trace.jsonl",
                "best_checkpoint_path": "checkpoint.pth",
                "num_checkpoints": 3,
            },
            out,
        )
        assert result == out
        payload = json.load(open(out))
        assert payload["best_checkpoint_path"] == "checkpoint.pth"


def test_generate_dagger_traces_wall_avoidance_runs():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "dagger.jsonl")
        summary = generate_dagger_traces(
            output_path=path,
            num_episodes=1,
            max_steps=1,
            student_policy="wall_avoidance",
            task="explore",
            seed_start=42,
            observation_version="v2",
        )
        assert summary["episodes"] == 1
        assert summary["rows"] == 1
        assert summary["all_multi_turn"] is False
        assert "teacher_match_rate" in summary


def test_generate_dagger_traces_can_use_bc_teacher(monkeypatch):
    def fail_select_task_action(*args, **kwargs):
        raise AssertionError("task teacher should not be used when teacher_bc_model_path is set")

    monkeypatch.setattr("rl.traces.select_task_action", fail_select_task_action)

    rows = [
        {
            "episode_id": "ep0",
            "seed": 42,
            "step": 0,
            "action": "east",
            "allowed_actions": ["east", "west"],
            "feature_vector": [0.0] * 160,
            "observation_version": "v2",
        },
        {
            "episode_id": "ep0",
            "seed": 42,
            "step": 1,
            "action": "west",
            "allowed_actions": ["east", "west"],
            "feature_vector": [0.1] * 160,
            "observation_version": "v2",
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        teacher_path = os.path.join(tmpdir, "teacher.pt")
        dagger_path = os.path.join(tmpdir, "dagger.jsonl")
        train_bc_model(rows, teacher_path, epochs=3, lr=1e-3, hidden_size=32, observation_version="v2")
        summary = generate_dagger_traces(
            output_path=dagger_path,
            num_episodes=1,
            max_steps=1,
            student_policy="wall_avoidance",
            task="explore",
            seed_start=42,
            observation_version="v2",
            teacher_bc_model_path=teacher_path,
        )
        assert summary["episodes"] == 1
        assert summary["rows"] == 1
        assert summary["teacher_policy"] == "bc"
        assert summary["teacher_bc_model_path"] == teacher_path


def test_run_dagger_iteration_accepts_appo_checkpoint_without_experiment(monkeypatch):
    calls = {}

    def fake_generate_dagger_traces(**kwargs):
        calls["appo_checkpoint_path"] = kwargs["appo_checkpoint_path"]
        with open(kwargs["output_path"], "w") as f:
            f.write(json.dumps({"episode_id": "dagger:appo:0", "step": 0, "step_index": 0, "observation_version": "v4", "features": [0.0], "action_index": 0}) + "\n")
        return {"episodes": 1, "rows": 1}

    def fake_load_trace_rows(path):
        return [{"observation_version": "v4", "action_index": 0}]

    def fake_build_merged_trace_rows(**kwargs):
        return [{"observation_version": "v4", "action_index": 0}]

    def fake_train_bc_model(rows, bc_output, **kwargs):
        assert kwargs["observation_version"] == "v4"
        return {"rows": len(rows), "bc_output": bc_output}

    def fake_evaluate_trace_policy(*args, **kwargs):
        return {"summary": {"match_rate": 0.5}}

    monkeypatch.setattr(dagger_module, "generate_dagger_traces", fake_generate_dagger_traces)
    monkeypatch.setattr(dagger_module, "load_trace_rows", fake_load_trace_rows)
    monkeypatch.setattr(dagger_module, "build_merged_trace_rows", fake_build_merged_trace_rows)
    monkeypatch.setattr(dagger_module, "train_bc_model", fake_train_bc_model)
    monkeypatch.setattr(dagger_module, "evaluate_trace_policy", fake_evaluate_trace_policy)

    with tempfile.TemporaryDirectory() as tmpdir:
        report = run_dagger_iteration(
            base_trace_input=os.path.join(tmpdir, "base.jsonl"),
            dagger_trace_output=os.path.join(tmpdir, "dagger.jsonl"),
            bc_output=os.path.join(tmpdir, "bc.pt"),
            student_policy="appo",
            appo_checkpoint_path=os.path.join(tmpdir, "checkpoint.pth"),
            observation_version="v4",
        )
        assert calls["appo_checkpoint_path"].endswith("checkpoint.pth")
        assert report["dagger_trace_summary"]["episodes"] == 1


def test_run_dagger_iteration_passes_teacher_bc_and_distill(monkeypatch):
    calls = {}

    def fake_generate_dagger_traces(**kwargs):
        calls["teacher_bc_model_path"] = kwargs["teacher_bc_model_path"]
        with open(kwargs["output_path"], "w") as f:
            f.write(json.dumps({"episode_id": "dagger:appo:0", "step": 0, "observation_version": "v4", "feature_vector": [0.0] * 302, "action": "east", "allowed_actions": ["east", "west"]}) + "\n")
        return {"episodes": 1, "rows": 1, "teacher_policy": "bc"}

    def fake_load_trace_rows(path):
        return [{"observation_version": "v4", "feature_vector": [0.0] * 302, "action": "east", "allowed_actions": ["east", "west"]}]

    def fake_build_merged_trace_rows(**kwargs):
        return [{"observation_version": "v4", "feature_vector": [0.0] * 302, "action": "east", "allowed_actions": ["east", "west"]}]

    def fake_train_bc_model(rows, bc_output, **kwargs):
        calls["distill_teacher_bc_path"] = kwargs["distill_teacher_bc_path"]
        calls["distill_loss_coef"] = kwargs["distill_loss_coef"]
        calls["distill_temperature"] = kwargs["distill_temperature"]
        return {"rows": len(rows), "bc_output": bc_output}

    def fake_evaluate_trace_policy(*args, **kwargs):
        return {"summary": {"match_rate": 0.5}}

    monkeypatch.setattr(dagger_module, "generate_dagger_traces", fake_generate_dagger_traces)
    monkeypatch.setattr(dagger_module, "load_trace_rows", fake_load_trace_rows)
    monkeypatch.setattr(dagger_module, "build_merged_trace_rows", fake_build_merged_trace_rows)
    monkeypatch.setattr(dagger_module, "train_bc_model", fake_train_bc_model)
    monkeypatch.setattr(dagger_module, "evaluate_trace_policy", fake_evaluate_trace_policy)

    with tempfile.TemporaryDirectory() as tmpdir:
        report = run_dagger_iteration(
            base_trace_input=os.path.join(tmpdir, "base.jsonl"),
            dagger_trace_output=os.path.join(tmpdir, "dagger.jsonl"),
            bc_output=os.path.join(tmpdir, "bc.pt"),
            student_policy="appo",
            appo_checkpoint_path=os.path.join(tmpdir, "checkpoint.pth"),
            teacher_bc_model_path=os.path.join(tmpdir, "teacher.pt"),
            observation_version="v4",
            distill_loss_coef=0.2,
            distill_temperature=2.0,
        )
        assert calls["teacher_bc_model_path"].endswith("teacher.pt")
        assert calls["distill_teacher_bc_path"].endswith("teacher.pt")
        assert calls["distill_loss_coef"] == 0.2
        assert calls["distill_temperature"] == 2.0
        assert report["dagger_trace_summary"]["teacher_policy"] == "bc"


def test_run_dagger_iteration_filters_dagger_rows_before_merge(monkeypatch):
    captured = {}

    def fake_generate_dagger_traces(**kwargs):
        with open(kwargs["output_path"], "w") as f:
            rows = [
                {
                    "episode_id": "dagger:appo:0",
                    "step": 0,
                    "observation_version": "v4",
                    "feature_vector": [0.0] * 302,
                    "action": "east",
                    "allowed_actions": ["east", "west"],
                    "behavior_action": "east",
                    "teacher_action": "east",
                },
                {
                    "episode_id": "dagger:appo:0",
                    "step": 1,
                    "observation_version": "v4",
                    "feature_vector": [0.0] * 302,
                    "action": "south",
                    "allowed_actions": ["south", "north"],
                    "behavior_action": "east",
                    "teacher_action": "south",
                    "is_disagreement_candidate": True,
                },
                {
                    "episode_id": "dagger:appo:0",
                    "step": 2,
                    "observation_version": "v4",
                    "feature_vector": [0.0] * 302,
                    "action": "west",
                    "allowed_actions": ["west", "east"],
                    "behavior_action": "east",
                    "teacher_action": "west",
                    "is_disagreement_candidate": True,
                },
            ]
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return {"episodes": 1}

    def fake_load_trace_rows(path):
        if path.endswith("base.jsonl"):
            return [
                {
                    "episode_id": "base",
                    "step": 0,
                    "observation_version": "v4",
                    "feature_vector": [0.0] * 302,
                    "action": "east",
                    "allowed_actions": ["east", "west"],
                }
            ]
        return [
            {
                "episode_id": "dagger:appo:0",
                "step": 0,
                "observation_version": "v4",
                "feature_vector": [0.0] * 302,
                "action": "east",
                "allowed_actions": ["east", "west"],
                "behavior_action": "east",
                "teacher_action": "east",
            },
            {
                "episode_id": "dagger:appo:0",
                "step": 1,
                "observation_version": "v4",
                "feature_vector": [0.0] * 302,
                "action": "south",
                "allowed_actions": ["south", "north"],
                "behavior_action": "east",
                "teacher_action": "south",
                "is_disagreement_candidate": True,
            },
            {
                "episode_id": "dagger:appo:0",
                "step": 2,
                "observation_version": "v4",
                "feature_vector": [0.0] * 302,
                "action": "west",
                "allowed_actions": ["west", "east"],
                "behavior_action": "east",
                "teacher_action": "west",
                "is_disagreement_candidate": True,
            },
        ]

    def fake_build_merged_trace_rows(**kwargs):
        captured["relabeled_rows"] = kwargs["relabeled_rows"]
        return kwargs["base_rows"] + kwargs["relabeled_rows"]

    def fake_train_bc_model(rows, bc_output, **kwargs):
        return {"rows": len(rows), "bc_output": bc_output}

    def fake_evaluate_trace_policy(*args, **kwargs):
        return {"summary": {"match_rate": 0.5}}

    monkeypatch.setattr(dagger_module, "generate_dagger_traces", fake_generate_dagger_traces)
    monkeypatch.setattr(dagger_module, "load_trace_rows", fake_load_trace_rows)
    monkeypatch.setattr(dagger_module, "build_merged_trace_rows", fake_build_merged_trace_rows)
    monkeypatch.setattr(dagger_module, "train_bc_model", fake_train_bc_model)
    monkeypatch.setattr(dagger_module, "evaluate_trace_policy", fake_evaluate_trace_policy)

    with tempfile.TemporaryDirectory() as tmpdir:
        report = run_dagger_iteration(
            base_trace_input=os.path.join(tmpdir, "base.jsonl"),
            dagger_trace_output=os.path.join(tmpdir, "dagger.jsonl"),
            bc_output=os.path.join(tmpdir, "bc.pt"),
            student_policy="appo",
            appo_checkpoint_path=os.path.join(tmpdir, "checkpoint.pth"),
            observation_version="v4",
            dagger_row_policy="disagreement",
            dagger_keep_match_ratio=0.0,
            dagger_confusion_pairs="east->south",
        )

    assert len(captured["relabeled_rows"]) == 1
    assert captured["relabeled_rows"][0]["teacher_action"] == "south"
    assert report["dagger_confusion_pairs"] == ["east->south"]
    assert report["selected_dagger_rows"] == 1
    assert report["selected_dagger_row_summary"]["disagreement_rows"] == 1
    assert report["selected_dagger_row_summary"]["confusion_pair_rows"] == 1


def test_rank_checkpoints_skips_unreadable_checkpoint(monkeypatch):
    checkpoints = [Path("/tmp/valid.pth"), Path("/tmp/broken.pth")]

    def fake_list_checkpoint_paths(experiment, train_dir):
        assert experiment == "exp"
        return checkpoints

    def fake_evaluate_trace_policy(**kwargs):
        if kwargs["appo_checkpoint_path"].endswith("broken.pth"):
            raise RuntimeError("partial checkpoint write")
        return {"summary": {"match_rate": 0.75, "invalid_action_rate": 0.0, "action_counts": {"east": 3}}}

    monkeypatch.setattr(checkpoint_tools, "list_checkpoint_paths", fake_list_checkpoint_paths)
    monkeypatch.setattr(checkpoint_tools, "evaluate_trace_policy", fake_evaluate_trace_policy)
    result = rank_appo_checkpoints_by_trace(
        experiment="exp",
        train_dir="train_dir/rl",
        trace_input="trace.jsonl",
    )
    assert result["num_checkpoints"] == 1
    assert result["num_skipped"] == 1
    assert result["best_checkpoint_path"] == "/tmp/valid.pth"
    assert result["skipped"][0]["checkpoint_path"] == "/tmp/broken.pth"


def test_experiment_lock_serializes_access():
    with tempfile.TemporaryDirectory() as tmpdir:
        lock_path = os.path.join(tmpdir, "exp", ".launch.lock")
        order = []

        def worker(name):
            with experiment_lock(lock_path):
                order.append(f"{name}:enter")
                import time
                time.sleep(0.05)
                order.append(f"{name}:exit")

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, "a"), pool.submit(worker, "b")]
            for future in futures:
                future.result()

        assert order in (
            ["a:enter", "a:exit", "b:enter", "b:exit"],
            ["b:enter", "b:exit", "a:enter", "a:exit"],
        )


def test_build_merged_trace_rows_policies():
    import random

    base_rows = [
        {"episode_id": "ep0", "step": 0},
        {"episode_id": "ep0", "step": 1},
        {"episode_id": "ep1", "step": 0},
        {"episode_id": "ep1", "step": 1},
    ]
    relabeled_rows = [
        {"episode_id": "ep2", "step": 0},
        {"episode_id": "ep2", "step": 1},
    ]
    merged_base = build_merged_trace_rows(
        base_rows=base_rows,
        relabeled_rows=relabeled_rows,
        merge_policy="base_only",
        merge_ratio=0.5,
        rng=random.Random(0),
    )
    assert len(merged_base) == 4
    assert merged_base[-1]["episode_id"] == "ep2"
    merged_uniform = build_merged_trace_rows(
        base_rows=base_rows,
        relabeled_rows=relabeled_rows,
        merge_policy="uniform_merge",
        merge_ratio=0.5,
        rng=random.Random(0),
    )
    assert len(merged_uniform) == 4
    merged_weighted = build_merged_trace_rows(
        base_rows=base_rows,
        relabeled_rows=relabeled_rows,
        merge_policy="weighted_recent",
        merge_ratio=0.5,
        rng=random.Random(0),
    )
    assert len(merged_weighted) == 4


def test_select_dagger_rows_targets_hard_cases_and_keeps_match_anchor():
    import random

    rows = [
        {"episode_id": "a", "step": 0, "behavior_action": "east", "teacher_action": "east"},
        {"episode_id": "a", "step": 1, "behavior_action": "north", "teacher_action": "south", "is_disagreement_candidate": True},
        {"episode_id": "a", "step": 2, "behavior_action": "west", "teacher_action": "west", "is_loop_risk": True},
        {"episode_id": "a", "step": 3, "behavior_action": "search", "teacher_action": "search", "is_weak_action": True},
    ]
    selected = select_dagger_rows(
        rows=rows,
        row_selection_policy="hard_only",
        keep_match_ratio=0.5,
        confusion_pairs=None,
        rng=random.Random(7),
    )
    assert len(selected) == 4
    assert sum(int(row.get("behavior_action") == row.get("teacher_action")) for row in selected) == 3

    disagreement_only = select_dagger_rows(
        rows=rows,
        row_selection_policy="disagreement",
        keep_match_ratio=0.0,
        confusion_pairs=None,
        rng=random.Random(7),
    )
    assert len(disagreement_only) == 1
    assert disagreement_only[0]["teacher_action"] == "south"

    pair_only = select_dagger_rows(
        rows=rows,
        row_selection_policy="disagreement",
        keep_match_ratio=0.0,
        confusion_pairs={("north", "south")},
        rng=random.Random(7),
    )
    assert len(pair_only) == 1
    assert pair_only[0]["behavior_action"] == "north"
    assert pair_only[0]["teacher_action"] == "south"


def test_trace_checkpoint_monitor_writes_best_metadata(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "exp" / "checkpoint_p0"
        checkpoint_dir.mkdir(parents=True)
        ckpt = checkpoint_dir / "checkpoint_000000001_1024.pth"
        torch.save({"model": {"weight": torch.tensor([1.0])}}, ckpt)

        monkeypatch.setattr(
            checkpoint_tools,
            "evaluate_checkpoint_trace_match",
            lambda **kwargs: {
                "checkpoint_path": str(ckpt),
                "env_steps": 1024,
                "match_rate": 0.75,
                "invalid_action_rate": 0.0,
                "action_counts": {"east": 3},
            },
        )
        monitor = TraceCheckpointMonitor(
            experiment="exp",
            train_dir=tmpdir,
            trace_input="trace.jsonl",
            interval_env_steps=512,
            poll_seconds=0.01,
        )
        monitor.start()
        monitor.stop()
        metadata = json.loads((checkpoint_dir / "best_trace_match.json").read_text())
        assert metadata["match_rate"] == 0.75
        assert metadata["env_steps"] == 1024
        assert (checkpoint_dir / "best_trace_match.pth").exists()


def test_record_warmstart_trace_match_writes_separate_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "exp" / "checkpoint_p0"
        checkpoint_dir.mkdir(parents=True)
        warmstart = checkpoint_dir / "checkpoint_000000000_0.pth"
        torch.save({"model": {"weight": torch.tensor([1.0])}}, warmstart)

        metadata = checkpoint_tools.record_warmstart_trace_match(
            experiment="exp",
            train_dir=tmpdir,
            trace_input="trace.jsonl",
            checkpoint_path=str(warmstart),
            evaluation={
                "env_steps": 0,
                "match_rate": 0.975,
                "invalid_action_rate": 0.0,
                "action_counts": {"east": 4},
            },
        )
        warmstart_metadata = json.loads((checkpoint_dir / "warmstart_trace_match.json").read_text())
        assert warmstart_metadata["match_rate"] == 0.975
        assert warmstart_metadata["env_steps"] == 0
        assert (checkpoint_dir / "warmstart_trace_match.pth").exists()
        assert metadata["alias_checkpoint_path"].endswith("warmstart_trace_match.pth")


def test_behavior_regularized_policy_trains():
    rows = [
        {
            "episode_id": "ep0",
            "seed": 42,
            "step": 0,
            "action": "east",
            "allowed_actions": ["east", "west"],
            "feature_vector": [0.0] * 160,
            "observation_version": "v2",
        },
        {
            "episode_id": "ep0",
            "seed": 42,
            "step": 1,
            "action": "west",
            "allowed_actions": ["east", "west"],
            "feature_vector": [0.1] * 160,
            "observation_version": "v2",
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "breg.pt")
        trace_path = os.path.join(tmpdir, "trace.jsonl")
        with open(trace_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        result = train_behavior_regularized_policy(
            rows,
            out,
            epochs=5,
            lr=1e-3,
            hidden_size=64,
            observation_version="v2",
            behavior_coef=0.1,
            temperature=1.0,
        )
        assert result["observation_version"] == "v2"
        assert os.path.exists(out)
        eval_summary = evaluate_trace_policy(trace_path, "bc", bc_model_path=out, summary_only=True)["summary"]
        assert eval_summary["rows"] == 2


def test_behavior_reg_masks_and_renormalizes_behavior_targets():
    action_masks = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    prior = torch.tensor([0.6, 0.2, 0.1, 0.1], dtype=torch.float32)
    targets = _masked_behavior_targets(action_masks, prior)
    assert torch.allclose(targets[0], torch.tensor([0.85714287, 0.0, 0.14285715, 0.0]), atol=1e-6)
    assert torch.allclose(targets[1], torch.tensor([0.0, 0.6666667, 0.0, 0.33333334]), atol=1e-6)


def test_run_dagger_schedule_produces_iteration_report():
    rows = [
        {
            "episode_id": "ep0",
            "seed": 42,
            "step": 0,
            "action": "east",
            "allowed_actions": ["east", "west"],
            "feature_vector": [0.0] * 160,
            "observation_version": "v2",
        },
        {
            "episode_id": "ep0",
            "seed": 42,
            "step": 1,
            "action": "west",
            "allowed_actions": ["east", "west"],
            "feature_vector": [0.1] * 160,
            "observation_version": "v2",
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = os.path.join(tmpdir, "trace.jsonl")
        bc_path = os.path.join(tmpdir, "seed_bc.pt")
        with open(trace_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        train_bc_model(rows, bc_path, epochs=3, lr=1e-3, hidden_size=32, observation_version="v2")
        result = run_dagger_schedule(
            base_trace_input=trace_path,
            output_dir=os.path.join(tmpdir, "dagger"),
            student_policy="bc",
            task="explore",
            iterations=1,
            num_episodes=1,
            max_steps=2,
            bc_model_path=bc_path,
            observation_version="v2",
            merge_ratio=0.5,
            merge_policy="uniform_merge",
            epochs=1,
            lr=1e-3,
            hidden_size=32,
        )
        assert len(result["reports"]) == 1
        assert os.path.exists(result["final_bc_model"])
