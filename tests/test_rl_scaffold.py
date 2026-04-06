import os
import sys
import json
import tempfile
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rl.config import RLConfig
from rl.feature_encoder import observation_dim, encode_observation
from rl.reward_model import reward_feature_dim
from rl.scheduler_model import scheduler_feature_dim
from rl.sf_env import NethackSkillEnv
from rl.options import build_skill_registry
from rl.scheduler import SchedulerContext, build_scheduler
from rl.trainer import APPOTrainerScaffold
from rl.train_bc import load_trace_rows, train_bc_model
from rl.bc_model import load_bc_model
from rl.debug_tools import check_policy_determinism, compare_actions_on_teacher_states
from rl.trace_eval import evaluate_trace_policy, trace_disagreement_report
from rl.evaluate import _load_checkpoint_payload
from rl.traces import shard_trace_file, generate_dagger_traces, generate_multi_turn_traces, verify_trace_file
from rl.checkpoint_tools import TraceCheckpointMonitor, write_trace_best_alias, rank_appo_checkpoints_by_trace
import rl.checkpoint_tools as checkpoint_tools
from pathlib import Path
import torch
from rl.timestep import build_policy_timestep
from rl.io_utils import experiment_lock
from rl.teacher_reg import patch_sample_factory_teacher_reg, _parse_teacher_action_boosts
from rl.env_adapter import SkillEnvAdapter, EpisodeContext
from rl.dagger import build_merged_trace_rows, run_dagger_schedule
from rl.train_behavior_reg import train_behavior_regularized_policy
from rl.train_appo import build_config as build_appo_config
from rl.train_world_model import train_world_model
from rl.world_model_dataset import build_world_model_examples, examples_to_arrays
from rl.world_model_eval import evaluate_world_model
from rl.world_model_features import (
    transform_trace_with_world_model,
    world_model_augmented_dim,
)


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
    config.appo.teacher_loss_coef = 0.25
    config.appo.teacher_loss_type = "ce"
    config.appo.teacher_action_boosts = "west=2.0,south=1.5"
    config.appo.teacher_loss_final_coef = 0.002
    config.appo.teacher_loss_warmup_env_steps = 1024
    config.appo.teacher_loss_decay_env_steps = 4096
    config.appo.param_anchor_coef = 0.005
    config.appo.learning_rate = 1e-4
    config.appo.entropy_coeff = 0.0
    config.appo.ppo_clip_ratio = 0.05
    config.model.hidden_size = 256
    config.model.normalize_input = False
    config.model.nonlinearity = "relu"
    trainer = APPOTrainerScaffold(config)
    argv = trainer.build_sf_argv()
    assert "--teacher_loss_coef=0.25" in argv
    assert "--teacher_loss_type=ce" in argv
    assert "--teacher_bc_path=/tmp/teacher.pt" in argv
    assert "--teacher_action_boosts=west=2.0,south=1.5" in argv
    assert "--teacher_loss_final_coef=0.002" in argv
    assert "--teacher_loss_warmup_env_steps=1024" in argv
    assert "--teacher_loss_decay_env_steps=4096" in argv
    assert "--param_anchor_coef=0.005" in argv
    assert "--learning_rate=0.0001" in argv
    assert "--exploration_loss_coeff=0.0" in argv
    assert "--ppo_clip_ratio=0.05" in argv
    assert "--normalize_input=False" in argv
    assert "--nonlinearity=relu" in argv
    idx = argv.index("--encoder_mlp_layers")
    assert argv[idx + 1:idx + 3] == ["256", "256"]


def test_build_appo_config_infers_hidden_size_from_bc_checkpoint():
    rows = [
        {"feature_vector": [0.0] * 160, "action": "east", "allowed_actions": ["east", "west"]},
        {"feature_vector": [0.1] * 160, "action": "west", "allowed_actions": ["east", "west"]},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "bc.pt")
        train_bc_model(rows, out, epochs=1, lr=1e-3, hidden_size=64, observation_version="v2")
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
        assert config.model.hidden_size == 64
        assert config.model.normalize_input is False
        assert config.model.nonlinearity == "relu"


def test_parse_teacher_action_boosts():
    boosts = _parse_teacher_action_boosts("west=2.0,south=1.5")
    assert boosts == {3: 2.0, 1: 1.5}


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
    trainer = APPOTrainerScaffold(config)
    argv = trainer.build_sf_argv()
    assert "--trace_eval_input=data/trace.jsonl" in argv
    assert "--trace_eval_interval_env_steps=2048" in argv
    assert "--trace_eval_top_k=7" in argv


def test_teacher_patch_is_idempotent():
    patch_sample_factory_teacher_reg()
    patch_sample_factory_teacher_reg()


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
        )
        assert train_result["num_examples"] > 0
        eval_result = evaluate_world_model(model_path, trace_path, horizon=2, observation_version="v4")
        assert eval_result["num_examples"] == train_result["num_examples"]
        assert eval_result["feature_mse"] >= 0.0
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
        meta = train_bc_model(rows, out, epochs=1, lr=1e-3, hidden_size=64, observation_version="v2")
        assert meta["observation_version"] == "v2"
        policy = load_bc_model(out)
        action = policy.act(rows[0]["feature_vector"], allowed_actions=["east", "west"])
        assert action in {"east", "west"}


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
