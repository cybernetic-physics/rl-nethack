import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rl.config import RLConfig
from rl.feature_encoder import observation_dim, encode_observation
from rl.reward_model import reward_feature_dim
from rl.scheduler_model import scheduler_feature_dim
from rl.sf_env import NethackSkillEnv
from rl.options import build_skill_registry
from rl.scheduler import SchedulerContext, build_scheduler
from rl.trainer import APPOTrainerScaffold
from rl.train_bc import train_bc_model
from rl.bc_model import load_bc_model


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


def test_feature_dims_are_stable():
    assert observation_dim("v1") == 106
    assert observation_dim("v2") == 160
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
