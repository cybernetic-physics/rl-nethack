import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rl.config import RLConfig
from rl.feature_encoder import observation_dim
from rl.reward_model import reward_feature_dim
from rl.scheduler_model import scheduler_feature_dim
from rl.sf_env import NethackSkillEnv
from rl.options import build_skill_registry
from rl.scheduler import SchedulerContext, build_scheduler
from rl.trainer import APPOTrainerScaffold


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
    assert observation_dim() == 106
    assert reward_feature_dim() == 37
    assert scheduler_feature_dim() > 0


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
