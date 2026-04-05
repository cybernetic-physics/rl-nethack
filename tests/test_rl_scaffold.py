import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rl.config import RLConfig
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
