from __future__ import annotations

from dataclasses import dataclass

from rl.options import SkillOption
from rl.scheduler_model import encode_scheduler_features, load_scheduler_model


@dataclass
class SchedulerContext:
    state: dict
    memory: object
    active_skill: str | None
    steps_in_skill: int
    available_skills: list[str]


class SkillScheduler:
    def select_skill(self, ctx: SchedulerContext) -> str:
        raise NotImplementedError


class RuleBasedScheduler(SkillScheduler):
    """Simple precedence-based scheduler, easy to replace later."""

    def select_skill(self, ctx: SchedulerContext) -> str:
        state = ctx.state
        if state.get("hp", 0) <= max(1, state.get("hp_max", 1) // 2) and "survive" in ctx.available_skills:
            return "survive"
        if state.get("visible_monsters") and "combat" in ctx.available_skills:
            return "combat"
        if "see here" in state.get("message", "").lower() and "resource" in ctx.available_skills:
            return "resource"
        if "descend" in ctx.available_skills and "stairs_down" in set(state.get("adjacent", {}).values()):
            return "descend"
        if "explore" in ctx.available_skills:
            return "explore"
        return ctx.available_skills[0]


class LearnedScheduler(SkillScheduler):
    def __init__(self, model_path: str):
        self.model = load_scheduler_model(model_path)

    def select_skill(self, ctx: SchedulerContext) -> str:
        features = encode_scheduler_features(
            state=ctx.state,
            memory=ctx.memory,
            active_skill=ctx.active_skill,
            steps_in_skill=ctx.steps_in_skill,
            available_skills=ctx.available_skills,
        )
        return self.model.select_skill(features, ctx.available_skills)


def build_scheduler(name: str, model_path: str | None = None) -> SkillScheduler:
    if name == "rule_based":
        return RuleBasedScheduler()
    if name == "learned":
        if not model_path:
            raise ValueError("learned scheduler requires a model path")
        return LearnedScheduler(model_path)
    raise ValueError(f"Unknown scheduler: {name}")
