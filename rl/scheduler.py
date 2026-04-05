from __future__ import annotations

from dataclasses import dataclass

from rl.options import SkillOption


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


def build_scheduler(name: str) -> SkillScheduler:
    if name == "rule_based":
        return RuleBasedScheduler()
    raise ValueError(f"Unknown scheduler: {name}")

