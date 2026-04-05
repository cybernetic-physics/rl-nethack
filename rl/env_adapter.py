from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import nle.env

from rl.options import build_skill_registry
from rl.scheduler import SchedulerContext, build_scheduler
from src.memory_tracker import MemoryTracker
from src.state_encoder import StateEncoder
from src.task_rewards import observation_hash, snapshot_memory


@dataclass
class EpisodeContext:
    active_skill: str
    steps_in_skill: int = 0
    recent_state_hashes: deque = field(default_factory=lambda: deque(maxlen=8))
    recent_positions: deque = field(default_factory=lambda: deque(maxlen=8))
    prev_action: str | None = None


class SkillEnvAdapter:
    """Thin env wrapper that tracks memory and option state.

    This is deliberately lightweight so the policy/trainer can change without
    forcing env logic rewrites.
    """

    def __init__(self, config):
        self.config = config
        self.encoder = StateEncoder()
        self.registry = build_skill_registry()
        self.scheduler = build_scheduler(config.options.scheduler)
        self.env = nle.env.NLE()
        self.memory = MemoryTracker()
        self.obs = None
        self.info = None
        self.ctx = None

    def reset(self, seed: int | None = None) -> dict:
        self.obs, self.info = self.env.reset(seed=seed if seed is not None else self.config.env.seed)
        self.memory = MemoryTracker()
        self.memory.update(self.obs)
        self.memory.detect_rooms()
        state = self.encoder.encode_full(self.obs)
        self.ctx = EpisodeContext(active_skill=self.config.env.active_skill_bootstrap)
        self.ctx.recent_state_hashes.append(observation_hash(self.obs))
        self.ctx.recent_positions.append(state["position"])
        return self._build_timestep(state)

    def _available_skills(self, state: dict) -> list[str]:
        return [
            name for name in self.config.options.enabled_skills
            if name in self.registry and self.registry[name].can_start(state, self.memory)
        ] or [self.config.env.active_skill_bootstrap]

    def maybe_switch_skill(self, state: dict) -> str:
        available = self._available_skills(state)
        current = self.registry[self.ctx.active_skill]
        if current.should_stop(state, self.memory, self.ctx.steps_in_skill):
            next_skill = self.scheduler.select_skill(
                SchedulerContext(
                    state=state,
                    memory=self.memory,
                    active_skill=self.ctx.active_skill,
                    steps_in_skill=self.ctx.steps_in_skill,
                    available_skills=available,
                )
            )
            self.ctx.active_skill = next_skill
            self.ctx.steps_in_skill = 0
        return self.ctx.active_skill

    def allowed_actions(self, state: dict) -> list[str]:
        skill = self.registry[self.ctx.active_skill]
        return skill.allowed_actions(state, self.memory)

    def step(self, action_idx: int, action_name: str) -> tuple[dict, float, bool, bool, dict]:
        state_before = self.encoder.encode_full(self.obs)
        mem_before = snapshot_memory(self.memory)
        obs_after, reward, terminated, truncated, info = self.env.step(action_idx)
        state_after = self.encoder.encode_full(obs_after)
        self.memory.update(obs_after)
        self.memory.detect_rooms()
        mem_after = snapshot_memory(self.memory)
        self.ctx.steps_in_skill += 1
        self.ctx.prev_action = action_name
        self.ctx.recent_state_hashes.append(observation_hash(obs_after))
        self.ctx.recent_positions.append(state_after["position"])
        self.obs = obs_after
        self.info = info
        timestep = self._build_timestep(state_after)
        timestep["transition"] = {
            "state_before": state_before,
            "state_after": state_after,
            "memory_before": mem_before,
            "memory_after": mem_after,
            "env_reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "action_name": action_name,
        }
        return timestep, float(reward), bool(terminated), bool(truncated), info

    def _build_timestep(self, state: dict) -> dict:
        active_skill = self.maybe_switch_skill(state)
        return {
            "state": state,
            "active_skill": active_skill,
            "allowed_actions": self.allowed_actions(state),
            "steps_in_skill": self.ctx.steps_in_skill,
            "memory_total_explored": self.memory.total_explored,
            "rooms_discovered": len(self.memory.rooms),
        }

    def close(self):
        self.env.close()

