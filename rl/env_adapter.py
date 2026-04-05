from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import nle.env

from nle_agent.agent_http import _build_action_map
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
        self.action_map = _build_action_map()
        self.registry = build_skill_registry()
        self.scheduler = build_scheduler(config.options.scheduler, config.options.scheduler_model_path)
        self.env = nle.env.NLE()
        self.memory = MemoryTracker()
        self.obs = None
        self.info = None
        self.ctx = None

    def _encode_state(self, obs: dict) -> dict:
        state = self.encoder.encode_full(obs)
        px, py = state["position"]
        tile_char = chr(int(obs["chars"][py, px])) if px >= 0 and py >= 0 else " "
        state["standing_on_down_stairs"] = tile_char == ">"
        state["standing_on_up_stairs"] = tile_char == "<"
        return state

    def reset(self, seed: int | None = None) -> dict:
        self.obs, self.info = self.env.reset(seed=seed if seed is not None else self.config.env.seed)
        self.memory = MemoryTracker()
        self.memory.update(self.obs)
        self.memory.detect_rooms()
        state = self._encode_state(self.obs)
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
        active_skill_before = self.ctx.active_skill
        raw_obs_before = self.obs
        state_before = self._encode_state(self.obs)
        mem_before = snapshot_memory(self.memory)
        recent_state_hashes_before = list(self.ctx.recent_state_hashes)
        recent_positions_before = list(self.ctx.recent_positions)
        prev_action_before = self.ctx.prev_action
        obs_after, reward, terminated, truncated, info = self.env.step(action_idx)
        state_after = self._encode_state(obs_after)
        self.memory.update(obs_after)
        self.memory.detect_rooms()
        mem_after = snapshot_memory(self.memory)
        next_hash = observation_hash(obs_after)
        repeated_state = next_hash in recent_state_hashes_before
        revisited_recent_tile = state_after["position"] in recent_positions_before
        repeated_action = action_name == prev_action_before
        self.ctx.steps_in_skill += 1
        self.ctx.prev_action = action_name
        self.ctx.recent_state_hashes.append(next_hash)
        self.ctx.recent_positions.append(state_after["position"])
        self.obs = obs_after
        self.info = info
        timestep = self._build_timestep(state_after)
        timestep["transition"] = {
            "state_before": state_before,
            "state_after": state_after,
            "obs_before": raw_obs_before,
            "obs_after": obs_after,
            "memory_before": mem_before,
            "memory_after": mem_after,
            "env_reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "action_name": action_name,
            "active_skill_before": active_skill_before,
            "repeated_state": repeated_state,
            "revisited_recent_tile": revisited_recent_tile,
            "repeated_action": repeated_action,
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
