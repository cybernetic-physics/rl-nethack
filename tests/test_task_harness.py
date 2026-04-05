import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.task_harness import _candidate_actions, evaluate_task_policy, run_task_episode
from src.task_rewards import compute_task_rewards


class DummyMemory:
    def __init__(self, total_explored, rooms):
        self.total_explored = total_explored
        self.rooms = rooms


def _make_obs(player_pos=(10, 10), hp=14, hp_max=14, depth=1, gold=0, score=0, xp=1, message=""):
    chars = np.full((21, 79), ord(' '), dtype=np.uint8)
    px, py = player_pos
    chars[py, px] = ord('@')
    chars[py, px + 1] = ord('.')
    chars[py, px - 1] = ord('.')
    chars[py - 1, px] = ord('.')
    chars[py + 1, px] = ord('.')

    blstats = np.zeros(27, dtype=np.int32)
    blstats[0] = px
    blstats[1] = py
    blstats[10] = hp
    blstats[11] = hp_max
    blstats[12] = depth
    blstats[13] = gold
    blstats[18] = xp
    blstats[20] = 1
    blstats[9] = score

    msg = np.zeros(256, dtype=np.uint8)
    encoded = message.encode("ascii", errors="replace")[:255]
    msg[:len(encoded)] = list(encoded)
    return {"chars": chars, "blstats": blstats, "message": msg}


def _make_state(position=(10, 10), message="", adjacent=None, visible_items=None):
    return {
        "position": position,
        "message": message,
        "adjacent": adjacent or {"north": "floor", "south": "floor", "east": "floor", "west": "floor"},
        "visible_items": visible_items or [],
    }


def test_compute_task_rewards_explore_values_new_tiles():
    obs_before = _make_obs()
    obs_after = _make_obs()
    state_before = _make_state()
    state_after = _make_state()
    memory_before = DummyMemory(total_explored=10, rooms=[{(1, 1)}])
    memory_after = DummyMemory(total_explored=16, rooms=[{(1, 1)}, {(2, 2)}])

    result = compute_task_rewards(
        task="explore",
        obs_before=obs_before,
        obs_after=obs_after,
        state_before=state_before,
        state_after=state_after,
        memory_before=memory_before,
        memory_after=memory_after,
        action_name="north",
        reward=0.0,
        terminated=False,
        truncated=False,
    )

    assert result.total > 0
    assert result.components["new_tiles"] == 6.0
    assert result.components["new_rooms"] == 5.0


def test_candidate_actions_add_pickup_and_stairs():
    obs = _make_obs()
    px, py = 10, 10
    obs["chars"][py, px] = ord(">")
    state = _make_state(
        position=(px, py),
        message="You see here 3 gold pieces.",
        visible_items=[{"type": "gold", "pos": (px, py)}],
    )
    actions = _candidate_actions("descend", obs, state)
    assert "pickup" in actions
    assert "down" in actions


def test_run_task_episode_wall_avoidance_smoke():
    result = run_task_episode(seed=42, task="explore", max_steps=3, policy="wall_avoidance")
    assert result["steps"] > 0
    assert result["policy"] == "wall_avoidance"
    assert "action_counts" in result
    assert "component_sums" in result


def test_run_task_episode_task_greedy_smoke():
    result = run_task_episode(seed=42, task="explore", max_steps=1, policy="task_greedy")
    assert result["steps"] == 1
    assert result["policy"] == "task_greedy"
    assert len(result["trajectory"]) == 1


def test_evaluate_task_policy_summary():
    result = evaluate_task_policy(task="explore", seeds=[42], max_steps=2, policy="wall_avoidance")
    assert result["summary"]["episodes"] == 1
    assert "avg_task_reward" in result["summary"]
    assert "action_counts" in result["summary"]
