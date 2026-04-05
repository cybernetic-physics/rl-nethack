import importlib.util
from pathlib import Path

import numpy as np

from src.memory_tracker import MemoryTracker, FLOOR, UNSEEN


MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "generate_training_data.py"
SPEC = importlib.util.spec_from_file_location("generate_training_data_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _make_obs(message=""):
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    chars[10, 10] = ord("@")
    for y, x in [(9, 10), (11, 10), (10, 9), (10, 11)]:
        chars[y, x] = ord(".")
    blstats = np.zeros(30, dtype=np.int64)
    blstats[0] = 10
    blstats[1] = 10
    blstats[10] = 14
    blstats[11] = 14
    blstats[12] = 1
    blstats[13] = 0
    blstats[20] = 1
    msg = np.zeros(256, dtype=np.uint8)
    encoded = message.encode("ascii")
    msg[: len(encoded)] = list(encoded)
    return {"chars": chars, "blstats": blstats, "message": msg}


def test_choose_fallback_move_prefers_frontier_not_fixed_north():
    memory = MemoryTracker()
    memory.explored.fill(FLOOR)
    memory.explored[10, 11] = FLOOR
    memory.explored[10, 12] = UNSEEN
    memory.visit_counts[9, 10] = 5
    memory.visit_counts[10, 11] = 0
    state = {
        "position": (10, 10),
        "adjacent": {"north": "floor", "south": "floor", "east": "floor", "west": "floor"},
    }

    action = MODULE.choose_fallback_move(state, memory)

    assert action == "east"


def test_choose_fallback_move_avoids_immediate_backtrack():
    memory = MemoryTracker()
    memory.explored.fill(FLOOR)
    memory.position_history = [(10, 11), (10, 10)]
    state = {
        "position": (10, 10),
        "adjacent": {"north": "floor", "south": "floor", "east": "floor", "west": "floor"},
    }

    action = MODULE.choose_fallback_move(state, memory)

    assert action != "east"


def test_sanitize_action_redirects_invalid_wait_to_frontier_move():
    memory = MemoryTracker()
    memory.explored.fill(FLOOR)
    memory.explored[10, 11] = FLOOR
    memory.explored[10, 12] = UNSEEN
    obs = _make_obs()
    encoder = MODULE.StateEncoder()

    action = MODULE.sanitize_action("wait", obs, history=[], encoder=encoder, memory=memory)

    assert action == "east"
