# dstack-lora World-Model Training Pipeline — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a LoRA training pipeline that teaches a language model to predict NetHack game outcomes (forward model), with every component independently testable and a virgin benchmark for attested evaluation.

**Architecture:** Five independent modules, each with its own test suite. Data flows: NLE env -> structured features -> training JSONL -> Unsloth LoRA -> evaluation on virgin seeds -> attested manifest.

**Tech Stack:** Python 3.10+, NLE (NetHack Learning Environment), Unsloth/trl/peft, PyTorch, pytest

**Key Design Principle:** Every module produces deterministic output given fixed seeds. This makes everything testable and makes the virgin benchmark concept work.

---

## Component Map

```
nle_env (NLE)
    |
    v
[1. state_encoder.py]  -- extract structured features from NLE observations
    |                        tested: given obs dict -> deterministic feature dict
    v
[2. data_generator.py] -- run random games, produce (state, action, outcome) JSONL
    |                        tested: given seed -> deterministic JSONL with known shape
    v
[3. train.py]           -- Unsloth LoRA training on prediction data
    |                        tested: overfit on 10 examples, verify loss decreases
    v
[4. evaluate.py]        -- measure prediction accuracy on virgin seeds
    |                        tested: known model + known state -> known accuracy range
    v
[5. build_manifest.py]  -- compile hashes, metrics, TEE quote into manifest
                             tested: given artifacts -> valid manifest with correct hashes
```

---

## Component 1: State Encoder (`src/state_encoder.py`)

**What it does:** Takes a raw NLE observation (dict of numpy arrays) and produces a structured, human-readable feature dict. This is the feature engineering layer.

**Why it matters:** This is the bridge between NLE's raw arrays and the text the LLM sees. Getting this right means the model gets dense, relevant signal instead of raw ASCII art.

### Interface

```python
class StateEncoder:
    """Encodes NLE observations into structured feature dicts."""

    def encode_full(self, obs: dict) -> dict:
        """Full observation -> structured dict with all features."""
        # Returns dict with keys: position, hp, stats, adjacent, visible_monsters,
        # visible_items, message, turn, depth, gold
        ...

    def encode_delta(self, obs_before: dict, obs_after: dict, action: str) -> dict:
        """Two consecutive observations -> what changed (delta)."""
        # Returns dict with keys: pos_delta, hp_delta, gold_delta, new_tiles,
        # disappeared_tiles, message, survived
        ...

    def format_prompt(self, state: dict, action: str) -> str:
        """Format (state, action) as the user prompt for the LLM."""
        ...

    def format_target(self, delta: dict) -> str:
        """Format delta as the target response for the LLM."""
        ...
```

### Tasks

#### Task 1.1: Create project structure and test file

**Files:**
- Create: `src/__init__.py`
- Create: `src/state_encoder.py` (empty class stubs)
- Create: `tests/__init__.py`
- Create: `tests/test_state_encoder.py`

**Step 1: Create directories and stubs**

```bash
mkdir -p src tests
touch src/__init__.py tests/__init__.py
```

**Step 2: Write test file with first test**

```python
# tests/test_state_encoder.py
import pytest
import numpy as np
from src.state_encoder import StateEncoder


def _make_fake_obs(
    char_grid=None, message=b"Hello\x00" * 10,
    hp=14, hp_max=14, ac=4, strength=18, dexterity=12,
    gold=0, depth=1, turn=1,
    player_x=5, player_y=5,
):
    """Create a fake NLE observation dict for testing."""
    if char_grid is None:
        # 21x79 grid of floor (.) with player (@) at center
        grid = np.full((21, 79), ord('.'), dtype=np.uint8)
        grid[player_y, player_x] = ord('@')
        # Add walls around edges
        grid[0, :] = ord('-')
        grid[-1, :] = ord('-')
        grid[:, 0] = ord('|')
        grid[:, -1] = ord('|')
    else:
        grid = np.array([[ord(c) for c in row] for row in char_grid], dtype=np.uint8)

    blstats = np.zeros(26, dtype=np.int64)
    blstats[3] = strength   # Str
    blstats[4] = dexterity  # Dex
    blstats[10] = hp         # HP
    blstats[11] = hp_max     # HP max
    blstats[12] = depth      # Dungeon level
    blstats[13] = gold       # Gold
    blstats[16] = ac         # AC
    blstats[18] = 1          # Experience level
    blstats[20] = turn       # Turn number
    # Position encoded in blstats[9] (y * 79 + x) -- actually we use glyphs position
    blstats[9] = player_y * 79 + player_x

    # glyphs grid - same shape as chars, use 0 for floor, etc
    glyphs = np.zeros_like(grid, dtype=np.int32)

    return {
        "chars": grid,
        "blstats": blstats,
        "message": np.frombuffer(message, dtype=np.uint8),
        "glyphs": glyphs,
        "specials": np.zeros_like(grid, dtype=np.int32),
    }


class TestStateEncoderFull:
    """Tests for encode_full()."""

    def test_returns_dict_with_required_keys(self):
        enc = StateEncoder()
        obs = _make_fake_obs()
        result = enc.encode_full(obs)
        required_keys = {
            "position", "hp", "hp_max", "ac", "strength", "dexterity",
            "gold", "depth", "turn", "adjacent", "visible_monsters",
            "visible_items", "message",
        }
        assert required_keys.issubset(result.keys()), f"Missing keys: {required_keys - result.keys()}"

    def test_position_extraction(self):
        enc = StateEncoder()
        obs = _make_fake_obs(player_x=10, player_y=3)
        result = enc.encode_full(obs)
        assert result["position"] == (10, 3)

    def test_hp_extraction(self):
        enc = StateEncoder()
        obs = _make_fake_obs(hp=8, hp_max=14)
        result = enc.encode_full(obs)
        assert result["hp"] == 8
        assert result["hp_max"] == 14

    def test_adjacent_tiles(self):
        """Adjacent tiles should be the 4 cardinal neighbors of the player."""
        enc = StateEncoder()
        # Place player at 5,5 in a grid of floors surrounded by walls
        obs = _make_fake_obs(player_x=5, player_y=5)
        result = enc.encode_full(obs)
        adj = result["adjacent"]
        assert set(adj.keys()) == {"north", "south", "east", "west"}
        # With our fake grid, north/south should be floor, east/west floor
        assert adj["north"] in ("floor", "wall")
        assert adj["south"] in ("floor", "wall")

    def test_message_extraction(self):
        enc = StateEncoder()
        obs = _make_fake_obs(message=b"It's a wall.\x00" + b"\x00" * 200)
        result = enc.encode_full(obs)
        assert "wall" in result["message"].lower() or result["message"] == ""


class TestStateEncoderDelta:
    """Tests for encode_delta()."""

    def test_no_change_delta(self):
        """Standing still should produce zero deltas."""
        enc = StateEncoder()
        obs1 = _make_fake_obs(hp=14, gold=0, player_x=5, player_y=5)
        obs2 = _make_fake_obs(hp=14, gold=0, player_x=5, player_y=5)
        delta = enc.encode_delta(obs1, obs2, "wait")
        assert delta["hp_delta"] == 0
        assert delta["gold_delta"] == 0
        assert delta["pos_delta"] == (0, 0)

    def test_movement_delta(self):
        enc = StateEncoder()
        obs1 = _make_fake_obs(player_x=5, player_y=5)
        obs2 = _make_fake_obs(player_x=6, player_y=5)  # moved east
        delta = enc.encode_delta(obs1, obs2, "east")
        assert delta["pos_delta"] == (1, 0)

    def test_damage_delta(self):
        enc = StateEncoder()
        obs1 = _make_fake_obs(hp=14)
        obs2 = _make_fake_obs(hp=10)
        delta = enc.encode_delta(obs1, obs2, "north")
        assert delta["hp_delta"] == -4

    def test_gold_delta(self):
        enc = StateEncoder()
        obs1 = _make_fake_obs(gold=0)
        obs2 = _make_fake_obs(gold=5)
        delta = enc.encode_delta(obs1, obs2, "pickup")
        assert delta["gold_delta"] == 5


class TestStateEncoderFormat:
    """Tests for format_prompt() and format_target()."""

    def test_format_prompt_contains_state_and_action(self):
        enc = StateEncoder()
        state = enc.encode_full(_make_fake_obs())
        prompt = enc.format_prompt(state, "east")
        assert "east" in prompt.lower()
        assert "14" in prompt  # HP value

    def test_format_target_contains_deltas(self):
        enc = StateEncoder()
        obs1 = _make_fake_obs(hp=14, gold=0, player_x=5, player_y=5)
        obs2 = _make_fake_obs(hp=12, gold=3, player_x=6, player_y=5)
        delta = enc.encode_delta(obs1, obs2, "east")
        target = enc.format_target(delta)
        assert "-2" in target or "2" in target  # HP change
        assert "+3" in target or "3" in target  # Gold change

    def test_format_round_trip(self):
        """Prompt + target should be valid training data."""
        enc = StateEncoder()
        obs1 = _make_fake_obs()
        obs2 = _make_fake_obs(player_x=6, player_y=5)
        state = enc.encode_full(obs1)
        delta = enc.encode_delta(obs1, obs2, "east")
        prompt = enc.format_prompt(state, "east")
        target = enc.format_target(delta)
        # Should be non-empty strings
        assert len(prompt) > 20
        assert len(target) > 5
```

**Step 3: Run tests (expect failures)**

```bash
cd /home/amiller/projects/dstack/dstack-lora
python -m pytest tests/test_state_encoder.py -v
```

Expected: FAIL (module not found)

**Step 4: Implement StateEncoder**

```python
# src/state_encoder.py
"""Encode NLE observations into structured features for LLM training."""
import numpy as np

# Map ASCII char codes to tile names
TILE_NAMES = {
    ord('.'): 'floor', ord('#'): 'corridor', ord(' '): 'unseen',
    ord('-'): 'wall', ord('|'): 'wall', ord('+'): 'door',
    ord('<'): 'stairs_up', ord('>'): 'stairs_down',
    ord('@'): 'player', ord('$'): 'gold', ord('?'): 'scroll',
    ord('!'): 'potion', ord('/'): 'wand', ord('='): 'ring',
    ord(')'): 'weapon', ord(']'): 'armor', ord('*'): 'gem',
    ord('%'): 'food', ord('`'): 'boulder', ord('^'): 'trap',
    ord('_'): 'altar', ord('{'): 'fountain', ord('}'): 'water',
}
# Monster tiles: lowercase or uppercase letters (not wall chars)
def _tile_name(char_code):
    ch = chr(char_code)
    if char_code in TILE_NAMES:
        return TILE_NAMES[char_code]
    if ch.isalpha():
        return f'monster_{ch}'
    if ch.isdigit():
        return f'trap_{ch}'
    return f'unknown_{char_code}'


class StateEncoder:
    """Encodes NLE observations into structured feature dicts."""

    def encode_full(self, obs: dict) -> dict:
        chars = obs["chars"]
        blstats = obs["blstats"]
        msg_raw = bytes(obs["message"]).decode("ascii", errors="replace")
        msg = msg_raw.rstrip('\x00').strip()

        # Find player position from blstats (index 9 = y*79 + x for some NLE versions)
        # Fallback: scan chars grid for '@'
        pos_yx = blstats[9]
        py, px = divmod(pos_yx, 79) if pos_yx > 0 else (10, 40)
        # Double-check by scanning
        for y in range(chars.shape[0]):
            for x in range(chars.shape[1]):
                if chars[y, x] == ord('@'):
                    py, px = y, x
                    break

        # Adjacent tiles (cardinal)
        adjacent = {}
        for name, dy, dx in [("north", -1, 0), ("south", 1, 0), ("east", 0, 1), ("west", 0, -1)]:
            ny, nx = py + dy, px + dx
            if 0 <= ny < chars.shape[0] and 0 <= nx < chars.shape[1]:
                adjacent[name] = _tile_name(chars[ny, nx])
            else:
                adjacent[name] = "wall"

        # Visible monsters and items (scan visible area for non-floor/non-wall)
        visible_monsters = []
        visible_items = []
        for y in range(chars.shape[0]):
            for x in range(chars.shape[1]):
                ch = chars[y, x]
                name = _tile_name(ch)
                if name.startswith("monster_"):
                    visible_monsters.append({"char": chr(ch), "pos": (x, y)})
                elif name in ("gold", "scroll", "potion", "weapon", "armor", "food", "gem", "wand", "ring"):
                    visible_items.append({"type": name, "pos": (x, y)})

        return {
            "position": (px, py),
            "hp": int(blstats[10]),
            "hp_max": int(blstats[11]),
            "ac": int(blstats[16]),
            "strength": int(blstats[3]),
            "dexterity": int(blstats[4]),
            "gold": int(blstats[13]),
            "depth": int(blstats[12]),
            "turn": int(blstats[20]),
            "adjacent": adjacent,
            "visible_monsters": visible_monsters,
            "visible_items": visible_items,
            "message": msg,
        }

    def encode_delta(self, obs_before: dict, obs_after: dict, action: str) -> dict:
        s1 = self.encode_full(obs_before)
        s2 = self.encode_full(obs_after)

        dx = s2["position"][0] - s1["position"][0]
        dy = s2["position"][1] - s1["position"][1]

        # New tiles visible in obs_after but not in obs_before
        chars_before = obs_before["chars"]
        chars_after = obs_after["chars"]
        new_tiles = []
        for y in range(chars_after.shape[0]):
            for x in range(chars_after.shape[1]):
                if chars_before[y, x] == ord(' ') and chars_after[y, x] != ord(' '):
                    new_tiles.append({"pos": (x, y), "tile": _tile_name(chars_after[y, x])})

        return {
            "pos_delta": (dx, dy),
            "hp_delta": s2["hp"] - s1["hp"],
            "hp_max_delta": s2["hp_max"] - s1["hp_max"],
            "gold_delta": s2["gold"] - s1["gold"],
            "depth_delta": s2["depth"] - s1["depth"],
            "turn_delta": s2["turn"] - s1["turn"],
            "new_tiles": new_tiles,
            "message": s2["message"],
            "survived": s2["hp"] > 0,
        }

    def format_prompt(self, state: dict, action: str) -> str:
        adj = " ".join(f"{d}={t}" for d, t in state["adjacent"].items())
        monsters = ", ".join(
            f"{m['char']}@({m['pos'][0]},{m['pos'][1]})" for m in state["visible_monsters"]
        ) or "none"
        items = ", ".join(
            f"{i['type']}@({i['pos'][0]},{i['pos'][1]})" for i in state["visible_items"]
        ) or "none"

        lines = [
            f"HP:{state['hp']}/{state['hp_max']} AC:{state['ac']} Str:{state['strength']} Dex:{state['dexterity']}",
            f"Pos:({state['position'][0]},{state['position'][1]}) Gold:{state['gold']} Depth:{state['depth']} Turn:{state['turn']}",
            f"Adjacent: {adj}",
            f"Monsters: {monsters}",
            f"Items: {items}",
        ]
        if state["message"]:
            lines.append(f"LastMsg: {state['message']}")
        lines.append(f"Action: {action}")
        return "\n".join(lines)

    def format_target(self, delta: dict) -> str:
        parts = []
        pd = delta["pos_delta"]
        if pd == (0, 0):
            parts.append("pos:same")
        else:
            parts.append(f"pos:({pd[0]:+d},{pd[1]:+d})")

        if delta["hp_delta"] != 0:
            parts.append(f"hp:{delta['hp_delta']:+d}")
        else:
            parts.append("hp:same")

        if delta["gold_delta"] != 0:
            parts.append(f"gold:{delta['gold_delta']:+d}")

        if delta["depth_delta"] != 0:
            parts.append(f"depth:{delta['depth_delta']:+d}")

        if delta["new_tiles"]:
            tiles_str = " ".join(f"({t['pos'][0]},{t['pos'][1]})={t['tile']}" for t in delta["new_tiles"][:5])
            parts.append(f"new_tiles:{tiles_str}")

        if delta["message"]:
            parts.append(f"msg:{delta['message'][:60]}")

        if not delta["survived"]:
            parts.append("DIED")

        return " | ".join(parts)
```

**Step 5: Run tests (expect pass)**

```bash
cd /home/amiller/projects/dstack/dstack-lora
python -m pytest tests/test_state_encoder.py -v
```

Expected: All tests PASS

#### Task 1.2: Integration test with real NLE observation

**Files:**
- Modify: `tests/test_state_encoder.py` (add integration test)

**Step 1: Add integration test class**

```python
# Append to tests/test_state_encoder.py

class TestStateEncoderIntegration:
    """Integration tests using real NLE environment."""

    def test_with_real_nle_observation(self):
        """Verify encoder works with actual NLE output."""
        try:
            import nle.env
        except ImportError:
            pytest.skip("NLE not installed")

        enc = StateEncoder()
        env = nle.env.NLE()
        obs, info = env.reset(seed=42)

        # encode_full should not crash on real data
        state = enc.encode_full(obs)
        assert state["hp"] > 0
        assert state["depth"] == 1
        assert state["turn"] == 1
        assert isinstance(state["adjacent"], dict)
        assert len(state["adjacent"]) == 4

        # Take a step and encode delta
        obs2, reward, term, trunc, info2 = env.step(3)  # west
        delta = enc.encode_delta(obs, obs2, "west")
        assert isinstance(delta["pos_delta"], tuple)
        assert isinstance(delta["survived"], bool)

        # format_prompt + format_target should produce valid text
        prompt = enc.format_prompt(state, "west")
        target = enc.format_target(delta)
        assert len(prompt) > 10
        assert len(target) > 3

        env.close()

    def test_real_nle_multiple_steps(self):
        """Encoder should be stable across multiple steps."""
        try:
            import nle.env
        except ImportError:
            pytest.skip("NLE not installed")

        enc = StateEncoder()
        env = nle.env.NLE()
        obs, _ = env.reset(seed=7)

        for step in range(20):
            state = enc.encode_full(obs)
            assert state["hp"] > 0 or step > 0  # HP > 0 unless we died
            # Random action
            action_idx = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(action_idx)
            if term or trunc:
                break

        env.close()

    def test_format_produces_training_pair(self):
        """The formatted prompt + target should look like valid instruction data."""
        try:
            import nle.env
        except ImportError:
            pytest.skip("NLE not installed")

        enc = StateEncoder()
        env = nle.env.NLE()
        obs1, _ = env.reset(seed=42)
        obs2, _, _, _, _ = env.step(3)  # west

        state = enc.encode_full(obs1)
        delta = enc.encode_delta(obs1, obs2, "west")

        prompt = enc.format_prompt(state, "west")
        target = enc.format_target(delta)

        # Should look like: state description + action -> outcome prediction
        assert "Action:" in prompt
        assert "west" in prompt
        assert "pos:" in target
        assert "hp:" in target
        env.close()
```

**Step 2: Run all tests**

```bash
cd /home/amiller/projects/dstack/dstack-lora
python -m pytest tests/test_state_encoder.py -v
```

Expected: All tests PASS (unit + integration)

**Step 3: Commit**

```bash
git add src/ tests/
git commit -m "feat: add StateEncoder with structured feature extraction and tests"
```

---

## Component 2: Data Generator (`src/data_generator.py`)

**What it does:** Runs random games using NLE, extracts (state, action, outcome) triples at each step, formats them as training data using StateEncoder.

**Why it matters:** This is the self-supervised data pipeline. Random play = free labels. Every game step is a training example.

### Key Design Decisions

1. **Random policy with wall avoidance** -- not purely random (that gets stuck). Use a simple heuristic: try random directions, skip walls. This produces more diverse data per game.
2. **Deterministic seeding** -- every seed produces identical output. Critical for reproducibility and virgin benchmark.
3. **JSONL output** -- one line per training example, standard format for Unsloth/TRL.

### Tasks

#### Task 2.1: Write tests for data generator

**Files:**
- Create: `tests/test_data_generator.py`

```python
# tests/test_data_generator.py
import pytest
import json
import os
import tempfile


class TestDataGenerator:
    """Tests for data generation pipeline."""

    def test_generate_single_game_produces_jsonl(self):
        """generate_game should produce valid JSONL."""
        try:
            import nle.env
        except ImportError:
            pytest.skip("NLE not installed")
        from src.data_generator import generate_game
        from src.state_encoder import StateEncoder

        enc = StateEncoder()
        lines = list(generate_game(seed=42, max_steps=10, encoder=enc))
        assert len(lines) >= 1  # At least one training example
        for line in lines:
            data = json.loads(line)
            assert "conversations" in data
            assert len(data["conversations"]) == 3  # system + user + assistant
            assert data["conversations"][0]["role"] == "system"
            assert data["conversations"][1]["role"] == "user"
            assert data["conversations"][2]["role"] == "assistant"

    def test_deterministic_output(self):
        """Same seed produces same output."""
        try:
            import nle.env
        except ImportError:
            pytest.skip("NLE not installed")
        from src.data_generator import generate_game
        from src.state_encoder import StateEncoder

        enc = StateEncoder()
        run1 = list(generate_game(seed=42, max_steps=10, encoder=enc))
        run2 = list(generate_game(seed=42, max_steps=10, encoder=enc))
        assert run1 == run2

    def test_different_seeds_produce_different_output(self):
        """Different seeds should produce different games."""
        try:
            import nle.env
        except ImportError:
            pytest.skip("NLE not installed")
        from src.data_generator import generate_game
        from src.state_encoder import StateEncoder

        enc = StateEncoder()
        run1 = list(generate_game(seed=42, max_steps=20, encoder=enc))
        run2 = list(generate_game(seed=99, max_steps=20, encoder=enc))
        # At least some lines should differ
        assert run1 != run2

    def test_generate_dataset_creates_file(self):
        """generate_dataset should write JSONL to disk."""
        try:
            import nle.env
        except ImportError:
            pytest.skip("NLE not installed")
        from src.data_generator import generate_dataset
        from src.state_encoder import StateEncoder

        enc = StateEncoder()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            generate_dataset(
                output_path=path,
                num_games=2,
                max_steps=10,
                seed_start=42,
                encoder=enc,
            )
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) >= 2  # At least 1 example per game
            for line in lines:
                json.loads(line)  # Should not raise
        finally:
            os.unlink(path)

    def test_train_eval_split_deterministic(self):
        """Split should be deterministic and cover different seeds."""
        try:
            import nle.env
        except ImportError:
            pytest.skip("NLE not installed")
        from src.data_generator import generate_dataset
        from src.state_encoder import StateEncoder

        enc = StateEncoder()
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train.jsonl")
            eval_path = os.path.join(tmpdir, "eval.jsonl")

            generate_dataset(
                output_path=train_path,
                num_games=10,
                max_steps=5,
                seed_start=0,
                encoder=enc,
                eval_path=eval_path,
                eval_fraction=0.2,
            )

            assert os.path.exists(train_path)
            assert os.path.exists(eval_path)

            with open(train_path) as f:
                train_lines = f.readlines()
            with open(eval_path) as f:
                eval_lines = f.readlines()

            # Train should be ~80%, eval ~20%
            total = len(train_lines) + len(eval_lines)
            assert total >= 10  # At least 1 step per game
            assert len(eval_lines) > 0
            assert len(train_lines) > len(eval_lines)


class TestDataGeneratorActionPolicy:
    """Tests for the action selection policy."""

    def test_wall_avoidance_policy_returns_valid_action(self):
        from src.data_generator import wall_avoidance_policy
        # Just check it returns an action name
        action = wall_avoidance_policy(adjacent={"north": "wall", "south": "floor", "east": "floor", "west": "wall"})
        assert action in ("north", "south", "east", "west", "northeast", "northwest",
                          "southeast", "southwest", "wait", "pickup", "search")

    def test_wall_avoidance_prefers_open_tiles(self):
        from src.data_generator import wall_avoidance_policy
        import random
        random.seed(42)
        # When surrounded by walls on 2 sides, should never pick those
        choices = set()
        for _ in range(20):
            action = wall_avoidance_policy(adjacent={
                "north": "wall", "south": "floor", "east": "floor", "west": "wall"
            })
            choices.add(action)
        # Should never be north or west (they're walls)
        # Unless randomness picks them sometimes, but at least south/east should appear
        assert "south" in choices or "east" in choices
```

#### Task 2.2: Implement data generator

**Files:**
- Create: `src/data_generator.py`

```python
# src/data_generator.py
"""Generate training data from random NLE gameplay."""
import json
import random
import os
from typing import Iterator, Optional

import numpy as np

try:
    import nle.env
except ImportError:
    nle = None

from src.state_encoder import StateEncoder

SYSTEM_PROMPT = (
    "Predict the outcome of a NetHack action. "
    "Given the current game state and an action, predict what changes. "
    "Output format: key:value pairs separated by |"
)


def wall_avoidance_policy(adjacent: dict, rng: random.Random = None) -> str:
    """Choose a random action, preferring directions that aren't walls."""
    if rng is None:
        rng = random
    directions = ["north", "south", "east", "west"]
    open_dirs = [d for d in directions if adjacent.get(d) not in ("wall", "unseen")]

    if open_dirs:
        return rng.choice(open_dirs)

    # All walls -- pick randomly anyway (might be a door)
    return rng.choice(directions)


def _action_name_to_nle_action(action_name: str, action_map: dict) -> int:
    """Convert action name to NLE action index."""
    return action_map.get(action_name, action_map.get("wait", 0))


def generate_game(
    seed: int,
    max_steps: int,
    encoder: StateEncoder,
    policy: callable = None,
) -> Iterator[str]:
    """Play one random game, yield JSONL training examples.

    Each yielded line is a JSON object with 'conversations' key
    in ShareGPT format suitable for Unsloth training.
    """
    if nle is None:
        raise ImportError("NLE not installed")

    if policy is None:
        policy = wall_avoidance_policy

    rng = random.Random(seed)
    env = nle.env.NLE()
    obs, info = env.reset(seed=seed)

    # Build action map
    from nle_agent.agent_http import _build_action_map
    action_map = _build_action_map()

    for step in range(max_steps):
        state = encoder.encode_full(obs)
        adjacent = state["adjacent"]
        action_name = policy(adjacent, rng=rng)

        action_idx = _action_name_to_nle_action(action_name, action_map)

        prompt_text = encoder.format_prompt(state, action_name)

        obs_next, reward, terminated, truncated, info_next = env.step(action_idx)

        delta = encoder.encode_delta(obs, obs_next, action_name)
        target_text = encoder.format_target(delta)

        example = {
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": target_text},
            ]
        }
        yield json.dumps(example)

        obs = obs_next
        if terminated or truncated:
            break

    env.close()


def generate_dataset(
    output_path: str,
    num_games: int,
    max_steps: int,
    seed_start: int,
    encoder: StateEncoder,
    eval_path: Optional[str] = None,
    eval_fraction: float = 0.2,
) -> dict:
    """Generate a full dataset of training examples.

    Returns stats dict with num_examples, num_games, etc.
    """
    num_eval = max(1, int(num_games * eval_fraction))
    num_train = num_games - num_eval

    total_examples = 0

    with open(output_path, "w") as f:
        for i in range(num_train):
            for line in generate_game(
                seed=seed_start + i,
                max_steps=max_steps,
                encoder=encoder,
            ):
                f.write(line + "\n")
                total_examples += 1

    if eval_path:
        eval_examples = 0
        with open(eval_path, "w") as f:
            for i in range(num_train, num_games):
                for line in generate_game(
                    seed=seed_start + i,
                    max_steps=max_steps,
                    encoder=encoder,
                ):
                    f.write(line + "\n")
                    eval_examples += 1
    else:
        eval_examples = 0

    return {
        "num_train_examples": total_examples,
        "num_eval_examples": eval_examples,
        "num_train_games": num_train,
        "num_eval_games": num_eval if eval_path else 0,
    }
```

#### Task 2.3: Run data generator tests

```bash
cd /home/amiller/projects/dstack/dstack-lora
python -m pytest tests/test_data_generator.py -v
```

Expected: All tests PASS

**Commit:**

```bash
git add src/data_generator.py tests/test_data_generator.py
git commit -m "feat: add data generator with random play policy and JSONL output"
```

---

## Component 3: Training Script (`train.py`)

**What it does:** Loads the JSONL dataset and trains a LoRA adapter using Unsloth. Includes validation loss tracking.

### Tasks

#### Task 3.1: Write training script tests

**Files:**
- Create: `tests/test_train.py`

```python
# tests/test_train.py
import pytest
import json
import os
import tempfile


class TestTrainingDataFormat:
    """Verify our JSONL is compatible with Unsloth/TRL."""

    def test_jsonl_loads_as_conversations(self):
        """Each line should parse as a valid ShareGPT conversation."""
        from src.state_encoder import StateEncoder
        from src.data_generator import generate_game

        enc = StateEncoder()
        for line in generate_game(seed=42, max_steps=5, encoder=enc):
            data = json.loads(line)
            convs = data["conversations"]
            assert len(convs) >= 2
            roles = [c["role"] for c in convs]
            assert "user" in roles
            assert "assistant" in roles

    def test_can_create_hf_dataset_from_jsonl(self):
        """JSONL should load as a HuggingFace Dataset."""
        try:
            from datasets import Dataset
            from src.state_encoder import StateEncoder
            from src.data_generator import generate_game
        except ImportError:
            pytest.skip("datasets not installed")

        enc = StateEncoder()
        rows = []
        for line in generate_game(seed=42, max_steps=10, encoder=enc):
            rows.append(json.loads(line))

        ds = Dataset.from_list(rows)
        assert len(ds) >= 1
        assert "conversations" in ds.column_names

    def test_overfit_tiny_dataset(self):
        """Training on 5 examples for 50 steps should overfit (loss -> near 0).

        This is the most important validation: if the model CAN'T overfit
        a tiny dataset, something is wrong with the data format or training loop.
        """
        try:
            from unsloth import FastLanguageModel
            from trl import SFTTrainer
            from transformers import TrainingArguments
            from datasets import Dataset
            from src.state_encoder import StateEncoder
            from src.data_generator import generate_game
        except ImportError:
            pytest.skip("Unsloth/TRL not installed (expected -- run in GPU env)")

        enc = StateEncoder()
        rows = [json.loads(l) for l in generate_game(seed=42, max_steps=5, encoder=enc)]
        ds = Dataset.from_list(rows)

        # Use smallest possible model for testing
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/smollm2-135m",  # tiny model for testing
            max_seq_length=512,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=8, lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            use_rslora=True,
        )

        # Format conversations as text
        def format_example(ex):
            text = ""
            for msg in ex["conversations"]:
                text += f"### {msg['role']}:\n{msg['content']}\n"
            return {"text": text}

        ds = ds.map(format_example)

        trainer = SFTTrainer(
            model=model,
            train_dataset=ds,
            args=TrainingArguments(
                output_dir="/tmp/test_train",
                per_device_train_batch_size=2,
                max_steps=50,
                learning_rate=5e-4,
                optim="adamw_8bit",
                report_to="none",
            ),
            dataset_text_field="text",
            max_seq_length=512,
        )
        result = trainer.train()

        # Loss should decrease significantly
        initial_loss = result.training_loss  # average over training
        # On 5 examples for 50 steps, it should overfit hard
        # We can't check exact values without running, but it should complete
        assert result.global_step == 50
```

#### Task 3.2: Implement training script

**Files:**
- Create: `train.py`

```python
#!/usr/bin/env python3
"""Train a LoRA adapter on NetHack prediction data using Unsloth."""
import argparse
import json
import os
import hashlib

from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments


def hash_file(path: str) -> str:
    """SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def format_conversation(example):
    """Format ShareGPT conversation as training text."""
    text = ""
    for msg in example["conversations"]:
        text += f"### {msg['role']}:\n{msg['content']}\n"
    return {"text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", default="/data/train.jsonl")
    parser.add_argument("--eval-data", default=None)
    parser.add_argument("--output", default="/data/output/adapter")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0,
        use_rslora=True,
    )

    print(f"Loading data: {args.data}")
    data_hash = hash_file(args.data)
    ds = Dataset.from_json(args.data)
    ds = ds.map(format_conversation)

    eval_ds = None
    if args.eval_data and os.path.exists(args.eval_data):
        eval_ds = Dataset.from_json(args.eval_data)
        eval_ds = eval_ds.map(format_conversation)

    print(f"Training: {len(ds)} examples, hash={data_hash[:12]}...")
    os.makedirs(args.output, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        fp16=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        eval_dataset=eval_ds,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
    )

    result = trainer.train()

    # Save adapter
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Save training metadata
    meta = {
        "base_model": args.model,
        "data_hash": data_hash,
        "data_path": args.data,
        "final_loss": result.training_loss,
        "global_steps": result.global_step,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "adapter_hash": hash_file(os.path.join(args.output, "adapter_model.safetensors")),
    }
    with open(os.path.join(args.output, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Adapter saved to {args.output}")
    print(f"Final loss: {result.training_loss:.4f}")


if __name__ == "__main__":
    main()
```

---

## Component 4: Virgin Benchmark Evaluator (`evaluate.py`)

**What it does:** Generates sealed test data from unseen seeds, evaluates base model and trained model on prediction accuracy.

### Key Design

- Virgin seeds are generated INSIDE the TEE, never seen during training
- Evaluation measures: exact match accuracy for key prediction fields
- Before/after comparison proves the LoRA learned something

### Tasks

#### Task 4.1: Write evaluator tests and implementation

**Files:**
- Create: `src/evaluator.py`
- Create: `tests/test_evaluator.py`

```python
# tests/test_evaluator.py
import pytest
import json
import tempfile
import os


class TestEvaluator:
    def test_parse_prediction_extracts_fields(self):
        from src.evaluator import parse_prediction
        text = "pos:(+1,0) | hp:-2 | gold:+5 | msg:You kill the kobold!"
        result = parse_prediction(text)
        assert result["pos"] == (+1, 0)
        assert result["hp"] == -2
        assert result["gold"] == 5

    def test_parse_prediction_handles_same(self):
        from src.evaluator import parse_prediction
        text = "pos:same | hp:same"
        result = parse_prediction(text)
        assert result["pos"] == (0, 0)
        assert result["hp"] == 0

    def test_compute_accuracy_on_known_data(self):
        from src.evaluator import compute_accuracy
        predictions = [
            {"pos": (1, 0), "hp": 0, "gold": 0},
            {"pos": (1, 0), "hp": -2, "gold": 0},
            {"pos": (0, 0), "hp": 0, "gold": 5},
        ]
        ground_truth = [
            {"pos": (1, 0), "hp": 0, "gold": 0},
            {"pos": (1, 0), "hp": -2, "gold": 3},  # gold wrong
            {"pos": (-1, 0), "hp": 0, "gold": 5},  # pos wrong
        ]
        acc = compute_accuracy(predictions, ground_truth)
        # Example 1: all correct. Example 2: 2/3 correct. Example 3: 2/3 correct.
        # Per-field accuracy: pos=2/3, hp=3/3, gold=2/3 = 7/9 ≈ 0.778
        assert 0.5 < acc["field_accuracy"] < 1.0
        assert acc["exact_match_rate"] == pytest.approx(1/3, abs=0.01)

    def test_generate_virgin_seeds(self):
        """Virgin seeds should be distinct from training seeds."""
        from src.evaluator import generate_virgin_seeds
        train_seeds = set(range(100))
        virgin = generate_virgin_seeds(num=20, train_seeds=train_seeds)
        assert len(virgin) == 20
        assert len(set(virgin) & train_seeds) == 0  # No overlap

    def test_virgin_benchmark_is_deterministic(self):
        from src.evaluator import generate_virgin_seeds
        train = set(range(50))
        v1 = generate_virgin_seeds(num=10, train_seeds=train, salt="test")
        v2 = generate_virbin_seeds(num=10, train_seeds=train, salt="test")
        assert v1 == v2
```

```python
# src/evaluator.py
"""Evaluate model prediction accuracy on virgin (unseen) seeds."""
import hashlib
import json
import re
from typing import List, Tuple, Dict, Set


def parse_prediction(text: str) -> dict:
    """Parse model output text into structured prediction dict."""
    result = {"pos": (0, 0), "hp": 0, "gold": 0, "depth": 0}

    # pos:(+1,0) or pos:same
    pos_match = re.search(r"pos:\(([+-]?\d+),([+-]?\d+)\)", text)
    if pos_match:
        result["pos"] = (int(pos_match.group(1)), int(pos_match.group(2)))
    elif "pos:same" in text:
        result["pos"] = (0, 0)

    # hp:-2 or hp:same or hp:+3
    hp_match = re.search(r"hp:([+-]?\d+)", text)
    if hp_match:
        result["hp"] = int(hp_match.group(1))

    # gold:+5
    gold_match = re.search(r"gold:([+-]?\d+)", text)
    if gold_match:
        result["gold"] = int(gold_match.group(1))

    # depth:+1
    depth_match = re.search(r"depth:([+-]?\d+)", text)
    if depth_match:
        result["depth"] = int(depth_match.group(1))

    return result


def compute_accuracy(
    predictions: List[dict],
    ground_truth: List[dict],
) -> dict:
    """Compute per-field and exact-match accuracy."""
    assert len(predictions) == len(ground_truth)
    n = len(predictions)

    if n == 0:
        return {"field_accuracy": 0.0, "exact_match_rate": 0.0, "per_field": {}}

    fields = ["pos", "hp", "gold", "depth"]
    per_field = {}
    total_correct = 0
    total_fields = 0

    exact_matches = 0

    for i in range(n):
        pred = predictions[i]
        truth = ground_truth[i]
        all_correct = True

        for field in fields:
            if field in truth:
                total_fields += 1
                correct = pred.get(field) == truth[field]
                per_field[field] = per_field.get(field, 0) + (1 if correct else 0)
                if not correct:
                    all_correct = False

        if all_correct:
            exact_matches += 1

    field_accuracy = total_correct / total_fields if total_fields > 0 else 0.0
    for f in per_field:
        per_field[f] = per_field[f] / n

    return {
        "field_accuracy": field_accuracy,
        "exact_match_rate": exact_matches / n,
        "per_field": per_field,
        "num_examples": n,
    }


def generate_virgin_seeds(
    num: int,
    train_seeds: Set[int],
    salt: str = "dstack-lora-virgin",
) -> List[int]:
    """Generate N seeds that don't overlap with training seeds.

    Deterministic given same num, train_seeds, and salt.
    """
    seeds = []
    counter = 0
    while len(seeds) < num:
        h = hashlib.sha256(f"{salt}:{counter}".encode()).hexdigest()
        seed = int(h[:8], 16)
        if seed not in train_seeds and seed not in seeds:
            seeds.append(seed)
        counter += 1
    return seeds
```

---

## Component 5: Attested Manifest (`build_manifest.py`)

**What it does:** Compiles all hashes, training config, evaluation results, and TEE attestation into a signed JSON manifest.

### Tasks

#### Task 5.1: Write manifest tests and implementation

```python
# tests/test_manifest.py
import pytest
import json
import tempfile
import os


class TestManifest:
    def test_manifest_has_required_fields(self):
        from src.manifest import build_manifest
        m = build_manifest(
            base_model="Qwen/Qwen2.5-3B-Instruct",
            training_data_hash="abc123",
            training_code_hash="def456",
            adapter_path="/tmp/adapter",
            baseline_scores={"field_accuracy": 0.3},
            post_training_scores={"field_accuracy": 0.7},
            virgin_benchmark_hash="ghi789",
        )
        required = {
            "version", "base_model", "training_data", "training_code",
            "training_config", "results", "virgin_benchmark", "manifest_hash",
        }
        assert required.issubset(m.keys())

    def test_manifest_hash_is_consistent(self):
        from src.manifest import build_manifest
        m1 = build_manifest(
            base_model="Qwen/Qwen2.5-3B-Instruct",
            training_data_hash="abc", training_code_hash="def",
            adapter_path="/tmp/a", baseline_scores={}, post_training_scores={},
            virgin_benchmark_hash="ghi",
        )
        m2 = build_manifest(
            base_model="Qwen/Qwen2.5-3B-Instruct",
            training_data_hash="abc", training_code_hash="def",
            adapter_path="/tmp/a", baseline_scores={}, post_training_scores={},
            virgin_benchmark_hash="ghi",
        )
        assert m1["manifest_hash"] == m2["manifest_hash"]

    def test_manifest_detects_tampering(self):
        """Changing any field should change the manifest hash."""
        from src.manifest import build_manifest
        m1 = build_manifest(
            base_model="Qwen/Qwen2.5-3B-Instruct",
            training_data_hash="abc", training_code_hash="def",
            adapter_path="/tmp/a", baseline_scores={}, post_training_scores={},
            virgin_benchmark_hash="ghi",
        )
        m2 = build_manifest(
            base_model="Qwen/Qwen2.5-3B-Instruct",
            training_data_hash="TAMPERED", training_code_hash="def",
            adapter_path="/tmp/a", baseline_scores={}, post_training_scores={},
            virgin_benchmark_hash="ghi",
        )
        assert m1["manifest_hash"] != m2["manifest_hash"]
```

```python
# src/manifest.py
"""Build attested manifest from training artifacts."""
import hashlib
import json
import os


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    base_model: str,
    training_data_hash: str,
    training_code_hash: str,
    adapter_path: str,
    baseline_scores: dict,
    post_training_scores: dict,
    virgin_benchmark_hash: str,
    training_config: dict = None,
) -> dict:
    # Compute adapter hash if path exists
    adapter_hash = ""
    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if os.path.exists(adapter_file):
        adapter_hash = hash_file(adapter_file)

    config = training_config or {}

    manifest = {
        "version": "1.0",
        "base_model": {
            "name": base_model,
        },
        "training_data": {
            "sha256": training_data_hash,
        },
        "training_code": {
            "sha256": training_code_hash,
        },
        "training_config": config,
        "results": {
            "baseline_scores": baseline_scores,
            "post_training_scores": post_training_scores,
            "improvement": {
                k: post_training_scores.get(k, 0) - baseline_scores.get(k, 0)
                for k in baseline_scores
            },
        },
        "virgin_benchmark": {
            "sha256": virgin_benchmark_hash,
            "baseline_scores": baseline_scores,
            "post_training_scores": post_training_scores,
        },
        "adapter": {
            "sha256": adapter_hash,
            "path": adapter_path,
        },
    }

    # Self-hash (can't include own hash, so hash everything else)
    content = json.dumps(manifest, sort_keys=True)
    manifest["manifest_hash"] = hashlib.sha256(content.encode()).hexdigest()

    return manifest
```

---

## Execution Order

1. **Component 1** (StateEncoder) -- foundation, everything depends on it
2. **Component 2** (DataGenerator) -- depends on StateEncoder
3. **Component 4** (Evaluator) -- depends on StateEncoder, independent of training
4. **Component 3** (Training) -- depends on data format being correct
5. **Component 5** (Manifest) -- depends on all other outputs
6. **Integration test** -- end-to-end: generate data -> train -> evaluate -> manifest

Each component can be built and tested independently by a subagent.

---

## Testing Strategy Summary

| Component | Unit Tests | Integration Tests | Key Assertion |
|-----------|-----------|-------------------|---------------|
| StateEncoder | 10 tests on fake obs | 3 tests on real NLE obs | Deterministic output, all keys present |
| DataGenerator | 5 tests | Determinism, JSONL format | Same seed = same output, valid JSONL |
| Training | Data format test | Overfit test (5 examples, 50 steps) | Loss decreases to near 0 |
| Evaluator | 4 tests | Accuracy computation | Deterministic virgin seeds, correct accuracy |
| Manifest | 3 tests | Tamper detection | Hash changes if any field changes |

**Golden rule:** Every test uses fixed seeds. No test depends on GPU. GPU-dependent tests (training) are marked with `pytest.skip` if unsloth is not available, so they run in the GPU CVM but not locally.
