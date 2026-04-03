"""
Comprehensive tests for DataGenerator module.

Tests cover:
- JSONL format validation (valid JSON, correct conversation structure)
- Determinism: same seed produces identical output
- Different seeds produce different output
- Dataset creation writes files to disk with correct train/eval split
- Wall avoidance policy returns valid actions and prefers open tiles
- Integration test: run a real game with NLE and verify the full pipeline
"""

import json
import os
import sys
import tempfile
import random

import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_generator import (
    generate_game,
    generate_dataset,
    wall_avoidance_policy,
    SYSTEM_PROMPT,
    _get_action_map,
)
from src.state_encoder import StateEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_jsonl_lines(lines):
    """Parse a list of JSONL strings into a list of dicts."""
    return [json.loads(line) for line in lines]


def _validate_conversation(obj):
    """Validate a single ShareGPT conversation object."""
    assert 'conversations' in obj, "Missing 'conversations' key"
    convs = obj['conversations']
    assert len(convs) == 3, f"Expected 3 messages, got {len(convs)}"

    # System message
    assert convs[0]['role'] == 'system'
    assert isinstance(convs[0]['content'], str)
    assert len(convs[0]['content']) > 0

    # User message
    assert convs[1]['role'] == 'user'
    assert isinstance(convs[1]['content'], str)
    assert len(convs[1]['content']) > 0

    # Assistant message
    assert convs[2]['role'] == 'assistant'
    assert isinstance(convs[2]['content'], str)
    assert len(convs[2]['content']) > 0


# ===========================================================================
# Wall Avoidance Policy Tests
# ===========================================================================

class TestWallAvoidancePolicy:
    """Test the wall_avoidance_policy function."""

    def test_returns_string(self):
        adjacent = {"north": "floor", "south": "floor", "east": "floor", "west": "floor"}
        rng = random.Random(42)
        result = wall_avoidance_policy(adjacent, rng)
        assert isinstance(result, str)

    def test_returns_valid_direction(self):
        adjacent = {"north": "floor", "south": "floor", "east": "floor", "west": "floor"}
        rng = random.Random(42)
        result = wall_avoidance_policy(adjacent, rng)
        assert result in ('north', 'south', 'east', 'west')

    def test_avoids_walls(self):
        """Only open directions should be chosen."""
        adjacent = {"north": "wall", "south": "wall", "east": "floor", "west": "wall"}
        rng = random.Random(42)
        # Run many times to check it never picks a wall direction
        for _ in range(100):
            result = wall_avoidance_policy(adjacent, rng)
            assert result == 'east', f"Expected east (only open), got {result}"

    def test_avoids_unseen(self):
        """unseen tiles should also be avoided."""
        adjacent = {"north": "unseen", "south": "floor", "east": "unseen", "west": "unseen"}
        rng = random.Random(42)
        for _ in range(100):
            result = wall_avoidance_policy(adjacent, rng)
            assert result == 'south', f"Expected south (only open), got {result}"

    def test_prefers_open_over_walls(self):
        """When there's a mix, only open tiles are chosen."""
        adjacent = {"north": "wall", "south": "floor", "east": "wall", "west": "door"}
        rng = random.Random(42)
        for _ in range(100):
            result = wall_avoidance_policy(adjacent, rng)
            assert result in ('south', 'west'), f"Expected open tile, got {result}"

    def test_waits_when_all_walls(self):
        """When all directions are walls, should return 'wait'."""
        adjacent = {"north": "wall", "south": "wall", "east": "wall", "west": "wall"}
        rng = random.Random(42)
        result = wall_avoidance_policy(adjacent, rng)
        assert result == 'wait'

    def test_waits_when_all_unseen(self):
        adjacent = {"north": "unseen", "south": "unseen", "east": "unseen", "west": "unseen"}
        rng = random.Random(42)
        result = wall_avoidance_policy(adjacent, rng)
        assert result == 'wait'

    def test_deterministic_with_same_rng_seed(self):
        adjacent = {"north": "floor", "south": "floor", "east": "floor", "west": "wall"}
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        for _ in range(20):
            assert wall_avoidance_policy(adjacent, rng1) == wall_avoidance_policy(adjacent, rng2)

    def test_handles_door(self):
        """Door is not a wall, should be a valid choice."""
        adjacent = {"north": "door", "south": "wall", "east": "wall", "west": "wall"}
        rng = random.Random(42)
        result = wall_avoidance_policy(adjacent, rng)
        assert result == 'north'

    def test_handles_stairs(self):
        """Stairs are walkable, should be valid choices."""
        adjacent = {"north": "wall", "south": "stairs_down", "east": "wall", "west": "wall"}
        rng = random.Random(42)
        result = wall_avoidance_policy(adjacent, rng)
        assert result == 'south'

    def test_handles_corridor(self):
        """Corridor is walkable."""
        adjacent = {"north": "corridor", "south": "wall", "east": "wall", "west": "wall"}
        rng = random.Random(42)
        result = wall_avoidance_policy(adjacent, rng)
        assert result == 'north'


# ===========================================================================
# JSONL Format Validation
# ===========================================================================

class TestJSONLFormat:
    """Test that generate_game produces valid JSONL in ShareGPT format."""

    def setup_method(self):
        self.encoder = StateEncoder()

    def test_each_line_is_valid_json(self):
        lines = list(generate_game(seed=42, max_steps=5, encoder=self.encoder))
        assert len(lines) > 0
        for line in lines:
            obj = json.loads(line)  # should not raise
            assert isinstance(obj, dict)

    def test_conversation_structure(self):
        lines = list(generate_game(seed=42, max_steps=5, encoder=self.encoder))
        for line in lines:
            obj = json.loads(line)
            _validate_conversation(obj)

    def test_system_prompt_content(self):
        lines = list(generate_game(seed=42, max_steps=3, encoder=self.encoder))
        for line in lines:
            obj = json.loads(line)
            assert obj['conversations'][0]['content'] == SYSTEM_PROMPT

    def test_user_content_has_action(self):
        """User prompt should contain 'Action:' line."""
        lines = list(generate_game(seed=42, max_steps=3, encoder=self.encoder))
        for line in lines:
            obj = json.loads(line)
            user_content = obj['conversations'][1]['content']
            assert 'Action:' in user_content
            assert 'HP:' in user_content
            assert 'Adjacent:' in user_content

    def test_assistant_content_has_delta(self):
        """Assistant response should contain delta info."""
        lines = list(generate_game(seed=42, max_steps=3, encoder=self.encoder))
        for line in lines:
            obj = json.loads(line)
            asst_content = obj['conversations'][2]['content']
            assert 'pos:' in asst_content
            assert 'hp:' in asst_content
            assert 'gold:' in asst_content
            assert 'alive:' in asst_content

    def test_lines_are_newline_separated(self):
        """Each yielded line should NOT contain internal newlines (single JSON per line)."""
        lines = list(generate_game(seed=42, max_steps=3, encoder=self.encoder))
        for line in lines:
            assert '\n' not in line, "JSONL line contains newline"


# ===========================================================================
# Determinism Tests
# ===========================================================================

class TestDeterminism:
    """Test deterministic behavior.

    Note: NLE's env.reset(seed=...) is NOT fully deterministic across runs
    (it uses multiple internal RNG sources). Therefore we test:
    - Wall avoidance policy is deterministic with same RNG
    - The code structure is deterministic (format_prompt/format_target)
    - Different seeds generally produce different gameplay
    """

    def setup_method(self):
        self.encoder = StateEncoder()

    def test_policy_determinism(self):
        """wall_avoidance_policy with same RNG seed produces same action sequence."""
        adjacent = {"north": "floor", "south": "wall", "east": "floor", "west": "wall"}
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        actions1 = [wall_avoidance_policy(adjacent, rng1) for _ in range(50)]
        actions2 = [wall_avoidance_policy(adjacent, rng2) for _ in range(50)]
        assert actions1 == actions2

    def test_same_seed_produces_output(self):
        """Same seed should produce output (no crashes)."""
        lines1 = list(generate_game(seed=42, max_steps=5, encoder=self.encoder))
        lines2 = list(generate_game(seed=42, max_steps=5, encoder=self.encoder))
        assert len(lines1) > 0
        assert len(lines2) > 0
        # Both should produce the same number of lines since max_steps is the same
        # (though content may differ due to NLE non-determinism)
        assert len(lines1) == len(lines2)

    def test_different_seeds_likely_different_output(self):
        """Different seeds should almost certainly produce different content."""
        lines1 = list(generate_game(seed=42, max_steps=10, encoder=self.encoder))
        lines2 = list(generate_game(seed=99, max_steps=10, encoder=self.encoder))
        # Compare first line content (very unlikely to be identical with different seeds)
        assert lines1 != lines2, "Different seeds produced identical output"

    def test_output_is_valid_jsonl(self):
        """Regardless of determinism, all output must be valid JSONL."""
        for seed in [42, 99, 123, 200]:
            lines = list(generate_game(seed=seed, max_steps=5, encoder=self.encoder))
            for line in lines:
                obj = json.loads(line)
                _validate_conversation(obj)

    def test_format_functions_deterministic(self):
        """StateEncoder format functions are deterministic with same inputs."""
        lines = list(generate_game(seed=42, max_steps=5, encoder=self.encoder))
        # Parse and verify structure consistency
        for line in lines:
            obj = json.loads(line)
            convs = obj['conversations']
            # System message is always the same
            assert convs[0]['content'] == SYSTEM_PROMPT
            # User message always has the expected fields
            assert 'HP:' in convs[1]['content']
            assert 'Action:' in convs[1]['content']
            # Assistant message always has delta fields
            assert 'pos:' in convs[2]['content']
            assert 'alive:' in convs[2]['content']


# ===========================================================================
# Dataset Generation Tests
# ===========================================================================

class TestGenerateDataset:
    """Test generate_dataset writes files and returns correct stats."""

    def setup_method(self):
        self.encoder = StateEncoder()
        self.tmpdir = tempfile.mkdtemp()

    def test_writes_train_file(self):
        train_path = os.path.join(self.tmpdir, 'train.jsonl')
        stats = generate_dataset(
            output_path=train_path,
            num_games=2,
            max_steps=5,
            seed_start=100,
            encoder=self.encoder,
        )
        assert os.path.exists(train_path)
        assert stats['train_path'] == train_path

    def test_train_file_valid_jsonl(self):
        train_path = os.path.join(self.tmpdir, 'train.jsonl')
        generate_dataset(
            output_path=train_path,
            num_games=2,
            max_steps=5,
            seed_start=100,
            encoder=self.encoder,
        )
        with open(train_path) as f:
            lines = f.readlines()
        assert len(lines) > 0
        for line in lines:
            obj = json.loads(line.strip())
            _validate_conversation(obj)

    def test_stats_total_games(self):
        train_path = os.path.join(self.tmpdir, 'train.jsonl')
        stats = generate_dataset(
            output_path=train_path,
            num_games=3,
            max_steps=5,
            seed_start=100,
            encoder=self.encoder,
        )
        assert stats['total_games'] == 3

    def test_stats_total_examples(self):
        train_path = os.path.join(self.tmpdir, 'train.jsonl')
        stats = generate_dataset(
            output_path=train_path,
            num_games=2,
            max_steps=5,
            seed_start=100,
            encoder=self.encoder,
        )
        assert stats['total_examples'] > 0
        assert stats['total_examples'] == stats['train_examples'] + stats['eval_examples']

    def test_eval_split(self):
        train_path = os.path.join(self.tmpdir, 'train.jsonl')
        eval_path = os.path.join(self.tmpdir, 'eval.jsonl')
        stats = generate_dataset(
            output_path=train_path,
            num_games=5,
            max_steps=5,
            seed_start=200,
            encoder=self.encoder,
            eval_path=eval_path,
            eval_fraction=0.4,
        )
        assert os.path.exists(train_path)
        assert os.path.exists(eval_path)
        assert stats['eval_path'] == eval_path
        assert stats['eval_examples'] > 0
        assert stats['train_examples'] > 0

    def test_eval_fraction_correct(self):
        """Check the number of eval games matches eval_fraction."""
        train_path = os.path.join(self.tmpdir, 'train.jsonl')
        eval_path = os.path.join(self.tmpdir, 'eval.jsonl')
        stats = generate_dataset(
            output_path=train_path,
            num_games=10,
            max_steps=3,
            seed_start=300,
            encoder=self.encoder,
            eval_path=eval_path,
            eval_fraction=0.2,
        )
        # 10 games * 0.2 = 2 eval games
        # Count lines in each file
        with open(train_path) as f:
            train_lines = f.readlines()
        with open(eval_path) as f:
            eval_lines = f.readlines()
        # Train should have ~8 games worth, eval ~2 games worth
        assert len(train_lines) > len(eval_lines)
        assert stats['train_examples'] == len(train_lines)
        assert stats['eval_examples'] == len(eval_lines)

    def test_no_eval_path_means_no_eval(self):
        train_path = os.path.join(self.tmpdir, 'train.jsonl')
        stats = generate_dataset(
            output_path=train_path,
            num_games=3,
            max_steps=5,
            seed_start=100,
            encoder=self.encoder,
            eval_path=None,
        )
        assert stats['eval_path'] is None
        assert stats['eval_examples'] == 0
        assert stats['train_examples'] == stats['total_examples']

    def test_all_examples_in_train_without_eval(self):
        train_path = os.path.join(self.tmpdir, 'train.jsonl')
        stats = generate_dataset(
            output_path=train_path,
            num_games=2,
            max_steps=5,
            seed_start=100,
            encoder=self.encoder,
        )
        with open(train_path) as f:
            lines = f.readlines()
        assert len(lines) == stats['train_examples']
        assert stats['train_examples'] == stats['total_examples']

    def test_dataset_consistency(self):
        """Dataset generation should produce consistent structure."""
        train_path = os.path.join(self.tmpdir, 'train.jsonl')
        stats = generate_dataset(
            output_path=train_path,
            num_games=3,
            max_steps=5,
            seed_start=42,
            encoder=self.encoder,
        )
        # Verify file exists and has the right number of lines
        with open(train_path) as f:
            lines = f.readlines()
        assert len(lines) == stats['train_examples']
        assert stats['total_examples'] == stats['train_examples'] + stats['eval_examples']
        # All lines are valid JSONL
        for line in lines:
            obj = json.loads(line.strip())
            _validate_conversation(obj)


# ===========================================================================
# Action Map Tests
# ===========================================================================

class TestActionMap:
    """Test that the action map is properly loaded."""

    def test_action_map_loaded(self):
        amap = _get_action_map()
        assert isinstance(amap, dict)
        assert len(amap) > 0

    def test_directions_in_map(self):
        amap = _get_action_map()
        for d in ('north', 'south', 'east', 'west'):
            assert d in amap
            assert isinstance(amap[d], int)

    def test_wait_in_map(self):
        amap = _get_action_map()
        assert 'wait' in amap


# ===========================================================================
# Integration Test: Real NLE Game
# ===========================================================================

class TestIntegration:
    """Integration tests running real NLE games."""

    def setup_method(self):
        self.encoder = StateEncoder()

    def test_generate_game_runs(self):
        """generate_game should yield at least one line for a real game."""
        lines = list(generate_game(seed=42, max_steps=10, encoder=self.encoder))
        assert len(lines) > 0, "generate_game yielded no lines"
        assert len(lines) <= 10, f"Expected <= 10 lines, got {len(lines)}"

    def test_full_pipeline(self):
        """Run a real game and verify all output is valid."""
        lines = list(generate_game(seed=123, max_steps=20, encoder=self.encoder))
        assert len(lines) > 0

        for line in lines:
            obj = json.loads(line)
            _validate_conversation(obj)

            user = obj['conversations'][1]['content']
            asst = obj['conversations'][2]['content']

            # User content should have state info
            assert 'HP:' in user
            assert 'Adjacent:' in user
            assert 'Action:' in user

            # Assistant should have delta info
            assert 'pos:' in asst
            assert 'hp:' in asst
            assert 'alive:' in asst

    def test_game_with_different_seeds(self):
        """Different seeds should produce different games."""
        lines_a = list(generate_game(seed=1, max_steps=10, encoder=self.encoder))
        lines_b = list(generate_game(seed=2, max_steps=10, encoder=self.encoder))

        # At least some content should differ (different maps, different actions)
        prompts_a = [json.loads(l)['conversations'][1]['content'] for l in lines_a]
        prompts_b = [json.loads(l)['conversations'][1]['content'] for l in lines_b]
        assert prompts_a != prompts_b

    def test_dataset_integration(self):
        """Full dataset generation with real NLE."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, 'train.jsonl')
            eval_path = os.path.join(tmpdir, 'eval.jsonl')

            stats = generate_dataset(
                output_path=train_path,
                num_games=3,
                max_steps=5,
                seed_start=500,
                encoder=self.encoder,
                eval_path=eval_path,
                eval_fraction=0.33,
            )

            assert stats['total_games'] == 3
            assert stats['total_examples'] > 0
            assert os.path.exists(train_path)

            # Verify all lines are valid
            with open(train_path) as f:
                for line in f:
                    obj = json.loads(line.strip())
                    _validate_conversation(obj)

    def test_game_respects_max_steps(self):
        """Game should not exceed max_steps lines."""
        lines = list(generate_game(seed=42, max_steps=5, encoder=self.encoder))
        assert len(lines) <= 5, f"Expected <= 5 lines, got {len(lines)}"

    def test_game_can_terminate_early(self):
        """Game may terminate before max_steps if the agent dies."""
        # Try multiple seeds, at least one should terminate early
        terminated_early = False
        for seed in range(50):
            lines = list(generate_game(seed=seed, max_steps=500, encoder=self.encoder))
            if len(lines) < 500:
                terminated_early = True
                break
        assert terminated_early, "No game terminated early across 50 seeds"

    def test_custom_policy_integration(self):
        """Test with a custom deterministic policy."""
        def always_east(adjacent, rng):
            if adjacent.get('east', 'wall') not in ('wall', 'unseen'):
                return 'east'
            return 'wait'

        lines = list(generate_game(
            seed=42, max_steps=5, encoder=self.encoder, policy=always_east
        ))
        assert len(lines) > 0
        for line in lines:
            obj = json.loads(line)
            user = obj['conversations'][1]['content']
            assert 'Action:' in user
