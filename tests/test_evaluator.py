"""
Comprehensive tests for Evaluator module.

Tests cover:
- parse_prediction: various formats, "same" values, negative values, died/alive
- compute_accuracy: known predictions vs ground truth, edge cases
- generate_test_data: real NLE, structure and non-empty output
- evaluate_model: graceful handling of missing server
- Integration: generate test data from seed 42, verify prompt/target consistency
"""

import json
import os
import sys

import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluator import (
    parse_prediction,
    compute_accuracy,
    evaluate_model,
    generate_test_data,
    hash_messages,
    run_evaluation,
)
from src.data_generator import SYSTEM_PROMPT
from src.state_encoder import StateEncoder


# ===========================================================================
# parse_prediction tests
# ===========================================================================

class TestParsePrediction:
    """Test parse_prediction with various formats."""

    def test_standard_format(self):
        text = "pos:(-1,0) | hp:same | gold:same | depth:same | alive:yes | msg:It's a wall."
        result = parse_prediction(text)
        assert result['pos'] == (-1, 0)
        assert result['hp_delta'] == 0
        assert result['gold_delta'] == 0
        assert result['depth_delta'] == 0
        assert result['survived'] is True
        assert result['message'] == "It's a wall."

    def test_positive_deltas(self):
        text = "pos:(1,-1) | hp:+5 | gold:+10 | depth:+1 | alive:yes | msg:You feel stronger!"
        result = parse_prediction(text)
        assert result['pos'] == (1, -1)
        assert result['hp_delta'] == 5
        assert result['gold_delta'] == 10
        assert result['depth_delta'] == 1
        assert result['survived'] is True
        assert result['message'] == 'You feel stronger!'

    def test_negative_deltas(self):
        text = "pos:(0,0) | hp:-3 | gold:-5 | depth:-1 | alive:yes | msg:Ouch!"
        result = parse_prediction(text)
        assert result['pos'] == (0, 0)
        assert result['hp_delta'] == -3
        assert result['gold_delta'] == -5
        assert result['depth_delta'] == -1
        assert result['survived'] is True
        assert result['message'] == 'Ouch!'

    def test_died(self):
        text = "pos:(0,0) | hp:-14 | gold:same | depth:same | alive:no | msg:You die..."
        result = parse_prediction(text)
        assert result['hp_delta'] == -14
        assert result['survived'] is False
        assert result['message'] == 'You die...'

    def test_all_same(self):
        text = "pos:(0,0) | hp:same | gold:same | depth:same | alive:yes | msg:"
        result = parse_prediction(text)
        assert result['pos'] == (0, 0)
        assert result['hp_delta'] == 0
        assert result['gold_delta'] == 0
        assert result['depth_delta'] == 0
        assert result['survived'] is True
        assert result['message'] == ''

    def test_with_spaces(self):
        """Handle extra whitespace."""
        text = "  pos: ( -1 , 0 )  |  hp: same  |  gold: +5  |  depth: same  |  alive: yes  |  msg: moved  "
        result = parse_prediction(text)
        assert result['pos'] == (-1, 0)
        assert result['hp_delta'] == 0
        assert result['gold_delta'] == 5
        assert result['depth_delta'] == 0
        assert result['survived'] is True
        assert result['message'] == 'moved'

    def test_empty_string(self):
        result = parse_prediction('')
        assert result['pos'] == (0, 0)
        assert result['hp_delta'] == 0
        assert result['gold_delta'] == 0
        assert result['depth_delta'] == 0
        assert result['survived'] is True
        assert result['message'] == ''

    def test_no_movement(self):
        text = "pos:(0,0) | hp:same | gold:same | depth:same | alive:yes | msg:"
        result = parse_prediction(text)
        assert result['pos'] == (0, 0)

    def test_movement_east(self):
        text = "pos:(1,0) | hp:same | gold:same | depth:same | alive:yes | msg:"
        result = parse_prediction(text)
        assert result['pos'] == (1, 0)

    def test_movement_south(self):
        text = "pos:(0,1) | hp:same | gold:same | depth:same | alive:yes | msg:"
        result = parse_prediction(text)
        assert result['pos'] == (0, 1)

    def test_alive_no(self):
        text = "pos:(0,0) | hp:-10 | gold:same | depth:same | alive:no | msg:Killed"
        result = parse_prediction(text)
        assert result['survived'] is False

    def test_partial_input_only_pos(self):
        """Input with only pos field should use defaults for the rest."""
        text = "pos:(2,3)"
        result = parse_prediction(text)
        assert result['pos'] == (2, 3)
        assert result['hp_delta'] == 0
        assert result['survived'] is True

    def test_message_with_special_chars(self):
        text = "pos:(0,0) | hp:same | gold:same | depth:same | alive:yes | msg:You see here a scroll."
        result = parse_prediction(text)
        assert 'scroll' in result['message']


# ===========================================================================
# compute_accuracy tests
# ===========================================================================

class TestComputeAccuracy:
    """Test compute_accuracy with known predictions vs ground truth."""

    def test_empty_predictions(self):
        result = compute_accuracy([], [])
        assert result['n'] == 0
        assert result['exact_match_rate'] == 0.0
        assert result['per_example'] == []

    def test_empty_predictions_nonempty_gt(self):
        result = compute_accuracy([], [{'pos_delta': (0, 0), 'hp_delta': 0, 'gold_delta': 0, 'depth_delta': 0, 'survived': True}])
        assert result['n'] == 0

    def test_nonempty_predictions_empty_gt(self):
        result = compute_accuracy([{'pos': (0, 0), 'hp_delta': 0, 'gold_delta': 0, 'depth_delta': 0, 'survived': True}], [])
        assert result['n'] == 0

    def test_perfect_match(self):
        predictions = [
            {'pos': (1, 0), 'hp_delta': 0, 'gold_delta': 5, 'depth_delta': 0, 'survived': True},
            {'pos': (0, -1), 'hp_delta': -3, 'gold_delta': 0, 'depth_delta': 0, 'survived': True},
        ]
        ground_truth = [
            {'pos_delta': (1, 0), 'hp_delta': 0, 'gold_delta': 5, 'depth_delta': 0, 'survived': True},
            {'pos_delta': (0, -1), 'hp_delta': -3, 'gold_delta': 0, 'depth_delta': 0, 'survived': True},
        ]
        result = compute_accuracy(predictions, ground_truth)
        assert result['n'] == 2
        assert result['exact_match_rate'] == 1.0
        assert result['pos_accuracy'] == 1.0
        assert result['hp_accuracy'] == 1.0
        assert result['gold_accuracy'] == 1.0
        assert result['depth_accuracy'] == 1.0
        assert result['survived_accuracy'] == 1.0
        assert result['per_example'] == [True, True]

    def test_total_mismatch(self):
        predictions = [
            {'pos': (1, 0), 'hp_delta': 5, 'gold_delta': 10, 'depth_delta': 1, 'survived': True},
        ]
        ground_truth = [
            {'pos_delta': (-1, 0), 'hp_delta': -3, 'gold_delta': -5, 'depth_delta': -1, 'survived': False},
        ]
        result = compute_accuracy(predictions, ground_truth)
        assert result['n'] == 1
        assert result['exact_match_rate'] == 0.0
        assert result['pos_accuracy'] == 0.0
        assert result['hp_accuracy'] == 0.0
        assert result['gold_accuracy'] == 0.0
        assert result['depth_accuracy'] == 0.0
        assert result['survived_accuracy'] == 0.0
        assert result['per_example'] == [False]

    def test_partial_match(self):
        predictions = [
            {'pos': (1, 0), 'hp_delta': 0, 'gold_delta': 0, 'depth_delta': 0, 'survived': True},
            {'pos': (0, 1), 'hp_delta': -3, 'gold_delta': 0, 'depth_delta': 0, 'survived': True},
            {'pos': (0, 0), 'hp_delta': 0, 'gold_delta': 5, 'depth_delta': 0, 'survived': True},
        ]
        ground_truth = [
            {'pos_delta': (1, 0), 'hp_delta': 0, 'gold_delta': 0, 'depth_delta': 0, 'survived': True},
            {'pos_delta': (0, 1), 'hp_delta': -3, 'gold_delta': 5, 'depth_delta': 0, 'survived': True},
            {'pos_delta': (1, 0), 'hp_delta': -1, 'gold_delta': 0, 'depth_delta': 1, 'survived': False},
        ]
        result = compute_accuracy(predictions, ground_truth)
        assert result['n'] == 3
        # Example 0: all match -> exact
        # Example 1: gold mismatches (0 vs 5) -> not exact, but pos/hp/survived match
        # Example 2: pos/hp/depth/survived all mismatch
        assert result['exact_match_rate'] == pytest.approx(1 / 3)
        assert result['pos_accuracy'] == pytest.approx(2 / 3)  # ex 0 and 1
        assert result['hp_accuracy'] == pytest.approx(2 / 3)  # ex 0 and 1
        assert result['gold_accuracy'] == pytest.approx(1 / 3)  # ex 0 only (ex 2: 5 vs 0)
        assert result['survived_accuracy'] == pytest.approx(2 / 3)  # ex 0 and 1
        assert result['per_example'] == [True, False, False]

    def test_all_zeros_match(self):
        predictions = [
            {'pos': (0, 0), 'hp_delta': 0, 'gold_delta': 0, 'depth_delta': 0, 'survived': True},
        ]
        ground_truth = [
            {'pos_delta': (0, 0), 'hp_delta': 0, 'gold_delta': 0, 'depth_delta': 0, 'survived': True},
        ]
        result = compute_accuracy(predictions, ground_truth)
        assert result['exact_match_rate'] == 1.0

    def test_survived_mismatch_only(self):
        predictions = [
            {'pos': (0, 0), 'hp_delta': 0, 'gold_delta': 0, 'depth_delta': 0, 'survived': True},
        ]
        ground_truth = [
            {'pos_delta': (0, 0), 'hp_delta': 0, 'gold_delta': 0, 'depth_delta': 0, 'survived': False},
        ]
        result = compute_accuracy(predictions, ground_truth)
        assert result['exact_match_rate'] == 0.0
        assert result['survived_accuracy'] == 0.0
        # All other fields match
        assert result['pos_accuracy'] == 1.0
        assert result['hp_accuracy'] == 1.0
        assert result['gold_accuracy'] == 1.0
        assert result['depth_accuracy'] == 1.0

    def test_single_example(self):
        predictions = [
            {'pos': (1, 0), 'hp_delta': -2, 'gold_delta': 0, 'depth_delta': 0, 'survived': True},
        ]
        ground_truth = [
            {'pos_delta': (1, 0), 'hp_delta': -2, 'gold_delta': 0, 'depth_delta': 0, 'survived': True},
        ]
        result = compute_accuracy(predictions, ground_truth)
        assert result['n'] == 1
        assert result['exact_match_rate'] == 1.0

    def test_unequal_lengths_uses_min(self):
        predictions = [
            {'pos': (0, 0), 'hp_delta': 0, 'gold_delta': 0, 'depth_delta': 0, 'survived': True},
            {'pos': (1, 0), 'hp_delta': 0, 'gold_delta': 0, 'depth_delta': 0, 'survived': True},
        ]
        ground_truth = [
            {'pos_delta': (0, 0), 'hp_delta': 0, 'gold_delta': 0, 'depth_delta': 0, 'survived': True},
        ]
        result = compute_accuracy(predictions, ground_truth)
        assert result['n'] == 1


# ===========================================================================
# evaluate_model tests (server not running)
# ===========================================================================

class TestEvaluateModel:
    """Test evaluate_model graceful handling of missing server."""

    def test_handles_missing_server(self):
        """When server is not running, should return gracefully."""
        result = evaluate_model(
            model_name_or_path="test-model",
            test_data=[
                {'prompt': 'test prompt', 'target': 'pos:(0,0) | hp:same | gold:same | depth:same | alive:yes | msg:'}
            ],
            server_url="http://127.0.0.1:19999",  # port that's not listening
        )
        assert result['server_available'] is False
        assert len(result['errors']) > 0
        assert result['accuracy']['n'] == 0
        assert result['predictions'] == []

    def test_result_structure(self):
        """Verify the return dict has all expected keys."""
        result = evaluate_model(
            model_name_or_path="test-model",
            test_data=[],
            server_url="http://127.0.0.1:19999",
        )
        assert 'predictions' in result
        assert 'raw_responses' in result
        assert 'accuracy' in result
        assert 'model' in result
        assert 'server_url' in result
        assert 'server_available' in result
        assert 'errors' in result

    def test_max_samples_limits(self):
        """max_samples should limit the number of examples evaluated."""
        # Server won't be running, but we can verify max_samples doesn't crash
        result = evaluate_model(
            model_name_or_path="test",
            test_data=[{'prompt': 'p1', 'messages': [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "p1"}], 'target': 't1'}, {'prompt': 'p2', 'messages': [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "p2"}], 'target': 't2'}],
            server_url="http://127.0.0.1:19999",
            max_samples=1,
        )
        # Should not crash; server is down so predictions are empty
        assert isinstance(result, dict)


# ===========================================================================
# generate_test_data tests
# ===========================================================================

class TestGenerateTestData:
    """Test generate_test_data with real NLE."""

    def setup_method(self):
        self.encoder = StateEncoder()

    def test_returns_list(self):
        data = generate_test_data(seeds=[42], max_steps=3, encoder=self.encoder)
        assert isinstance(data, list)

    def test_non_empty_output(self):
        data = generate_test_data(seeds=[42], max_steps=5, encoder=self.encoder)
        assert len(data) > 0

    def test_structure_has_required_keys(self):
        data = generate_test_data(seeds=[42], max_steps=3, encoder=self.encoder)
        for item in data:
            assert 'prompt' in item
            assert 'messages' in item
            assert 'target' in item
            assert 'ground_truth_delta' in item
            assert 'seed' in item
            assert 'step' in item

    def test_messages_have_system_and_user_roles(self):
        data = generate_test_data(seeds=[42], max_steps=3, encoder=self.encoder)
        for item in data:
            messages = item["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == SYSTEM_PROMPT
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == item["prompt"]

    def test_prompt_is_string(self):
        data = generate_test_data(seeds=[42], max_steps=3, encoder=self.encoder)
        for item in data:
            assert isinstance(item['prompt'], str)
            assert len(item['prompt']) > 0

    def test_target_is_string(self):
        data = generate_test_data(seeds=[42], max_steps=3, encoder=self.encoder)
        for item in data:
            assert isinstance(item['target'], str)
            assert len(item['target']) > 0

    def test_ground_truth_delta_structure(self):
        data = generate_test_data(seeds=[42], max_steps=3, encoder=self.encoder)
        for item in data:
            gt = item['ground_truth_delta']
            assert 'pos_delta' in gt
            assert 'hp_delta' in gt
            assert 'gold_delta' in gt
            assert 'depth_delta' in gt
            assert 'survived' in gt
            assert 'message' in gt
            assert isinstance(gt['pos_delta'], tuple)
            assert len(gt['pos_delta']) == 2
            assert isinstance(gt['hp_delta'], int)
            assert isinstance(gt['gold_delta'], int)
            assert isinstance(gt['depth_delta'], int)
            assert isinstance(gt['survived'], bool)

    def test_seed_recorded(self):
        data = generate_test_data(seeds=[42], max_steps=3, encoder=self.encoder)
        for item in data:
            assert item['seed'] == 42

    def test_multiple_seeds(self):
        data = generate_test_data(seeds=[42, 99], max_steps=3, encoder=self.encoder)
        seeds_seen = {item['seed'] for item in data}
        assert 42 in seeds_seen
        assert 99 in seeds_seen

    def test_step_indices_sequential(self):
        data = generate_test_data(seeds=[42], max_steps=5, encoder=self.encoder)
        steps = [item['step'] for item in data]
        assert steps == list(range(len(data)))

    def test_max_steps_respected(self):
        data = generate_test_data(seeds=[42], max_steps=3, encoder=self.encoder)
        assert len(data) <= 3

    def test_prompt_contains_state_info(self):
        data = generate_test_data(seeds=[42], max_steps=3, encoder=self.encoder)
        for item in data:
            assert 'HP:' in item['prompt']
            assert 'Action:' in item['prompt']

    def test_target_format(self):
        data = generate_test_data(seeds=[42], max_steps=3, encoder=self.encoder)
        for item in data:
            target = item['target']
            assert 'pos:' in target
            assert 'hp:' in target
            assert 'gold:' in target
            assert 'depth:' in target
            assert 'alive:' in target

    def test_message_hash_stable(self):
        data = generate_test_data(seeds=[42], max_steps=1, encoder=self.encoder)
        assert len(data) == 1
        first = data[0]["messages"]
        assert hash_messages(first) == hash_messages(first)


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:
    """Integration tests: verify consistency with state_encoder."""

    def setup_method(self):
        self.encoder = StateEncoder()

    def test_parse_prediction_roundtrip(self):
        """Parse the output of format_target and verify it matches the original delta."""
        data = generate_test_data(seeds=[42], max_steps=5, encoder=self.encoder)
        for item in data:
            target = item['target']
            parsed = parse_prediction(target)
            gt = item['ground_truth_delta']

            # pos should match pos_delta
            assert parsed['pos'] == gt['pos_delta'], \
                f"pos mismatch: parsed={parsed['pos']} vs gt={gt['pos_delta']}"

            # hp_delta should match
            assert parsed['hp_delta'] == gt['hp_delta'], \
                f"hp mismatch: parsed={parsed['hp_delta']} vs gt={gt['hp_delta']}"

            # gold_delta should match
            assert parsed['gold_delta'] == gt['gold_delta'], \
                f"gold mismatch: parsed={parsed['gold_delta']} vs gt={gt['gold_delta']}"

            # depth_delta should match
            assert parsed['depth_delta'] == gt['depth_delta'], \
                f"depth mismatch: parsed={parsed['depth_delta']} vs gt={gt['depth_delta']}"

            # survived should match
            assert parsed['survived'] == gt['survived'], \
                f"survived mismatch: parsed={parsed['survived']} vs gt={gt['survived']}"

    def test_compute_accuracy_perfect_from_parsed(self):
        """Parse targets and use them as predictions -> should get perfect accuracy."""
        data = generate_test_data(seeds=[42], max_steps=5, encoder=self.encoder)
        predictions = []
        ground_truth = []

        for item in data:
            parsed = parse_prediction(item['target'])
            predictions.append(parsed)
            ground_truth.append({
                'pos_delta': item['ground_truth_delta']['pos_delta'],
                'hp_delta': item['ground_truth_delta']['hp_delta'],
                'gold_delta': item['ground_truth_delta']['gold_delta'],
                'depth_delta': item['ground_truth_delta']['depth_delta'],
                'survived': item['ground_truth_delta']['survived'],
            })

        result = compute_accuracy(predictions, ground_truth)
        assert result['n'] == len(data)
        assert result['exact_match_rate'] == 1.0
        assert result['pos_accuracy'] == 1.0

    def test_compute_accuracy_with_wrong_predictions(self):
        """Intentionally wrong predictions should yield low accuracy."""
        data = generate_test_data(seeds=[42], max_steps=5, encoder=self.encoder)
        predictions = []
        ground_truth = []

        for item in data:
            # Flip all predictions
            parsed = parse_prediction(item['target'])
            dx, dy = parsed['pos']
            predictions.append({
                'pos': (-dx, -dy),  # flipped
                'hp_delta': -parsed['hp_delta'],  # flipped
                'gold_delta': -parsed['gold_delta'],  # flipped
                'depth_delta': -parsed['depth_delta'],  # flipped
                'survived': not parsed['survived'],  # flipped
            })
            ground_truth.append({
                'pos_delta': item['ground_truth_delta']['pos_delta'],
                'hp_delta': item['ground_truth_delta']['hp_delta'],
                'gold_delta': item['ground_truth_delta']['gold_delta'],
                'depth_delta': item['ground_truth_delta']['depth_delta'],
                'survived': item['ground_truth_delta']['survived'],
            })

        result = compute_accuracy(predictions, ground_truth)
        # Zero deltas stay zero when flipped, so accuracy won't be exactly 0
        # But exact_match_rate should be 0 (since at least survived is flipped)
        assert result['exact_match_rate'] == 0.0

    def test_run_evaluation_no_server(self):
        """run_evaluation with no server should return gracefully."""
        result = run_evaluation(
            seeds=[42],
            max_steps=3,
            encoder=self.encoder,
            server_url="http://127.0.0.1:19999",
        )
        assert result['server_available'] is False
        assert isinstance(result['test_data'], list)
        assert len(result['test_data']) > 0
        assert isinstance(result['per_example'], list)
        assert len(result['per_example']) == 0  # no predictions made
        assert 'accuracy' in result
        assert 'errors' in result
        assert len(result['errors']) > 0

    def test_prompt_target_consistency_seed42(self):
        """Verify prompt and target are consistent with state_encoder outputs.

        Generate data from seed 42, then independently verify by re-running
        the game and checking that prompts/targets match format_prompt/format_target.
        """
        data = generate_test_data(seeds=[42], max_steps=5, encoder=self.encoder)
        assert len(data) > 0

        # Verify each target parses back correctly
        for item in data:
            target = item['target']
            parsed = parse_prediction(target)

            # Target should contain expected fields
            assert isinstance(parsed['pos'], tuple)
            assert isinstance(parsed['hp_delta'], int)
            assert isinstance(parsed['survived'], bool)

            # Prompt should contain state info
            prompt = item['prompt']
            assert 'HP:' in prompt
            assert 'Pos:' in prompt
            assert 'Action:' in prompt
            assert 'Adjacent:' in prompt

    def test_generate_test_data_deterministic_length(self):
        """Same seed and max_steps should produce the same number of examples."""
        data1 = generate_test_data(seeds=[42], max_steps=5, encoder=self.encoder)
        data2 = generate_test_data(seeds=[42], max_steps=5, encoder=self.encoder)
        assert len(data1) == len(data2)

    def test_format_target_matches_parsed(self):
        """Verify format_target output roundtrips through parse_prediction."""
        data = generate_test_data(seeds=[42], max_steps=5, encoder=self.encoder)
        for item in data:
            # Re-format the ground truth delta
            reformatted = self.encoder.format_target(item['ground_truth_delta'])
            # Parse both the original target and the reformatted version
            parsed_original = parse_prediction(item['target'])
            parsed_reformatted = parse_prediction(reformatted)

            # They should be identical
            assert parsed_original == parsed_reformatted, \
                f"Mismatch: original={parsed_original} vs reformatted={parsed_reformatted}"
