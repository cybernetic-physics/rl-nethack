"""
Comprehensive tests for the Game Reporter module.

Tests cover:
- format_replay: non-empty text with step numbers, HP, actions
- format_html_replay: valid HTML with proper tags and expected elements
- format_summary: compact one-line summary
- run_and_report: correct return structure with all expected keys
- Integration with real NLE environment (seed=42, 10 steps)
- HTML output is viewable (html/head/body tags)
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.reporter import format_replay, format_html_replay, format_summary, run_and_report
from src.state_encoder import StateEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_step(step_idx, hp, hp_max, pos, action, hp_delta=0, gold_delta=0,
               survived=True, new_tiles=None, message='', depth=1, gold=0):
    """Build a minimal step_data dict for unit testing."""
    return {
        'step': step_idx,
        'obs': None,
        'obs_after': None,
        'state': {
            'hp': hp,
            'hp_max': hp_max,
            'position': pos,
            'gold': gold,
            'depth': depth,
            'ac': 5,
            'strength': 16,
            'dexterity': 14,
            'turn': step_idx,
            'adjacent': {'north': 'floor', 'south': 'floor', 'east': 'floor', 'west': 'floor'},
            'visible_monsters': [],
            'visible_items': [],
            'message': '',
        },
        'action': action,
        'delta': {
            'pos_delta': (0, 0),
            'hp_delta': hp_delta,
            'gold_delta': gold_delta,
            'depth_delta': 0,
            'turn_delta': 1,
            'new_tiles': new_tiles or [],
            'message': message,
            'survived': survived,
        },
        'prompt': '',
        'target': '',
    }


# ===========================================================================
# format_replay tests
# ===========================================================================

class TestFormatReplay:
    """Test the text replay formatter."""

    def test_non_empty_string(self):
        steps = [_make_step(0, 14, 14, (5, 3), 'east')]
        result = format_replay(steps, seed=42)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_header_with_seed_and_steps(self):
        steps = [
            _make_step(0, 14, 14, (5, 3), 'east'),
            _make_step(1, 14, 14, (6, 3), 'north'),
        ]
        result = format_replay(steps, seed=42)
        assert 'Seed 42' in result
        assert '2 steps' in result

    def test_step_numbers_present(self):
        steps = [
            _make_step(0, 14, 14, (5, 3), 'east'),
            _make_step(1, 14, 14, (6, 3), 'north'),
            _make_step(2, 12, 14, (6, 2), 'wait', hp_delta=-2),
        ]
        result = format_replay(steps, seed=1)
        assert 'Step  0' in result
        assert 'Step  1' in result
        assert 'Step  2' in result

    def test_hp_values_present(self):
        steps = [
            _make_step(0, 14, 14, (5, 3), 'east'),
            _make_step(1, 10, 14, (6, 3), 'north', hp_delta=-4),
        ]
        result = format_replay(steps, seed=10)
        assert 'HP:14/14' in result
        assert 'HP:10/14' in result

    def test_actions_present(self):
        steps = [
            _make_step(0, 14, 14, (5, 3), 'east'),
            _make_step(1, 14, 14, (6, 3), 'north'),
            _make_step(2, 14, 14, (6, 2), 'wait'),
        ]
        result = format_replay(steps, seed=5)
        assert 'Action: east' in result
        assert 'Action: north' in result
        assert 'Action: wait' in result

    def test_gold_event(self):
        steps = [
            _make_step(0, 14, 14, (5, 3), 'east', gold_delta=3, gold=3),
        ]
        result = format_replay(steps, seed=1)
        assert 'Gold+3' in result

    def test_damage_event(self):
        steps = [
            _make_step(0, 10, 14, (5, 3), 'north', hp_delta=-4),
        ]
        result = format_replay(steps, seed=1)
        assert 'HP-4' in result

    def test_result_footer_survived(self):
        steps = [_make_step(0, 14, 14, (5, 3), 'east')]
        result = format_replay(steps, seed=42)
        assert 'Survived' in result

    def test_result_footer_died(self):
        steps = [
            _make_step(0, 0, 14, (5, 3), 'east', hp_delta=-14, survived=False),
        ]
        result = format_replay(steps, seed=42)
        assert 'Died at step 0' in result

    def test_empty_steps(self):
        result = format_replay([], seed=99)
        assert '0 steps' in result
        assert 'No steps recorded' in result

    def test_exploration_event(self):
        steps = [
            _make_step(0, 14, 14, (5, 3), 'east',
                       new_tiles=[{'tile': 'floor', 'pos': (6, 3)}]),
        ]
        result = format_replay(steps, seed=1)
        assert '+1 tile' in result

    def test_multiple_new_tiles(self):
        steps = [
            _make_step(0, 14, 14, (5, 3), 'east',
                       new_tiles=[
                           {'tile': 'floor', 'pos': (6, 3)},
                           {'tile': 'corridor', 'pos': (7, 3)},
                       ]),
        ]
        result = format_replay(steps, seed=1)
        assert '+2 tiles' in result


# ===========================================================================
# format_html_replay tests
# ===========================================================================

class TestFormatHtmlReplay:
    """Test the HTML replay formatter."""

    def test_produces_string(self):
        steps = [_make_step(0, 14, 14, (5, 3), 'east')]
        result = format_html_replay(steps, seed=42)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_has_html_tags(self):
        steps = [_make_step(0, 14, 14, (5, 3), 'east')]
        result = format_html_replay(steps, seed=42)
        assert '<html>' in result.lower()
        assert '</html>' in result.lower()
        assert '<head>' in result.lower()
        assert '</head>' in result.lower()
        assert '<body>' in result.lower()
        assert '</body>' in result.lower()

    def test_has_doctype(self):
        steps = [_make_step(0, 14, 14, (5, 3), 'east')]
        result = format_html_replay(steps, seed=42)
        assert '<!DOCTYPE html>' in result

    def test_has_title_with_seed(self):
        steps = [_make_step(0, 14, 14, (5, 3), 'east')]
        result = format_html_replay(steps, seed=42)
        assert '<title>' in result
        assert 'Seed 42' in result

    def test_has_inline_css(self):
        steps = [_make_step(0, 14, 14, (5, 3), 'east')]
        result = format_html_replay(steps, seed=42)
        assert '<style>' in result
        assert '</style>' in result

    def test_header_info(self):
        steps = [_make_step(0, 14, 14, (5, 3), 'east')]
        result = format_html_replay(steps, seed=42)
        assert '<b>Seed:</b> 42' in result
        assert '<b>Steps:</b> 1' in result

    def test_step_content(self):
        steps = [_make_step(0, 14, 14, (5, 3), 'east')]
        result = format_html_replay(steps, seed=42)
        assert 'Step 0' in result
        assert 'Action: east' in result

    def test_hp_bar_present(self):
        steps = [_make_step(0, 14, 14, (5, 3), 'east')]
        result = format_html_replay(steps, seed=42)
        assert 'hp-bar' in result
        assert 'hp-fill' in result
        assert '14/14' in result

    def test_death_class(self):
        steps = [_make_step(0, 0, 14, (5, 3), 'east', hp_delta=-14, survived=False)]
        result = format_html_replay(steps, seed=42)
        assert 'death' in result

    def test_gold_class(self):
        steps = [_make_step(0, 14, 14, (5, 3), 'east', gold_delta=5, gold=5)]
        result = format_html_replay(steps, seed=42)
        assert 'gold' in result

    def test_damage_class(self):
        steps = [_make_step(0, 10, 14, (5, 3), 'north', hp_delta=-4)]
        result = format_html_replay(steps, seed=42)
        assert 'damage' in result

    def test_explore_class(self):
        steps = [
            _make_step(0, 14, 14, (5, 3), 'east',
                       new_tiles=[{'tile': 'floor', 'pos': (6, 3)}]),
        ]
        result = format_html_replay(steps, seed=42)
        assert 'explore' in result

    def test_empty_steps_html(self):
        result = format_html_replay([], seed=99)
        assert '<b>Steps:</b> 0' in result
        assert '<html>' in result.lower()

    def test_outcome_survived(self):
        steps = [_make_step(0, 14, 14, (5, 3), 'east')]
        result = format_html_replay(steps, seed=42)
        assert 'survived' in result

    def test_outcome_died(self):
        steps = [_make_step(0, 0, 14, (5, 3), 'east', survived=False)]
        result = format_html_replay(steps, seed=42)
        assert 'died' in result


# ===========================================================================
# format_summary tests
# ===========================================================================

class TestFormatSummary:
    """Test the one-line summary formatter."""

    def test_non_empty_string(self):
        steps = [_make_step(0, 14, 14, (5, 3), 'east')]
        result = format_summary(steps, seed=42)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_single_line(self):
        steps = [_make_step(0, 14, 14, (5, 3), 'east')]
        result = format_summary(steps, seed=42)
        assert '\n' not in result

    def test_contains_seed(self):
        steps = [_make_step(0, 14, 14, (5, 3), 'east')]
        result = format_summary(steps, seed=42)
        assert 'Seed 42' in result

    def test_contains_step_count(self):
        steps = [
            _make_step(0, 14, 14, (5, 3), 'east'),
            _make_step(1, 14, 14, (6, 3), 'north'),
            _make_step(2, 12, 14, (6, 2), 'wait', hp_delta=-2),
        ]
        result = format_summary(steps, seed=10)
        assert '3 steps' in result

    def test_contains_gold(self):
        steps = [_make_step(0, 14, 14, (5, 3), 'east', gold=7)]
        result = format_summary(steps, seed=1)
        assert '7 gold' in result

    def test_contains_damage(self):
        steps = [
            _make_step(0, 10, 14, (5, 3), 'east', hp_delta=-4, gold=0),
            _make_step(1, 6, 14, (6, 3), 'north', hp_delta=-4, gold=0),
        ]
        result = format_summary(steps, seed=1)
        assert '8 damage taken' in result

    def test_contains_tiles_explored(self):
        steps = [
            _make_step(0, 14, 14, (5, 3), 'east',
                       new_tiles=[{'tile': 'floor', 'pos': (6, 3)},
                                  {'tile': 'wall', 'pos': (7, 3)}]),
            _make_step(1, 14, 14, (6, 3), 'north',
                       new_tiles=[{'tile': 'corridor', 'pos': (6, 2)}]),
        ]
        result = format_summary(steps, seed=1)
        assert '3 tiles explored' in result

    def test_survived_outcome(self):
        steps = [_make_step(0, 14, 14, (5, 3), 'east')]
        result = format_summary(steps, seed=42)
        assert 'survived' in result

    def test_died_outcome(self):
        steps = [_make_step(0, 0, 14, (5, 3), 'east', survived=False)]
        result = format_summary(steps, seed=42)
        assert 'died at step 0' in result

    def test_empty_steps(self):
        result = format_summary([], seed=99)
        assert '0 steps' in result
        assert 'no data' in result

    def test_depth(self):
        steps = [_make_step(0, 14, 14, (5, 3), 'east', depth=3)]
        result = format_summary(steps, seed=1)
        assert 'depth 3' in result


# ===========================================================================
# run_and_report tests (real NLE)
# ===========================================================================

class TestRunAndReport:
    """Test run_and_report with the real NLE environment."""

    def setup_method(self):
        self.encoder = StateEncoder()

    def test_returns_dict(self):
        result = run_and_report(seed=42, max_steps=5, encoder=self.encoder)
        assert isinstance(result, dict)

    def test_has_all_expected_keys(self):
        result = run_and_report(seed=42, max_steps=5, encoder=self.encoder)
        for key in ('seed', 'steps', 'outcome', 'total_gold', 'total_damage',
                     'tiles_explored', 'step_data'):
            assert key in result, f"Missing key: {key}"

    def test_seed_matches(self):
        result = run_and_report(seed=42, max_steps=5, encoder=self.encoder)
        assert result['seed'] == 42

    def test_steps_positive(self):
        result = run_and_report(seed=42, max_steps=10, encoder=self.encoder)
        assert result['steps'] > 0
        assert result['steps'] <= 10

    def test_outcome_value(self):
        result = run_and_report(seed=42, max_steps=10, encoder=self.encoder)
        assert result['outcome'] in ('survived', 'died')

    def test_total_gold_non_negative(self):
        result = run_and_report(seed=42, max_steps=10, encoder=self.encoder)
        assert result['total_gold'] >= 0

    def test_total_damage_non_negative(self):
        result = run_and_report(seed=42, max_steps=10, encoder=self.encoder)
        assert result['total_damage'] >= 0

    def test_tiles_explored_non_negative(self):
        result = run_and_report(seed=42, max_steps=10, encoder=self.encoder)
        assert result['tiles_explored'] >= 0

    def test_step_data_is_list(self):
        result = run_and_report(seed=42, max_steps=5, encoder=self.encoder)
        assert isinstance(result['step_data'], list)
        assert len(result['step_data']) == result['steps']

    def test_step_data_structure(self):
        result = run_and_report(seed=42, max_steps=5, encoder=self.encoder)
        for sd in result['step_data']:
            assert 'step' in sd
            assert 'obs' in sd
            assert 'obs_after' in sd
            assert 'state' in sd
            assert 'action' in sd
            assert 'delta' in sd
            assert 'prompt' in sd
            assert 'target' in sd
            # Check state sub-fields
            assert 'hp' in sd['state']
            assert 'hp_max' in sd['state']
            assert 'position' in sd['state']
            assert 'gold' in sd['state']
            # Check delta sub-fields
            assert 'hp_delta' in sd['delta']
            assert 'gold_delta' in sd['delta']
            assert 'survived' in sd['delta']

    def test_step_numbers_sequential(self):
        result = run_and_report(seed=42, max_steps=10, encoder=self.encoder)
        indices = [sd['step'] for sd in result['step_data']]
        assert indices == list(range(len(indices)))

    def test_actions_are_strings(self):
        result = run_and_report(seed=42, max_steps=5, encoder=self.encoder)
        for sd in result['step_data']:
            assert isinstance(sd['action'], str)

    def test_prompt_and_target_non_empty(self):
        result = run_and_report(seed=42, max_steps=5, encoder=self.encoder)
        for sd in result['step_data']:
            assert isinstance(sd['prompt'], str)
            assert len(sd['prompt']) > 0
            assert isinstance(sd['target'], str)
            assert len(sd['target']) > 0


# ===========================================================================
# HTML file output tests
# ===========================================================================

class TestHtmlOutput:
    """Test that HTML reports can be written to disk and are viewable."""

    def setup_method(self):
        self.encoder = StateEncoder()
        self.tmpdir = tempfile.mkdtemp()

    def test_writes_html_file(self):
        result = run_and_report(
            seed=42, max_steps=5,
            encoder=self.encoder,
            output_dir=self.tmpdir,
        )
        assert 'html_path' in result
        assert os.path.exists(result['html_path'])
        assert result['html_path'].endswith('.html')

    def test_html_file_has_content(self):
        result = run_and_report(
            seed=42, max_steps=5,
            encoder=self.encoder,
            output_dir=self.tmpdir,
        )
        with open(result['html_path']) as f:
            content = f.read()
        assert len(content) > 100  # non-trivial HTML

    def test_html_file_is_valid(self):
        result = run_and_report(
            seed=42, max_steps=5,
            encoder=self.encoder,
            output_dir=self.tmpdir,
        )
        with open(result['html_path']) as f:
            content = f.read()
        assert '<!DOCTYPE html>' in content
        assert '<html>' in content.lower()
        assert '<head>' in content.lower()
        assert '<body>' in content.lower()
        assert '</html>' in content.lower()

    def test_html_file_name_contains_seed(self):
        result = run_and_report(
            seed=42, max_steps=5,
            encoder=self.encoder,
            output_dir=self.tmpdir,
        )
        assert 'seed_42' in result['html_path']


# ===========================================================================
# Integration: format functions with real game data
# ===========================================================================

class TestIntegration:
    """Integration tests using real NLE game data."""

    def setup_method(self):
        self.encoder = StateEncoder()

    def test_format_replay_with_real_game(self):
        result = run_and_report(seed=42, max_steps=10, encoder=self.encoder)
        replay = format_replay(result['step_data'], seed=42)
        assert isinstance(replay, str)
        assert 'Seed 42' in replay
        assert 'Step' in replay
        assert 'HP:' in replay

    def test_format_html_with_real_game(self):
        result = run_and_report(seed=42, max_steps=10, encoder=self.encoder)
        html = format_html_replay(result['step_data'], seed=42)
        assert '<!DOCTYPE html>' in html
        assert 'Seed 42' in html
        assert 'Step 0' in html

    def test_format_summary_with_real_game(self):
        result = run_and_report(seed=42, max_steps=10, encoder=self.encoder)
        summary = format_summary(result['step_data'], seed=42)
        assert 'Seed 42' in summary
        assert 'steps' in summary
        assert '\n' not in summary

    def test_consistent_results(self):
        """All three formatters should agree on step count."""
        result = run_and_report(seed=42, max_steps=10, encoder=self.encoder)
        step_count = result['steps']

        replay = format_replay(result['step_data'], seed=42)
        html = format_html_replay(result['step_data'], seed=42)
        summary = format_summary(result['step_data'], seed=42)

        assert f'{step_count} steps' in replay
        assert f'<b>Steps:</b> {step_count}' in html
        assert f'{step_count} steps' in summary
