"""
Comprehensive tests for StateEncoder module.

Tests cover:
- Unit tests with fake (constructed) NLE observations
- Integration tests with real NLE environment
- Determinism: same obs -> same output
- format_prompt / format_target output validation
- Delta detection: position, HP, gold changes
"""

import sys
import os
import pytest
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.state_encoder import StateEncoder, _tile_name, _find_player, _is_monster_char

# ---------------------------------------------------------------------------
# Helpers for constructing fake NLE observations
# ---------------------------------------------------------------------------

def make_fake_obs(
    chars_2d=None,
    blstats=None,
    message=b'\x00' * 256,
    chars_shape=(21, 79),
):
    """Build a fake NLE observation dict with sensible defaults."""
    if chars_2d is None:
        # Default: all unseen (spaces)
        chars_2d = np.full(chars_shape, ord(' '), dtype=np.uint8)
        # Place a small room with the player
        # Row 10: wall-wall-wall-door-wall-wall
        # Row 11: wall-floor-floor-player-floor-wall
        # Row 12: wall-wall-wall-wall-wall-wall
        for x in range(30, 36):
            chars_2d[10][x] = ord('-')
            chars_2d[12][x] = ord('-')
        chars_2d[11][30] = ord('|')
        chars_2d[11][35] = ord('|')
        chars_2d[11][31] = ord('.')
        chars_2d[11][32] = ord('.')
        chars_2d[11][33] = ord('@')  # player at (33, 11)
        chars_2d[11][34] = ord('.')
        chars_2d[10][33] = ord('+')  # door north of player

    if blstats is None:
        # Default blstats: player at x=33, y=11
        blstats = np.zeros(27, dtype=np.int64)
        blstats[0] = 33   # x
        blstats[1] = 11   # y
        blstats[3] = 16   # str
        blstats[4] = 14   # dex
        blstats[10] = 14  # hp
        blstats[11] = 14  # hp_max
        blstats[12] = 1   # depth
        blstats[13] = 0   # gold
        blstats[16] = 5   # ac
        blstats[18] = 1   # xp level
        blstats[20] = 1   # turn

    if isinstance(message, str):
        msg_bytes = message.encode('ascii') + b'\x00' * 256
        message = np.frombuffer(msg_bytes[:256], dtype=np.uint8).copy()
    elif isinstance(message, bytes):
        padded = message + b'\x00' * 256
        message = np.frombuffer(padded[:256], dtype=np.uint8).copy()
    else:
        message = np.array(message, dtype=np.uint8)

    return {
        'chars': np.array(chars_2d, dtype=np.uint8),
        'blstats': np.array(blstats, dtype=np.int64),
        'message': message,
    }


# ===========================================================================
# Unit tests with fake observations
# ===========================================================================

class TestTileName:
    """Test the _tile_name helper."""

    def test_floor(self):
        assert _tile_name(ord('.')) == 'floor'

    def test_wall_dash(self):
        assert _tile_name(ord('-')) == 'wall'

    def test_wall_pipe(self):
        assert _tile_name(ord('|')) == 'wall'

    def test_door(self):
        assert _tile_name(ord('+')) == 'door'

    def test_corridor(self):
        assert _tile_name(ord('#')) == 'corridor'

    def test_player(self):
        assert _tile_name(ord('@')) == 'player'

    def test_gold(self):
        assert _tile_name(ord('$')) == 'gold'

    def test_scroll(self):
        assert _tile_name(ord('?')) == 'scroll'

    def test_potion(self):
        assert _tile_name(ord('!')) == 'potion'

    def test_food(self):
        assert _tile_name(ord('%')) == 'food'

    def test_armor(self):
        assert _tile_name(ord('[')) == 'armor'

    def test_weapon(self):
        assert _tile_name(ord(')')) == 'weapon'

    def test_monster_lowercase(self):
        assert _tile_name(ord('d')) == 'monster_d'

    def test_monster_uppercase(self):
        assert _tile_name(ord('F')) == 'monster_F'

    def test_unseen(self):
        assert _tile_name(ord(' ')) == 'unseen'

    def test_stairs_down(self):
        assert _tile_name(ord('>')) == 'stairs_down'

    def test_stairs_up(self):
        assert _tile_name(ord('<')) == 'stairs_up'


class TestIsMonsterChar:
    def test_lowercase_monster(self):
        assert _is_monster_char(ord('d')) is True

    def test_uppercase_monster(self):
        assert _is_monster_char(ord('F')) is True

    def test_player_not_monster(self):
        assert _is_monster_char(ord('@')) is False

    def test_I_not_monster(self):
        assert _is_monster_char(ord('I')) is False

    def test_floor_not_monster(self):
        assert _is_monster_char(ord('.')) is False

    def test_space_not_monster(self):
        assert _is_monster_char(ord(' ')) is False


class TestFindPlayer:
    def test_finds_at_sign(self):
        chars = np.full((21, 79), ord(' '), dtype=np.uint8)
        chars[5][10] = ord('@')
        assert _find_player(chars) == (10, 5)

    def test_no_player(self):
        chars = np.full((21, 79), ord('.'), dtype=np.uint8)
        assert _find_player(chars) == (-1, -1)


class TestEncodeFull:
    """Test encode_full with constructed observations."""

    def setup_method(self):
        self.enc = StateEncoder()

    def test_basic_fields(self):
        obs = make_fake_obs()
        state = self.enc.encode_full(obs)
        assert state['position'] == (33, 11)
        assert state['hp'] == 14
        assert state['hp_max'] == 14
        assert state['ac'] == 5
        assert state['strength'] == 16
        assert state['dexterity'] == 14
        assert state['gold'] == 0
        assert state['depth'] == 1
        assert state['turn'] == 1

    def test_adjacent_tiles(self):
        obs = make_fake_obs()
        state = self.enc.encode_full(obs)
        adj = state['adjacent']
        assert 'north' in adj
        assert 'south' in adj
        assert 'east' in adj
        assert 'west' in adj
        # Player at (33, 11): north is row 10 col 33 = '+'
        assert adj['north'] == 'door'
        # south is row 12 col 33 = '-'
        assert adj['south'] == 'wall'
        # east is row 11 col 34 = '.'
        assert adj['east'] == 'floor'
        # west is row 11 col 32 = '.'
        assert adj['west'] == 'floor'

    def test_visible_monsters(self):
        obs = make_fake_obs()
        chars = obs['chars']
        # Place a monster (cat 'f') at (34, 11)
        chars[11][34] = ord('f')
        state = self.enc.encode_full(obs)
        monsters = state['visible_monsters']
        assert len(monsters) == 1
        assert monsters[0]['char'] == 'f'
        assert monsters[0]['pos'] == (34, 11)

    def test_no_monsters(self):
        obs = make_fake_obs()
        state = self.enc.encode_full(obs)
        assert state['visible_monsters'] == []

    def test_visible_items(self):
        obs = make_fake_obs()
        chars = obs['chars']
        # Place gold at (31, 11) and a scroll at (34, 10)
        chars[11][31] = ord('$')
        chars[10][34] = ord('?')
        state = self.enc.encode_full(obs)
        items = state['visible_items']
        types = {it['type'] for it in items}
        assert 'gold' in types
        assert 'scroll' in types

    def test_message_decoding(self):
        obs = make_fake_obs(message='Hello NetHack!')
        state = self.enc.encode_full(obs)
        assert state['message'] == 'Hello NetHack!'

    def test_empty_message(self):
        obs = make_fake_obs(message=b'\x00' * 256)
        state = self.enc.encode_full(obs)
        assert state['message'] == ''

    def test_custom_blstats(self):
        bl = np.zeros(27, dtype=np.int64)
        bl[3] = 18   # str
        bl[4] = 12   # dex
        bl[10] = 10  # hp
        bl[11] = 20  # hp_max
        bl[12] = 3   # depth
        bl[13] = 50  # gold
        bl[16] = 2   # ac
        bl[20] = 100 # turn
        obs = make_fake_obs(blstats=bl)
        state = self.enc.encode_full(obs)
        assert state['strength'] == 18
        assert state['dexterity'] == 12
        assert state['hp'] == 10
        assert state['hp_max'] == 20
        assert state['depth'] == 3
        assert state['gold'] == 50
        assert state['ac'] == 2
        assert state['turn'] == 100

    def test_determinism(self):
        """Same observation -> same output, twice."""
        obs = make_fake_obs()
        enc = StateEncoder()
        state1 = enc.encode_full(obs)
        state2 = enc.encode_full(obs)
        assert state1 == state2

    def test_player_position_from_grid(self):
        """Position should come from scanning chars grid, not blstats."""
        chars = np.full((21, 79), ord('.'), dtype=np.uint8)
        chars[7][20] = ord('@')
        bl = np.zeros(27, dtype=np.int64)
        bl[0] = 0  # intentionally wrong x
        bl[1] = 0  # intentionally wrong y
        bl[10] = 10
        bl[11] = 10
        obs = make_fake_obs(chars_2d=chars, blstats=bl)
        state = self.enc.encode_full(obs)
        assert state['position'] == (20, 7)


class TestEncodeDelta:
    """Test encode_delta with constructed observations."""

    def setup_method(self):
        self.enc = StateEncoder()

    def test_no_change(self):
        obs = make_fake_obs()
        delta = self.enc.encode_delta(obs, obs, 'wait')
        assert delta['pos_delta'] == (0, 0)
        assert delta['hp_delta'] == 0
        assert delta['gold_delta'] == 0
        assert delta['depth_delta'] == 0
        assert delta['turn_delta'] == 0
        assert delta['survived'] is True
        assert delta['new_tiles'] == []

    def test_position_change(self):
        obs_before = make_fake_obs()

        # After: player moved east (from col 33 to 34)
        chars_after = obs_before['chars'].copy()
        chars_after[11][33] = ord('.')   # old pos becomes floor
        chars_after[11][34] = ord('@')   # new pos

        bl_after = obs_before['blstats'].copy()
        bl_after[0] = 34
        bl_after[20] = 2  # turn incremented

        obs_after = make_fake_obs(
            chars_2d=chars_after,
            blstats=bl_after,
        )
        delta = self.enc.encode_delta(obs_before, obs_after, 'east')
        assert delta['pos_delta'] == (1, 0)  # dx=+1, dy=0
        assert delta['turn_delta'] == 1

    def test_hp_change(self):
        obs_before = make_fake_obs()
        bl_after = obs_before['blstats'].copy()
        bl_after[BL_HP] = 10  # was 14

        # Copy chars with player still in same position
        obs_after = make_fake_obs(blstats=bl_after)
        obs_after['chars'] = obs_before['chars'].copy()

        delta = self.enc.encode_delta(obs_before, obs_after, 'wait')
        assert delta['hp_delta'] == -4
        assert delta['survived'] is True

    def test_gold_change(self):
        obs_before = make_fake_obs()
        bl_after = obs_before['blstats'].copy()
        bl_after[BL_GOLD] = 25  # was 0

        obs_after = make_fake_obs(blstats=bl_after)
        obs_after['chars'] = obs_before['chars'].copy()

        delta = self.enc.encode_delta(obs_before, obs_after, 'pickup')
        assert delta['gold_delta'] == 25

    def test_death(self):
        obs_before = make_fake_obs()
        bl_after = obs_before['blstats'].copy()
        bl_after[BL_HP] = 0

        obs_after = make_fake_obs(blstats=bl_after)
        obs_after['chars'] = obs_before['chars'].copy()

        delta = self.enc.encode_delta(obs_before, obs_after, 'west')
        assert delta['hp_delta'] == -14
        assert delta['survived'] is False

    def test_new_tiles(self):
        obs_before = make_fake_obs()
        chars_after = obs_before['chars'].copy()
        # Reveal a previously unseen tile
        assert chars_after[0][0] == ord(' ')
        chars_after[0][0] = ord('.')
        chars_after[0][1] = ord('#')

        obs_after = make_fake_obs(chars_2d=chars_after)
        obs_after['blstats'] = obs_before['blstats'].copy()
        obs_after['message'] = obs_before['message'].copy()

        delta = self.enc.encode_delta(obs_before, obs_after, 'search')
        assert len(delta['new_tiles']) >= 2
        tile_names = {t['tile'] for t in delta['new_tiles']}
        assert 'floor' in tile_names
        assert 'corridor' in tile_names

    def test_depth_change(self):
        obs_before = make_fake_obs()
        bl_after = obs_before['blstats'].copy()
        bl_after[BL_DEPTH] = 2

        obs_after = make_fake_obs(blstats=bl_after)
        obs_after['chars'] = obs_before['chars'].copy()

        delta = self.enc.encode_delta(obs_before, obs_after, 'down')
        assert delta['depth_delta'] == 1

    def test_message_in_delta(self):
        obs_before = make_fake_obs(message='You see here a scroll.')
        obs_after = make_fake_obs(message='You picked up a scroll.')
        obs_after['chars'] = obs_before['chars'].copy()
        obs_after['blstats'] = obs_before['blstats'].copy()

        delta = self.enc.encode_delta(obs_before, obs_after, 'pickup')
        assert delta['message'] == 'You picked up a scroll.'


# Need BL constants for test usage
from src.state_encoder import BL_HP, BL_GOLD, BL_DEPTH


class TestFormatPrompt:
    """Test format_prompt output."""

    def setup_method(self):
        self.enc = StateEncoder()

    def test_basic_output(self):
        obs = make_fake_obs()
        state = self.enc.encode_full(obs)
        prompt = self.enc.format_prompt(state, 'north')
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert 'HP:14/14' in prompt
        assert 'AC:5' in prompt
        assert 'Str:16' in prompt
        assert 'Dex:14' in prompt
        assert 'Action: north' in prompt
        assert 'Pos:(33, 11)' in prompt

    def test_monsters_in_prompt(self):
        obs = make_fake_obs()
        obs['chars'][11][34] = ord('f')  # place a cat
        state = self.enc.encode_full(obs)
        prompt = self.enc.format_prompt(state, 'east')
        assert 'Monsters:' in prompt
        assert 'f@' in prompt

    def test_items_in_prompt(self):
        obs = make_fake_obs()
        obs['chars'][11][31] = ord('$')  # gold
        state = self.enc.encode_full(obs)
        prompt = self.enc.format_prompt(state, 'wait')
        assert 'Items:' in prompt
        assert 'gold@' in prompt

    def test_no_monsters_shows_none(self):
        obs = make_fake_obs()
        state = self.enc.encode_full(obs)
        prompt = self.enc.format_prompt(state, 'wait')
        assert 'Monsters: none' in prompt

    def test_no_items_shows_none(self):
        obs = make_fake_obs()
        state = self.enc.encode_full(obs)
        prompt = self.enc.format_prompt(state, 'wait')
        assert 'Items: none' in prompt

    def test_adjacent_in_prompt(self):
        obs = make_fake_obs()
        state = self.enc.encode_full(obs)
        prompt = self.enc.format_prompt(state, 'north')
        assert 'Adjacent:' in prompt
        assert 'north=door' in prompt
        assert 'south=wall' in prompt
        assert 'east=floor' in prompt
        assert 'west=floor' in prompt

    def test_gold_depth_turn(self):
        bl = np.zeros(27, dtype=np.int64)
        bl[3] = 18
        bl[4] = 12
        bl[10] = 10
        bl[11] = 20
        bl[12] = 3
        bl[13] = 50
        bl[16] = 2
        bl[20] = 100
        obs = make_fake_obs(blstats=bl)
        state = self.enc.encode_full(obs)
        prompt = self.enc.format_prompt(state, 'wait')
        assert 'Gold:50' in prompt
        assert 'Depth:3' in prompt
        assert 'Turn:100' in prompt


class TestFormatTarget:
    """Test format_target output."""

    def setup_method(self):
        self.enc = StateEncoder()

    def test_no_change(self):
        delta = {
            'pos_delta': (0, 0),
            'hp_delta': 0,
            'gold_delta': 0,
            'depth_delta': 0,
            'turn_delta': 1,
            'new_tiles': [],
            'message': '',
            'survived': True,
        }
        result = self.enc.format_target(delta)
        assert 'pos:(0,0)' in result
        assert 'hp:same' in result
        assert 'gold:same' in result
        assert 'depth:same' in result
        assert 'alive:yes' in result

    def test_positive_deltas(self):
        delta = {
            'pos_delta': (1, -1),
            'hp_delta': 5,
            'gold_delta': 10,
            'depth_delta': 0,
            'turn_delta': 1,
            'new_tiles': [],
            'message': 'You feel stronger!',
            'survived': True,
        }
        result = self.enc.format_target(delta)
        assert 'pos:(1,-1)' in result
        assert 'hp:+5' in result
        assert 'gold:+10' in result
        assert 'depth:same' in result
        assert "msg:You feel stronger!" in result

    def test_negative_deltas(self):
        delta = {
            'pos_delta': (0, 0),
            'hp_delta': -3,
            'gold_delta': -5,
            'depth_delta': -1,
            'turn_delta': 1,
            'new_tiles': [],
            'message': 'Ouch!',
            'survived': True,
        }
        result = self.enc.format_target(delta)
        assert 'hp:-3' in result
        assert 'gold:-5' in result
        assert 'depth:-1' in result

    def test_death(self):
        delta = {
            'pos_delta': (0, 0),
            'hp_delta': -14,
            'gold_delta': 0,
            'depth_delta': 0,
            'turn_delta': 0,
            'new_tiles': [],
            'message': 'You die...',
            'survived': False,
        }
        result = self.enc.format_target(delta)
        assert 'alive:no' in result
        assert 'hp:-14' in result

    def test_empty_message(self):
        delta = {
            'pos_delta': (0, 0),
            'hp_delta': 0,
            'gold_delta': 0,
            'depth_delta': 0,
            'turn_delta': 1,
            'new_tiles': [],
            'message': '',
            'survived': True,
        }
        result = self.enc.format_target(delta)
        assert 'msg:' in result

    def test_long_message_truncated(self):
        delta = {
            'pos_delta': (0, 0),
            'hp_delta': 0,
            'gold_delta': 0,
            'depth_delta': 0,
            'turn_delta': 1,
            'new_tiles': [],
            'message': 'A' * 100,
            'survived': True,
        }
        result = self.enc.format_target(delta)
        assert '...' in result
        # Should be truncated, not full 100 chars
        assert len(result) < 200

    def test_format_target_is_single_line(self):
        delta = {
            'pos_delta': (1, 0),
            'hp_delta': 0,
            'gold_delta': 5,
            'depth_delta': 0,
            'turn_delta': 1,
            'new_tiles': [],
            'message': 'test',
            'survived': True,
        }
        result = self.enc.format_target(delta)
        assert '\n' not in result


# ===========================================================================
# Integration tests with real NLE
# ===========================================================================

def _nle_available():
    """Check if NLE is importable."""
    try:
        import nle.env
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _nle_available(), reason="NLE not installed")
class TestWithRealNLE:
    """Integration tests using a real NLE environment."""

    def setup_method(self):
        self.enc = StateEncoder()

    @pytest.fixture(autouse=True)
    def _env(self):
        import nle.env
        self.env = nle.env.NLE()
        self.obs, self.info = self.env.reset(seed=42)
        yield
        self.env.close()

    def test_encode_full_returns_all_keys(self):
        state = self.enc.encode_full(self.obs)
        expected_keys = {
            'position', 'hp', 'hp_max', 'ac', 'strength', 'dexterity',
            'gold', 'depth', 'turn', 'adjacent', 'visible_monsters',
            'visible_items', 'message',
        }
        assert set(state.keys()) == expected_keys

    def test_position_is_valid(self):
        state = self.enc.encode_full(self.obs)
        px, py = state['position']
        assert 0 <= px < 79
        assert 0 <= py < 21
        # Verify player is actually at that position
        assert self.obs['chars'][py][px] == ord('@')

    def test_hp_positive(self):
        state = self.enc.encode_full(self.obs)
        assert state['hp'] > 0
        assert state['hp_max'] > 0
        assert state['hp'] <= state['hp_max']

    def test_depth_positive(self):
        state = self.enc.encode_full(self.obs)
        assert state['depth'] >= 1

    def test_adjacent_four_directions(self):
        state = self.enc.encode_full(self.obs)
        assert set(state['adjacent'].keys()) == {'north', 'south', 'east', 'west'}

    def test_format_prompt_with_real_obs(self):
        state = self.enc.encode_full(self.obs)
        prompt = self.enc.format_prompt(state, 'north')
        assert len(prompt) > 50
        assert 'HP:' in prompt
        assert 'Action: north' in prompt

    def test_delta_after_step(self):
        """Take a step and verify delta detection works."""
        # NLE uses action indices into env.actions, not raw ASCII
        # Index 18 = MiscDirection.WAIT (ord('.') = 46)
        wait_idx = 18
        obs_after, reward, terminated, truncated, info = self.env.step(wait_idx)

        delta = self.enc.encode_delta(self.obs, obs_after, 'wait')

        assert isinstance(delta['pos_delta'], tuple)
        assert len(delta['pos_delta']) == 2
        assert isinstance(delta['hp_delta'], int)
        assert isinstance(delta['gold_delta'], int)
        assert isinstance(delta['survived'], bool)

    def test_delta_after_multiple_steps(self):
        """Take several steps and verify delta accumulates correctly."""
        obs_before = self.obs.copy() if hasattr(self.obs, 'copy') else self.obs
        obs = self.obs
        # Index 0 = CompassDirection.N (north), 18 = WAIT
        for _ in range(5):
            obs_next, _, term, trunc, _ = self.env.step(18)  # wait
            if term or trunc:
                break
            obs = obs_next

        delta = self.enc.encode_delta(self.obs, obs, 'wait')
        # Turn may or may not advance on waits, but the delta should be valid
        assert isinstance(delta['turn_delta'], int)
        assert isinstance(delta['hp_delta'], int)

    def test_format_target_after_step(self):
        obs_after, _, _, _, _ = self.env.step(18)  # wait
        delta = self.enc.encode_delta(self.obs, obs_after, 'wait')
        target = self.enc.format_target(delta)
        assert isinstance(target, str)
        assert len(target) > 0
        assert 'pos:' in target
        assert 'hp:' in target

    def test_determinism_real_env(self):
        """Same real obs -> same encode_full output."""
        state1 = self.enc.encode_full(self.obs)
        state2 = self.enc.encode_full(self.obs)
        assert state1 == state2

    def test_pet_detected_as_monster(self):
        """NLE games start with a pet (usually 'd' for dog or 'f' for cat).
        Check if it's detected as a visible monster."""
        state = self.enc.encode_full(self.obs)
        # The pet may or may not be in view initially, but the test
        # shouldn't fail if it's not -- just check the structure.
        for m in state['visible_monsters']:
            assert 'char' in m
            assert 'pos' in m
            assert isinstance(m['char'], str)
            assert len(m['char']) == 1
            assert isinstance(m['pos'], tuple)
            assert len(m['pos']) == 2

    def test_items_structure(self):
        """Check visible_items structure with real obs."""
        state = self.enc.encode_full(self.obs)
        for item in state['visible_items']:
            assert 'type' in item
            assert 'pos' in item
            assert isinstance(item['type'], str)
            assert isinstance(item['pos'], tuple)

    def test_message_decoded(self):
        """Message from real env should be a non-empty string on game start."""
        state = self.enc.encode_full(self.obs)
        # NetHack usually shows a message on game start
        assert isinstance(state['message'], str)
        # It may or may not be empty depending on the seed, but should not crash


# ===========================================================================
# Edge case tests
# ===========================================================================

class TestEdgeCases:
    """Edge case handling."""

    def setup_method(self):
        self.enc = StateEncoder()

    def test_player_at_grid_edge(self):
        """Player at position (0, 0) -- adjacent tiles may be out of bounds."""
        chars = np.full((21, 79), ord(' '), dtype=np.uint8)
        chars[0][0] = ord('@')
        bl = np.zeros(27, dtype=np.int64)
        bl[10] = 10
        bl[11] = 10
        obs = make_fake_obs(chars_2d=chars, blstats=bl)
        state = self.enc.encode_full(obs)
        assert state['position'] == (0, 0)
        # North and west are out of bounds
        assert state['adjacent']['north'] == 'unseen'
        assert state['adjacent']['west'] == 'unseen'

    def test_player_at_bottom_right(self):
        """Player at (78, 20) -- east and south out of bounds."""
        chars = np.full((21, 79), ord(' '), dtype=np.uint8)
        chars[20][78] = ord('@')
        bl = np.zeros(27, dtype=np.int64)
        bl[10] = 10
        bl[11] = 10
        obs = make_fake_obs(chars_2d=chars, blstats=bl)
        state = self.enc.encode_full(obs)
        assert state['position'] == (78, 20)
        assert state['adjacent']['south'] == 'unseen'
        assert state['adjacent']['east'] == 'unseen'

    def test_multiple_monsters(self):
        chars = np.full((21, 79), ord('.'), dtype=np.uint8)
        chars[5][10] = ord('@')
        chars[5][11] = ord('d')  # dog
        chars[3][10] = ord('F')  # fungus
        chars[7][15] = ord('r')  # rat
        bl = np.zeros(27, dtype=np.int64)
        bl[10] = 10
        bl[11] = 10
        obs = make_fake_obs(chars_2d=chars, blstats=bl)
        state = self.enc.encode_full(obs)
        assert len(state['visible_monsters']) == 3
        chars_set = {m['char'] for m in state['visible_monsters']}
        assert chars_set == {'d', 'F', 'r'}

    def test_zero_hp_survived_false(self):
        obs = make_fake_obs()
        bl = obs['blstats'].copy()
        bl[BL_HP] = 0
        obs_after = make_fake_obs(blstats=bl)
        obs_after['chars'] = obs['chars'].copy()
        delta = self.enc.encode_delta(obs, obs_after, 'wait')
        assert delta['survived'] is False
