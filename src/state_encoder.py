"""
StateEncoder: Convert raw NLE observations into structured features for LLM training.

Handles three concerns:
1. Full state encoding (encode_full)
2. Delta/diff encoding between two timesteps (encode_delta)
3. Text formatting for LLM prompts and targets (format_prompt, format_target)
"""

import numpy as np
from typing import Dict, List, Tuple, Any

# blstats indices (from nle.nethack constants)
BL_X = 0
BL_Y = 1
BL_STR = 3      # STR125
BL_DEX = 4
BL_HP = 10
BL_HPMAX = 11
BL_DEPTH = 12
BL_GOLD = 13
BL_AC = 16
BL_XP = 18      # experience level
BL_TIME = 20    # turn counter

# Tile name mapping: ASCII code -> human-readable name
TILE_NAMES = {
    ord('.'): 'floor',
    ord('#'): 'corridor',
    ord(' '): 'unseen',
    ord('-'): 'wall',
    ord('|'): 'wall',
    ord('+'): 'door',
    ord('@'): 'player',
    ord('$'): 'gold',
    ord('?'): 'scroll',
    ord('!'): 'potion',
    ord('/'): 'wand',
    ord('='): 'ring',
    ord('*'): 'gem',
    ord('"'): 'amulet',
    ord('('): 'tool',
    ord(')'): 'weapon',
    ord('['): 'armor',
    ord('%'): 'food',
    ord('^'): 'trap',
    ord('>'): 'stairs_down',
    ord('<'): 'stairs_up',
    ord('\\'): 'throne',
    ord('_'): 'altar',
    ord('{'): 'fountain',
    ord('}'): 'water',
    ord('`'): 'boulder',
    ord('0'): 'iron_ball',
    ord('I'): 'remembered_unseen',
    ord('8'): 'fence',
}

# Direction offsets (dy, dx)
DIRECTIONS = {
    'north': (-1, 0),
    'south': (1, 0),
    'east': (0, 1),
    'west': (0, -1),
}


def _is_monster_char(c: int) -> bool:
    """Check if an ASCII code represents a monster character."""
    if c < 33 or c > 126:
        return False
    ch = chr(c)
    # Lowercase and uppercase letters (except @ which is the player,
    # and some special letters). In NetHack monsters are letters
    # a-z (pets/peaceful/hostile) and A-Z (uppercase monsters).
    # Exclude '@' (player) which is handled separately.
    return ch.isalpha() and ch != '@' and ch != 'I'


def _tile_name(c: int) -> str:
    """Convert an ASCII char code to a tile name."""
    if c == ord('@'):
        return 'player'
    if _is_monster_char(c):
        return f'monster_{chr(c)}'
    if c in TILE_NAMES:
        return TILE_NAMES[c]
    return f'unknown_{chr(c)}' if 33 <= c <= 126 else 'unseen'


def _decode_message(msg_array: np.ndarray) -> str:
    """Decode the NLE message uint8 array to a string."""
    raw = bytes(msg_array).decode('ascii', errors='replace')
    return raw.strip().rstrip('\x00').strip()


def _find_player(chars: np.ndarray) -> Tuple[int, int]:
    """Find player position by scanning for '@'. Returns (x, y)."""
    positions = np.argwhere(chars == ord('@'))
    if len(positions) == 0:
        # Fallback to blstats
        return -1, -1
    y, x = positions[0]
    return int(x), int(y)


class StateEncoder:
    """Encode NLE observations into structured feature dicts for LLM training."""

    def encode_full(self, obs: dict) -> dict:
        """Extract structured features from an NLE observation dict.

        Args:
            obs: NLE observation with keys 'chars', 'blstats', 'message'
                 (all numpy arrays).

        Returns:
            Dict with keys:
                position: (x, y) tuple
                hp, hp_max, ac, strength, dexterity, gold, depth, turn: ints
                adjacent: dict of north/south/east/west -> tile name
                visible_monsters: list of {char, pos} dicts
                visible_items: list of {type, pos} dicts
                message: string
        """
        chars = obs['chars']
        blstats = obs['blstats']
        msg_array = obs['message']

        px, py = _find_player(chars)

        # Stats from blstats
        hp = int(blstats[BL_HP])
        hp_max = int(blstats[BL_HPMAX])
        ac = int(blstats[BL_AC])
        strength = int(blstats[BL_STR])
        dexterity = int(blstats[BL_DEX])
        gold = int(blstats[BL_GOLD])
        depth = int(blstats[BL_DEPTH])
        turn = int(blstats[BL_TIME])

        # Adjacent tiles
        adjacent = {}
        for dir_name, (dy, dx) in DIRECTIONS.items():
            ny, nx = py + dy, px + dx
            if 0 <= ny < chars.shape[0] and 0 <= nx < chars.shape[1]:
                adjacent[dir_name] = _tile_name(int(chars[ny, nx]))
            else:
                adjacent[dir_name] = 'unseen'

        # Visible monsters (scan entire chars grid)
        visible_monsters = []
        for y in range(chars.shape[0]):
            for x in range(chars.shape[1]):
                c = int(chars[y, x])
                if _is_monster_char(c):
                    visible_monsters.append({
                        'char': chr(c),
                        'pos': (x, y),
                    })

        # Visible items
        # Item codes: pickupable/interactable objects only.
        # Architectural features (doors '+', stairs '>', '<') are NOT items.
        item_codes = {
            ord('$'): 'gold',
            ord('?'): 'scroll',
            ord('!'): 'potion',
            ord('/'): 'wand',
            ord('='): 'ring',
            ord('*'): 'gem',
            ord('"'): 'amulet',
            ord('('): 'tool',
            ord(')'): 'weapon',
            ord('['): 'armor',
            ord('%'): 'food',
        }
        visible_items = []
        for y in range(chars.shape[0]):
            for x in range(chars.shape[1]):
                c = int(chars[y, x])
                if c in item_codes:
                    visible_items.append({
                        'type': item_codes[c],
                        'pos': (x, y),
                    })

        message = _decode_message(msg_array)

        return {
            'position': (px, py),
            'hp': hp,
            'hp_max': hp_max,
            'ac': ac,
            'strength': strength,
            'dexterity': dexterity,
            'gold': gold,
            'depth': depth,
            'turn': turn,
            'adjacent': adjacent,
            'visible_monsters': visible_monsters,
            'visible_items': visible_items,
            'message': message,
        }

    def encode_delta(self, obs_before: dict, obs_after: dict, action: str) -> dict:
        """Compute what changed between two observations.

        Args:
            obs_before: NLE observation before the action.
            obs_after: NLE observation after the action.
            action: The action name that was taken.

        Returns:
            Dict with keys:
                pos_delta: (dx, dy) tuple
                hp_delta, gold_delta, depth_delta, turn_delta: ints
                new_tiles: list of tile descriptions that appeared from darkness
                message: string (from obs_after)
                survived: bool (True if HP > 0 in obs_after)
        """
        chars_before = obs_before['chars']
        chars_after = obs_after['chars']
        bl_before = obs_before['blstats']
        bl_after = obs_after['blstats']

        # Position deltas
        px_before, py_before = _find_player(chars_before)
        px_after, py_after = _find_player(chars_after)
        pos_delta = (px_after - px_before, py_after - py_before)

        # Stat deltas
        hp_delta = int(bl_after[BL_HP]) - int(bl_before[BL_HP])
        gold_delta = int(bl_after[BL_GOLD]) - int(bl_before[BL_GOLD])
        depth_delta = int(bl_after[BL_DEPTH]) - int(bl_before[BL_DEPTH])
        turn_delta = int(bl_after[BL_TIME]) - int(bl_before[BL_TIME])

        # New tiles: tiles that were unseen (space) before but now visible
        new_tiles = []
        for y in range(chars_after.shape[0]):
            for x in range(chars_after.shape[1]):
                c_before = int(chars_before[y, x])
                c_after = int(chars_after[y, x])
                if c_before == ord(' ') and c_after != ord(' '):
                    new_tiles.append({
                        'tile': _tile_name(c_after),
                        'pos': (x, y),
                    })

        message = _decode_message(obs_after['message'])
        survived = int(bl_after[BL_HP]) > 0

        return {
            'pos_delta': pos_delta,
            'hp_delta': hp_delta,
            'gold_delta': gold_delta,
            'depth_delta': depth_delta,
            'turn_delta': turn_delta,
            'new_tiles': new_tiles,
            'message': message,
            'survived': survived,
        }

    def format_prompt(self, state: dict, action: str) -> str:
        """Format state + action as a compact text prompt for the LLM.

        Args:
            state: Output of encode_full().
            action: The action name being considered/taken.

        Returns:
            Multi-line string prompt.
        """
        pos = state['position']
        adjacent_parts = []
        for d in ('north', 'south', 'east', 'west'):
            adjacent_parts.append(f"{d}={state['adjacent'].get(d, 'unknown')}")

        # Monsters: compact format like "f@(4,3)"
        monster_parts = []
        for m in state.get('visible_monsters', []):
            monster_parts.append(f"{m['char']}@{m['pos']}")
        monsters_str = ' '.join(monster_parts) if monster_parts else 'none'

        # Items: compact format like "gold@(5,6) scroll@(3,1)"
        item_parts = []
        for it in state.get('visible_items', []):
            item_parts.append(f"{it['type']}@{it['pos']}")
        items_str = ' '.join(item_parts) if item_parts else 'none'

        lines = [
            f"HP:{state['hp']}/{state['hp_max']} AC:{state['ac']} "
            f"Str:{state['strength']} Dex:{state['dexterity']}",
            f"Pos:{pos} Gold:{state['gold']} Depth:{state['depth']} "
            f"Turn:{state['turn']}",
            f"Adjacent: {' '.join(adjacent_parts)}",
            f"Monsters: {monsters_str}",
            f"Items: {items_str}",
            f"Action: {action}",
        ]
        return '\n'.join(lines)

    def format_target(self, delta: dict) -> str:
        """Format delta as the target prediction string.

        Args:
            delta: Output of encode_delta().

        Returns:
            Compact single-line string like:
            "pos:(-1,0) | hp:same | gold:same | msg:It's a wall."
        """
        dx, dy = delta['pos_delta']
        pos_str = f"pos:({dx},{dy})"

        hp_d = delta['hp_delta']
        hp_str = f"hp:{'same' if hp_d == 0 else ('+' + str(hp_d) if hp_d > 0 else str(hp_d))}"

        gold_d = delta['gold_delta']
        gold_str = f"gold:{'same' if gold_d == 0 else ('+' + str(gold_d) if gold_d > 0 else str(gold_d))}"

        depth_d = delta['depth_delta']
        depth_str = f"depth:{'same' if depth_d == 0 else ('+' + str(depth_d) if depth_d > 0 else str(depth_d))}"

        msg = delta['message']
        # Truncate long messages for compactness
        if len(msg) > 60:
            msg = msg[:57] + '...'
        msg_str = f"msg:{msg}" if msg else "msg:"

        survived_str = f"alive:{'yes' if delta['survived'] else 'no'}"

        return f"{pos_str} | {hp_str} | {gold_str} | {depth_str} | {survived_str} | {msg_str}"
