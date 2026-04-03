"""
Memory-Augmented Forward Model Training Data.

This module implements a memory tracker that accumulates explored map data,
items, and monsters across timesteps, and produces enriched training pairs
for a forward model that predicts game dynamics from exploration history.

Training pair format:
  PROMPT:  [explored map summary] + [current observation] + [action]
  TARGET:  [delta: hp/gold/depth/msg/encounters]

The key insight: predictions that require MEMORY of past observations
are non-trivial and teach the model game dynamics, not just physics.
  - "I saw a goblin in room 3 eight turns ago, I'm heading there now"
  - "I left a healing potion on the floor near the stairs"
  - "The unexplored door is probably a closet (60% chance)"
"""

import numpy as np
from collections import defaultdict
from typing import Optional, Tuple, List, Dict


# Tile categories for the explored map
UNSEEN = 0
FLOOR = 1
WALL = 2
CORRIDOR = 3
DOOR_OPEN = 4
DOOR_CLOSED = 5
STAIRS_UP = 6
STAIRS_DOWN = 7
FOUNTAIN = 8
SINK = 9
THRONE = 10
ALTAR = 11
TRAP = 12
SHOP = 13
WATER = 14
LAVA = 15
ICE = 16

CHAR_TO_TILE = {
    ord('.'): FLOOR,
    ord('#'): CORRIDOR,
    ord('-'): WALL,
    ord('|'): WALL,
    ord('+'): DOOR_CLOSED,
    ord('>'): STAIRS_DOWN,
    ord('<'): STAIRS_UP,
    ord('{'): FOUNTAIN,
    ord(')'): SINK,
    ord('\\'): THRONE,
    ord('_'): ALTAR,
    ord('^'): TRAP,
    ord('$'): FLOOR,  # gold on floor -- still floor
    ord('?'): FLOOR,  # scroll on floor
    ord('!'): FLOOR,  # potion on floor
    ord(')'): FLOOR,  # weapon on floor -- wait, conflicts with sink
    ord('('): FLOOR,  # tool/sink on floor
    ord('*'): FLOOR,  # gem on floor
    ord('/'): FLOOR,  # wand on floor
    ord('='): FLOOR,  # ring on floor
    ord('"'): FLOOR,  # amulet on floor
    ord('%'): FLOOR,  # food on floor
}

TILE_NAMES = {
    UNSEEN: 'unseen', FLOOR: 'floor', WALL: 'wall', CORRIDOR: 'corridor',
    DOOR_OPEN: 'open_door', DOOR_CLOSED: 'closed_door',
    STAIRS_UP: 'stairs_up', STAIRS_DOWN: 'stairs_down',
    FOUNTAIN: 'fountain', SINK: 'sink', THRONE: 'throne',
    ALTAR: 'altar', TRAP: 'trap', SHOP: 'shop',
    WATER: 'water', LAVA: 'lava', ICE: 'ice',
}


def _classify_tile(char_code, glyph_code=0, desc=""):
    """Classify a tile character into a category."""
    ch = chr(int(char_code))
    if ch == '.':
        if 'doorway' in desc:
            return DOOR_OPEN
        return FLOOR
    elif ch in '-|':
        return WALL
    elif ch == '#':
        if 'corridor' in desc or 'passage' in desc:
            return CORRIDOR
        elif 'sink' in desc:
            return SINK
        elif 'throne' in desc:
            return THRONE
        elif 'altar' in desc:
            return ALTAR
        return CORRIDOR
    elif ch == '+':
        return DOOR_CLOSED
    elif ch == '>':
        return STAIRS_DOWN
    elif ch == '<':
        return STAIRS_UP
    elif ch == '{':
        return FOUNTAIN
    elif ch == '^':
        return TRAP
    else:
        # Items on floor or monsters -- the underlying tile is floor/corridor
        if 'room' in desc or 'floor' in desc:
            return FLOOR
        elif 'corridor' in desc or 'passage' in desc:
            return CORRIDOR
        elif 'doorway' in desc:
            return DOOR_OPEN
        return FLOOR  # default


class MemoryTracker:
    """Accumulates explored map, item locations, and monster sightings across timesteps.
    
    At each step, call update() with the NLE observation. The tracker maintains:
    - explored_map: 21x79 grid of tile categories (UNSEEN for unexplored)
    - items_on_floor: dict of (y,x) -> {type, description, last_seen_turn}
    - monster_memory: dict of (y,x) -> {char, description, first_seen_turn, last_seen_turn}
    - rooms: list of connected regions discovered so far
    - visit_counts: how many times the agent has visited each tile
    """

    def __init__(self, map_height=21, map_width=79):
        self.map_h = map_height
        self.map_w = map_width
        self.explored = np.full((map_height, map_width), UNSEEN, dtype=np.int8)
        self.items_on_floor: Dict[Tuple[int,int], dict] = {}
        self.monster_memory: Dict[Tuple[int,int], dict] = {}
        self.rooms: List[set] = []
        self.visit_counts = np.zeros((map_height, map_width), dtype=np.int16)
        self.turn = 0
        self.total_explored = 0
        self._prev_obs = None
        # Track position history for the trail
        self.position_history: List[Tuple[int,int]] = []

    def update(self, obs, turn: Optional[int] = None):
        """Update memory with a new observation.
        
        Args:
            obs: NLE observation dict with 'chars', 'specials', 'blstats', 
                 'screen_descriptions' keys.
            turn: Current game turn (from blstats[20] if not given).
        """
        chars = obs['chars']
        descs = obs['screen_descriptions']
        bl = obs['blstats']
        player_y, player_x = int(bl[1]), int(bl[0])
        self.turn = turn if turn is not None else int(bl[20])
        
        # Track position
        self.position_history.append((player_y, player_x))
        self.visit_counts[player_y, player_x] += 1

        # Update explored map and items/monsters from current observation
        new_tiles = 0
        current_monsters = set()
        
        for y in range(self.map_h):
            for x in range(self.map_w):
                ch = chars[y][x]
                if ch == ord(' '):
                    continue  # empty space, skip
                
                desc = bytes(descs[y][x]).decode('latin-1').strip().rstrip('\x00')
                tile = _classify_tile(ch, desc=desc)
                
                if self.explored[y][x] == UNSEEN:
                    new_tiles += 1
                
                self.explored[y][x] = tile
                
                # Detect items on floor (not player, not pet, not wall)
                if ch not in (ord('@'), ord(' '), ord('.'), ord('-'), ord('|'),
                             ord('#'), ord('+'), ord('>'), ord('<')):
                    if desc and 'tame' not in desc and 'called Agent' not in desc:
                        # It's an item or monster
                        is_monster = any(w in desc for w in 
                            ['newt', 'goblin', 'jackal', 'lichen', 'sewer rat',
                             'kobold', 'bat', 'gnome', 'dwarf', 'orc', 'troll',
                             'dragon', 'demon', 'vampire', 'ghost', 'zombie',
                             'mummy', 'golem', 'skeleton', 'warrior', 'wizard',
                             'priest', 'shopkeeper', 'guard', 'soldier', 'captain',
                             'watchman', 'watch captain', 'monk', 'rogue',
                             'samurai', 'knight', 'caveman', 'archaeologist',
                             'healer', 'tourist', 'valkyrie', 'ranger', 'wizard',
                             'mage', 'ninja', 'rogue', 'lord', 'king', 'queen',
                             'Prisoner', 'Nazgul'])
                        
                        # Check if it's a tame pet
                        if 'tame' in desc:
                            continue
                        
                        # Check by char -- lowercase = monster, else item
                        ch_char = chr(ch)
                        if ch_char.isalpha() and ch_char.islower() and 'called' not in desc:
                            # Monster
                            current_monsters.add((y, x))
                            if (y, x) in self.monster_memory:
                                self.monster_memory[(y, x)]['last_seen_turn'] = self.turn
                                self.monster_memory[(y, x)]['desc'] = desc
                            else:
                                self.monster_memory[(y, x)] = {
                                    'char': ch_char,
                                    'desc': desc,
                                    'first_seen': self.turn,
                                    'last_seen': self.turn,
                                }
                        elif not ch_char.isalpha() or ch_char.isupper():
                            # Item on floor
                            self.items_on_floor[(y, x)] = {
                                'char': ch_char,
                                'desc': desc,
                                'last_seen': self.turn,
                            }
        
        self.total_explored += new_tiles
        
        # Remove monsters that are no longer visible (they moved or died)
        # Keep them in memory with a "last_seen" for prediction
        # But mark visible ones
        
        self._prev_obs = obs
        return new_tiles

    def forget_items_at(self, y, x):
        """Remove items at a position (e.g., after pickup)."""
        self.items_on_floor.pop((y, x), None)

    def get_room_at(self, y, x) -> Optional[int]:
        """Find which room contains position (y, x)."""
        for i, room in enumerate(self.rooms):
            if (y, x) in room:
                return i
        return None

    def detect_rooms(self):
        """Run flood-fill on explored floor tiles to find connected regions."""
        floor_tiles = set()
        for y in range(self.map_h):
            for x in range(self.map_w):
                if self.explored[y][x] in (FLOOR, DOOR_OPEN, DOOR_CLOSED, 
                                           STAIRS_UP, STAIRS_DOWN, FOUNTAIN,
                                           SINK, THRONE, ALTAR, TRAP):
                    floor_tiles.add((y, x))
        
        visited = set()
        self.rooms = []
        for start in floor_tiles:
            if start in visited:
                continue
            region = set()
            queue = [start]
            while queue:
                cy, cx = queue.pop()
                if (cy, cx) in visited or (cy, cx) not in floor_tiles:
                    continue
                visited.add((cy, cx))
                region.add((cy, cx))
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = cy+dy, cx+dx
                    if 0 <= ny < self.map_h and 0 <= nx < self.map_w:
                        if (ny, nx) not in visited:
                            queue.append((ny, nx))
            if len(region) >= 4:  # Skip tiny 1-3 tile regions
                self.rooms.append(region)
        return len(self.rooms)

    def render_explored_map(self) -> str:
        """Render the explored map as ASCII, using ' ' for unseen tiles."""
        TILE_CHARS = {
            UNSEEN: ' ', FLOOR: '.', WALL: '#', CORRIDOR: '#',
            DOOR_OPEN: '+', DOOR_CLOSED: '+', STAIRS_UP: '<',
            STAIRS_DOWN: '>', FOUNTAIN: '{', SINK: ')', THRONE: '\\',
            ALTAR: '_', TRAP: '^', WATER: '}', LAVA: '}', ICE: '}',
        }
        lines = []
        for y in range(self.map_h):
            row = []
            for x in range(self.map_w):
                row.append(TILE_CHARS.get(self.explored[y][x], '?'))
            line = ''.join(row).rstrip()
            if line.strip():
                lines.append(line)
        return '\n'.join(lines)

    def format_memory_summary(self) -> str:
        """Format a compact text summary of exploration memory."""
        self.detect_rooms()
        
        parts = []
        parts.append(f"Turn:{self.turn} | Explored:{self.total_explored} tiles | Rooms:{len(self.rooms)}")
        
        # Room summaries
        for i, room in enumerate(self.rooms):
            ys = [p[0] for p in room]
            xs = [p[1] for p in room]
            h, w = max(ys)-min(ys)+1, max(xs)-min(xs)+1
            
            # Items in this room
            room_items = []
            for (iy, ix), item in self.items_on_floor.items():
                if (iy, ix) in room:
                    room_items.append(f"{item['desc']}@({ix},{iy})")
            
            # Stairs in this room
            room_stairs = []
            for (sy, sx) in room:
                if self.explored[sy][sx] == STAIRS_DOWN:
                    room_stairs.append(f"stairs_down@({sx},{sy})")
                elif self.explored[sy][sx] == STAIRS_UP:
                    room_stairs.append(f"stairs_up@({sx},{sy})")
            
            features = room_stairs + room_items
            feat_str = ', '.join(features) if features else 'empty'
            parts.append(f"  Room {i} ({w}x{h}): {feat_str}")
        
        # Active monsters (seen recently)
        recent_monsters = {pos: m for pos, m in self.monster_memory.items()
                          if self.turn - m['last_seen_turn'] <= 3}
        if recent_monsters:
            mon_strs = [f"{m['desc'].split()[-1]}@({pos[1]},{pos[0]}) t-{self.turn - m['last_seen_turn']}"
                       for pos, m in recent_monsters.items()]
            parts.append(f"Monsters: {', '.join(mon_strs)}")
        else:
            parts.append("Monsters: none visible")
        
        # Items on floor (all remembered)
        if self.items_on_floor:
            item_strs = [f"{it['desc']}@({pos[1]},{pos[0]}) t-{self.turn - it['last_seen']}"
                        for pos, it in self.items_on_floor.items()]
            parts.append(f"Floor items: {', '.join(item_strs)}")
        
        return '\n'.join(parts)


def format_enriched_prompt(obs, memory: MemoryTracker, action: str) -> str:
    """Format the enriched prompt for the forward model.
    
    Combines:
    - Memory summary (rooms, items, monsters, explored map)
    - Current viewport (full visible ASCII map)
    - Stats (HP, gold, depth, etc.)
    - The proposed action
    """
    bl = obs['blstats']
    
    parts = []
    
    # 1. Memory context
    parts.append("=== MEMORY ===")
    parts.append(memory.format_memory_summary())
    
    # 2. Current viewport (full ASCII map)
    parts.append("\n=== VIEWPORT ===")
    chars = obs['chars']
    for y in range(chars.shape[0]):
        row = bytes(chars[y]).decode('ascii', errors='replace').rstrip()
        if row.strip():
            parts.append(row)
    
    # 3. Stats
    msg = bytes(obs['message']).decode('ascii', errors='replace').strip().rstrip('\x00')
    parts.append(f"\n=== STATUS ===")
    parts.append(f"HP:{bl[10]}/{bl[11]} AC:{bl[16]} Str:{bl[3]} Dex:{bl[4]} "
                f"Gold:{bl[13]} Depth:{bl[12]} Turn:{bl[20]}")
    if msg:
        parts.append(f"Last msg: {msg}")
    
    # 4. Action
    parts.append(f"\nAction: {action}")
    
    return '\n'.join(parts)


def format_enriched_target(delta: dict, obs_after=None, memory: MemoryTracker = None) -> str:
    """Format the target prediction for the enriched forward model.
    
    Predicts:
    - Position delta
    - HP change
    - Gold change
    - Depth change
    - Survival
    - Game message
    - Whether any new tiles were explored
    - Whether any monsters appeared/disappeared
    """
    parts = []
    
    # Position delta
    dx, dy = delta.get('pos_delta', (0, 0))
    parts.append(f"pos:({dx},{dy})")
    
    # HP change
    hp_d = delta.get('hp_delta', 0)
    parts.append(f"hp:{'same' if hp_d == 0 else ('+' + str(hp_d) if hp_d > 0 else str(hp_d))}")
    
    # Gold change
    gold_d = delta.get('gold_delta', 0)
    parts.append(f"gold:{'same' if gold_d == 0 else ('+' + str(gold_d) if gold_d > 0 else str(gold_d))}")
    
    # Depth change
    depth_d = delta.get('depth_delta', 0)
    parts.append(f"depth:{'same' if depth_d == 0 else ('+' + str(depth_d) if depth_d > 0 else str(depth_d))}")
    
    # Survival
    survived = delta.get('survived', True)
    parts.append(f"alive:{'yes' if survived else 'no'}")
    
    # New tiles explored
    new_tiles = len(delta.get('new_tiles', []))
    parts.append(f"explored:+{new_tiles}")
    
    # Message
    msg = delta.get('message', '')
    if len(msg) > 80:
        msg = msg[:77] + '...'
    parts.append(f"msg:{msg}")
    
    return ' | '.join(parts)
