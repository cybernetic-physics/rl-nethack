"""
Trace recorder for AutoAscend - patches EnvWrapper to capture game state at each step.
Outputs JSON compatible with game_viewer.html format.
"""
import json
import os
import numpy as np
from pathlib import Path

# Action name mapping for NLE
ACTION_NAMES = {
    0: 'north', 1: 'south', 2: 'east', 3: 'west',
    4: 'northeast', 5: 'northwest', 6: 'southeast', 7: 'southwest',
    8: 'up_stairs', 9: 'down_stairs', 10: 'wait', 11: 'pick_up',
    12: 'open', 13: 'kick', 14: 'search',
}

class TraceRecorder:
    def __init__(self, output_path, seed):
        self.output_path = output_path
        self.seed = seed
        self.steps = []
        self.kills = 0
        self.damage_taken = 0
        self.min_hp = 9999
        self.max_depth = 0
        
    def record_step(self, obs, action_idx, prev_hp=None):
        """Record a single step's observation."""
        blstats = obs['blstats']
        hp = int(blstats[10])
        max_hp = int(blstats[11])
        depth = int(blstats[12])  # dungeon_level in NLE
        score = int(blstats[9])
        
        # Track stats
        if hp < self.min_hp:
            self.min_hp = hp
        if depth > self.max_depth:
            self.max_depth = depth
        if prev_hp is not None and hp < prev_hp:
            self.damage_taken += (prev_hp - hp)
            
        # Get message
        message = bytes(obs['message']).decode('ascii', errors='replace').strip()
        
        # Get map (chars)
        chars = obs['chars']
        tty_chars = obs.get('tty_chars', chars)
        
        # Extract the visible portion (21x79 tty display)
        if tty_chars.shape == (24, 80):
            # Use tty display, skip bottom status lines
            map_text = ''
            for row in tty_chars[:21]:
                map_text += bytes(row).decode('ascii', errors='replace') + '\n'
            map_text = map_text.rstrip('\n')
        else:
            # Fallback: use full chars array centered on player
            map_text = ''
            y, x = int(blstats[1]), int(blstats[0])
            for row in range(max(0, y-10), min(chars.shape[0], y+11)):
                line = ''
                for col in range(max(0, x-39), min(chars.shape[1], x+40)):
                    line += chr(chars[row, col])
                map_text += line + '\n'
            map_text = map_text.rstrip('\n')
        
        action_name = ACTION_NAMES.get(action_idx, f'action_{action_idx}')
        
        step_data = {
            'step': len(self.steps),
            'action': action_name,
            'map': map_text,
            'player_pos': [int(blstats[0]), int(blstats[1])],
            'hp': hp,
            'max_hp': max_hp,
            'depth': depth,
            'score': score,
            'message': message,
            'level': int(blstats[20]) if len(blstats) > 20 else 1,  # experience level
        }
        
        # Check for kill messages
        if 'you kill' in message.lower() or 'destroy' in message.lower():
            self.kills += 1
            step_data['event_type'] = 'kill'
        elif prev_hp is not None and hp < prev_hp:
            step_data['event_type'] = 'damage'
            
        self.steps.append(step_data)
        return hp
    
    def save(self, died=False, end_reason=''):
        game_data = {
            'seed': self.seed,
            'label': f'Seed {self.seed}: {self.kills} kills, {self.damage_taken} dmg (depth {self.max_depth})',
            'total_steps': len(self.steps),
            'kills': self.kills,
            'damage_taken': self.damage_taken,
            'min_hp': self.min_hp if self.min_hp < 9999 else 0,
            'max_depth': self.max_depth,
            'died': died,
            'end_reason': end_reason,
            'steps': self.steps,
        }
        
        # Load existing or create new
        games = []
        if os.path.exists(self.output_path):
            with open(self.output_path) as f:
                games = json.load(f)
        games.append(game_data)
        
        os.makedirs(os.path.dirname(self.output_path) or '.', exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(games, f)
        
        print(f'[trace] Saved game seed={self.seed}, {len(self.steps)} steps, '
              f'kills={self.kills}, depth={self.max_depth}, died={died}')
        return game_data
