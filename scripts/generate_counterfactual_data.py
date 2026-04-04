#!/usr/bin/env python3
"""Generate counterfactual training data from NLE using os.fork()."""
import sys, os, json, random, argparse, signal
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import nle.env
import numpy as np
from nle_agent.agent_http import _build_action_map
from src.state_encoder import StateEncoder
from src.memory_tracker import MemoryTracker

AMAP = _build_action_map()
ENCODER = StateEncoder()
ACTIONS = ['north', 'south', 'east', 'west', 'northeast', 'northwest',
           'southeast', 'southwest', 'wait']
EXPLORE = ['north', 'south', 'east', 'west', 'wait', 'search']

SYSTEM_PROMPT = "You are a NetHack agent. Predict the outcome of the given action in the current game state."


def detect_skills(obs):
    """Return list of (skill_name, detail) tuples for interesting moments."""
    chars = obs['chars']
    bl = obs['blstats']
    px, py = int(bl[0]), int(bl[1])
    hp, hp_max = int(bl[10]), int(bl[11])
    skills = []

    # Adjacent hostile monsters -> combat
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            ny, nx = py + dy, px + dx
            if 0 <= ny < 21 and 0 <= nx < 79:
                c = int(chars[ny, nx])
                ch = chr(c)
                if ch.isalpha() and ch != '@' and ch != 'I' and ch.islower():
                    desc = bytes(obs['screen_descriptions'][ny][nx]).decode('latin-1').strip().rstrip('\x00')
                    if 'tame' not in desc and 'statue' not in desc:
                        skills.append(('combat', ch, desc, (nx, ny)))

    # Visible hostile at distance -> threat
    for y in range(21):
        for x in range(79):
            c = int(chars[y, x])
            ch = chr(c)
            if ch.isalpha() and ch != '@' and ch != 'I' and ch.islower():
                desc = bytes(obs['screen_descriptions'][y][x]).decode('latin-1').strip().rstrip('\x00')
                if 'tame' not in desc and 'statue' not in desc:
                    dist = abs(y - py) + abs(x - px)
                    if dist > 1:
                        skills.append(('threat', ch, desc, dist))

    # Low HP -> survival
    if 0 < hp <= hp_max // 2:
        skills.append(('survival', hp, hp_max))

    # Stairs nearby -> descent
    for y in range(21):
        for x in range(79):
            if int(chars[y, x]) == ord('>'):
                dist = abs(y - py) + abs(x - px)
                if dist <= 5:
                    skills.append(('descent', dist))

    # Item underfoot -> pickup
    c = int(chars[py, px])
    if c in (ord('$'), ord('?'), ord('!'), ord('%'), ord('/'), ord('='), ord(')'), ord('[')):
        desc = bytes(obs['screen_descriptions'][py][px]).decode('latin-1').strip().rstrip('\x00')
        skills.append(('pickup', desc))

    return skills


def fork_counterfactual(env, obs, seed_idx, step, skill_names):
    """Fork and try all ACTIONS, return list of (action, result_dict)."""
    results = []
    state = ENCODER.encode_full(obs)
    prompt = ENCODER.format_prompt(state, 'north')  # representative
    pre_hp = int(obs['blstats'][10])

    tmpdir = f'/tmp/cf_{os.getpid()}'
    os.makedirs(tmpdir, exist_ok=True)

    children = []
    for action_name in ACTIONS:
        pid = os.fork()
        if pid == 0:
            # Child
            try:
                obs_after, reward, term, trunc, _ = env.step(AMAP[action_name])
                delta = ENCODER.encode_delta(obs, obs_after, action_name)
                target = ENCODER.format_target(delta)
                hp_after = int(obs_after['blstats'][10])
                msg = bytes(obs_after['message']).decode('ascii', errors='replace').strip().rstrip('\x00')
                result = {
                    'target': target,
                    'hp_delta': hp_after - pre_hp,
                    'died': bool(term),
                    'msg': msg[:100],
                    'reward': float(reward),
                }
                with open(f'{tmpdir}/{action_name}.json', 'w') as f:
                    json.dump(result, f)
            except Exception as e:
                with open(f'{tmpdir}/{action_name}.json', 'w') as f:
                    json.dump({'error': str(e)}, f)
            env.close()
            os._exit(0)
        else:
            children.append(pid)

    # Wait for all children
    for pid in children:
        os.waitpid(pid, 0)

    # Read results
    pairs = []
    for action_name in ACTIONS:
        path = f'{tmpdir}/{action_name}.json'
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)
            if 'error' not in r:
                pair = {
                    'conversations': [
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': prompt.replace('north', action_name)},
                        {'role': 'assistant', 'content': r['target']},
                    ],
                    'metadata': {
                        'seed': seed_idx,
                        'step': step,
                        'skills': skill_names,
                        'action': action_name,
                        'hp_delta': r['hp_delta'],
                        'died': r['died'],
                        'msg': r['msg'],
                        'reward': r['reward'],
                    }
                }
                pairs.append(pair)
            os.unlink(path)

    # Clean up tmpdir
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass

    return pairs


def seq_counterfactual(env, obs, seed_idx, step, skill_names):
    """Sequential (no-fork) version for debugging."""
    results = []
    state = ENCODER.encode_full(obs)
    prompt = ENCODER.format_prompt(state, 'north')
    pre_hp = int(obs['blstats'][10])

    pairs = []
    for action_name in ACTIONS:
        # We can't undo in NLE, so just note the action
        # In no-fork mode we skip counterfactuals and just record the state
        delta = ENCODER.encode_delta(obs, obs, action_name)
        target = ENCODER.format_target(delta)
        pair = {
            'conversations': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt.replace('north', action_name)},
                {'role': 'assistant', 'content': target},
            ],
            'metadata': {
                'seed': seed_idx,
                'step': step,
                'skills': skill_names,
                'action': action_name,
                'hp_delta': 0,
                'died': False,
                'msg': '',
                'reward': 0.0,
            }
        }
        pairs.append(pair)

    return pairs


def run(num_games, max_steps, seed_start, output_path, no_fork):
    """Main generation loop."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    stats = {'games': 0, 'steps': 0, 'moments': 0, 'pairs': 0,
             'kills': 0, 'deaths': 0, 'hp_changes': 0}
    skill_counts = {}

    counter_fn = seq_counterfactual if no_fork else fork_counterfactual

    for gi in range(num_games):
        seed = seed_start + gi
        env = nle.env.NLE()
        obs, _ = env.reset(seed=seed)
        memory = MemoryTracker()
        memory.update(obs)

        game_pairs = 0
        for step in range(max_steps):
            stats['steps'] += 1
            skills = detect_skills(obs)

            if skills:
                stats['moments'] += 1
                skill_names = sorted(set(s[0] for s in skills))
                for s in skill_names:
                    skill_counts[s] = skill_counts.get(s, 0) + 1

                pairs = counter_fn(env, obs, seed, step, skill_names)
                all_pairs.extend(pairs)
                game_pairs += len(pairs)
                stats['pairs'] += len(pairs)

                # Update kill/death stats
                for p in pairs:
                    m = p['metadata']
                    if m['died']:
                        stats['deaths'] += 1
                    if m['hp_delta'] != 0:
                        stats['hp_changes'] += 1
                    if 'kill' in m['msg'].lower():
                        stats['kills'] += 1

            # Random walk to continue game
            action = random.choice(EXPLORE)
            obs, reward, term, trunc, _ = env.step(AMAP[action])
            memory.update(obs)

            if term or trunc:
                break

        env.close()
        stats['games'] += 1
        print(f"  Seed {seed}: {game_pairs} pairs, {step+1} steps", flush=True)

    # Write output
    with open(output_path, 'w') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + '\n')

    print(f"\nDone! {stats['pairs']} pairs from {stats['games']} games")
    print(f"  Steps: {stats['steps']}, Moments: {stats['moments']}")
    print(f"  Kills: {stats['kills']}, Deaths: {stats['deaths']}, HP changes: {stats['hp_changes']}")
    print(f"  Skills: {skill_counts}")
    print(f"  Output: {output_path}")
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate counterfactual NLE training data')
    parser.add_argument('--num-games', type=int, default=50)
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--output', default='data/counterfactual_training_pairs.jsonl')
    parser.add_argument('--no-fork', action='store_true', help='Sequential mode (no counterfactuals)')
    args = parser.parse_args()

    run(args.num_games, args.max_steps, args.seed_start, args.output, args.no_fork)
