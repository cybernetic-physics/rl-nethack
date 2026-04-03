"""
Game Reporter: Visualize NLE gameplay as text replays and HTML reports.

Produces human-readable gameplay summaries in three formats:
1. Text replay  - compact turn-by-turn log
2. HTML replay  - scrollable browser page with map, actions, color coding
3. Summary      - one-line game overview

Also provides run_and_report() to execute a full game and collect all data.
"""

import os
import random
from typing import Dict, List, Optional

import nle.env

from src.state_encoder import StateEncoder
from nle_agent.agent_http import _build_action_map


# ---------------------------------------------------------------------------
# Helper: describe what happened in a single step
# ---------------------------------------------------------------------------

def _describe_event(action: str, delta: dict, state: dict) -> str:
    """Build a short human-readable description of one step's outcome."""
    parts = []

    # Did the player move?
    dx, dy = delta['pos_delta']
    if dx != 0 or dy != 0:
        parts.append('Moved')
    elif action == 'wait':
        parts.append('Waited')
    else:
        parts.append('Acted')

    # HP change
    hp_d = delta['hp_delta']
    if hp_d < 0:
        parts.append(f'HP{hp_d}!')

    # Gold change
    gold_d = delta['gold_delta']
    if gold_d > 0:
        parts.append(f'Gold+{gold_d}')

    # New tiles explored
    new_count = len(delta.get('new_tiles', []))
    if new_count > 0:
        parts.append(f'+{new_count} tile{"s" if new_count > 1 else ""}')

    # Message snippet (short)
    msg = delta.get('message', '')
    if msg:
        # Keep it brief
        snippet = msg[:50] + ('...' if len(msg) > 50 else '')
        parts.append(snippet)

    return '. '.join(parts)


# ---------------------------------------------------------------------------
# 1. Text replay
# ---------------------------------------------------------------------------

def format_replay(steps: List[dict], seed: int = 0) -> str:
    """Produce a compact turn-by-turn text log of a game.

    Args:
        steps: List of step_data dicts (from run_and_report).
        seed:  Game seed for the header.

    Returns:
        Multi-line string with header, per-step lines, and a result footer.
    """
    lines = []
    lines.append(f"=== Seed {seed}: {len(steps)} steps ===")

    for sd in steps:
        step_idx = sd['step']
        state = sd['state']
        action = sd['action']
        delta = sd['delta']

        hp = state['hp']
        hp_max = state['hp_max']
        pos = state['position']

        event = _describe_event(action, delta, state)

        line = (
            f"Step {step_idx:2d} | "
            f"HP:{hp}/{hp_max} | "
            f"Pos:{pos} | "
            f"Action: {action:<6s} -> "
            f"{event}"
        )
        lines.append(line)

    # Result footer
    if steps:
        last = steps[-1]
        last_state = last['state']
        last_delta = last['delta']
        outcome = 'survived' if last_delta['survived'] else 'died'
        if outcome == 'died':
            outcome = f'Died at step {last["step"]}'
        else:
            outcome = 'Survived'
        lines.append(
            f"=== Result: {outcome}. "
            f"Final HP:{last_state['hp']}, "
            f"Gold:{last_state['gold']}, "
            f"Depth:{last_state['depth']} ==="
        )
    else:
        lines.append("=== No steps recorded ===")

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. HTML replay
# ---------------------------------------------------------------------------

def _render_map_html(chars) -> str:
    """Render the chars grid as a <pre> block of ASCII."""
    rows = []
    for i in range(chars.shape[0]):
        row = bytes(chars[i]).decode('ascii', errors='replace').rstrip()
        if row.strip():
            rows.append(row)
    return '\n'.join(rows)


def _hp_bar(hp: int, hp_max: int) -> str:
    """Return an inline HP bar as HTML."""
    if hp_max <= 0:
        pct = 0
    else:
        pct = int(100 * hp / hp_max)
    if pct > 60:
        color = '#2d2'
    elif pct > 30:
        color = '#da2'
    else:
        color = '#d22'
    return (
        f'<div class="hp-bar">'
        f'<div class="hp-fill" style="width:{pct}%;background:{color}"></div>'
        f'<span class="hp-text">{hp}/{hp_max}</span>'
        f'</div>'
    )


def _step_class(delta: dict) -> str:
    """Return a CSS class name for colour-coding a step."""
    if not delta.get('survived', True):
        return 'death'
    if delta.get('gold_delta', 0) > 0:
        return 'gold'
    if delta.get('hp_delta', 0) < 0:
        return 'damage'
    if len(delta.get('new_tiles', [])) > 0:
        return 'explore'
    return 'normal'


def format_html_replay(steps: List[dict], seed: int = 0) -> str:
    """Produce a self-contained HTML page showing the game replay.

    Args:
        steps: List of step_data dicts.
        seed:  Game seed for the header.

    Returns:
        Complete HTML string (inline CSS, no external deps).
    """
    total_steps = len(steps)
    outcome = 'in progress'
    final_gold = 0
    final_depth = 1
    if steps:
        last = steps[-1]
        final_gold = last['state']['gold']
        final_depth = last['state']['depth']
        if last['delta']['survived']:
            outcome = 'survived'
        else:
            outcome = 'died'

    # Build step rows
    step_rows = []
    for sd in steps:
        idx = sd['step']
        state = sd['state']
        delta = sd['delta']
        action = sd['action']

        hp = state['hp']
        hp_max = state['hp_max']
        pos = state['position']

        event = _describe_event(action, delta, state)
        cls = _step_class(delta)
        hp_bar_html = _hp_bar(hp, hp_max)

        # Map from obs (before action)
        map_html = ''
        obs = sd.get('obs')
        if obs is not None:
            map_text = _render_map_html(obs['chars'])
            map_html = (
                f'<pre class="map">{_html_escape(map_text)}</pre>'
            )

        # Delta info
        gold_d = delta.get('gold_delta', 0)
        hp_d = delta.get('hp_delta', 0)
        new_tiles = len(delta.get('new_tiles', []))

        step_rows.append(
            f'<div class="step {cls}">'
            f'<div class="step-header">'
            f'<span class="step-num">Step {idx}</span>'
            f'<span class="step-action">Action: {action}</span>'
            f'{hp_bar_html}'
            f'<span class="step-pos">Pos: {pos}</span>'
            f'</div>'
            f'{map_html}'
            f'<div class="step-detail">'
            f'{_html_escape(event)}'
            f' | Gold{"+" if gold_d >= 0 else ""}{gold_d}'
            f' | HP{"+" if hp_d >= 0 else ""}{hp_d}'
            f' | New tiles: {new_tiles}'
            f'</div>'
            f'</div>'
        )

    steps_html = '\n'.join(step_rows)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NLE Game Replay - Seed {seed}</title>
<style>
  body {{ font-family: monospace; background: #1a1a2e; color: #e0e0e0; margin: 0; padding: 20px; }}
  h1 {{ color: #e0e0e0; }}
  .header {{ background: #16213e; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
  .header span {{ margin-right: 20px; }}
  .step {{ background: #16213e; margin-bottom: 10px; border-radius: 6px; padding: 10px; border-left: 4px solid #555; }}
  .step.gold {{ border-left-color: #2d2; }}
  .step.damage {{ border-left-color: #d22; }}
  .step.death {{ border-left-color: #f00; background: #2a1020; }}
  .step.explore {{ border-left-color: #28f; }}
  .step-header {{ display: flex; align-items: center; gap: 15px; margin-bottom: 5px; }}
  .step-num {{ font-weight: bold; min-width: 70px; }}
  .step-action {{ color: #8af; min-width: 120px; }}
  .step-pos {{ color: #aaa; }}
  .hp-bar {{ width: 120px; height: 18px; background: #333; border-radius: 3px; position: relative; overflow: hidden; }}
  .hp-fill {{ height: 100%; border-radius: 3px; }}
  .hp-text {{ position: absolute; top: 0; left: 5px; font-size: 12px; line-height: 18px; color: #fff; text-shadow: 1px 1px 1px #000; }}
  .map {{ font-size: 10px; line-height: 11px; background: #0d1117; padding: 5px; border-radius: 4px; margin: 5px 0; overflow-x: auto; white-space: pre; max-height: 200px; }}
  .step-detail {{ color: #bbb; font-size: 13px; }}
</style>
</head>
<body>
<h1>NLE Game Replay</h1>
<div class="header">
  <span><b>Seed:</b> {seed}</span>
  <span><b>Steps:</b> {total_steps}</span>
  <span><b>Outcome:</b> {outcome}</span>
  <span><b>Gold:</b> {final_gold}</span>
  <span><b>Depth:</b> {final_depth}</span>
</div>
{steps_html}
</body>
</html>"""
    return html


def _html_escape(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
    )


# ---------------------------------------------------------------------------
# 3. Summary
# ---------------------------------------------------------------------------

def format_summary(steps: List[dict], seed: int = 0) -> str:
    """Produce a one-line game summary.

    Args:
        steps: List of step_data dicts.
        seed:  Game seed.

    Returns:
        Single-line summary string.
    """
    if not steps:
        return f"Seed {seed}: 0 steps, no data"

    last = steps[-1]
    last_state = last['state']
    last_delta = last['delta']

    total_steps = len(steps)
    depth = last_state['depth']
    gold = last_state['gold']

    # Accumulate damage taken
    total_damage = sum(
        -sd['delta']['hp_delta'] for sd in steps if sd['delta']['hp_delta'] < 0
    )

    # Count tiles explored (sum of new_tiles across all steps)
    tiles_explored = sum(
        len(sd['delta'].get('new_tiles', [])) for sd in steps
    )

    # Outcome
    if last_delta['survived']:
        outcome = 'survived'
    else:
        outcome = f'died at step {last["step"]}'

    return (
        f"Seed {seed}: {total_steps} steps, {outcome}, "
        f"depth {depth}, {gold} gold collected, "
        f"{total_damage} damage taken, {tiles_explored} tiles explored"
    )


# ---------------------------------------------------------------------------
# 4. Game runner
# ---------------------------------------------------------------------------

def run_and_report(
    seed: int,
    max_steps: int,
    encoder: StateEncoder,
    output_dir: Optional[str] = None,
) -> dict:
    """Run a complete NLE game, collect all step data, optionally write HTML.

    Args:
        seed:       Random seed for the environment.
        max_steps:  Maximum number of steps to play.
        encoder:    StateEncoder instance.
        output_dir: If provided, write an HTML report to this directory.

    Returns:
        Dict with keys:
            seed, steps (count), outcome, total_gold, total_damage,
            tiles_explored, step_data (list)
    """
    action_map = _build_action_map()
    rng = random.Random(seed)

    env = nle.env.NLE()
    obs, info = env.reset(seed=seed)

    step_data: List[dict] = []
    terminated = False
    truncated = False

    for step in range(max_steps):
        state = encoder.encode_full(obs)

        # Choose action via wall_avoidance_policy
        adjacent = state['adjacent']
        # Inline wall avoidance (same logic as data_generator)
        open_dirs = [
            d for d in ('north', 'south', 'east', 'west')
            if adjacent.get(d, 'unseen') not in ('wall', 'unseen')
        ]
        if open_dirs:
            action_name = rng.choice(open_dirs)
        else:
            action_name = 'wait'

        if action_name in action_map:
            action_idx = action_map[action_name]
        else:
            action_idx = action_map.get('wait', 18)
            action_name = 'wait'

        # Format prompt/target
        prompt_text = encoder.format_prompt(state, action_name)

        # Take the action
        obs_after, reward, terminated, truncated, info_after = env.step(action_idx)

        delta = encoder.encode_delta(obs, obs_after, action_name)
        target_text = encoder.format_target(delta)

        step_data.append({
            'step': step,
            'obs': obs,
            'obs_after': obs_after,
            'state': state,
            'action': action_name,
            'delta': delta,
            'prompt': prompt_text,
            'target': target_text,
        })

        obs = obs_after

        if terminated or truncated:
            break

    env.close()

    # Compute aggregates
    total_steps = len(step_data)
    last_state = step_data[-1]['state'] if step_data else None
    last_delta = step_data[-1]['delta'] if step_data else None

    outcome = 'survived'
    if last_delta and not last_delta['survived']:
        outcome = 'died'

    total_gold = last_state['gold'] if last_state else 0
    total_damage = sum(
        -sd['delta']['hp_delta'] for sd in step_data if sd['delta']['hp_delta'] < 0
    )
    tiles_explored = sum(
        len(sd['delta'].get('new_tiles', [])) for sd in step_data
    )

    result = {
        'seed': seed,
        'steps': total_steps,
        'outcome': outcome,
        'total_gold': total_gold,
        'total_damage': total_damage,
        'tiles_explored': tiles_explored,
        'step_data': step_data,
    }

    # Optionally write HTML report
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(output_dir, f'game_seed_{seed}.html')
        html_content = format_html_replay(step_data, seed)
        with open(html_path, 'w') as f:
            f.write(html_content)
        result['html_path'] = html_path

    return result
