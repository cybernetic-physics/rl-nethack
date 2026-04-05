#!/usr/bin/env python3
"""
Generate an animated interactive HTML demo report showing 5 NetHack games
with step-by-step replay and training data visualization.

Usage:
    python generate_demo_report.py

Output:
    output/demo_report.html  (self-contained, ~100KB+)
"""

import json
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.state_encoder import StateEncoder
from src.reporter import run_and_report

SEEDS = [42, 99, 7, 1337, 2024]
MAX_STEPS = 30
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'demo_report.html')


def render_map_text(chars_array) -> str:
    """Render chars numpy array as plain text lines."""
    rows = []
    for i in range(chars_array.shape[0]):
        row = bytes(chars_array[i]).decode('ascii', errors='replace').rstrip()
        if row.strip():
            rows.append(row)
    return '\n'.join(rows)


def find_player_pos(chars_array):
    """Find player position (row, col) in chars array."""
    positions = np.argwhere(chars_array == ord('@'))
    if len(positions) == 0:
        return None
    return int(positions[0][0]), int(positions[0][1])


def serialize_game_data(result: dict) -> dict:
    """Convert a game result dict into a JSON-serializable structure for the HTML viewer."""
    steps = []
    for sd in result['step_data']:
        obs = sd['obs']
        obs_after = sd.get('obs_after')
        state = sd['state']
        delta = sd['delta']

        # Map: use obs_after (state after action) so map shows where player moved TO
        map_obs = obs_after if obs_after is not None else obs
        map_text = render_map_text(map_obs['chars'])
        player_pos = find_player_pos(map_obs['chars'])

        # Build step info
        step_info = {
            'step': sd['step'],
            'action': sd['action'],
            'map': map_text,
            'player_pos': player_pos,  # [row, col] or null
            'hp': state['hp'],
            'hp_max': state['hp_max'],
            'gold': state['gold'],
            'depth': state['depth'],
            'position': list(state['position']),
            'ac': state['ac'],
            'strength': state['strength'],
            'dexterity': state['dexterity'],
            'turn': state['turn'],
            'message': delta.get('message', ''),
            'hp_delta': delta['hp_delta'],
            'gold_delta': delta['gold_delta'],
            'depth_delta': delta['depth_delta'],
            'survived': delta['survived'],
            'new_tiles_count': len(delta.get('new_tiles', [])),
            'pos_delta': list(delta['pos_delta']),
            'prompt': sd['prompt'],
            'target': sd['target'],
        }
        steps.append(step_info)

    return {
        'seed': result['seed'],
        'total_steps': result['steps'],
        'outcome': result['outcome'],
        'total_gold': result['total_gold'],
        'total_damage': result['total_damage'],
        'tiles_explored': result['tiles_explored'],
        'steps': steps,
    }


def build_html(all_games: list) -> str:
    """Build the complete self-contained HTML file with embedded game data."""

    games_json = json.dumps(all_games, indent=None, separators=(',', ':'))

    # Compute overall stats
    total_steps_all = sum(g['total_steps'] for g in all_games)
    total_examples = total_steps_all  # each step = one training example

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>rl-nethack: NetHack Forward Model Training Data</title>
<style>
/* === RESET & BASE === */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: 'Courier New', 'Consolas', 'Monaco', monospace;
  background: #0a0e17;
  color: #c9d1d9;
  line-height: 1.5;
  min-height: 100vh;
}}

/* === HEADER === */
.header {{
  background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
  border-bottom: 2px solid #21262d;
  padding: 20px 30px;
  position: sticky; top: 0; z-index: 100;
}}
.header h1 {{
  font-size: 22px;
  color: #58a6ff;
  margin-bottom: 10px;
  letter-spacing: 1px;
}}
.header .subtitle {{
  font-size: 12px;
  color: #8b949e;
  margin-bottom: 12px;
}}
.stats-bar {{
  display: flex;
  gap: 25px;
  flex-wrap: wrap;
}}
.stat-item {{
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 6px;
  padding: 8px 14px;
  font-size: 13px;
}}
.stat-item .stat-label {{
  color: #8b949e;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}
.stat-item .stat-value {{
  color: #f0f6fc;
  font-weight: bold;
  font-size: 18px;
}}

/* === GAME SELECTOR TABS === */
.game-tabs {{
  display: flex;
  gap: 4px;
  padding: 15px 30px 0;
  background: #0d1117;
}}
.tab-btn {{
  background: #161b22;
  border: 1px solid #30363d;
  border-bottom: none;
  color: #8b949e;
  padding: 10px 20px;
  cursor: pointer;
  font-family: inherit;
  font-size: 13px;
  border-radius: 8px 8px 0 0;
  transition: all 0.2s;
  outline: none;
}}
.tab-btn:hover {{ color: #c9d1d9; background: #1c2128; }}
.tab-btn.active {{
  background: #0d1117;
  color: #58a6ff;
  border-color: #58a6ff;
  border-bottom: 2px solid #0d1117;
  font-weight: bold;
}}

/* === GAME PANEL === */
.game-panel {{
  display: none;
  padding: 20px 30px;
}}
.game-panel.active {{ display: block; }}

/* === GAME HEADER === */
.game-header {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
  flex-wrap: wrap;
  gap: 10px;
}}
.game-meta {{
  display: flex;
  gap: 15px;
  flex-wrap: wrap;
}}
.game-meta .meta-item {{
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 6px;
  padding: 6px 12px;
  font-size: 12px;
}}
.meta-label {{ color: #8b949e; }}
.meta-value {{ color: #f0f6fc; font-weight: bold; }}

/* === CONTROLS === */
.controls {{
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 15px;
  flex-wrap: wrap;
}}
.ctrl-btn {{
  background: #21262d;
  border: 1px solid #30363d;
  color: #c9d1d9;
  padding: 8px 16px;
  cursor: pointer;
  font-family: inherit;
  font-size: 13px;
  border-radius: 6px;
  transition: all 0.15s;
  outline: none;
}}
.ctrl-btn:hover {{ background: #30363d; }}
.ctrl-btn.active {{ background: #388bfd; color: #fff; border-color: #58a6ff; }}
.speed-btns {{ display: flex; gap: 4px; }}
.step-counter {{
  color: #8b949e;
  font-size: 13px;
  margin-left: 10px;
}}

/* === PROGRESS BAR === */
.progress-container {{
  background: #161b22;
  border-radius: 4px;
  height: 6px;
  margin-bottom: 15px;
  overflow: hidden;
}}
.progress-fill {{
  height: 100%;
  background: linear-gradient(90deg, #388bfd, #58a6ff);
  border-radius: 4px;
  transition: width 0.3s ease;
  width: 0%;
}}

/* === MAIN LAYOUT === */
.replay-layout {{
  display: flex;
  gap: 15px;
  align-items: flex-start;
}}
.map-col {{
  width: 42%;
  flex-shrink: 0;
  position: sticky;
  top: 10px;
  z-index: 10;
}}
.info-col {{
  width: 58%;
  min-width: 0;
  flex-shrink: 0;
}}
@media (max-width: 1000px) {{
  .replay-layout {{ flex-direction: column; }}
  .map-col, .info-col {{ width: 100%; position: static; }}
}}

/* === MAP PANEL === */
.map-panel {{
  background: #0d1117;
  border: 1px solid #30363d;
  border-radius: 8px;
  overflow: hidden;
}}
.panel-title {{
  background: #161b22;
  padding: 8px 14px;
  font-size: 12px;
  color: #58a6ff;
  text-transform: uppercase;
  letter-spacing: 1px;
  border-bottom: 1px solid #30363d;
}}
.map-container {{
  padding: 10px;
  font-size: 13px;
  line-height: 15px;
  height: 420px;
  overflow: hidden;
  white-space: pre;
  position: relative;
}}
.map-line {{
  display: block;
}}
.map-char {{
  display: inline;
  transition: color 0.3s, text-shadow 0.3s;
}}
.map-char.player {{
  color: #ff0;
  text-shadow: 0 0 8px #ff0, 0 0 15px #ff8;
  animation: blink 0.8s infinite;
  font-weight: bold;
}}
@keyframes blink {{
  0%, 100% {{ opacity: 1; }}
  50% {{ opacity: 0.5; }}
}}
.map-char.wall {{ color: #6e7681; }}
.map-char.floor {{ color: #484f58; }}
.map-char.door {{ color: #d29922; }}
.map-char.corridor {{ color: #6e7681; }}
.map-char.gold-item {{ color: #3fb950; font-weight: bold; }}
.map-char.monster {{ color: #f85149; font-weight: bold; }}
.map-char.item {{ color: #d2a8ff; }}
.map-char.stairs {{ color: #79c0ff; }}
.map-char.unseen {{ color: #0d1117; }}
.map-char.new-tile {{
  animation: fadein 0.5s ease-out;
}}
@keyframes fadein {{
  from {{ color: #0d1117; }}
}}
/* Trail: fading path showing last ~5 player positions */
.map-char.trail {{
  border-radius: 2px;
}}
.map-char.trail-age-1 {{ background: rgba(88,166,255,0.22); }}
.map-char.trail-age-2 {{ background: rgba(88,166,255,0.16); }}
.map-char.trail-age-3 {{ background: rgba(88,166,255,0.11); }}
.map-char.trail-age-4 {{ background: rgba(88,166,255,0.07); }}
.map-char.trail-age-5 {{ background: rgba(88,166,255,0.04); }}
/* Tile flash: brief pulse when a tile changes */
@keyframes tilePulse {{
  0% {{ background: rgba(255,255,100,0.5); }}
  100% {{ background: transparent; }}
}}
.map-char.tile-flash {{
  animation: tilePulse 0.6s ease-out;
}}

/* === HP BAR === */
.hp-section {{
  padding: 10px 14px;
  border-top: 1px solid #30363d;
}}
.hp-bar-outer {{
  background: #21262d;
  border-radius: 4px;
  height: 20px;
  position: relative;
  overflow: hidden;
}}
.hp-bar-fill {{
  height: 100%;
  border-radius: 4px;
  transition: width 0.4s ease, background 0.4s ease;
}}
.hp-bar-text {{
  position: absolute;
  top: 2px;
  left: 8px;
  font-size: 12px;
  color: #fff;
  text-shadow: 0 0 3px #000;
  line-height: 16px;
}}

/* === INFO PANEL === */
.info-panel {{
  display: flex;
  flex-direction: column;
  gap: 15px;
}}
.info-card {{
  background: #0d1117;
  border: 1px solid #30363d;
  border-radius: 8px;
  overflow: hidden;
}}
.info-card .card-body {{
  padding: 12px 14px;
}}

/* === ACTION DISPLAY === */
.action-display {{
  font-size: 16px;
  padding: 10px 14px;
}}
.action-name {{
  color: #58a6ff;
  font-weight: bold;
  font-size: 18px;
}}
.delta-info {{
  margin-top: 8px;
  font-size: 13px;
}}
.delta-tag {{
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  margin-right: 6px;
  margin-bottom: 4px;
  font-size: 12px;
}}
.delta-tag.gold {{ background: #0d2818; color: #3fb950; border: 1px solid #1b4332; }}
.delta-tag.damage {{ background: #2d1215; color: #f85149; border: 1px solid #6e2d30; }}
.delta-tag.explore {{ background: #0d1d33; color: #58a6ff; border: 1px solid #1b3a5c; }}
.delta-tag.heal {{ background: #0d2818; color: #56d364; border: 1px solid #1b4332; }}
.delta-tag.depth {{ background: #2b1a3d; color: #bc8cff; border: 1px solid #4b2d6e; }}
.delta-tag.death {{ background: #3d0a0a; color: #ff4444; border: 1px solid #8b0000; }}
.delta-tag.move {{ background: #161b22; color: #8b949e; border: 1px solid #30363d; }}

/* === EVENT MESSAGE === */
.event-msg {{
  padding: 8px 14px;
  font-size: 12px;
  color: #8b949e;
  border-top: 1px solid #21262d;
  font-style: italic;
  min-height: 20px;
}}

/* === TRAINING DATA === */
.training-card .card-body {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}}
@media (max-width: 800px) {{
  .training-card .card-body {{ grid-template-columns: 1fr; }}
}}
.code-block {{
  background: #161b22;
  border: 1px solid #21262d;
  border-radius: 6px;
  padding: 10px;
  font-size: 11px;
  line-height: 1.6;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 250px;
  overflow-y: auto;
}}
.code-block .label {{
  color: #ff7b72;
  font-weight: bold;
}}
.code-block .value {{ color: #a5d6ff; }}
.code-block .key {{ color: #d2a8ff; }}
.code-block .action-label {{ color: #79c0ff; font-weight: bold; }}
.block-label {{
  font-size: 11px;
  color: #8b949e;
  text-transform: uppercase;
  margin-bottom: 5px;
  letter-spacing: 0.5px;
}}

/* === STEP LIST (mini timeline) === */
.step-timeline {{
  background: #0d1117;
  border: 1px solid #30363d;
  border-radius: 8px;
  padding: 10px;
  margin-top: 15px;
  overflow-x: auto;
  white-space: nowrap;
}}
.step-dot {{
  display: inline-block;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: #21262d;
  border: 2px solid #30363d;
  text-align: center;
  line-height: 18px;
  font-size: 9px;
  color: #8b949e;
  cursor: pointer;
  margin-right: 4px;
  transition: all 0.15s;
  vertical-align: middle;
}}
.step-dot:hover {{ border-color: #58a6ff; }}
.step-dot.active {{ background: #388bfd; color: #fff; border-color: #58a6ff; }}
.step-dot.has-gold {{ border-color: #3fb950; }}
.step-dot.has-damage {{ border-color: #f85149; }}
.step-dot.has-explore {{ border-color: #58a6ff; }}
.step-dot.has-death {{ background: #3d0a0a; border-color: #f85149; color: #f85149; }}

/* Scrollbar styling */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: #0d1117; }}
::-webkit-scrollbar-thumb {{ background: #30363d; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: #484f58; }}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <h1>&#x1F3AE; rl-nethack: NetHack Forward Model Training Data</h1>
  <div class="subtitle">Interactive replay viewer with step-by-step animation and training pair visualization</div>
  <div class="stats-bar">
    <div class="stat-item">
      <div class="stat-label">Games</div>
      <div class="stat-value" id="stat-games">5</div>
    </div>
    <div class="stat-item">
      <div class="stat-label">Total Steps</div>
      <div class="stat-value" id="stat-steps">0</div>
    </div>
    <div class="stat-item">
      <div class="stat-label">Training Examples</div>
      <div class="stat-value" id="stat-examples">0</div>
    </div>
    <div class="stat-item">
      <div class="stat-label">Seeds</div>
      <div class="stat-value">42, 99, 7, 1337, 2024</div>
    </div>
  </div>
</div>

<!-- GAME TABS -->
<div class="game-tabs" id="game-tabs"></div>

<!-- GAME PANELS -->
<div id="game-panels"></div>

<script>
// ===== GAME DATA (embedded) =====
const GAMES = {games_json};

// ===== GLOBALS =====
const CHAR_STYLES = {{
  '@': 'player',
  '-': 'wall', '|': 'wall',
  '.': 'floor',
  '#': 'corridor',
  '+': 'door',
  '$': 'gold-item',
  '>': 'stairs', '<': 'stairs',
  '?': 'item', '!': 'item', '/': 'item', '=': 'item',
  '*': 'item', '"': 'item', '(': 'item', ')': 'item',
  '[': 'item', '%': 'item', '^': 'item',
  ' ': 'unseen',
}};
function isMonsterChar(c) {{
  return /[a-zA-Z]/.test(c) && c !== '@' && c !== 'I' && c !== ' ';
}}

let currentGame = 0;
let currentStep = 0;
let isPlaying = false;
let playSpeed = 1000; // ms
let playTimer = null;

// Persistent map grid state per game
const mapGrids = {{}};      // gi -> {{ rows, cols, spans:[][] }}
const prevMapText = {{}};     // gi -> previous map text for change detection

// ===== INIT =====
function init() {{
  // Set stats
  let totalSteps = GAMES.reduce((s, g) => s + g.total_steps, 0);
  document.getElementById('stat-steps').textContent = totalSteps;
  document.getElementById('stat-examples').textContent = totalSteps;

  // Build tabs
  const tabsEl = document.getElementById('game-tabs');
  const panelsEl = document.getElementById('game-panels');

  GAMES.forEach((game, gi) => {{
    // Tab button
    const btn = document.createElement('button');
    btn.className = 'tab-btn' + (gi === 0 ? ' active' : '');
    btn.textContent = 'Seed ' + game.seed;
    btn.id = 'tab-' + gi;
    btn.onclick = () => switchGame(gi);
    tabsEl.appendChild(btn);

    // Panel
    const panel = document.createElement('div');
    panel.className = 'game-panel' + (gi === 0 ? ' active' : '');
    panel.id = 'panel-' + gi;
    panel.innerHTML = buildGamePanelHTML(game, gi);
    panelsEl.appendChild(panel);
  }});

  // Show first game, first step
  renderStep(0, 0);
}}

function buildGamePanelHTML(game, gi) {{
  const outcomeColor = game.outcome === 'survived' ? '#3fb950' : '#f85149';
  return `
    <div class="game-header">
      <div class="game-meta">
        <div class="meta-item"><span class="meta-label">Seed:</span> <span class="meta-value">${{game.seed}}</span></div>
        <div class="meta-item"><span class="meta-label">Steps:</span> <span class="meta-value">${{game.total_steps}}</span></div>
        <div class="meta-item"><span class="meta-label">Outcome:</span> <span class="meta-value" style="color:${{outcomeColor}}">${{game.outcome}}</span></div>
        <div class="meta-item"><span class="meta-label">Gold:</span> <span class="meta-value" style="color:#3fb950">${{game.total_gold}}</span></div>
        <div class="meta-item"><span class="meta-label">Damage:</span> <span class="meta-value" style="color:#f85149">${{game.total_damage}}</span></div>
        <div class="meta-item"><span class="meta-label">Tiles Explored:</span> <span class="meta-value" style="color:#58a6ff">${{game.tiles_explored}}</span></div>
      </div>
    </div>

    <!-- Controls -->
    <div class="controls">
      <button class="ctrl-btn" id="play-btn-${{gi}}" onclick="togglePlay(${{gi}})">&#9654; Play</button>
      <button class="ctrl-btn" onclick="stepBack(${{gi}})">&#9664;&#9664; Prev</button>
      <button class="ctrl-btn" onclick="stepForward(${{gi}})">&#9654;&#9654; Next</button>
      <button class="ctrl-btn" onclick="resetPlayback(${{gi}})">&#8634; Reset</button>
      <div class="speed-btns">
        <button class="ctrl-btn speed-btn" data-speed="2000" onclick="setSpeed(2000, this)">0.5x</button>
        <button class="ctrl-btn speed-btn active" data-speed="1000" onclick="setSpeed(1000, this)">1x</button>
        <button class="ctrl-btn speed-btn" data-speed="500" onclick="setSpeed(500, this)">2x</button>
        <button class="ctrl-btn speed-btn" data-speed="200" onclick="setSpeed(200, this)">5x</button>
      </div>
      <span class="step-counter" id="step-counter-${{gi}}">Step 0 / ${{game.total_steps}}</span>
    </div>

    <!-- Progress Bar -->
    <div class="progress-container">
      <div class="progress-fill" id="progress-${{gi}}"></div>
    </div>

    <!-- Main replay layout -->
    <div class="replay-layout">
      <!-- Left: Map (fixed viewport) -->
      <div class="map-col">
        <div class="map-panel">
          <div class="panel-title">&#128506; Dungeon Map</div>
          <div class="map-container" id="map-${{gi}}"></div>
          <div class="hp-section">
            <div class="hp-bar-outer">
              <div class="hp-bar-fill" id="hp-fill-${{gi}}"></div>
              <div class="hp-bar-text" id="hp-text-${{gi}}">HP: ?/?</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Right: Info (scrollable) -->
      <div class="info-col">
        <div class="info-panel">
        <!-- Action -->
        <div class="info-card">
          <div class="panel-title">&#127918; Action &amp; Outcome</div>
          <div class="action-display" id="action-${{gi}}">
            <div>Action: <span class="action-name">--</span></div>
          </div>
          <div class="delta-info" id="delta-${{gi}}"></div>
          <div class="event-msg" id="event-msg-${{gi}}"></div>
        </div>

        <!-- Training Data -->
        <div class="info-card training-card">
          <div class="panel-title">&#129302; Training Data Pair</div>
          <div class="card-body">
            <div>
              <div class="block-label">Prompt (input)</div>
              <div class="code-block" id="prompt-${{gi}}">--</div>
            </div>
            <div>
              <div class="block-label">Target (prediction)</div>
              <div class="code-block" id="target-${{gi}}">--</div>
            </div>
          </div>
        </div>
      </div>
      </div>
    </div>

    <!-- Step Timeline -->
    <div class="step-timeline" id="timeline-${{gi}}"></div>
  `;
}}

// ===== RENDERING =====
function renderStep(gi, si) {{
  const game = GAMES[gi];
  if (si < 0) si = 0;
  if (si >= game.steps.length) si = game.steps.length - 1;

  const step = game.steps[si];

  // Update step counter
  document.getElementById('step-counter-' + gi).textContent =
    'Step ' + (si + 1) + ' / ' + game.total_steps;

  // Update progress bar
  const pct = game.total_steps > 0 ? ((si + 1) / game.total_steps * 100) : 0;
  document.getElementById('progress-' + gi).style.width = pct + '%';

  // Render map (persistent grid, updates in-place)
  renderMap(gi, step, si);

  // HP bar
  renderHP(gi, step);

  // Action & delta
  renderAction(gi, step);

  // Training data
  renderTraining(gi, step);

  // Timeline dots
  renderTimeline(gi, si);
}}

function initMapGrid(gi) {{
  const game = GAMES[gi];
  let maxRows = 0, maxCols = 0;
  game.steps.forEach(step => {{
    const lines = step.map.split('\\n');
    maxRows = Math.max(maxRows, lines.length);
    lines.forEach(line => {{ maxCols = Math.max(maxCols, line.length); }});
  }});

  const container = document.getElementById('map-' + gi);
  container.innerHTML = '';
  const spans = [];
  for (let r = 0; r < maxRows; r++) {{
    const rowSpans = [];
    const lineSpan = document.createElement('span');
    lineSpan.className = 'map-line';
    for (let c = 0; c < maxCols; c++) {{
      const sp = document.createElement('span');
      sp.className = 'map-char unseen';
      sp.textContent = ' ';
      lineSpan.appendChild(sp);
      rowSpans.push(sp);
    }}
    container.appendChild(lineSpan);
    if (r < maxRows - 1) container.appendChild(document.createTextNode('\\n'));
    spans.push(rowSpans);
  }}
  mapGrids[gi] = {{ rows: maxRows, cols: maxCols, spans }};
  prevMapText[gi] = '';
}}

function renderMap(gi, step, stepIndex) {{
  if (!mapGrids[gi]) initMapGrid(gi);

  const grid = mapGrids[gi];
  const game = GAMES[gi];
  const lines = step.map.split('\\n');
  const playerPos = step.player_pos;
  const prevText = prevMapText[gi];
  const prevLines = prevText ? prevText.split('\\n') : [];

  // Build trail from previous steps (last 5 positions before current)
  const trailMap = {{}}; // "r,c" -> age (1=newest, 5=oldest)
  const trailLen = 5;
  for (let i = Math.max(0, stepIndex - trailLen); i < stepIndex; i++) {{
    const pp = game.steps[i] && game.steps[i].player_pos;
    if (pp) {{
      const key = pp[0] + ',' + pp[1];
      if (!trailMap[key]) {{
        trailMap[key] = stepIndex - i;
      }}
    }}
  }}

  for (let r = 0; r < grid.rows; r++) {{
    const line = r < lines.length ? lines[r] : '';
    const prevLine = r < prevLines.length ? prevLines[r] : '';
    for (let c = 0; c < grid.cols; c++) {{
      const ch = c < line.length ? line[c] : ' ';
      const prevCh = c < prevLine.length ? prevLine[c] : ' ';
      const span = grid.spans[r][c];

      let cls = CHAR_STYLES[ch] || 'unseen';
      if (isMonsterChar(ch)) cls = 'monster';
      if (ch === 'I') cls = 'unseen';

      const isPlayer = playerPos && r === playerPos[0] && c === playerPos[1];
      if (isPlayer) cls = 'player';

      // Trail highlight (not on current player position)
      let trailCls = '';
      if (!isPlayer) {{
        const tKey = r + ',' + c;
        const age = trailMap[tKey];
        if (age) trailCls = ' trail trail-age-' + Math.min(age, 5);
      }}

      // Detect changed tiles: newly visible, entity appeared/disappeared
      let flashCls = '';
      const becameVisible = (prevCh === ' ' || !prevCh) && ch !== ' ' && cls !== 'unseen';
      const entityChanged = prevCh !== ch && ch !== ' ' && prevCh !== ' ' &&
                            (isMonsterChar(ch) || '$?!/='.includes(ch) ||
                             isMonsterChar(prevCh) || '$?!/='.includes(prevCh));
      if (becameVisible || entityChanged) flashCls = ' tile-flash';

      span.textContent = ch;
      span.className = 'map-char ' + cls + trailCls + flashCls;
    }}
  }}

  prevMapText[gi] = step.map;
}}

function renderHP(gi, step) {{
  const hp = step.hp;
  const hpMax = step.hp_max;
  const pct = hpMax > 0 ? Math.round(hp / hpMax * 100) : 0;
  let color = '#3fb950';
  if (pct <= 30) color = '#f85149';
  else if (pct <= 60) color = '#d29922';

  const fill = document.getElementById('hp-fill-' + gi);
  fill.style.width = pct + '%';
  fill.style.background = color;
  document.getElementById('hp-text-' + gi).textContent = 'HP: ' + hp + ' / ' + hpMax;
}}

function renderAction(gi, step) {{
  const actionEl = document.getElementById('action-' + gi);
  const deltaEl = document.getElementById('delta-' + gi);
  const msgEl = document.getElementById('event-msg-' + gi);

  actionEl.innerHTML = '<div>Action: <span class="action-name">' + step.action.toUpperCase() + '</span></div>';

  let tags = '';
  // Move tag
  const dx = step.pos_delta[0], dy = step.pos_delta[1];
  if (dx !== 0 || dy !== 0) {{
    tags += '<span class="delta-tag move">Moved (' + dx + ',' + dy + ')</span>';
  }} else if (step.action === 'wait') {{
    tags += '<span class="delta-tag move">Waited</span>';
  }} else {{
    tags += '<span class="delta-tag move">Acted</span>';
  }}

  // Gold
  if (step.gold_delta > 0) {{
    tags += '<span class="delta-tag gold">Gold +' + step.gold_delta + '</span>';
  }}
  // Damage/heal
  if (step.hp_delta < 0) {{
    tags += '<span class="delta-tag damage">HP ' + step.hp_delta + '</span>';
  }} else if (step.hp_delta > 0) {{
    tags += '<span class="delta-tag heal">HP +' + step.hp_delta + '</span>';
  }}
  // Explore
  if (step.new_tiles_count > 0) {{
    tags += '<span class="delta-tag explore">+' + step.new_tiles_count + ' tile' + (step.new_tiles_count > 1 ? 's' : '') + '</span>';
  }}
  // Depth
  if (step.depth_delta !== 0) {{
    tags += '<span class="delta-tag depth">Depth ' + (step.depth_delta > 0 ? '+' : '') + step.depth_delta + '</span>';
  }}
  // Death
  if (!step.survived) {{
    tags += '<span class="delta-tag death">&#9760; DIED</span>';
  }}

  deltaEl.innerHTML = tags;
  msgEl.textContent = step.message || '';
}}

function renderTraining(gi, step) {{
  const promptEl = document.getElementById('prompt-' + gi);
  const targetEl = document.getElementById('target-' + gi);

  promptEl.innerHTML = syntaxHighlight(step.prompt);
  targetEl.innerHTML = syntaxHighlight(step.target);
}}

function syntaxHighlight(text) {{
  if (!text) return '';
  let escaped = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Highlight labels like "HP:" "Action:" etc.
  escaped = escaped.replace(/\\b(HP|AC|Str|Dex|Pos|Gold|Depth|Turn|Adjacent|Monsters|Items|Action|pos|hp|gold|depth|alive|msg)\\b/g,
    '<span class="label">$1</span>');

  // Highlight numbers
  escaped = escaped.replace(/\\b(\\d+)\\b/g, '<span class="value">$1</span>');

  // Highlight special values
  escaped = escaped.replace(/\\b(same|none|yes|no)\\b/g, '<span class="key">$1</span>');

  // Highlight direction names
  escaped = escaped.replace(/\\b(north|south|east|west)\\b/g, '<span class="action-label">$1</span>');

  return escaped;
}}

function renderTimeline(gi, activeStep) {{
  const game = GAMES[gi];
  const container = document.getElementById('timeline-' + gi);

  let html = '';
  game.steps.forEach((step, i) => {{
    let dotClass = 'step-dot';
    if (i === activeStep) dotClass += ' active';
    if (step.gold_delta > 0) dotClass += ' has-gold';
    if (step.hp_delta < 0) dotClass += ' has-damage';
    if (step.new_tiles_count > 0) dotClass += ' has-explore';
    if (!step.survived) dotClass += ' has-death';
    html += '<span class="' + dotClass + '" onclick="jumpStep(' + gi + ',' + i + ')" title="Step ' + (i+1) + '">' + (i+1) + '</span>';
  }});
  container.innerHTML = html;
}}

// ===== PLAYBACK CONTROLS =====
function switchGame(gi) {{
  stopPlay();
  currentGame = gi;

  // Update tabs
  document.querySelectorAll('.tab-btn').forEach((btn, i) => {{
    btn.classList.toggle('active', i === gi);
  }});

  // Update panels
  document.querySelectorAll('.game-panel').forEach((panel, i) => {{
    panel.classList.toggle('active', i === gi);
  }});

  // Render first step (grid will init if needed)
  currentStep = 0;
  renderStep(gi, 0);
}}

function togglePlay(gi) {{
  if (isPlaying) {{
    stopPlay();
  }} else {{
    startPlay(gi);
  }}
}}

function startPlay(gi) {{
  isPlaying = true;
  const btn = document.getElementById('play-btn-' + gi);
  btn.innerHTML = '&#9646;&#9646; Pause';
  btn.classList.add('active');

  const game = GAMES[gi];
  // If at end, restart
  if (currentStep >= game.steps.length - 1) {{
    currentStep = 0;
  }}

  playTimer = setInterval(() => {{
    const game = GAMES[currentGame];
    if (currentStep < game.steps.length - 1) {{
      currentStep++;
      renderStep(currentGame, currentStep);
    }} else {{
      stopPlay();
    }}
  }}, playSpeed);
}}

function stopPlay() {{
  isPlaying = false;
  if (playTimer) {{
    clearInterval(playTimer);
    playTimer = null;
  }}
  const btn = document.getElementById('play-btn-' + currentGame);
  if (btn) {{
    btn.innerHTML = '&#9654; Play';
    btn.classList.remove('active');
  }}
}}

function stepForward(gi) {{
  stopPlay();
  const game = GAMES[gi];
  if (currentStep < game.steps.length - 1) {{
    currentStep++;
    renderStep(gi, currentStep);
  }}
}}

function stepBack(gi) {{
  stopPlay();
  if (currentStep > 0) {{
    currentStep--;
    renderStep(gi, currentStep);
  }}
}}

function jumpStep(gi, si) {{
  stopPlay();
  currentStep = si;
  renderStep(gi, si);
}}

function resetPlayback(gi) {{
  stopPlay();
  currentStep = 0;
  renderStep(gi, 0);
}}

function setSpeed(ms, el) {{
  playSpeed = ms;
  document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
  el.classList.add('active');
  // If currently playing, restart timer with new speed
  if (isPlaying) {{
    stopPlay();
    startPlay(currentGame);
  }}
}}

// ===== BOOT =====
document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>""";

    return html


def main():
    print("=" * 60)
    print("Generating NetHack Demo Report")
    print("=" * 60)
    print(f"Seeds: {SEEDS}")
    print(f"Steps per game: {MAX_STEPS}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    encoder = StateEncoder()
    all_games = []

    for seed in SEEDS:
        print(f"Running game with seed={seed} ({MAX_STEPS} steps)...")
        try:
            result = run_and_report(
                seed=seed,
                max_steps=MAX_STEPS,
                encoder=encoder,
            )
            serialized = serialize_game_data(result)
            all_games.append(serialized)
            print(f"  -> {serialized['total_steps']} steps, outcome={serialized['outcome']}, "
                  f"gold={serialized['total_gold']}, damage={serialized['total_damage']}, "
                  f"tiles={serialized['tiles_explored']}")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            import traceback
            traceback.print_exc()
            print(f"  -> Skipping seed {seed}")

    if not all_games:
        print("ERROR: No games were successfully run. Cannot generate report.")
        sys.exit(1)

    print(f"\nSuccessfully ran {len(all_games)} games.")
    print("Building HTML report...")

    html = build_html(all_games)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        f.write(html)

    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"\nReport written to: {OUTPUT_PATH}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")

    if file_size < 50_000:
        print("WARNING: File size is smaller than expected (< 50KB). The data may be incomplete.")
    else:
        print("File size looks good! (> 50KB)")

    print("\nDone! Open the report in a browser to view the interactive demo.")


if __name__ == '__main__':
    main()
