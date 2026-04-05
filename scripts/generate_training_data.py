#!/usr/bin/env python3
"""
Generate enriched training data using an LLM policy.

Supports: ZAI (Zhipu GLM), OpenRouter, or a local OpenAI-compatible server such
as vLLM.
Uses concurrent game execution for throughput.

Usage:
  python scripts/generate_training_data.py --api-key KEY --backend zai --model glm-4.5-flash --num-games 5 --max-steps 30
  python scripts/generate_training_data.py --backend vllm --model Qwen/Qwen2.5-1.5B-Instruct --server-url http://127.0.0.1:8000/v1 --workers 64
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import nle.env

from src.memory_tracker import (
    MemoryTracker,
    UNSEEN,
    format_enriched_prompt,
    format_enriched_target,
)
from src.state_encoder import StateEncoder
from nle_agent.agent_http import (
    _build_action_map,
    parse_action,
)

SYSTEM_PROMPT_FORWARD = (
    "Predict the outcome of a NetHack action. "
    "Given accumulated exploration memory, the current viewport, stats, and an action, "
    "predict the resulting changes: position delta, HP change, gold change, depth change, "
    "survival, new tiles explored, and game message."
)

# Tighter system prompt for the policy -- reduces verbose responses
POLICY_SYSTEM_PROMPT = (
    "You are choosing the next NetHack action. "
    "Reply with exactly one action word and nothing else. "
    "Priorities: survive, fight adjacent threats, pick up useful items when standing on them, "
    "open doors that block progress, and keep exploring. "
    "Avoid repeating wait, search, pickup, or open if the previous attempt did not help. "
    "Do not choose inventory actions unless the state explicitly suggests them. "
    "Valid actions: north, south, east, west, northeast, northwest, southeast, southwest, "
    "wait, pickup, open, search, eat, drink, kick, or drop."
)

ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"

import threading
_rate_lock = threading.Lock()
_rate_last_call = 0.0
_server_rr_lock = threading.Lock()
_server_rr_index = 0

def _rate_limit(min_interval=2.5):
    """Ensure minimum interval between API calls across all threads."""
    global _rate_last_call
    with _rate_lock:
        now = time.time()
        wait = min_interval - (now - _rate_last_call)
        if wait > 0:
            time.sleep(wait)
        _rate_last_call = time.time()

# Valid action names for extraction from reasoning
VALID_ACTIONS = [
    "north", "south", "east", "west",
    "northeast", "northwest", "southeast", "southwest",
    "wait", "pickup", "open", "search", "eat", "drink",
    "kick", "drop", "up", "down",
]

DEFAULT_LOCAL_SERVER_URL = "http://127.0.0.1:8000/v1"
WALKABLE_TILES = {
    "floor", "corridor", "stairs_down", "stairs_up", "gold", "scroll",
    "potion", "wand", "ring", "gem", "amulet", "tool", "weapon",
    "armor", "food", "fountain", "altar", "throne", "water",
}


def extract_action(text, reasoning=""):
    """Extract a valid action from model output, checking content then reasoning."""
    for source in [text, reasoning]:
        if not source:
            continue
        # Try exact match first
        lower = source.strip().lower().rstrip(".,!;")
        if lower in VALID_ACTIONS:
            return lower
        # Try first word
        first = lower.split()[0] if lower.split() else ""
        if first in VALID_ACTIONS:
            return first
        # Search for any valid action in the text
        for action in sorted(VALID_ACTIONS, key=len, reverse=True):
            if action in lower:
                return action
    return "wait"


def query_zai(state_text, history, model="glm-4.5-flash", api_key="", base_url=""):
    """Query ZAI / Zhipu GLM API via subprocess curl (avoids urllib timeout issues)."""
    import subprocess
    messages = build_policy_messages(state_text, history)

    url = (base_url or ZAI_BASE_URL).rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.2,
    })

    for attempt in range(3):
        try:
            _rate_limit()
            result = subprocess.run([
                "curl", "-s", "-m", "90",
                url,
                "-X", "POST",
                "-H", "Content-Type: application/json",
                "-H", f"Authorization: Bearer {api_key}",
                "-d", payload,
            ], capture_output=True, text=True, timeout=100)

            if result.returncode != 0:
                print(f"  [CURL ERROR] {result.stderr[:200]}", file=sys.stderr)
                return "wait"

            data = json.loads(result.stdout)
            if "error" in data:
                err = data["error"]
                code = err.get("code", "")
                if "429" in str(code) or "rate" in str(code).lower():
                    wait = 2 ** attempt * 5
                    print(f"  [429 rate limited, waiting {wait}s]", file=sys.stderr)
                    time.sleep(wait)
                    continue
                print(f"  [ZAI ERROR] {err}", file=sys.stderr)
                return "wait"

            msg = data["choices"][0]["message"]
            content = msg.get("content", "")
            reasoning = msg.get("reasoning_content", "")
            action = extract_action(content, reasoning)
            return action
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT attempt {attempt+1}]", file=sys.stderr)
        except Exception as e:
            print(f"  [ZAI ERROR] {e}", file=sys.stderr)
            return "wait"
    return "wait"


def query_openai_compatible(state_text, history, model, base_url, api_key="local-token"):
    """Query a local or remote OpenAI-compatible chat completions endpoint."""
    messages = build_policy_messages(state_text, history)

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": 8,
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"].strip()
            return extract_action(content)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        print(f"  [OPENAI-COMPAT ERROR {e.code}] {body}", file=sys.stderr)
    except Exception as e:
        print(f"  [LOCAL SERVER ERROR] {e}", file=sys.stderr)
    return "wait"


def build_policy_messages(state_text, history):
    """Build chat messages for the policy model."""
    messages = [{"role": "system", "content": POLICY_SYSTEM_PROMPT}]
    window = history[-4:]
    for h in window:
        messages.append({"role": "user", "content": h["state"]})
        messages.append({"role": "assistant", "content": h["action"]})
    messages.append({"role": "user", "content": state_text})
    return messages


def parse_server_urls(server_url_value):
    """Parse one or more comma-separated OpenAI-compatible server URLs."""
    urls = [u.strip().rstrip("/") for u in server_url_value.split(",") if u.strip()]
    return urls or [DEFAULT_LOCAL_SERVER_URL.rstrip("/")]


def choose_server_url(server_urls):
    """Round-robin across multiple local inference replicas."""
    if len(server_urls) == 1:
        return server_urls[0]

    global _server_rr_index
    with _server_rr_lock:
        url = server_urls[_server_rr_index % len(server_urls)]
        _server_rr_index += 1
    return url


def build_policy_state_text(obs, memory, history, encoder):
    """Build a tighter policy prompt from structured state instead of raw map only."""
    state = encoder.encode_full(obs)
    bl = obs["blstats"]
    parts = [
        f"HP:{state['hp']}/{state['hp_max']} Gold:{state['gold']} Depth:{state['depth']} Turn:{state['turn']}",
        f"Position:{state['position']}",
        "Adjacent: " + " ".join(f"{d}={state['adjacent'].get(d, 'unknown')}" for d in ("north", "south", "east", "west")),
    ]
    msg = state.get("message", "")
    if msg:
        parts.append(f"Message: {msg}")

    if state["visible_monsters"]:
        mons = ", ".join(f"{m['char']}@{m['pos']}" for m in state["visible_monsters"][:6])
        parts.append(f"Visible monsters: {mons}")
    else:
        parts.append("Visible monsters: none")

    if state["visible_items"]:
        items = ", ".join(f"{it['type']}@{it['pos']}" for it in state["visible_items"][:6])
        parts.append(f"Visible items: {items}")
    else:
        parts.append("Visible items: none")

    parts.append(f"Explored tiles: {memory.total_explored} Rooms: {len(memory.rooms)}")
    if history:
        recent = ", ".join(h["action"] for h in history[-4:])
        parts.append(f"Recent actions: {recent}")
    else:
        parts.append("Recent actions: none")

    parts.append(
        "Choose one action. Prefer movement over wait when open tiles exist. "
        "Only use pickup if the message says there is something here. "
        "Only use open if a door is adjacent."
    )
    return "\n".join(parts)


def query_vllm_batched(llm, state_history_pairs):
    """Run one in-process vLLM batch over multiple policy states."""
    from vllm import SamplingParams

    messages_batch = [
        build_policy_messages(state_text, history)
        for state_text, history in state_history_pairs
    ]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8,
    )
    outputs = llm.chat(messages_batch, sampling_params=sampling_params, use_tqdm=False)

    actions = []
    for output in outputs:
        text = output.outputs[0].text.strip() if output.outputs else ""
        actions.append(extract_action(text))
    return actions


def _count_unseen_neighbors(memory, y, x):
    unseen = 0
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < memory.map_h and 0 <= nx < memory.map_w:
            if int(memory.explored[ny, nx]) == UNSEEN:
                unseen += 1
    return unseen


def choose_fallback_move(state, memory):
    """Choose a movement fallback that prefers frontiers and low-visit tiles."""
    px, py = state["position"]
    previous_pos = memory.position_history[-2] if len(memory.position_history) >= 2 else None
    candidates = []

    for direction, (dy, dx) in {
        "north": (-1, 0),
        "south": (1, 0),
        "east": (0, 1),
        "west": (0, -1),
    }.items():
        if state["adjacent"].get(direction) not in WALKABLE_TILES:
            continue

        ny, nx = py + dy, px + dx
        if not (0 <= ny < memory.map_h and 0 <= nx < memory.map_w):
            continue

        frontier_score = _count_unseen_neighbors(memory, ny, nx)
        visit_count = int(memory.visit_counts[ny, nx])
        backtrack_penalty = 1 if previous_pos == (ny, nx) else 0
        tile = state["adjacent"].get(direction)
        feature_bonus = 1 if tile in {"stairs_down", "stairs_up", "gold"} else 0
        score = (frontier_score, feature_bonus, -visit_count, -backtrack_penalty)
        candidates.append((score, direction))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return "wait"


def sanitize_action(action_name, obs, history, encoder, memory):
    """Clamp obviously bad repeated no-op behavior into simpler exploration behavior."""
    state = encoder.encode_full(obs)
    msg = state.get("message", "").lower()
    repeated = len(history) >= 2 and history[-1]["action"] == action_name and history[-2]["action"] == action_name
    fallback_move = choose_fallback_move(state, memory)
    adjacent_tiles = set(state["adjacent"].values())

    if action_name == "wait" and fallback_move != "wait":
        return fallback_move
    if repeated and action_name in {"wait", "search", "pickup", "open"} and fallback_move != "wait":
        return fallback_move
    if action_name == "pickup" and "see here" not in msg and "you feel here" not in msg and fallback_move != "wait":
        return fallback_move
    if action_name == "open" and "door" not in adjacent_tiles and fallback_move != "wait":
        return fallback_move
    if action_name in {"eat", "drink", "drop"} and fallback_move != "wait":
        return fallback_move
    if action_name == "kick" and "door" not in adjacent_tiles and "wall" not in adjacent_tiles and fallback_move != "wait":
        return fallback_move
    return action_name


@dataclass
class GameRollout:
    seed: int
    env: object
    encoder: StateEncoder
    memory: MemoryTracker
    obs: dict
    history: list = field(default_factory=list)
    pairs: list = field(default_factory=list)
    total_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    step_count: int = 0


def initialize_rollout(seed):
    env = nle.env.NLE()
    obs, _ = env.reset(seed=seed)
    memory = MemoryTracker()
    memory.update(obs)
    return GameRollout(
        seed=seed,
        env=env,
        encoder=StateEncoder(),
        memory=memory,
        obs=obs,
    )


def close_rollouts(rollouts):
    for rollout in rollouts:
        try:
            rollout.env.close()
        except Exception:
            pass


def append_pair(rollout, step, action_name, raw_action, obs_before, obs_after, reward, model):
    prompt_text = format_enriched_prompt(obs_before, rollout.memory, action_name)
    delta = rollout.encoder.encode_delta(obs_before, obs_after, action_name)
    new_count = sum(
        1 for y in range(rollout.memory.map_h) for x in range(rollout.memory.map_w)
        if rollout.memory.explored[y][x] != 0
    )
    delta["new_tiles"] = [{"tile": "explored", "count": new_count}]
    target_text = format_enriched_target(delta, obs_after, rollout.memory)

    bl = obs_before["blstats"]
    rollout.pairs.append({
        "prompt": prompt_text,
        "target": target_text,
        "metadata": {
            "seed": rollout.seed,
            "step": step,
            "model": model or "local",
            "action": action_name,
            "raw_action": raw_action,
            "reward": float(reward),
            "total_reward": float(rollout.total_reward),
            "hp_before": int(bl[10]),
            "hp_after": int(obs_after["blstats"][10]),
            "turn": int(bl[20]),
            "explored_tiles": rollout.memory.total_explored,
            "rooms_found": len(rollout.memory.rooms),
            "monsters_seen": len(rollout.memory.monster_memory),
            "items_found": len(rollout.memory.items_on_floor),
        },
    })


def step_rollout(rollout, raw_action, model, action_map):
    obs_before = rollout.obs
    state_text = build_policy_state_text(obs_before, rollout.memory, rollout.history, rollout.encoder)
    action_int, action_name = parse_action(raw_action)
    action_name = sanitize_action(
        action_name, rollout.obs, rollout.history, rollout.encoder, rollout.memory
    )
    action_int = action_map.get(action_name, action_map.get("wait", action_int))
    obs_after, reward, terminated, truncated, _ = rollout.env.step(action_int)
    rollout.total_reward += reward
    rollout.memory.update(obs_after)
    append_pair(
        rollout,
        rollout.step_count,
        action_name,
        raw_action,
        obs_before,
        obs_after,
        reward,
        model,
    )
    rollout.history.append({"state": state_text, "action": action_name})
    if len(rollout.history) > 8:
        rollout.history = rollout.history[-8:]
    rollout.obs = obs_after
    rollout.step_count += 1
    rollout.terminated = terminated
    rollout.truncated = truncated


def run_batched_vllm_games(llm, seeds, max_steps, model, verbose=True):
    """Play many NetHack games with one in-process vLLM batch per turn."""
    action_map = _build_action_map()
    rollouts = [initialize_rollout(seed) for seed in seeds]

    try:
        for step in range(max_steps):
            active = [r for r in rollouts if not (r.terminated or r.truncated)]
            if not active:
                break

            state_history_pairs = [
                (build_policy_state_text(r.obs, r.memory, r.history, r.encoder), r.history)
                for r in active
            ]
            raw_actions = query_vllm_batched(llm, state_history_pairs)

            for rollout, _state_history, raw_action in zip(active, state_history_pairs, raw_actions):
                step_rollout(rollout, raw_action, model, action_map)

                if verbose and step % 5 == 0:
                    hp = int(rollout.obs["blstats"][10])
                    hp_max = int(rollout.obs["blstats"][11])
                    print(
                        f"  Seed {rollout.seed:4d} | step {step:3d} | action={rollout.history[-1]['action']:10s} | "
                        f"HP={hp}/{hp_max} | explored={rollout.memory.total_explored} | "
                        f"reward={rollout.total_reward:.0f}"
                    )
    finally:
        close_rollouts(rollouts)

    return {rollout.seed: rollout.pairs for rollout in rollouts}


def run_game_with_memory(seed, max_steps, model=None, verbose=True, query_fn=None):
    """Play one NetHack game with an LLM policy, collecting enriched training pairs."""
    action_map = _build_action_map()
    encoder = StateEncoder()
    memory = MemoryTracker()
    env = nle.env.NLE()

    obs, info = env.reset(seed=seed)
    memory.update(obs)

    history = []
    pairs = []
    total_reward = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Game seed={seed} | model={model or 'local'} | max_steps={max_steps}")
        print(f"{'='*60}")

    for step in range(max_steps):
        state_text = build_policy_state_text(obs, memory, history, encoder)

        if query_fn:
            raw_action = query_fn(state_text, history)
        else:
            raw_action = query_model(state_text, history, model=model)
        action_int, action_name = parse_action(raw_action)
        action_name = sanitize_action(action_name, obs, history, encoder, memory)
        action_int = action_map.get(action_name, action_map.get("wait", action_int))

        prompt_text = format_enriched_prompt(obs, memory, action_name)

        obs_before = obs
        obs_after, reward, terminated, truncated, info = env.step(action_int)
        total_reward += reward

        memory.update(obs_after)
        delta = encoder.encode_delta(obs_before, obs_after, action_name)

        new_count = sum(1 for y in range(memory.map_h) for x in range(memory.map_w)
                        if memory.explored[y][x] != 0)
        delta['new_tiles'] = [{'tile': 'explored', 'count': new_count}]

        target_text = format_enriched_target(delta, obs_after, memory)

        bl = obs_before['blstats']
        pair = {
            "prompt": prompt_text,
            "target": target_text,
            "metadata": {
                "seed": seed, "step": step, "model": model or "local",
                "action": action_name, "raw_action": raw_action,
                "reward": float(reward), "total_reward": float(total_reward),
                "hp_before": int(bl[10]), "hp_after": int(obs_after['blstats'][10]),
                "turn": int(bl[20]), "explored_tiles": memory.total_explored,
                "rooms_found": len(memory.rooms),
                "monsters_seen": len(memory.monster_memory),
                "items_found": len(memory.items_on_floor),
            },
        }
        pairs.append(pair)

        history.append({"state": state_text, "action": action_name})
        if len(history) > 8:
            history = history[-8:]

        if verbose and step % 5 == 0:
            hp = int(obs_before['blstats'][10])
            hp_max = int(obs_before['blstats'][11])
            print(f"  Step {step:3d} | action={action_name:10s} | "
                  f"HP={hp}/{hp_max} | explored={memory.total_explored} | "
                  f"rooms={len(memory.rooms)} | reward={total_reward:.0f}")

        obs = obs_after
        if terminated or truncated:
            msg = bytes(obs['message']).decode('ascii', errors='replace').strip().rstrip('\x00')
            if verbose:
                print(f"  *** GAME OVER at step {step+1}: {msg}")
            break

    env.close()
    if verbose:
        bl = obs['blstats']
        print(f"  Final: {len(pairs)} pairs | reward={total_reward:.0f} | "
              f"depth={int(bl[12])} | gold={int(bl[13])}")
    return pairs


def save_checkpoint(path, completed_seeds, stats):
    with open(path, 'w') as f:
        json.dump({'completed_seeds': completed_seeds, 'stats': stats,
                   'timestamp': time.time()}, f, indent=2)


def load_checkpoint(path):
    if not os.path.exists(path):
        return [], {}
    with open(path) as f:
        data = json.load(f)
    return data.get('completed_seeds', []), data.get('stats', {})


def write_pairs(output_path, pairs):
    with open(output_path, 'a') as outf:
        for pair in pairs:
            conversation = {
                "conversations": [
                    {"role": "system", "content": SYSTEM_PROMPT_FORWARD},
                    {"role": "user", "content": pair["prompt"]},
                    {"role": "assistant", "content": pair["target"]},
                ],
                "metadata": pair["metadata"],
            }
            outf.write(json.dumps(conversation) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Generate enriched training data for NetHack forward model")
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--backend",
        choices=["auto", "zai", "openrouter", "vllm", "local", "vllm-batch"],
        default="auto",
        help="Inference backend. 'vllm' and 'local' use an OpenAI-compatible local server; 'vllm-batch' uses in-process vLLM batching",
    )
    parser.add_argument("--model", type=str, default="glm-4.5-flash")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="OpenAI-compatible local server base URL, or comma-separated URLs for replicas",
    )
    parser.add_argument("--output", type=str, default="data/training_pairs.jsonl")
    parser.add_argument("--eval-output", type=str, default="data/eval_pairs.jsonl")
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent games (default: 1)")
    parser.add_argument("--cooldown", type=float, default=None, help="Seconds between games (default: auto)")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--vllm-max-model-len", type=int, default=2048)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=128)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Resolve API key
    api_key = (args.api_key
               or os.environ.get("GLM_API_KEY")
               or os.environ.get("ZAI_API_KEY")
               or os.environ.get("OPENROUTER_API_KEY")
               or "")

    backend = args.backend
    base_url = args.base_url or ""
    server_url = args.server_url or os.environ.get("LLM_SERVER_URL") or DEFAULT_LOCAL_SERVER_URL
    server_urls = parse_server_urls(server_url)
    query_fn = None

    if backend == "auto":
        if api_key and "/" not in args.model:
            backend = "zai"
        elif api_key and "/" in args.model:
            backend = "openrouter"
        else:
            backend = "vllm-batch"

    if backend == "zai":
        print(f"Using ZAI endpoint: model={args.model} workers={args.workers}")
        _m, _k, _u = args.model, api_key, base_url
        query_fn = lambda st, h: query_zai(st, h, model=_m, api_key=_k, base_url=_u)
    elif backend == "openrouter":
        os.environ["OPENROUTER_API_KEY"] = api_key
        print(f"Using OpenRouter: model={args.model}")
    elif backend == "vllm-batch":
        print(
            f"Using in-process vLLM batching: model={args.model} "
            f"workers={args.workers} tp={args.vllm_tensor_parallel_size}"
        )
    else:
        server_label = ", ".join(server_urls)
        print(f"Using local OpenAI-compatible server: model={args.model} server={server_label} workers={args.workers}")
        _m, _us = args.model, server_urls
        query_fn = lambda st, h: query_openai_compatible(
            st, h, model=_m, base_url=choose_server_url(_us)
        )

    if args.cooldown is None:
        args.cooldown = 45.0 if backend == "zai" else 0.0

    # Dry run
    if args.dry_run:
        print(f"=== DRY RUN: 1 game, 10 steps (model={args.model}) ===")
        pairs = run_game_with_memory(seed=42, max_steps=10, model=args.model,
                                     verbose=True, query_fn=query_fn)
        print(f"\n--- Generated {len(pairs)} pairs ---")
        if pairs:
            print("\nFirst pair:")
            print("PROMPT:")
            print(pairs[0]['prompt'][:500])
            print("...\nTARGET:")
            print(pairs[0]['target'])
            print("\nMETADATA:")
            print(json.dumps(pairs[0]['metadata'], indent=2))
        return

    # Resume support
    checkpoint_path = args.resume or args.output.replace('.jsonl', '_checkpoint.json')
    completed_seeds, stats = load_checkpoint(checkpoint_path) if args.resume else ([], {})

    all_seeds = list(range(args.seed_start, args.seed_start + args.num_games))
    remaining_seeds = [s for s in all_seeds if s not in completed_seeds]

    if completed_seeds:
        print(f"Resuming: {len(completed_seeds)} done, {len(remaining_seeds)} remaining")

    num_eval = int(args.num_games * args.eval_fraction)
    eval_seeds = set(all_seeds[-num_eval:]) if num_eval > 0 else set()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    if args.eval_output:
        os.makedirs(os.path.dirname(args.eval_output) or '.', exist_ok=True)

    start_time = time.time()
    total_pairs = stats.get('total_pairs', 0)
    total_train = stats.get('total_train', 0)
    total_eval = stats.get('total_eval', 0)
    total_tokens_est = stats.get('total_tokens_est', 0)
    games_done = len(completed_seeds)

    def run_one(seed):
        return seed, run_game_with_memory(
            seed=seed, max_steps=args.max_steps, model=args.model,
            verbose=True, query_fn=query_fn
        )

    if backend == "vllm-batch":
        seed_batches = [
            remaining_seeds[i:i + args.workers]
            for i in range(0, len(remaining_seeds), args.workers)
        ]
        try:
            from vllm import LLM

            llm = LLM(
                model=args.model,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_len=args.vllm_max_model_len,
                max_num_seqs=args.vllm_max_num_seqs,
                enforce_eager=args.vllm_enforce_eager,
                disable_log_stats=True,
            )
            for seed_batch in seed_batches:
                batch_pairs = run_batched_vllm_games(
                    llm,
                    seed_batch,
                    max_steps=args.max_steps,
                    model=args.model,
                    verbose=False,
                )

                for seed in seed_batch:
                    pairs = batch_pairs[seed]
                    is_eval = seed in eval_seeds
                    outf_path = args.eval_output if is_eval else args.output
                    write_pairs(outf_path, pairs)

                    total_pairs += len(pairs)
                    if is_eval:
                        total_eval += len(pairs)
                    else:
                        total_train += len(pairs)
                    total_tokens_est += len(pairs) * 450
                    completed_seeds.append(seed)
                    games_done += 1

                elapsed = time.time() - start_time
                games_left = len(all_seeds) - games_done
                eta = (elapsed / games_done * games_left) if games_done > 0 else 0
                cost_est = total_tokens_est * 0.0000005
                print(
                    f"\n  Progress: {games_done}/{len(all_seeds)} games | "
                    f"{total_pairs} pairs (train={total_train}, eval={total_eval}) | "
                    f"~{total_tokens_est:,} tokens | ~${cost_est:.2f} | ETA: {eta/60:.0f}min"
                )

                if games_done % args.checkpoint_every == 0:
                    save_checkpoint(checkpoint_path, completed_seeds, {
                        'total_pairs': total_pairs, 'total_train': total_train,
                        'total_eval': total_eval, 'total_tokens_est': total_tokens_est,
                    })
                    print("  Checkpoint saved")

        except KeyboardInterrupt:
            print("\nInterrupted. Saving checkpoint...")
            save_checkpoint(checkpoint_path, completed_seeds, {
                'total_pairs': total_pairs, 'total_train': total_train,
                'total_eval': total_eval, 'total_tokens_est': total_tokens_est,
            })
            raise

        elapsed = time.time() - start_time
        cost_est = total_tokens_est * 0.0000005
        print(f"\n{'='*60}")
        print("  DONE")
        print(f"  Games:    {games_done}")
        print(f"  Pairs:    {total_pairs} (train={total_train}, eval={total_eval})")
        print(f"  Tokens:   ~{total_tokens_est:,}")
        print(f"  Cost:     ~${cost_est:.2f}")
        print(f"  Time:     {elapsed/60:.1f}min")
        print(f"  Train:    {args.output}")
        print(f"  Eval:     {args.eval_output}")
        print(f"{'='*60}")
        return

    # Run games concurrently
    pending = list(remaining_seeds)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            # Seed initial batch
            for seed in pending[:args.workers]:
                futures[pool.submit(run_one, seed)] = seed
            pending = pending[args.workers:]

            while futures:
                done, _ = as_completed(futures).__next__(), None
                seed, pairs = done.result()
                del futures[done]

                is_eval = seed in eval_seeds
                outf_path = args.eval_output if is_eval else args.output
                write_pairs(outf_path, pairs)

                total_pairs += len(pairs)
                if is_eval:
                    total_eval += len(pairs)
                else:
                    total_train += len(pairs)
                total_tokens_est += len(pairs) * 450

                completed_seeds.append(seed)
                games_done += 1

                elapsed = time.time() - start_time
                games_left = len(pending) + len(futures)
                eta = (elapsed / games_done * games_left) if games_done > 0 else 0
                cost_est = total_tokens_est * 0.0000005

                print(f"\n  Progress: {games_done}/{len(all_seeds)} games | "
                      f"{total_pairs} pairs (train={total_train}, eval={total_eval}) | "
                      f"~{total_tokens_est:,} tokens | ~${cost_est:.2f} | "
                      f"ETA: {eta/60:.0f}min")

                if games_done % args.checkpoint_every == 0:
                    save_checkpoint(checkpoint_path, completed_seeds, {
                        'total_pairs': total_pairs, 'total_train': total_train,
                        'total_eval': total_eval, 'total_tokens_est': total_tokens_est,
                    })
                    print(f"  Checkpoint saved")

                # Cooldown between games to respect rate limits
                if args.cooldown > 0 and pending:
                    print(f"  Cooldown {args.cooldown:.0f}s...")
                    time.sleep(args.cooldown)

                # Submit next game
                if pending:
                    next_seed = pending.pop(0)
                    futures[pool.submit(run_one, next_seed)] = next_seed

    except KeyboardInterrupt:
        print(f"\n\nInterrupted! {games_done}/{len(all_seeds)} games, {total_pairs} pairs")
    finally:
        save_checkpoint(checkpoint_path, completed_seeds, {
            'total_pairs': total_pairs, 'total_train': total_train,
            'total_eval': total_eval, 'total_tokens_est': total_tokens_est,
        })

    elapsed = time.time() - start_time
    cost_est = total_tokens_est * 0.0000005
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Games:    {games_done}")
    print(f"  Pairs:    {total_pairs} (train={total_train}, eval={total_eval})")
    print(f"  Tokens:   ~{total_tokens_est:,}")
    print(f"  Cost:     ~${cost_est:.2f}")
    print(f"  Time:     {elapsed/60:.1f}min")
    print(f"  Train:    {args.output}")
    print(f"  Eval:     {args.eval_output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
