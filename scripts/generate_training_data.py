#!/usr/bin/env python3
"""
Generate enriched training data using an LLM policy.

Supports: ZAI (Zhipu GLM), OpenRouter, or local llama-server.
Uses concurrent game execution for throughput.

Usage:
  python scripts/generate_training_data.py --api-key KEY --model glm-4.5-flash --num-games 5 --max-steps 30
  python scripts/generate_training_data.py --dry-run --api-key KEY --model glm-4.5-flash
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import nle.env

from src.memory_tracker import (
    MemoryTracker,
    format_enriched_prompt,
    format_enriched_target,
)
from src.state_encoder import StateEncoder
from nle_agent.agent_http import (
    _build_action_map,
    query_model,
    parse_action,
    render_state,
    SYSTEM_PROMPT,
)

SYSTEM_PROMPT_FORWARD = (
    "Predict the outcome of a NetHack action. "
    "Given accumulated exploration memory, the current viewport, stats, and an action, "
    "predict the resulting changes: position delta, HP change, gold change, depth change, "
    "survival, new tiles explored, and game message."
)

# Tighter system prompt for the policy -- reduces verbose responses
POLICY_SYSTEM_PROMPT = (
    "Reply with exactly one action word: north, south, east, west, northeast, northwest, "
    "southeast, southwest, wait, pickup, open, search, eat, drink, kick, or drop."
)

ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"

import threading
_rate_lock = threading.Lock()
_rate_last_call = 0.0

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
    messages = [{"role": "system", "content": POLICY_SYSTEM_PROMPT}]
    window = history[-4:]
    for h in window:
        messages.append({"role": "user", "content": h["state"]})
        messages.append({"role": "assistant", "content": h["action"]})
    messages.append({"role": "user", "content": state_text})

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
        state_text = render_state(obs)

        if query_fn:
            raw_action = query_fn(state_text, history)
        else:
            raw_action = query_model(state_text, history, model=model)
        action_int, action_name = parse_action(raw_action)

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

        history.append({"state": state_text.split("\n")[-1], "action": action_name})
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


def main():
    parser = argparse.ArgumentParser(description="Generate enriched training data for NetHack forward model")
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--model", type=str, default="glm-4.5-flash")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--output", type=str, default="data/training_pairs.jsonl")
    parser.add_argument("--eval-output", type=str, default="data/eval_pairs.jsonl")
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent games (default: 1)")
    parser.add_argument("--cooldown", type=float, default=60, help="Seconds between games (default: 60)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Resolve API key
    api_key = (args.api_key
               or os.environ.get("GLM_API_KEY")
               or os.environ.get("ZAI_API_KEY")
               or os.environ.get("OPENROUTER_API_KEY")
               or "")

    is_openrouter = "/" in args.model
    base_url = args.base_url or ""
    query_fn = None

    if not is_openrouter and api_key:
        print(f"Using ZAI endpoint: model={args.model} workers={args.workers}")
        _m, _k, _u = args.model, api_key, base_url
        query_fn = lambda st, h: query_zai(st, h, model=_m, api_key=_k, base_url=_u)
    elif is_openrouter and api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
        print(f"Using OpenRouter: model={args.model}")
    elif not api_key:
        print("Using local llama-server (no API key)")
        args.model = None
    else:
        print(f"Using local llama-server with model={args.model}")

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

    def run_one(seed):
        return seed, run_game_with_memory(
            seed=seed, max_steps=args.max_steps, model=args.model,
            verbose=True, query_fn=query_fn
        )

    # Run games concurrently
    pending = list(remaining_seeds)
    games_done = len(completed_seeds)

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
                with open(outf_path, 'a') as outf:
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
