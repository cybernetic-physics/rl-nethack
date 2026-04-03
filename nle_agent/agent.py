#!/usr/bin/env python3
"""
NetHack LLM Agent - baseline zero-shot play via Qwen 2.5 3B (llama.cpp)

Reads NetHack tty screen, sends to model as text, parses action from response.
"""

import subprocess
import json
import time
import sys
import os
import argparse

import nle.env

LLAMA_CLI = os.environ.get(
    "LLAMA_CLI",
    "/home/amiller/projects/llama_reproducible/llama.cpp/build/bin/llama-cli"
)
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/home/amiller/projects/llama_reproducible/models/qwen2.5-3b-instruct-q4_k_m.gguf"
)

# NetHack actions we allow the model to use
# NLE uses integer actions; these map to NetHack commands
ACTION_MAP = {
    "north": ord("k"),
    "south": ord("j"),
    "east": ord("l"),
    "west": ord("h"),
    "northwest": ord("y"),
    "northeast": ord("u"),
    "southwest": ord("b"),
    "southeast": ord("n"),
    "wait": ord("."),
    "pickup": ord(","),
    "open": ord("o"),
    "kick": ord("K"),  # Ctrl-K doesn't work, use Kick
    "search": ord("s"),
    "up": ord("<"),
    "down": ord(">"),
    "eat": ord("e"),
    "drink": ord("q"),
    "read": ord("r"),
    "zap": ord("z"),
    "fire": ord("f"),
    "throw": ord("t"),
    "wear": ord("W"),
    "takeoff": ord("T"),
    "wield": ord("w"),
    "drop": ord("d"),
    "inventory": ord("i"),
    # For misc raw commands
    "rest": ord("."),
    "attack_north": ord("k"),
    "attack_south": ord("j"),
    "attack_east": ord("l"),
    "attack_west": ord("h"),
}

# Direction aliases
DIRECTION_ALIASES = {
    "up": "north", "down": "south", "left": "west", "right": "east",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "move up": "north", "move down": "south", "move left": "west", "move right": "east",
    "go up": "north", "go down": "south", "go left": "west", "go right": "east",
    "go north": "north", "go south": "south", "go east": "east", "go west": "west",
}


def render_nethack_state(obs):
    """Convert NLE observation to a text representation for the LLM."""
    lines = []

    # Get the message bar
    msg = bytes(obs["message"]).decode("ascii", errors="replace").strip().rstrip("\x00")
    if msg:
        lines.append(f"Message: {msg}")
        lines.append("")

    # Render the map from chars (21x79 dungeon view)
    chars = obs["chars"]
    lines.append("Dungeon Map:")
    for i in range(chars.shape[0]):
        row = bytes(chars[i]).decode("ascii", errors="replace").rstrip()
        if row.strip():  # skip blank rows
            lines.append(row)
    lines.append("")

    # Player stats from blstats
    bl = obs["blstats"]
    stats = {
        "depth": int(bl[12]),
        "gold": int(bl[13]),
        "hp": int(bl[10]),
        "max_hp": int(bl[11]),
        "energy": int(bl[14]),
        "max_energy": int(bl[15]),
        "ac": int(bl[16]),
        "level": int(bl[18]),
        "str": int(bl[3]),
        "dex": int(bl[4]),
        "con": int(bl[5]),
        "int": int(bl[6]),
        "wis": int(bl[7]),
        "cha": int(bl[8]),
    }
    lines.append(f"Stats: HP:{stats['hp']}/{stats['max_hp']} "
                 f"Lvl:{stats['level']} AC:{stats['ac']} "
                 f"Str:{stats['str']} Dex:{stats['dex']} "
                 f"Gold:{stats['gold']} Depth:{stats['depth']}")

    return "\n".join(lines)


SYSTEM_PROMPT = """You are an AI playing NetHack. You will see the current game state as a text map and stats. Choose exactly ONE action from this list:

move: north, south, east, west, northeast, northwest, southeast, southwest
other: wait, pickup, open, search, eat, drink, read, wield, drop, up, down, kick, zap

Reply with ONLY the action word, nothing else. Think carefully about survival.
Common NetHack tips:
- @ is you, d is your pet dog
- Letters are monsters (be careful!)
- . is floor, # is corridor, | and - are walls
- < goes upstairs, > goes downstairs
- , picks up items on the ground
- s searches for hidden doors and traps
- Always check HP before fighting"""


def query_model(state_text, history, timeout=30):
    """Send the game state to the model via llama-cli and get an action."""
    # Build conversation
    prompt_parts = [SYSTEM_PROMPT]

    # Include last few moves for context (keep it short to stay fast)
    for h in history[-4:]:
        prompt_parts.append(f"State:\n{h['state']}\nAction: {h['action']}")

    prompt_parts.append(f"State:\n{state_text}\nAction:")

    full_prompt = "\n\n".join(prompt_parts)

    try:
        result = subprocess.run(
            [
                LLAMA_CLI,
                "-m", MODEL_PATH,
                "-p", full_prompt,
                "-n", "10",  # max tokens to generate
                "--threads", "6",
                "--temp", "0.3",  # low temp for more deterministic actions
                "--no-display-prompt",
                "-ngl", "0",  # CPU only
                "--log-disable",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        response = result.stdout.strip()

        # Clean up the response - take just the first line/word
        for line in response.split("\n"):
            line = line.strip()
            if line and not line.startswith("[") and not line.startswith("llama"):
                return line.strip().rstrip(".!,").lower()
        return "wait"

    except subprocess.TimeoutExpired:
        return "wait"
    except Exception as e:
        print(f"  [MODEL ERROR] {e}", file=sys.stderr)
        return "wait"


def parse_action(raw_action):
    """Parse the model's text output into an NLE action integer."""
    action = raw_action.strip().lower()

    # Handle direction aliases
    if action in DIRECTION_ALIASES:
        action = DIRECTION_ALIASES[action]

    # Direct match
    if action in ACTION_MAP:
        return ACTION_MAP[action], action

    # Try partial match
    for key in ACTION_MAP:
        if key.startswith(action) or action.startswith(key):
            return ACTION_MAP[key], key

    # Handle "move X" patterns
    for word in action.split():
        if word in ACTION_MAP:
            return ACTION_MAP[word], word
        if word in DIRECTION_ALIASES:
            resolved = DIRECTION_ALIASES[word]
            return ACTION_MAP[resolved], resolved

    # Default: wait
    return ord("."), "wait"


def run_game(seed=None, max_steps=200, render=True, verbose=True):
    """Run a complete NetHack game with the LLM agent."""
    env = nle.env.NLE()
    obs, info = env.reset(seed=seed)

    total_reward = 0
    step = 0
    history = []
    start_time = time.time()
    all_states = []

    if verbose:
        print("=" * 60)
        print(f"  NETHACK LLM AGENT - Baseline Run")
        print(f"  Model: {os.path.basename(MODEL_PATH)}")
        print(f"  Seed: {seed}")
        print(f"  Max steps: {max_steps}")
        print("=" * 60)
        print()

    for step in range(max_steps):
        state_text = render_nethack_state(obs)

        if verbose and step % 5 == 0:
            print(f"--- Step {step} ---")
            print(state_text)
            print()

        # Get action from model
        raw_action = query_model(state_text, history)
        action_int, action_name = parse_action(raw_action)

        if verbose:
            print(f"  Model says: '{raw_action}' -> action: {action_name} ({action_int})")

        # Record history (keep compact)
        history.append({
            "state": state_text.split("\n")[-1] if state_text else "",  # just stats line
            "action": action_name,
        })
        # Keep only last 8 moves in history
        if len(history) > 8:
            history = history[-8:]

        # Save full state for replay
        all_states.append({
            "step": step,
            "state": state_text,
            "raw_action": raw_action,
            "action": action_name,
            "hp": int(obs["blstats"][10]),
            "reward": float(info.get("reward", 0)) if isinstance(info, dict) else 0,
        })

        # Execute action
        obs, reward, terminated, truncated, info = env.step(action_int)
        total_reward += reward

        if verbose:
            bl = obs["blstats"]
            print(f"  HP: {bl[10]}/{bl[11]} | Gold: {bl[13]} | Depth: {bl[12]} | Reward: {reward}")

        if terminated or truncated:
            if verbose:
                print()
                print(f"  *** GAME OVER at step {step} ***")
                msg = bytes(obs["message"]).decode("ascii", errors="replace").strip().rstrip("\x00")
                if msg:
                    print(f"  Last message: {msg}")
            break

    elapsed = time.time() - start_time

    if verbose:
        print()
        print("=" * 60)
        print(f"  FINAL RESULTS")
        print(f"  Steps survived: {step + 1}")
        print(f"  Total reward: {total_reward}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/(step+1):.1f}s/step)")
        bl = obs["blstats"]
        print(f"  Final HP: {bl[10]}/{bl[11]}")
        print(f"  Final depth: {bl[12]}")
        print(f"  Final gold: {bl[13]}")
        print(f"  Final level: {bl[18]}")
        print("=" * 60)

    env.close()

    return {
        "seed": seed,
        "steps": step + 1,
        "total_reward": float(total_reward),
        "elapsed_seconds": elapsed,
        "steps_per_second": (step + 1) / elapsed,
        "final_hp": int(obs["blstats"][10]),
        "final_depth": int(obs["blstats"][12]),
        "final_gold": int(obs["blstats"][13]),
        "final_level": int(obs["blstats"][18]),
        "states": all_states,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NetHack LLM Agent")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per game")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    parser.add_argument("--save", type=str, help="Save game log to JSON file")
    args = parser.parse_args()

    result = run_game(seed=args.seed, max_steps=args.max_steps, verbose=not args.quiet)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Game log saved to {args.save}")
