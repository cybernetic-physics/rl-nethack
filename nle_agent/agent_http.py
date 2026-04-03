#!/usr/bin/env python3
"""
NetHack LLM Agent - baseline zero-shot play via Qwen 2.5 3B (llama-server HTTP)

Start the server first:
  llama-server -m model.gguf --threads 6 --port 8765 -c 2048

Then run:
  python agent_http.py --seed 42 --max-steps 100
"""

import json
import time
import sys
import os
import argparse
import urllib.request

import nle.env
import nle.nethack as nh

SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8765")


def _build_action_map():
    """Build action name -> NLE action index from the env's actual action list."""
    # ASCII values for NetHack commands
    cmd_ascii = {
        "north": ord("k"), "south": ord("j"), "east": ord("l"), "west": ord("h"),
        "northeast": ord("u"), "northwest": ord("y"), "southeast": ord("n"), "southwest": ord("b"),
        "wait": ord("."), "more": ord("\r"),
        "up": ord("<"), "down": ord(">"),
        "pickup": ord(","), "open": ord("o"), "close": ord("c"),
        "search": ord("s"), "kick": 0x04,  # Ctrl-D
        "eat": ord("e"), "drink": ord("q"), "read": ord("r"),
        "zap": ord("z"), "fire": ord("f"), "throw": ord("t"),
        "wear": ord("W"), "takeoff": ord("T"), "wield": ord("w"),
        "drop": ord("d"), "apply": ord("a"),
    }
    # Create a dummy env to get its actual action list
    env = nle.env.NLE()
    actions = env.actions
    env.close()

    # Build reverse map: action enum value -> index in env.actions
    val_to_idx = {}
    for i, action in enumerate(actions):
        # action can be an enum with .value, or a raw int
        val = action.value if hasattr(action, "value") else action
        val_to_idx[val] = i

    # Map command names to indices
    result = {}
    for name, ascii_val in cmd_ascii.items():
        if ascii_val in val_to_idx:
            result[name] = val_to_idx[ascii_val]

    return result


ACTION_MAP = _build_action_map()

DIRECTION_ALIASES = {
    "up": "north", "down": "south", "left": "west", "right": "east",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "move up": "north", "move down": "south", "move left": "west", "move right": "east",
    "go up": "north", "go down": "south", "go left": "west", "go right": "east",
    "go north": "north", "go south": "south", "go east": "east", "go west": "west",
    "move north": "north", "move south": "south", "move east": "east", "move west": "west",
}


def render_state(obs):
    """Convert NLE observation to compact text for the LLM."""
    lines = []
    msg = bytes(obs["message"]).decode("ascii", errors="replace").strip().rstrip("\x00")
    if msg:
        lines.append(f"Message: {msg}")

    # Compact map - only non-empty rows
    chars = obs["chars"]
    map_lines = []
    for i in range(chars.shape[0]):
        row = bytes(chars[i]).decode("ascii", errors="replace").rstrip()
        if row.strip():
            map_lines.append(row)
    lines.append("Map:")
    lines.extend(map_lines)

    bl = obs["blstats"]
    lines.append(
        f"HP:{bl[10]}/{bl[11]} Lvl:{bl[18]} AC:{bl[16]} "
        f"Str:{bl[3]} Dex:{bl[4]} Gold:{bl[13]} Depth:{bl[12]} "
        f"Turn:{bl[20]}"
    )
    return "\n".join(lines)


SYSTEM_PROMPT = "You play NetHack. Reply with one action: north/south/east/west/northeast/northwest/southeast/southwest/wait/pickup/open/search/eat/drink/up/down/kick. @=you d=pet letters=monsters .=floor <>-|=stairs/walls $=gold"


def query_model(state_text, history):
    """Query llama-server via HTTP API."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include recent history for context (keep very short for speed)
    for h in history[-2:]:
        messages.append({"role": "user", "content": h["state"]})
        messages.append({"role": "assistant", "content": h["action"]})

    messages.append({"role": "user", "content": state_text})

    payload = json.dumps({
        "messages": messages,
        "max_tokens": 8,
        "temperature": 0.2,
    }).encode()

    req = urllib.request.Request(
        f"{SERVER_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"].strip()
            return content
    except Exception as e:
        print(f"  [MODEL ERROR] {e}", file=sys.stderr)
        return "wait"


def parse_action(raw):
    """Parse model output -> NLE action integer."""
    action = raw.strip().lower().rstrip(".!,")

    if action in DIRECTION_ALIASES:
        action = DIRECTION_ALIASES[action]
    if action in ACTION_MAP:
        return ACTION_MAP[action], action

    # Partial match
    for key in ACTION_MAP:
        if action.startswith(key) or key.startswith(action):
            return ACTION_MAP[key], key

    for word in action.split():
        if word in DIRECTION_ALIASES:
            action = DIRECTION_ALIASES[word]
            if action in ACTION_MAP:
                return ACTION_MAP[action], action
        if word in ACTION_MAP:
            return ACTION_MAP[word], word

    return ord("."), "wait"


def run_game(seed=None, max_steps=100, verbose=True):
    """Run a complete NetHack game."""
    env = nle.env.NLE()
    obs, info = env.reset(seed=seed)

    total_reward = 0
    history = []
    all_states = []
    start_time = time.time()

    if verbose:
        print("=" * 60)
        print(f"  NETHACK LLM AGENT - Baseline Zero-Shot")
        print(f"  Model: Qwen 2.5 3B Instruct (Q4_K_M via llama-server)")
        print(f"  Seed: {seed}  |  Max steps: {max_steps}")
        print("=" * 60)
        print()

    for step in range(max_steps):
        state_text = render_state(obs)

        raw_action = query_model(state_text, history)
        action_int, action_name = parse_action(raw_action)

        hp = int(obs["blstats"][10])
        max_hp = int(obs["blstats"][11])

        if verbose:
            if step % 5 == 0:
                print(f"--- Step {step} ---")
                print(state_text)
                print()
            print(f"  Step {step:3d} | model: '{raw_action}' -> {action_name:10s} | HP: {hp}/{max_hp}")

        # Compact history for context window
        history.append({
            "state": state_text.split("\n")[-1],  # just stats line
            "action": action_name,
        })
        if len(history) > 8:
            history = history[-8:]

        all_states.append({
            "step": step,
            "raw_action": raw_action,
            "action": action_name,
            "hp": hp,
        })

        obs, reward, terminated, truncated, info = env.step(action_int)
        total_reward += reward

        if terminated or truncated:
            msg = bytes(obs["message"]).decode("ascii", errors="replace").strip().rstrip("\x00")
            if verbose:
                print()
                print(f"  *** GAME OVER at step {step+1} ***")
                if msg:
                    print(f"  Last message: {msg}")
            break

    elapsed = time.time() - start_time
    bl = obs["blstats"]

    result = {
        "seed": seed,
        "steps": step + 1,
        "total_reward": float(total_reward),
        "elapsed_seconds": round(elapsed, 1),
        "seconds_per_step": round(elapsed / (step + 1), 2),
        "final_hp": int(bl[10]),
        "final_depth": int(bl[12]),
        "final_gold": int(bl[13]),
        "final_level": int(bl[18]),
        "died": terminated,
        "states": all_states,
    }

    if verbose:
        print()
        print("=" * 60)
        print(f"  RESULTS")
        print(f"  Steps survived: {result['steps']}")
        print(f"  Total reward:   {result['total_reward']}")
        print(f"  Final HP:       {result['final_hp']}")
        print(f"  Depth reached:  {result['final_depth']}")
        print(f"  Gold collected: {result['final_gold']}")
        print(f"  Level:          {result['final_level']}")
        print(f"  Time:           {result['elapsed_seconds']}s ({result['seconds_per_step']}s/step)")
        print("=" * 60)

    env.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--save", type=str, help="Save JSON log")
    args = parser.parse_args()

    result = run_game(seed=args.seed, max_steps=args.max_steps)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Log saved to {args.save}")
