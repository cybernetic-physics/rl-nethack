"""
DataGenerator: Generate training data from random NLE gameplay.

Produces JSONL files in ShareGPT conversation format for LLM fine-tuning.
Each line is one (state, action) -> (delta) training example.

Uses StateEncoder for state encoding and format_prompt/format_target for
text representation. Imports the action map from nle_agent.agent_http.
"""

import json
import os
import random
from typing import Callable, Dict, Iterator, Optional

import nle.env

from src.state_encoder import StateEncoder
from nle_agent.agent_http import _build_action_map

# System prompt for ShareGPT conversations
SYSTEM_PROMPT = (
    "Predict the outcome of a NetHack action. "
    "Given the current state and an action, predict the resulting changes "
    "(position delta, HP change, gold change, depth change, survival, message)."
)

# Build the action map once at module level
_ACTION_MAP: Optional[Dict[str, int]] = None


def _get_action_map() -> Dict[str, int]:
    """Lazily build and cache the action map."""
    global _ACTION_MAP
    if _ACTION_MAP is None:
        _ACTION_MAP = _build_action_map()
    return _ACTION_MAP


def build_messages(prompt_text: str, target_text: Optional[str] = None) -> list[dict]:
    """Build the canonical message list used by training and evaluation."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    if target_text is not None:
        messages.append({"role": "assistant", "content": target_text})
    return messages


# Directions that the wall_avoidance_policy can choose from
_DIRECTIONS = ['north', 'south', 'east', 'west']


def wall_avoidance_policy(adjacent: dict, rng: random.Random) -> str:
    """Simple policy that picks random directions but avoids walls.

    Takes an adjacent dict like {"north": "floor", "south": "wall",
    "east": "door", "west": "wall"} and returns an action name.

    Prefers tiles that are not walls or unseen. Falls back to any
    direction if all are walls.

    Args:
        adjacent: Dict mapping direction names to tile names.
        rng: A random.Random instance for reproducibility.

    Returns:
        An action name string (e.g. 'north', 'east', 'wait').
    """
    # "open" tiles: not wall and not unseen
    open_dirs = [
        d for d in _DIRECTIONS
        if adjacent.get(d, 'unseen') not in ('wall', 'unseen')
    ]

    if open_dirs:
        return rng.choice(open_dirs)
    else:
        # All blocked -- just wait
        return 'wait'


def generate_game(
    seed: int,
    max_steps: int,
    encoder: StateEncoder,
    policy: Optional[Callable] = None,
) -> Iterator[str]:
    """Play one NLE game with a random policy, yielding JSONL lines.

    Each yielded line is a JSON object in ShareGPT conversation format:
    {
        "conversations": [
            {"role": "system", "content": "<system prompt>"},
            {"role": "user", "content": "<format_prompt output>"},
            {"role": "assistant", "content": "<format_target output>"}
        ]
    }

    Args:
        seed: Random seed for NLE environment and policy RNG.
        max_steps: Maximum number of steps to play.
        encoder: StateEncoder instance for encoding observations.
        policy: Callable(adjacent_dict, rng) -> action_name.
                Defaults to wall_avoidance_policy if None.

    Yields:
        JSONL-formatted strings, one per step.
    """
    if policy is None:
        policy = wall_avoidance_policy

    rng = random.Random(seed)
    action_map = _get_action_map()

    env = nle.env.NLE()
    obs, info = env.reset(seed=seed)

    for step in range(max_steps):
        # Encode current state
        state = encoder.encode_full(obs)

        # Choose action via policy
        action_name = policy(state['adjacent'], rng)

        # Look up NLE action index
        if action_name in action_map:
            action_idx = action_map[action_name]
        else:
            # Fallback to wait
            action_idx = action_map.get('wait', 18)
            action_name = 'wait'

        # Format the prompt (state + proposed action)
        prompt_text = encoder.format_prompt(state, action_name)

        # Take the action
        obs_after, reward, terminated, truncated, info_after = env.step(action_idx)

        # Encode the delta
        delta = encoder.encode_delta(obs, obs_after, action_name)

        # Format the target
        target_text = encoder.format_target(delta)

        # Build ShareGPT conversation
        conversation = {"conversations": build_messages(prompt_text, target_text)}

        yield json.dumps(conversation)

        # Advance observation
        obs = obs_after

        if terminated or truncated:
            break

    env.close()


def generate_dataset(
    output_path: str,
    num_games: int,
    max_steps: int,
    seed_start: int,
    encoder: StateEncoder,
    eval_path: Optional[str] = None,
    eval_fraction: float = 0.2,
    policy: Optional[Callable] = None,
) -> dict:
    """Run multiple games, write JSONL, split into train/eval, return stats.

    Generates num_games games starting from seed_start, seed_start+1, ...
    The last ceil(num_games * eval_fraction) games are written to eval_path
    if provided, the rest to output_path.

    Args:
        output_path: Path for the training JSONL file.
        num_games: Number of games to play.
        max_steps: Maximum steps per game.
        seed_start: Starting seed (games use seed_start, seed_start+1, ...).
        encoder: StateEncoder instance.
        eval_path: Optional path for evaluation JSONL file.
        eval_fraction: Fraction of games for evaluation (default 0.2).
        policy: Optional policy function.

    Returns:
        Dict with stats:
            - total_games: number of games played
            - total_examples: total training examples generated
            - train_examples: examples in training set
            - eval_examples: examples in eval set
            - train_path: path to training file
            - eval_path: path to eval file (or None)
    """
    num_eval = int(num_games * eval_fraction) if eval_path else 0
    num_train = num_games - num_eval

    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    if eval_path:
        os.makedirs(os.path.dirname(eval_path) if os.path.dirname(eval_path) else '.', exist_ok=True)

    total_examples = 0
    train_examples = 0
    eval_examples = 0

    # Collect all lines by game for splitting
    train_lines = []
    eval_lines = []

    for i in range(num_games):
        game_seed = seed_start + i
        game_lines = list(generate_game(game_seed, max_steps, encoder, policy))
        total_examples += len(game_lines)

        if i < num_train:
            train_lines.extend(game_lines)
            train_examples += len(game_lines)
        else:
            eval_lines.extend(game_lines)
            eval_examples += len(game_lines)

    # Write training file
    with open(output_path, 'w') as f:
        for line in train_lines:
            f.write(line + '\n')

    # Write eval file
    actual_eval_path = None
    if eval_path and eval_lines:
        with open(eval_path, 'w') as f:
            for line in eval_lines:
                f.write(line + '\n')
        actual_eval_path = eval_path

    return {
        'total_games': num_games,
        'total_examples': total_examples,
        'train_examples': train_examples,
        'eval_examples': eval_examples,
        'train_path': output_path,
        'eval_path': actual_eval_path,
    }
