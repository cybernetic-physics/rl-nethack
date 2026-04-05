"""
Evaluator: Measure model prediction accuracy on NLE game outcome predictions.

Provides:
1. parse_prediction -- Parse model text output into structured prediction dict
2. compute_accuracy -- Compare predictions vs ground truth, return per-field accuracy
3. evaluate_model  -- Load model via llama-server HTTP API, run on test data
4. generate_test_data -- Generate test data from specific seeds
5. run_evaluation   -- Full pipeline: generate data, query model, compute accuracy
"""

import json
import random
import re
import hashlib
import sys
import urllib.request
from typing import Dict, List, Optional

import nle.env

from src.state_encoder import StateEncoder
from src.data_generator import build_messages, wall_avoidance_policy
from nle_agent.agent_http import _build_action_map

LIVE_ENV_WARNING = (
    "Held-out seed evaluation is only a diagnostic in this repo because plain "
    "nle.env.NLE().reset(seed=...) is not reproducible across runs."
)


# ---------------------------------------------------------------------------
# 1. parse_prediction
# ---------------------------------------------------------------------------

def parse_prediction(text: str) -> dict:
    """Parse a model's text output into a structured prediction dict.

    Expected format:
        pos:(-1,0) | hp:same | gold:same | depth:same | alive:yes | msg:It's a wall.

    Returns:
        Dict with keys:
            pos (dx, dy) tuple
            hp_delta (int)
            gold_delta (int)
            depth_delta (int)
            survived (bool)
            message (str)
    """
    text = text.strip()

    # Split on pipe delimiters
    parts = [p.strip() for p in text.split('|')]

    result = {
        'pos': (0, 0),
        'hp_delta': 0,
        'gold_delta': 0,
        'depth_delta': 0,
        'survived': True,
        'message': '',
    }

    for part in parts:
        # pos:(dx,dy) -- allow spaces around parens
        pos_match = re.match(r'pos:\s*\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?', part)
        if pos_match:
            result['pos'] = (int(pos_match.group(1)), int(pos_match.group(2)))
            continue

        # hp:same | hp:+N | hp:-N
        hp_match = re.match(r'hp:\s*(.+)', part)
        if hp_match:
            val = hp_match.group(1).strip()
            if val == 'same':
                result['hp_delta'] = 0
            else:
                result['hp_delta'] = int(val)
            continue

        # gold:same | gold:+N | gold:-N
        gold_match = re.match(r'gold:\s*(.+)', part)
        if gold_match:
            val = gold_match.group(1).strip()
            if val == 'same':
                result['gold_delta'] = 0
            else:
                result['gold_delta'] = int(val)
            continue

        # depth:same | depth:+N | depth:-N
        depth_match = re.match(r'depth:\s*(.+)', part)
        if depth_match:
            val = depth_match.group(1).strip()
            if val == 'same':
                result['depth_delta'] = 0
            else:
                result['depth_delta'] = int(val)
            continue

        # alive:yes | alive:no
        alive_match = re.match(r'alive:\s*(.+)', part)
        if alive_match:
            val = alive_match.group(1).strip().lower()
            result['survived'] = val in ('yes', 'true', '1')
            continue

        # msg:<message text>
        msg_match = re.match(r'msg:\s*(.*)', part)
        if msg_match:
            result['message'] = msg_match.group(1).strip()
            continue

    return result


# ---------------------------------------------------------------------------
# 2. compute_accuracy
# ---------------------------------------------------------------------------

def compute_accuracy(predictions: list, ground_truth: list) -> dict:
    """Compare predictions to ground truth deltas and compute per-field accuracy.

    Each prediction is a dict with: pos, hp_delta, gold_delta, depth_delta, survived
    Each ground_truth is a dict with: pos_delta, hp_delta, gold_delta, depth_delta, survived

    Returns:
        Dict with:
            n: number of examples compared
            exact_match_rate: fraction where all fields match
            pos_accuracy: fraction where pos matches
            hp_accuracy: fraction where hp_delta matches
            gold_accuracy: fraction where gold_delta matches
            depth_accuracy: fraction where depth_delta matches
            survived_accuracy: fraction where survived matches
            per_example: list of bools (True = exact match for that example)
    """
    if not predictions or not ground_truth:
        return {
            'n': 0,
            'exact_match_rate': 0.0,
            'pos_accuracy': 0.0,
            'hp_accuracy': 0.0,
            'gold_accuracy': 0.0,
            'depth_accuracy': 0.0,
            'survived_accuracy': 0.0,
            'per_example': [],
        }

    n = min(len(predictions), len(ground_truth))
    pos_correct = 0
    hp_correct = 0
    gold_correct = 0
    depth_correct = 0
    survived_correct = 0
    exact_match = 0
    per_example = []

    for i in range(n):
        pred = predictions[i]
        gt = ground_truth[i]

        # Map prediction.pos to ground_truth pos_delta
        pred_pos = pred.get('pos', (0, 0))
        gt_pos = gt.get('pos_delta', (0, 0))

        pos_ok = (pred_pos == gt_pos)
        hp_ok = (pred.get('hp_delta', 0) == gt.get('hp_delta', 0))
        gold_ok = (pred.get('gold_delta', 0) == gt.get('gold_delta', 0))
        depth_ok = (pred.get('depth_delta', 0) == gt.get('depth_delta', 0))
        survived_ok = (pred.get('survived', True) == gt.get('survived', True))

        if pos_ok:
            pos_correct += 1
        if hp_ok:
            hp_correct += 1
        if gold_ok:
            gold_correct += 1
        if depth_ok:
            depth_correct += 1
        if survived_ok:
            survived_correct += 1

        all_ok = pos_ok and hp_ok and gold_ok and depth_ok and survived_ok
        if all_ok:
            exact_match += 1
        per_example.append(all_ok)

    return {
        'n': n,
        'exact_match_rate': exact_match / n if n > 0 else 0.0,
        'pos_accuracy': pos_correct / n if n > 0 else 0.0,
        'hp_accuracy': hp_correct / n if n > 0 else 0.0,
        'gold_accuracy': gold_correct / n if n > 0 else 0.0,
        'depth_accuracy': depth_correct / n if n > 0 else 0.0,
        'survived_accuracy': survived_correct / n if n > 0 else 0.0,
        'per_example': per_example,
    }


# ---------------------------------------------------------------------------
# 3. evaluate_model
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name_or_path: str,
    test_data: list,
    server_url: str = "http://127.0.0.1:8765",
    max_samples: Optional[int] = None,
) -> dict:
    """Evaluate a model by querying it via llama-server HTTP API.

    Args:
        model_name_or_path: Model identifier (for logging purposes).
        test_data: List of dicts with 'prompt' and 'target' keys.
        server_url: URL of the llama-server instance.
        max_samples: If set, only evaluate on first max_samples examples.

    Returns:
        Dict with:
            predictions: list of parsed prediction dicts
            raw_responses: list of raw model response strings
            accuracy: dict from compute_accuracy
            model: model_name_or_path
            server_url: the server URL used
            server_available: bool
            errors: list of error messages
    """
    if max_samples is not None:
        test_data = test_data[:max_samples]

    predictions = []
    raw_responses = []
    errors = []
    server_available = False

    # Check server availability first
    try:
        req = urllib.request.Request(
            f"{server_url}/health",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            server_available = True
    except Exception:
        server_available = False

    if not server_available:
        return {
            'predictions': [],
            'raw_responses': [],
            'accuracy': {
                'n': 0,
                'exact_match_rate': 0.0,
                'pos_accuracy': 0.0,
                'hp_accuracy': 0.0,
                'gold_accuracy': 0.0,
                'depth_accuracy': 0.0,
                'survived_accuracy': 0.0,
                'per_example': [],
            },
            'model': model_name_or_path,
            'server_url': server_url,
            'server_available': False,
            'errors': ['Server not available at ' + server_url],
        }

    for item in test_data:
        messages = item.get('messages')
        if messages is None:
            prompt = item['prompt']
            messages = build_messages(prompt)

        payload = json.dumps({
            "messages": messages,
            "max_tokens": 128,
            "temperature": 0.2,
        }).encode()

        req = urllib.request.Request(
            f"{server_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                content = data["choices"][0]["message"]["content"].strip()
                raw_responses.append(content)
                predictions.append(parse_prediction(content))
        except Exception as e:
            errors.append(str(e))
            raw_responses.append('')
            predictions.append(parse_prediction(''))

    # Compute accuracy against ground truth targets
    ground_truth = []
    for item in test_data:
        target = item.get('target', '')
        gt = parse_prediction(target)
        # Convert to ground_truth format (pos_delta instead of pos)
        ground_truth.append({
            'pos_delta': gt['pos'],
            'hp_delta': gt['hp_delta'],
            'gold_delta': gt['gold_delta'],
            'depth_delta': gt['depth_delta'],
            'survived': gt['survived'],
        })

    accuracy = compute_accuracy(predictions, ground_truth)

    return {
        'predictions': predictions,
        'raw_responses': raw_responses,
        'accuracy': accuracy,
        'model': model_name_or_path,
        'server_url': server_url,
        'server_available': True,
        'errors': errors,
    }


# ---------------------------------------------------------------------------
# 4. generate_test_data
# ---------------------------------------------------------------------------

def generate_test_data(
    seeds: list,
    max_steps: int,
    encoder: StateEncoder,
) -> list:
    """Generate test data from specific seeds.

    Args:
        seeds: List of random seeds for NLE environment.
        max_steps: Maximum number of steps per game.
        encoder: StateEncoder instance.

    Returns:
        List of dicts, each with:
            prompt (str): formatted state + action
            messages (list[dict]): canonical message list for model calls
            target (str): formatted delta
            ground_truth_delta (dict): structured delta from encode_delta
            seed (int): the seed used
            step (int): step index within the game

    Notes:
        This path is not suitable for strict regression gating because NLE seed
        resets are not fully reproducible across runs.
    """
    action_map = _build_action_map()
    test_data = []

    for seed in seeds:
        rng = random.Random(seed)
        env = nle.env.NLE()
        obs, info = env.reset(seed=seed)

        for step in range(max_steps):
            state = encoder.encode_full(obs)

            # Choose action via wall avoidance policy
            action_name = wall_avoidance_policy(state['adjacent'], rng)

            if action_name in action_map:
                action_idx = action_map[action_name]
            else:
                action_idx = action_map.get('wait', 18)
                action_name = 'wait'

            prompt_text = encoder.format_prompt(state, action_name)

            obs_after, reward, terminated, truncated, info_after = env.step(action_idx)

            delta = encoder.encode_delta(obs, obs_after, action_name)
            target_text = encoder.format_target(delta)
            messages = build_messages(prompt_text)

            test_data.append({
                'prompt': prompt_text,
                'messages': messages,
                'target': target_text,
                'ground_truth_delta': delta,
                'seed': seed,
                'step': step,
            })

            obs = obs_after

            if terminated or truncated:
                break

        env.close()

    return test_data


# ---------------------------------------------------------------------------
# 5. run_evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    seeds: list,
    max_steps: int,
    encoder: StateEncoder,
    server_url: str = "http://127.0.0.1:8765",
) -> dict:
    """Full evaluation pipeline.

    1. Generate test data from seeds
    2. Query model for each prompt
    3. Parse predictions
    4. Compute accuracy

    Args:
        seeds: List of random seeds.
        max_steps: Max steps per game.
        encoder: StateEncoder instance.
        server_url: llama-server URL.

    Returns:
        Dict with:
            test_data: list of test data dicts
            accuracy: dict from compute_accuracy
            per_example: list of dicts with prompt, target, prediction, match info
            server_available: bool
    """
    test_data = generate_test_data(seeds, max_steps, encoder)

    # Try to query model
    eval_result = evaluate_model(
        model_name_or_path="llama-server",
        test_data=test_data,
        server_url=server_url,
    )

    # Build per-example details
    per_example = []
    if eval_result['predictions']:
        for i, item in enumerate(test_data):
            if i < len(eval_result['predictions']):
                pred = eval_result['predictions'][i]
                gt = item['ground_truth_delta']
                match = (
                    pred.get('pos', (0, 0)) == gt.get('pos_delta', (0, 0))
                    and pred.get('hp_delta', 0) == gt.get('hp_delta', 0)
                    and pred.get('gold_delta', 0) == gt.get('gold_delta', 0)
                    and pred.get('depth_delta', 0) == gt.get('depth_delta', 0)
                    and pred.get('survived', True) == gt.get('survived', True)
                )
                per_example.append({
                    'seed': item['seed'],
                    'step': item['step'],
                    'prompt': item['prompt'],
                    'target': item['target'],
                    'raw_response': eval_result['raw_responses'][i] if i < len(eval_result['raw_responses']) else '',
                    'prediction': pred,
                    'ground_truth': gt,
                    'exact_match': match,
                })

    return {
        'test_data': test_data,
        'accuracy': eval_result['accuracy'],
        'per_example': per_example,
        'server_available': eval_result['server_available'],
        'model': eval_result['model'],
        'errors': eval_result['errors'],
        'evaluation_mode': 'live_env_seeded',
        'warning': LIVE_ENV_WARNING,
    }


def hash_messages(messages: list[dict]) -> str:
    """Return a stable hash for a model input message list."""
    payload = json.dumps(messages, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
