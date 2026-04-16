"""
Closed-loop live evaluation for the canonical long-context policy interface.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import nle.env

from nle_agent.agent_http import _build_action_map
from src.long_sequence_dataset import RenderedTurn
from src.policy_actions import canonicalize_action
from src.policy_replay import render_policy_state_from_obs


def build_inference_messages(
    *,
    history_turns: list[RenderedTurn],
    current_state_text: str,
    episode_id: str,
    step_index: int,
    max_context_tokens: int,
    reserve_output_tokens: int = 16,
    tokenizer=None,
) -> list[dict]:
    from src.long_sequence_dataset import build_messages

    messages, _ = build_messages(
        history_turns=history_turns,
        current_state_text=current_state_text,
        current_state_tokens=0,
        target_action="wait",
        episode_id=episode_id,
        step_index=step_index,
        max_context_tokens=max_context_tokens,
        reserve_output_tokens=reserve_output_tokens,
        tokenizer=tokenizer,
    )
    return messages[:-1]


def evaluate_live_long_sequence_policy(
    *,
    seeds: list[int],
    max_steps: int,
    server_url: str,
    model_name: str,
    encoder,
    max_context_tokens: int = 8192,
    board_mode: str = "tokenized",
    reserve_output_tokens: int = 16,
    tokenizer=None,
) -> dict:
    from src.long_sequence_dataset import render_turn
    from src.long_sequence_eval import query_openai_chat

    action_map = _build_action_map()
    results = []
    total_actions = Counter()
    invalid_or_odd_actions = 0
    total_steps = 0
    for seed in seeds:
        env = nle.env.NLE()
        obs, _ = env.reset(seed=seed)
        history_turns: list[RenderedTurn] = []
        per_seed_actions = Counter()
        steps = 0
        final_depth = 1
        final_hp = 0
        for step in range(max_steps):
            rendered = render_policy_state_from_obs(
                obs,
                state_index=step,
                encoder=encoder,
                board_mode=board_mode,
                persist_dual_views=False,
                tokenizer=tokenizer,
            )
            messages = build_inference_messages(
                history_turns=history_turns,
                current_state_text=rendered.text,
                episode_id=f"live-seed-{seed}",
                step_index=step,
                max_context_tokens=max_context_tokens,
                reserve_output_tokens=reserve_output_tokens,
                tokenizer=tokenizer,
            )
            raw_prediction = query_openai_chat(server_url, messages, model_name=model_name)
            normalized = canonicalize_action(raw_prediction)
            action_name = normalized.normalized if normalized.should_keep and normalized.normalized in action_map else "wait"
            if not normalized.should_keep or action_name == "wait" and raw_prediction.strip().lower() not in {"wait", ".", "rest"}:
                invalid_or_odd_actions += 1
            per_seed_actions[action_name] += 1
            total_actions[action_name] += 1
            action_idx = action_map.get(action_name, action_map["wait"])
            history_turns.append(
                render_turn(
                    obs=None,
                    state_text=rendered.text,
                    action=action_name,
                    encoder=encoder,
                    turn_index=step,
                    board_mode=board_mode,
                    tokenizer=tokenizer,
                )
            )
            obs, reward, terminated, truncated, _ = env.step(action_idx)
            steps = step + 1
            final_depth = int(obs["blstats"][12])
            final_hp = int(obs["blstats"][10])
            if terminated or truncated:
                break
        env.close()
        total_steps += steps
        results.append(
            {
                "seed": seed,
                "steps": steps,
                "final_depth": final_depth,
                "final_hp": final_hp,
                "action_counts": dict(per_seed_actions),
            }
        )

    return {
        "seeds": seeds,
        "max_steps": max_steps,
        "server_url": server_url,
        "model_name": model_name,
        "episodes": results,
        "summary": {
            "episodes": len(results),
            "total_steps": total_steps,
            "avg_steps": (total_steps / len(results)) if results else 0.0,
            "avg_final_depth": (sum(item["final_depth"] for item in results) / len(results)) if results else 0.0,
            "avg_final_hp": (sum(item["final_hp"] for item in results) / len(results)) if results else 0.0,
            "invalid_or_odd_action_rate": (invalid_or_odd_actions / total_steps) if total_steps else 0.0,
            "action_counts": dict(total_actions),
        },
    }
