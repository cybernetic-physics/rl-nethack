from __future__ import annotations

import math

import numpy as np

from rl.feature_encoder import ACTION_SET


def plan_action_sequence_with_cem(
    inference,
    initial_features: np.ndarray | list[float],
    *,
    prompt_text: str | None = None,
    allowed_actions: list[str] | None = None,
    root_action: str | None = None,
    task_index: int = 0,
    planning_horizon: int = 8,
    population_size: int = 64,
    elite_frac: float = 0.25,
    iterations: int = 4,
    risk_coef: float = 0.25,
    done_penalty: float = 1.0,
    invalid_action_penalty: float = 1.5,
    bootstrap_value_coef: float = 0.5,
    root_prior_coef: float = 0.25,
    uncertainty_coef: float = 0.0,
    disagreement_coef: float = 0.0,
    rollout_samples: int = 1,
    rng: np.random.Generator | None = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng()
    action_count = len(ACTION_SET)
    elite_count = max(1, int(math.ceil(population_size * elite_frac)))
    probs = np.full((planning_horizon, action_count), 1.0 / action_count, dtype=np.float32)
    root_action_idx = None
    if root_action is not None and root_action in ACTION_SET:
        root_action_idx = ACTION_SET.index(root_action)
    if allowed_actions:
        allowed = np.zeros(action_count, dtype=np.float32)
        for action_name in allowed_actions:
            if action_name in ACTION_SET:
                allowed[ACTION_SET.index(action_name)] = 1.0
        if allowed.sum() > 0:
            probs[0] = allowed / allowed.sum()
    if root_action_idx is not None:
        probs[0] = np.zeros(action_count, dtype=np.float32)
        probs[0][root_action_idx] = 1.0

    best_sequence = [0] * planning_horizon
    best_score = -float("inf")
    observed = inference.observe(initial_features, prompt_texts=[prompt_text or ""])
    root_prior_logits = observed.get("planner_action_logits")
    root_prior = None if root_prior_logits is None else np.asarray(root_prior_logits, dtype=np.float32).reshape(-1)

    for _ in range(iterations):
        candidates = np.stack(
            [np.array([rng.choice(action_count, p=probs[t]) for t in range(planning_horizon)]) for _ in range(population_size)],
            axis=0,
        )
        scores = []
        for candidate in candidates:
            rollout = inference.rollout(
                initial_features,
                candidate.tolist(),
                tasks=[task_index] * planning_horizon,
                prompt_text=prompt_text,
                deterministic=rollout_samples <= 1,
                num_samples=rollout_samples,
            )
            score = float(np.sum(rollout["pred_rewards"]))
            if rollout["pred_values"].size:
                score += float(bootstrap_value_coef) * float(rollout["pred_values"][-1])
            if root_prior is not None:
                score += float(root_prior_coef) * float(root_prior[int(candidate[0])])
            score -= float(done_penalty) * float(np.sum(rollout["pred_done_probs"]))
            score -= float(risk_coef) * float(np.std(rollout["pred_values"]))
            if "pred_latent_uncertainty" in rollout:
                score -= float(uncertainty_coef) * float(np.sum(rollout["pred_latent_uncertainty"]))
            if "reward_disagreement" in rollout:
                score -= float(disagreement_coef) * float(np.sum(rollout["reward_disagreement"]))
            if "value_disagreement" in rollout:
                score -= float(disagreement_coef) * float(np.sum(rollout["value_disagreement"]))
            valid_probs = rollout["pred_action_valid_probs"]
            for step in range(planning_horizon):
                action_idx = int(candidate[step])
                if step == 0 and allowed_actions and ACTION_SET[action_idx] not in set(allowed_actions):
                    score -= float(invalid_action_penalty) * 2.0
                    continue
                if step + 1 < planning_horizon:
                    next_action_idx = int(candidate[step + 1])
                    score -= float(invalid_action_penalty) * float(1.0 - valid_probs[step, next_action_idx])
            scores.append(score)
        scores_np = np.asarray(scores, dtype=np.float32)
        elite_indices = np.argsort(scores_np)[-elite_count:]
        elites = candidates[elite_indices]
        probs = np.full_like(probs, 1e-4)
        for step in range(planning_horizon):
            counts = np.bincount(elites[:, step], minlength=action_count).astype(np.float32)
            probs[step] = counts / counts.sum()
        if root_action_idx is not None:
            probs[0] = np.zeros(action_count, dtype=np.float32)
            probs[0][root_action_idx] = 1.0
        max_idx = int(np.argmax(scores_np))
        if float(scores_np[max_idx]) > best_score:
            best_score = float(scores_np[max_idx])
            best_sequence = candidates[max_idx].tolist()

    return {
        "best_action_indices": best_sequence,
        "best_actions": [ACTION_SET[idx] for idx in best_sequence],
        "best_score": best_score,
        "final_action_probs": probs.tolist(),
        "uncertainty_coef": float(uncertainty_coef),
        "disagreement_coef": float(disagreement_coef),
        "rollout_samples": int(rollout_samples),
    }


def score_action_candidates(
    inference,
    initial_features: np.ndarray | list[float],
    candidate_actions: list[str],
    *,
    prompt_text: str | None = None,
    allowed_actions: list[str] | None = None,
    task_index: int = 0,
    planning_horizon: int = 8,
    population_size: int = 64,
    elite_frac: float = 0.25,
    iterations: int = 4,
    risk_coef: float = 0.25,
    done_penalty: float = 1.0,
    invalid_action_penalty: float = 1.5,
    bootstrap_value_coef: float = 0.5,
    root_prior_coef: float = 0.25,
    uncertainty_coef: float = 0.0,
    disagreement_coef: float = 0.0,
    rollout_samples: int = 1,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    if rng is None:
        rng = np.random.default_rng()
    scored = []
    for action_name in candidate_actions:
        plan = plan_action_sequence_with_cem(
            inference,
            initial_features,
            prompt_text=prompt_text,
            allowed_actions=allowed_actions,
            root_action=action_name,
            task_index=task_index,
            planning_horizon=planning_horizon,
            population_size=population_size,
            elite_frac=elite_frac,
            iterations=iterations,
            risk_coef=risk_coef,
            done_penalty=done_penalty,
            invalid_action_penalty=invalid_action_penalty,
            bootstrap_value_coef=bootstrap_value_coef,
            root_prior_coef=root_prior_coef,
            uncertainty_coef=uncertainty_coef,
            disagreement_coef=disagreement_coef,
            rollout_samples=rollout_samples,
            rng=rng,
        )
        scored.append(
            {
                "action": action_name,
                "score": float(plan["best_score"]),
                "plan": plan,
            }
        )
    scored.sort(key=lambda row: row["score"], reverse=True)
    return scored
