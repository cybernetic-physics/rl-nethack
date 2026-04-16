from __future__ import annotations

import argparse
import json
from statistics import mean

from rl.sequence_world_model import load_sequence_world_model
from rl.train_bc import load_trace_rows
from rl.world_model_planner import plan_action_sequence_with_cem, score_action_candidates


def _rank_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    x_order = {idx: rank for rank, idx in enumerate(sorted(range(len(xs)), key=lambda i: xs[i]))}
    y_order = {idx: rank for rank, idx in enumerate(sorted(range(len(ys)), key=lambda i: ys[i]))}
    n = len(xs)
    d2 = sum((x_order[i] - y_order[i]) ** 2 for i in range(n))
    return float(1.0 - (6.0 * d2) / (n * (n * n - 1)))


def evaluate_sequence_planner_replay(
    model_path: str,
    trace_path: str,
    *,
    planning_horizon: int = 6,
    population_size: int = 48,
    iterations: int = 4,
    max_rows: int | None = None,
    bootstrap_value_coef: float = 0.5,
    root_prior_coef: float = 0.25,
    uncertainty_coef: float = 0.0,
    disagreement_coef: float = 0.0,
    rollout_samples: int = 1,
    random_seed: int = 0,
) -> dict:
    rows = load_trace_rows(trace_path)
    inference = load_sequence_world_model(model_path)
    rng = __import__("numpy").random.default_rng(int(random_seed))
    evaluated = 0
    exact_matches = 0
    teacher_margin_hits = 0
    invalid_root = 0
    rank_corrs = []
    chosen_teacher_scores = []
    chosen_model_scores = []
    predicted_teacher_gaps = []

    for row in rows:
        planner_trace = row.get("planner_trace") or []
        if len(planner_trace) < 2:
            continue
        candidate_actions = [str(candidate["action"]) for candidate in planner_trace]
        teacher_scores = [float(candidate["total"]) for candidate in planner_trace]
        teacher_best = candidate_actions[max(range(len(candidate_actions)), key=lambda idx: teacher_scores[idx])]

        scored = score_action_candidates(
            inference,
            row["feature_vector"],
            candidate_actions,
            prompt_text=row.get("prompt"),
            allowed_actions=row.get("allowed_actions"),
            planning_horizon=planning_horizon,
            population_size=population_size,
            iterations=iterations,
            bootstrap_value_coef=bootstrap_value_coef,
            root_prior_coef=root_prior_coef,
            uncertainty_coef=uncertainty_coef,
            disagreement_coef=disagreement_coef,
            rollout_samples=rollout_samples,
            rng=rng,
        )
        predicted_best = scored[0]["action"]
        model_scores = [next(item["score"] for item in scored if item["action"] == action) for action in candidate_actions]
        rank_corrs.append(_rank_correlation(teacher_scores, model_scores))

        exact_matches += int(predicted_best == teacher_best)
        invalid_root += int(bool(row.get("allowed_actions")) and predicted_best not in set(row.get("allowed_actions", [])))
        predicted_teacher_gaps.append(float(max(teacher_scores) - teacher_scores[candidate_actions.index(predicted_best)]))
        chosen_teacher_scores.append(float(teacher_scores[candidate_actions.index(predicted_best)]))
        chosen_model_scores.append(float(scored[0]["score"]))

        if row.get("action") == teacher_best and predicted_best == teacher_best:
            teacher_margin_hits += 1

        evaluated += 1
        if max_rows is not None and evaluated >= int(max_rows):
            break

    return {
        "model_path": model_path,
        "trace_path": trace_path,
        "rows_evaluated": evaluated,
        "exact_match_rate": 0.0 if evaluated == 0 else float(exact_matches / evaluated),
        "teacher_best_recovery_rate": 0.0 if evaluated == 0 else float(exact_matches / evaluated),
        "teacher_trace_agreement_rate": 0.0 if evaluated == 0 else float(teacher_margin_hits / evaluated),
        "invalid_root_rate": 0.0 if evaluated == 0 else float(invalid_root / evaluated),
        "mean_rank_correlation": 0.0 if not rank_corrs else float(mean(rank_corrs)),
        "mean_teacher_score_of_chosen_action": 0.0 if not chosen_teacher_scores else float(mean(chosen_teacher_scores)),
        "mean_model_score_of_chosen_action": 0.0 if not chosen_model_scores else float(mean(chosen_model_scores)),
        "mean_teacher_gap_from_predicted_best": 0.0 if not predicted_teacher_gaps else float(mean(predicted_teacher_gaps)),
        "planning_horizon": planning_horizon,
        "population_size": population_size,
        "iterations": iterations,
        "bootstrap_value_coef": bootstrap_value_coef,
        "root_prior_coef": float(root_prior_coef),
        "uncertainty_coef": float(uncertainty_coef),
        "disagreement_coef": float(disagreement_coef),
        "rollout_samples": int(rollout_samples),
        "random_seed": int(random_seed),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Replay-evaluate sequence planner against trace planner totals")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--planning-horizon", type=int, default=6)
    parser.add_argument("--population-size", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--bootstrap-value-coef", type=float, default=0.5)
    parser.add_argument("--root-prior-coef", type=float, default=0.25)
    parser.add_argument("--uncertainty-coef", type=float, default=0.0)
    parser.add_argument("--disagreement-coef", type=float, default=0.0)
    parser.add_argument("--rollout-samples", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = evaluate_sequence_planner_replay(
        args.model,
        args.input,
        planning_horizon=args.planning_horizon,
        population_size=args.population_size,
        iterations=args.iterations,
        max_rows=args.max_rows,
        bootstrap_value_coef=args.bootstrap_value_coef,
        root_prior_coef=args.root_prior_coef,
        uncertainty_coef=args.uncertainty_coef,
        disagreement_coef=args.disagreement_coef,
        rollout_samples=args.rollout_samples,
        random_seed=args.random_seed,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
