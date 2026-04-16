from __future__ import annotations

import argparse
import json

from rl.sequence_planner_eval import evaluate_sequence_planner_replay


def tune_sequence_planner_replay(
    model_path: str,
    trace_path: str,
    *,
    planning_horizon: int = 6,
    population_size: int = 16,
    iterations: int = 2,
    max_rows: int | None = 20,
    bootstrap_value_coefs: list[float] | None = None,
    root_prior_coefs: list[float] | None = None,
    uncertainty_coefs: list[float] | None = None,
    disagreement_coefs: list[float] | None = None,
    rollout_samples: int = 1,
    random_seed: int = 7,
) -> dict:
    if bootstrap_value_coefs is None:
        bootstrap_value_coefs = [0.25, 0.5, 0.75]
    if root_prior_coefs is None:
        root_prior_coefs = [0.0, 0.25, 0.5]
    if uncertainty_coefs is None:
        uncertainty_coefs = [0.0, 0.25, 0.5]
    if disagreement_coefs is None:
        disagreement_coefs = [0.0]

    trials = []
    for bootstrap_value_coef in bootstrap_value_coefs:
        for root_prior_coef in root_prior_coefs:
            for uncertainty_coef in uncertainty_coefs:
                for disagreement_coef in disagreement_coefs:
                    result = evaluate_sequence_planner_replay(
                        model_path,
                        trace_path,
                        planning_horizon=planning_horizon,
                        population_size=population_size,
                        iterations=iterations,
                        max_rows=max_rows,
                        bootstrap_value_coef=bootstrap_value_coef,
                        root_prior_coef=root_prior_coef,
                        uncertainty_coef=uncertainty_coef,
                        disagreement_coef=disagreement_coef,
                        rollout_samples=rollout_samples,
                        random_seed=random_seed,
                    )
                    score = (
                        float(result["mean_rank_correlation"])
                        + 0.5 * float(result["exact_match_rate"])
                        - 0.5 * float(result["mean_teacher_gap_from_predicted_best"])
                    )
                    result["selection_score"] = score
                    trials.append(result)

    best = max(trials, key=lambda row: float(row["selection_score"])) if trials else None
    return {
        "model_path": model_path,
        "trace_path": trace_path,
        "planning_horizon": planning_horizon,
        "population_size": population_size,
        "iterations": iterations,
        "max_rows": max_rows,
        "random_seed": int(random_seed),
        "trials": trials,
        "best": best,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Tune replay planner hyperparameters for a sequence world model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--planning-horizon", type=int, default=6)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--bootstrap-value-coefs", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--root-prior-coefs", nargs="+", type=float, default=[0.0, 0.25, 0.5])
    parser.add_argument("--uncertainty-coefs", nargs="+", type=float, default=[0.0, 0.25, 0.5])
    parser.add_argument("--disagreement-coefs", nargs="+", type=float, default=[0.0])
    parser.add_argument("--rollout-samples", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=7)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = tune_sequence_planner_replay(
        args.model,
        args.input,
        planning_horizon=args.planning_horizon,
        population_size=args.population_size,
        iterations=args.iterations,
        max_rows=args.max_rows,
        bootstrap_value_coefs=list(args.bootstrap_value_coefs),
        root_prior_coefs=list(args.root_prior_coefs),
        uncertainty_coefs=list(args.uncertainty_coefs),
        disagreement_coefs=list(args.disagreement_coefs),
        rollout_samples=args.rollout_samples,
        random_seed=args.random_seed,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
