from __future__ import annotations

import argparse
import json
from statistics import mean, pstdev

from rl.sequence_planner_tune import tune_sequence_planner_replay


def evaluate_sequence_planner_multiseed(
    model_path: str,
    trace_path: str,
    *,
    planning_horizon: int = 6,
    population_size: int = 16,
    iterations: int = 2,
    max_rows: int | None = 20,
    seeds: list[int] | None = None,
    bootstrap_value_coefs: list[float] | None = None,
    root_prior_coefs: list[float] | None = None,
    uncertainty_coefs: list[float] | None = None,
    disagreement_coefs: list[float] | None = None,
    rollout_samples: int = 1,
) -> dict:
    if seeds is None:
        seeds = [7, 17, 27]

    per_seed = []
    for seed in seeds:
        tuned = tune_sequence_planner_replay(
            model_path,
            trace_path,
            planning_horizon=planning_horizon,
            population_size=population_size,
            iterations=iterations,
            max_rows=max_rows,
            bootstrap_value_coefs=bootstrap_value_coefs,
            root_prior_coefs=root_prior_coefs,
            uncertainty_coefs=uncertainty_coefs,
            disagreement_coefs=disagreement_coefs,
            rollout_samples=rollout_samples,
            random_seed=int(seed),
        )
        best = tuned.get("best") or {}
        per_seed.append(
            {
                "seed": int(seed),
                "best": best,
                "tuned": tuned,
            }
        )

    exact_rates = [float(item["best"].get("exact_match_rate", 0.0)) for item in per_seed]
    rank_corrs = [float(item["best"].get("mean_rank_correlation", 0.0)) for item in per_seed]
    teacher_gaps = [float(item["best"].get("mean_teacher_gap_from_predicted_best", 0.0)) for item in per_seed]

    return {
        "model_path": model_path,
        "trace_path": trace_path,
        "planning_horizon": planning_horizon,
        "population_size": population_size,
        "iterations": iterations,
        "max_rows": max_rows,
        "seeds": [int(seed) for seed in seeds],
        "per_seed": per_seed,
        "summary": {
            "mean_exact_match_rate": float(mean(exact_rates)) if exact_rates else 0.0,
            "std_exact_match_rate": float(pstdev(exact_rates)) if len(exact_rates) > 1 else 0.0,
            "mean_rank_correlation": float(mean(rank_corrs)) if rank_corrs else 0.0,
            "std_rank_correlation": float(pstdev(rank_corrs)) if len(rank_corrs) > 1 else 0.0,
            "mean_teacher_gap": float(mean(teacher_gaps)) if teacher_gaps else 0.0,
            "std_teacher_gap": float(pstdev(teacher_gaps)) if len(teacher_gaps) > 1 else 0.0,
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Multi-seed replay planner evaluation for sequence world models")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--planning-horizon", type=int, default=6)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--bootstrap-value-coefs", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--root-prior-coefs", nargs="+", type=float, default=[0.0, 0.25, 0.5])
    parser.add_argument("--uncertainty-coefs", nargs="+", type=float, default=[0.0, 0.25, 0.5])
    parser.add_argument("--disagreement-coefs", nargs="+", type=float, default=[0.0])
    parser.add_argument("--rollout-samples", type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = evaluate_sequence_planner_multiseed(
        args.model,
        args.input,
        planning_horizon=args.planning_horizon,
        population_size=args.population_size,
        iterations=args.iterations,
        max_rows=args.max_rows,
        seeds=list(args.seeds),
        bootstrap_value_coefs=list(args.bootstrap_value_coefs),
        root_prior_coefs=list(args.root_prior_coefs),
        uncertainty_coefs=list(args.uncertainty_coefs),
        disagreement_coefs=list(args.disagreement_coefs),
        rollout_samples=args.rollout_samples,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
