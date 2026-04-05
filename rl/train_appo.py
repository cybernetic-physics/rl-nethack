from __future__ import annotations

import argparse
import json

from rl.config import RLConfig
from rl.trainer import APPOTrainerScaffold


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="APPO + options RL scaffold")
    parser.add_argument("--experiment", type=str, default="appo_options_scaffold")
    parser.add_argument("--train-dir", type=str, default="train_dir/rl")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-envs-per-worker", type=int, default=16)
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument("--recurrence", type=int, default=32)
    parser.add_argument("--train-for-env-steps", type=int, default=50_000_000)
    parser.add_argument("--scheduler", type=str, default="rule_based")
    parser.add_argument("--reward-source", type=str, default="hand_shaped")
    parser.add_argument("--enabled-skills", type=str, default="explore,survive,combat,descend,resource")
    parser.add_argument("--write-plan", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_config(args) -> RLConfig:
    config = RLConfig()
    config.experiment = args.experiment
    config.train_dir = args.train_dir
    config.rollout.num_workers = args.num_workers
    config.rollout.num_envs_per_worker = args.num_envs_per_worker
    config.rollout.rollout_length = args.rollout_length
    config.rollout.recurrence = args.recurrence
    config.appo.train_for_env_steps = args.train_for_env_steps
    config.options.scheduler = args.scheduler
    config.reward.source = args.reward_source
    config.options.enabled_skills = [s.strip() for s in args.enabled_skills.split(",") if s.strip()]
    return config


def main(argv=None):
    args = parse_args(argv)
    config = build_config(args)
    trainer = APPOTrainerScaffold(config)

    if args.write_plan:
        path = trainer.write_plan(args.write_plan)
        print(f"Wrote APPO plan to: {path}")

    result = trainer.launch(dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
