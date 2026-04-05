#!/usr/bin/env python3
"""
CLI entry point for the rl-nethack training pipeline.

Subcommands:
  generate    -- Generate training data from NLE gameplay
  report      -- Run a game and produce HTML + text reports
  evaluate    -- Evaluate model prediction accuracy
  task-evaluate -- Evaluate task-directed control policies
  rl-train-appo -- Run the modular APPO + options scaffold
  rl-evaluate-appo -- Evaluate a trained APPO checkpoint
  rl-train-reward -- Train a learned reward model from task-harness preferences
  rl-train-scheduler -- Train a learned scheduler from rule-based labels
  golden-generate -- Build a tiny golden debug episode
  golden-evaluate -- Evaluate a model against a saved golden episode
  manifest    -- Build and save a training manifest
  smoke-test  -- Quick end-to-end validation (no GPU needed)
"""

import argparse
import json
import os
import sys
import tempfile


def cmd_generate(args):
    """Generate training data using generate_dataset()."""
    from src.state_encoder import StateEncoder
    from src.data_generator import generate_dataset

    encoder = StateEncoder()

    eval_path = args.eval_output if args.eval_output else None
    eval_fraction = args.eval_fraction if eval_path else 0.0

    print(f"Generating {args.num_games} games (max {args.max_steps} steps each)...")
    print(f"  Seed range: {args.seed_start} .. {args.seed_start + args.num_games - 1}")
    print(f"  Training output: {args.output}")
    if eval_path:
        print(f"  Eval output:     {eval_path} (fraction={eval_fraction})")

    stats = generate_dataset(
        output_path=args.output,
        num_games=args.num_games,
        max_steps=args.max_steps,
        seed_start=args.seed_start,
        encoder=encoder,
        eval_path=eval_path,
        eval_fraction=eval_fraction,
    )

    print()
    print("Generation complete:")
    print(f"  Total games:    {stats['total_games']}")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Train examples: {stats['train_examples']}")
    print(f"  Eval examples:  {stats['eval_examples']}")
    print(f"  Train file:     {stats['train_path']}")
    if stats['eval_path']:
        print(f"  Eval file:      {stats['eval_path']}")

    return 0


def cmd_report(args):
    """Run a game and generate HTML + text reports."""
    from src.state_encoder import StateEncoder
    from src.reporter import run_and_report, format_replay, format_summary

    encoder = StateEncoder()

    print(f"Running game (seed={args.seed}, max_steps={args.max_steps})...")

    result = run_and_report(
        seed=args.seed,
        max_steps=args.max_steps,
        encoder=encoder,
        output_dir=args.output_dir,
    )

    # Print text replay
    replay_text = format_replay(result['step_data'], seed=args.seed)
    print()
    print(replay_text)

    # Print summary
    summary = format_summary(result['step_data'], seed=args.seed)
    print()
    print(summary)

    # Print file paths
    if 'html_path' in result:
        print()
        print(f"HTML report written to: {result['html_path']}")

    text_path = os.path.join(args.output_dir, f'game_seed_{args.seed}.txt')
    with open(text_path, 'w') as f:
        f.write(replay_text)
        f.write('\n\n')
        f.write(summary)
        f.write('\n')
    print(f"Text report written to: {text_path}")

    return 0


def cmd_evaluate(args):
    """Evaluate model prediction accuracy on given seeds."""
    from src.state_encoder import StateEncoder
    from src.evaluator import run_evaluation

    encoder = StateEncoder()
    seeds = [int(s) for s in args.seeds.split(',')]

    print(f"Evaluating on seeds: {seeds}")
    print(f"  Max steps per game: {args.max_steps}")
    print(f"  Server URL: {args.server_url}")

    result = run_evaluation(
        seeds=seeds,
        max_steps=args.max_steps,
        encoder=encoder,
        server_url=args.server_url,
    )

    if not result['server_available']:
        print()
        print(f"WARNING: Server not available at {args.server_url}")
        print("Make sure llama-server is running, or use a different --server-url.")
        return 0

    acc = result['accuracy']
    print()
    print("Evaluation Results:")
    print(f"  Examples evaluated: {acc['n']}")
    print(f"  Exact match rate:   {acc['exact_match_rate']:.1%}")
    print(f"  Position accuracy:  {acc['pos_accuracy']:.1%}")
    print(f"  HP accuracy:        {acc['hp_accuracy']:.1%}")
    print(f"  Gold accuracy:      {acc['gold_accuracy']:.1%}")
    print(f"  Depth accuracy:     {acc['depth_accuracy']:.1%}")
    print(f"  Survived accuracy:  {acc['survived_accuracy']:.1%}")

    if result.get('errors'):
        print()
        print(f"  Errors: {len(result['errors'])}")
        for err in result['errors'][:5]:
            print(f"    - {err}")

    return 0


def cmd_manifest(args):
    """Build and save manifest."""
    import json
    from src.manifest import build_manifest, save_manifest

    baseline_scores = json.loads(args.baseline_scores)
    post_scores = json.loads(args.post_scores)

    print("Building manifest...")
    print(f"  Base model:     {args.base_model}")
    print(f"  Training data:  {args.training_data}")
    print(f"  Adapter path:   {args.adapter}")
    print(f"  Output:         {args.output}")

    manifest = build_manifest(
        base_model=args.base_model,
        training_data_path=args.training_data,
        adapter_path=args.adapter,
        baseline_scores=baseline_scores,
        post_training_scores=post_scores,
    )

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    save_manifest(manifest, args.output)

    print()
    print("Manifest saved successfully.")
    print(f"  File:          {args.output}")
    print(f"  Manifest hash: {manifest['manifest_hash']}")
    print(f"  Data SHA256:   {manifest['training_data']['sha256']}")
    print(f"  Data lines:    {manifest['training_data']['num_lines']}")

    return 0


def cmd_golden_generate(args):
    """Build a tiny golden debug episode JSONL."""
    from src.closed_loop_debug import build_golden_episode
    from src.state_encoder import StateEncoder

    encoder = StateEncoder()
    result = build_golden_episode(
        seed=args.seed,
        max_steps=args.max_steps,
        encoder=encoder,
        output_path=args.output,
    )
    print("Golden episode saved.")
    print(f"  Seed:      {result['seed']}")
    print(f"  Examples:  {result['examples']}")
    print(f"  Output:    {result['path']}")
    return 0


def cmd_golden_evaluate(args):
    """Evaluate a model against a saved golden debug episode."""
    from src.closed_loop_debug import evaluate_golden_episode

    result = evaluate_golden_episode(
        path=args.input,
        server_url=args.server_url,
        model_name_or_path=args.model_name,
        max_samples=args.max_samples,
    )

    if not result["server_available"]:
        print()
        print(f"WARNING: Server not available at {args.server_url}")
        return 0

    acc = result["accuracy"]
    print("Golden episode evaluation:")
    print(f"  Examples evaluated: {acc['n']}")
    print(f"  Exact match rate:   {acc['exact_match_rate']:.1%}")
    print(f"  Position accuracy:  {acc['pos_accuracy']:.1%}")
    print(f"  HP accuracy:        {acc['hp_accuracy']:.1%}")
    print(f"  Gold accuracy:      {acc['gold_accuracy']:.1%}")
    print(f"  Depth accuracy:     {acc['depth_accuracy']:.1%}")
    print(f"  Survived accuracy:  {acc['survived_accuracy']:.1%}")

    mismatches = [row for row in result["comparisons"] if not row["exact_match"]]
    if mismatches:
        first = mismatches[0]
        print(f"  First mismatch step: {first['step']}")
        print(f"  Action:              {first['action']}")
        print(f"  Message hash:        {first['message_hash']}")
    else:
        print("  All steps matched.")
    return 0


def cmd_task_evaluate(args):
    """Evaluate task-directed policies on fixed seeds."""
    from src.task_harness import evaluate_task_policy

    seeds = [int(s) for s in args.seeds.split(',')]
    policies = [args.policy]
    if args.compare_baseline and args.policy != "wall_avoidance":
        policies = ["wall_avoidance", args.policy]

    all_results = []
    print(f"Task evaluation for {args.task}")
    print(f"  Directive: {args.task}")
    print(f"  Seeds: {seeds}")
    print(f"  Max steps: {args.max_steps}")

    for policy in policies:
        result = evaluate_task_policy(
            task=args.task,
            seeds=seeds,
            max_steps=args.max_steps,
            policy=policy,
        )
        all_results.append(result)
        summary = result["summary"]
        print()
        print(f"Policy: {policy}")
        print(f"  Episodes:            {summary['episodes']}")
        print(f"  Avg task reward:     {summary['avg_task_reward']:.2f}")
        print(f"  Avg env reward:      {summary['avg_env_reward']:.2f}")
        print(f"  Avg unique tiles:    {summary['avg_unique_tiles']:.2f}")
        print(f"  Avg rooms:           {summary['avg_rooms_discovered']:.2f}")
        print(f"  Avg final HP:        {summary['avg_final_hp']:.2f}")
        print(f"  Avg final depth:     {summary['avg_final_depth']:.2f}")
        print(f"  Survival rate:       {summary['survival_rate']:.1%}")
        print(f"  Repeated state rate: {summary['repeated_state_rate']:.1%}")
        print(f"  Repeated action rate:{summary['repeated_action_rate']:.1%}")
        if summary["action_counts"]:
            top_actions = sorted(summary["action_counts"].items(), key=lambda kv: (-kv[1], kv[0]))[:6]
            print(f"  Top actions:         {top_actions}")

    if args.output:
        payload = all_results[0] if len(all_results) == 1 else {"results": all_results}
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print()
        print(f"Saved task evaluation to: {args.output}")
    return 0


def cmd_rl_train_appo(args):
    """Run the APPO + options scaffold entrypoint."""
    from rl.bootstrap import ensure_sample_factory_backend
    from rl.train_appo import main as train_appo_main

    installed = ensure_sample_factory_backend()
    if installed:
        print("Installed Sample Factory APPO backend into the project venv.")

    argv = [
        "--experiment", args.experiment,
        "--train-dir", args.train_dir,
        "--serial-mode" if args.serial_mode else "",
        "--async-rl" if args.async_rl else "",
        "--num-workers", str(args.num_workers),
        "--num-envs-per-worker", str(args.num_envs_per_worker),
        "--rollout-length", str(args.rollout_length),
        "--recurrence", str(args.recurrence),
        "--batch-size", str(args.batch_size),
        "--num-batches-per-epoch", str(args.num_batches_per_epoch),
        "--ppo-epochs", str(args.ppo_epochs),
        "--train-for-env-steps", str(args.train_for_env_steps),
        "--scheduler", args.scheduler,
        "--reward-source", args.reward_source,
        "--enabled-skills", args.enabled_skills,
        "--disable-action-mask" if args.disable_action_mask else "",
    ]
    argv = [arg for arg in argv if arg != ""]
    if args.learned_reward_path:
        argv.extend(["--learned-reward-path", args.learned_reward_path])
    if args.scheduler_model_path:
        argv.extend(["--scheduler-model-path", args.scheduler_model_path])
    if args.write_plan:
        argv.extend(["--write-plan", args.write_plan])
    if args.dry_run:
        argv.append("--dry-run")
    return train_appo_main(argv)


def cmd_rl_evaluate_appo(args):
    from rl.evaluate import evaluate_appo_policy

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    result = evaluate_appo_policy(
        experiment=args.experiment,
        train_dir=args.train_dir,
        seeds=seeds,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        mask_actions=not args.disable_eval_mask,
        compare_baseline=args.compare_baseline,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_rl_train_reward(args):
    from rl.train_reward_model import main as train_reward_main

    argv = [
        "--task", args.task,
        "--seeds", args.seeds,
        "--max-steps", str(args.max_steps),
        "--output", args.output,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
    ]
    if args.dataset_output:
        argv.extend(["--dataset-output", args.dataset_output])
    if args.input:
        argv.extend(["--input", args.input])
    return train_reward_main(argv)


def cmd_rl_train_scheduler(args):
    from rl.train_scheduler import main as train_scheduler_main

    argv = [
        "--seeds", args.seeds,
        "--max-steps", str(args.max_steps),
        "--output", args.output,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
    ]
    if args.dataset_output:
        argv.extend(["--dataset-output", args.dataset_output])
    return train_scheduler_main(argv)


def cmd_smoke_test(args):
    """Quick end-to-end test: generate 2 games, verify JSONL, build manifest, verify."""
    from src.state_encoder import StateEncoder
    from src.data_generator import generate_game
    from src.manifest import build_manifest, save_manifest, load_manifest, verify_manifest

    print("Running smoke test...")
    print()

    encoder = StateEncoder()
    errors = []

    # Step 1: Generate 2 games
    print("  [1/4] Generating 2 games (5 steps each)...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, 'train.jsonl')

            all_lines = []
            for seed in (0, 1):
                lines = list(generate_game(seed, 5, encoder))
                all_lines.extend(lines)

            # Write JSONL
            with open(train_path, 'w') as f:
                for line in all_lines:
                    f.write(line + '\n')

            line_count = len(all_lines)
            print(f"        Generated {line_count} training examples")

            # Step 2: Verify JSONL format
            print("  [2/4] Verifying JSONL format...")
            with open(train_path, 'r') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        convs = obj.get('conversations', [])
                        roles = [c['role'] for c in convs]
                        if roles != ['system', 'user', 'assistant']:
                            errors.append(f"Line {i}: unexpected roles {roles}")
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {i}: invalid JSON: {e}")

            if errors:
                print(f"        FAIL: {len(errors)} JSONL errors")
                for e in errors[:5]:
                    print(f"          {e}")
            else:
                print(f"        OK: {line_count} valid JSONL lines")

            # Step 3: Build manifest with dummy scores
            print("  [3/4] Building manifest with dummy scores...")
            adapter_dir = os.path.join(tmpdir, 'adapter')
            os.makedirs(adapter_dir, exist_ok=True)

            manifest = build_manifest(
                base_model='test-model',
                training_data_path=train_path,
                adapter_path=adapter_dir,
                baseline_scores={'field_accuracy': 0.3},
                post_training_scores={'field_accuracy': 0.7},
            )

            manifest_path = os.path.join(tmpdir, 'manifest.json')
            save_manifest(manifest, manifest_path)
            print(f"        Manifest hash: {manifest['manifest_hash']}")

            # Step 4: Verify manifest
            print("  [4/4] Verifying manifest...")
            loaded = load_manifest(manifest_path)
            verification = verify_manifest(loaded)

            if verification['valid']:
                print(f"        OK: manifest self-hash verified")
            else:
                errors.append(f"Manifest hash mismatch: stored={verification['stored_hash']} computed={verification['computed_hash']}")
                print(f"        FAIL: manifest hash mismatch")

    except Exception as e:
        errors.append(str(e))

    print()
    if errors:
        print(f"SMOKE TEST: FAIL ({len(errors)} error(s))")
        for e in errors:
            print(f"  - {e}")
        return 1
    else:
        print("SMOKE TEST: PASS")
        return 0


def main():
    parser = argparse.ArgumentParser(
        prog='cli.py',
        description='rl-nethack training pipeline CLI',
    )
    subparsers = parser.add_subparsers(dest='command', help='Available subcommands')

    # --- generate ---
    p_gen = subparsers.add_parser('generate', help='Generate training data from NLE gameplay')
    p_gen.add_argument('--num-games', type=int, default=100,
                       help='Number of games to play (default: 100)')
    p_gen.add_argument('--max-steps', type=int, default=50,
                       help='Maximum steps per game (default: 50)')
    p_gen.add_argument('--seed-start', type=int, default=0,
                       help='Starting random seed (default: 0)')
    p_gen.add_argument('--output', type=str, default='data/train.jsonl',
                       help='Output path for training JSONL (default: data/train.jsonl)')
    p_gen.add_argument('--eval-output', type=str, default=None,
                       help='Output path for eval JSONL (optional)')
    p_gen.add_argument('--eval-fraction', type=float, default=0.2,
                       help='Fraction of games for eval split (default: 0.2)')

    # --- report ---
    p_rep = subparsers.add_parser('report', help='Run a game and generate reports')
    p_rep.add_argument('--seed', type=int, default=42,
                       help='Random seed for the game (default: 42)')
    p_rep.add_argument('--max-steps', type=int, default=30,
                       help='Maximum steps to play (default: 30)')
    p_rep.add_argument('--output-dir', type=str, default='output/reports',
                       help='Directory for report output (default: output/reports)')

    # --- evaluate ---
    p_eval = subparsers.add_parser('evaluate', help='Evaluate model prediction accuracy')
    p_eval.add_argument('--seeds', type=str, default='100,101,102',
                        help='Comma-separated list of seeds (default: 100,101,102)')
    p_eval.add_argument('--max-steps', type=int, default=20,
                        help='Maximum steps per game (default: 20)')
    p_eval.add_argument('--server-url', type=str, default='http://127.0.0.1:8765',
                        help='llama-server URL (default: http://127.0.0.1:8765)')

    # --- task-evaluate ---
    p_task = subparsers.add_parser('task-evaluate', help='Evaluate task-directed control policies')
    p_task.add_argument('--task', type=str, default='explore',
                        choices=['explore', 'survive', 'combat', 'descend', 'resource'],
                        help='Task to evaluate')
    p_task.add_argument('--policy', type=str, default='task_greedy',
                        choices=['task_greedy', 'wall_avoidance'],
                        help='Policy to run')
    p_task.add_argument('--compare-baseline', action='store_true',
                        help='Also run wall_avoidance for comparison')
    p_task.add_argument('--seeds', type=str, default='42,43,44',
                        help='Comma-separated list of seeds')
    p_task.add_argument('--max-steps', type=int, default=20,
                        help='Maximum steps per episode')
    p_task.add_argument('--output', type=str, default=None,
                        help='Optional JSON output path')

    # --- rl-train-appo ---
    p_rl = subparsers.add_parser('rl-train-appo', help='Run the modular APPO + options scaffold')
    p_rl.add_argument('--experiment', type=str, default='appo_options_scaffold',
                      help='Experiment name')
    p_rl.add_argument('--train-dir', type=str, default='train_dir/rl',
                      help='Output directory for RL artifacts')
    p_rl.add_argument('--serial-mode', action='store_true',
                      help='Run in serial mode for debugging/smoke tests')
    p_rl.add_argument('--async-rl', action='store_true',
                      help='Enable async RL collection explicitly')
    p_rl.add_argument('--num-workers', type=int, default=8,
                      help='Rollout workers')
    p_rl.add_argument('--num-envs-per-worker', type=int, default=16,
                      help='Parallel envs per worker')
    p_rl.add_argument('--rollout-length', type=int, default=64,
                      help='Rollout fragment length')
    p_rl.add_argument('--recurrence', type=int, default=32,
                      help='Recurrent unroll length')
    p_rl.add_argument('--batch-size', type=int, default=4096,
                      help='APPO minibatch size')
    p_rl.add_argument('--num-batches-per-epoch', type=int, default=1,
                      help='APPO batches per epoch')
    p_rl.add_argument('--ppo-epochs', type=int, default=1,
                      help='APPO epochs per update')
    p_rl.add_argument('--train-for-env-steps', type=int, default=50000000,
                      help='Target environment steps')
    p_rl.add_argument('--scheduler', type=str, default='rule_based',
                      help='Skill scheduler')
    p_rl.add_argument('--reward-source', type=str, default='hand_shaped',
                      help='Reward source')
    p_rl.add_argument('--learned-reward-path', type=str, default=None,
                      help='Path to learned reward model when using --reward-source learned')
    p_rl.add_argument('--scheduler-model-path', type=str, default=None,
                      help='Path to learned scheduler model when using --scheduler learned')
    p_rl.add_argument('--enabled-skills', type=str, default='explore,survive,combat,descend,resource',
                      help='Comma-separated skill list')
    p_rl.add_argument('--disable-action-mask', action='store_true',
                      help='Disable env-side invalid action clamping')
    p_rl.add_argument('--write-plan', type=str, default=None,
                      help='Optional JSON output path for the resolved training plan')
    p_rl.add_argument('--dry-run', action='store_true',
                      help='Print scaffold plan without launching training')

    p_rl_eval = subparsers.add_parser('rl-evaluate-appo', help='Evaluate a trained APPO checkpoint')
    p_rl_eval.add_argument('--experiment', type=str, required=True,
                           help='Experiment name under train_dir/rl')
    p_rl_eval.add_argument('--train-dir', type=str, default='train_dir/rl',
                           help='RL train directory')
    p_rl_eval.add_argument('--seeds', type=str, default='42,43,44',
                           help='Comma-separated seed list')
    p_rl_eval.add_argument('--max-steps', type=int, default=50,
                           help='Max steps per episode')
    p_rl_eval.add_argument('--stochastic', action='store_true',
                           help='Use sampled actions instead of argmax')
    p_rl_eval.add_argument('--disable-eval-mask', action='store_true',
                           help='Disable action masking during checkpoint evaluation')
    p_rl_eval.add_argument('--compare-baseline', action='store_true',
                           help='Also evaluate task_greedy if the checkpoint is single-skill')

    p_rl_rew = subparsers.add_parser('rl-train-reward', help='Train a learned reward model')
    p_rl_rew.add_argument('--task', type=str, default='explore',
                          choices=['explore', 'survive', 'combat', 'descend', 'resource'])
    p_rl_rew.add_argument('--seeds', type=str, default='42,43,44,45,46,47')
    p_rl_rew.add_argument('--max-steps', type=int, default=30)
    p_rl_rew.add_argument('--dataset-output', type=str, default=None)
    p_rl_rew.add_argument('--input', type=str, default=None)
    p_rl_rew.add_argument('--output', type=str, required=True)
    p_rl_rew.add_argument('--epochs', type=int, default=20)
    p_rl_rew.add_argument('--lr', type=float, default=1e-3)

    p_rl_sched = subparsers.add_parser('rl-train-scheduler', help='Train a learned scheduler')
    p_rl_sched.add_argument('--seeds', type=str, default='42,43,44,45,46,47')
    p_rl_sched.add_argument('--max-steps', type=int, default=30)
    p_rl_sched.add_argument('--dataset-output', type=str, default=None)
    p_rl_sched.add_argument('--output', type=str, required=True)
    p_rl_sched.add_argument('--epochs', type=int, default=20)
    p_rl_sched.add_argument('--lr', type=float, default=1e-3)

    # --- manifest ---
    p_man = subparsers.add_parser('manifest', help='Build and save a training manifest')
    p_man.add_argument('--base-model', type=str, required=True,
                       help='Base model name or path')
    p_man.add_argument('--training-data', type=str, required=True,
                       help='Path to training data JSONL file')
    p_man.add_argument('--adapter', type=str, required=True,
                       help='Path to adapter directory')
    p_man.add_argument('--baseline-scores', type=str, required=True,
                       help='Baseline scores as JSON string')
    p_man.add_argument('--post-scores', type=str, required=True,
                       help='Post-training scores as JSON string')
    p_man.add_argument('--output', type=str, required=True,
                       help='Output manifest JSON path')

    # --- golden-generate ---
    p_gold_gen = subparsers.add_parser('golden-generate', help='Build a tiny golden debug episode')
    p_gold_gen.add_argument('--seed', type=int, default=42,
                            help='Random seed for the episode (default: 42)')
    p_gold_gen.add_argument('--max-steps', type=int, default=10,
                            help='Maximum steps to record (default: 10)')
    p_gold_gen.add_argument('--output', type=str, default='data/golden_episode.jsonl',
                            help='Output JSONL path (default: data/golden_episode.jsonl)')

    # --- golden-evaluate ---
    p_gold_eval = subparsers.add_parser('golden-evaluate', help='Evaluate a model against a golden episode')
    p_gold_eval.add_argument('--input', type=str, default='data/golden_episode.jsonl',
                             help='Input golden episode JSONL path')
    p_gold_eval.add_argument('--server-url', type=str, default='http://127.0.0.1:8765',
                             help='Model server URL')
    p_gold_eval.add_argument('--model-name', type=str, default='llama-server',
                             help='Model name for logging only')
    p_gold_eval.add_argument('--max-samples', type=int, default=None,
                             help='Optional cap on golden examples to evaluate')

    # --- smoke-test ---
    subparsers.add_parser('smoke-test', help='Quick end-to-end validation')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    try:
        if args.command == 'generate':
            return cmd_generate(args)
        elif args.command == 'report':
            return cmd_report(args)
        elif args.command == 'evaluate':
            return cmd_evaluate(args)
        elif args.command == 'task-evaluate':
            return cmd_task_evaluate(args)
        elif args.command == 'rl-train-appo':
            return cmd_rl_train_appo(args)
        elif args.command == 'rl-evaluate-appo':
            return cmd_rl_evaluate_appo(args)
        elif args.command == 'rl-train-reward':
            return cmd_rl_train_reward(args)
        elif args.command == 'rl-train-scheduler':
            return cmd_rl_train_scheduler(args)
        elif args.command == 'manifest':
            return cmd_manifest(args)
        elif args.command == 'golden-generate':
            return cmd_golden_generate(args)
        elif args.command == 'golden-evaluate':
            return cmd_golden_evaluate(args)
        elif args.command == 'smoke-test':
            return cmd_smoke_test(args)
        else:
            parser.print_help()
            return 1
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
