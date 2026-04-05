#!/usr/bin/env python3
"""
CLI entry point for the rl-nethack training pipeline.

Subcommands:
  generate    -- Generate training data from NLE gameplay
  report      -- Run a game and produce HTML + text reports
  evaluate    -- Evaluate model prediction accuracy
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
        elif args.command == 'manifest':
            return cmd_manifest(args)
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
