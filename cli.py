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
  rl-generate-traces -- Generate explicit multi-turn traces
  rl-verify-traces -- Verify trace files and report if they are multi-turn
  rl-train-bc -- Train a behavior cloning policy from traces
  rl-evaluate-bc -- Evaluate a behavior cloning policy
  rl-train-world-model -- Train a short-horizon latent world model on traces
  rl-evaluate-world-model -- Evaluate a short-horizon latent world model
  rl-transform-traces-world-model -- Rewrite trace features using a trained world model encoder
  rl-relabel-traces-bc -- Relabel trace actions with a BC teacher
  rl-build-proxy-dataset -- Build teacher-derived proxy labels from traces
  rl-train-proxy -- Train a teacher-derived proxy reward model
  rl-evaluate-proxy -- Evaluate a teacher-derived proxy reward model
  rl-check-determinism -- Repeat the same evaluation and diff action traces
  rl-compare-actions -- Compare teacher, BC, and APPO on teacher-induced states
  rl-short-benchmark -- Train BC on a trace shard and run the fast debug checks
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

    if result.get("warning"):
        print(f"  Note: {result['warning']}")

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
        "--learning-rate", str(args.learning_rate),
        "--gamma", str(args.gamma),
        "--gae-lambda", str(args.gae_lambda),
        "--value-loss-coeff", str(args.value_loss_coeff),
        "--reward-scale", str(args.reward_scale),
        "--entropy-coeff", str(args.entropy_coeff),
        "--ppo-clip-ratio", str(args.ppo_clip_ratio),
        "--train-for-env-steps", str(args.train_for_env_steps),
        "--scheduler", args.scheduler,
        "--reward-source", args.reward_source,
        "--episodic-explore-bonus-scale", str(args.episodic_explore_bonus_scale),
        "--episodic-explore-bonus-mode", args.episodic_explore_bonus_mode,
        "--enabled-skills", args.enabled_skills,
        "--observation-version", args.observation_version,
        "--env-max-episode-steps", str(args.env_max_episode_steps),
        "--trace-eval-top-k", str(args.trace_eval_top_k),
        "--save-every-sec", str(args.save_every_sec),
        "--save-best-every-sec", str(args.save_best_every_sec),
        "--no-rnn" if args.no_rnn else "",
        "--use-rnn" if getattr(args, "use_rnn", False) else "",
        "--disable-input-normalization" if getattr(args, "disable_input_normalization", False) else "",
        "--disable-action-mask" if args.disable_action_mask else "",
    ]
    argv = [arg for arg in argv if arg != ""]
    if args.learned_reward_path:
        argv.extend(["--learned-reward-path", args.learned_reward_path])
    if getattr(args, "proxy_reward_path", None):
        argv.extend(["--proxy-reward-path", args.proxy_reward_path])
    if float(getattr(args, "proxy_reward_weight", 1.0)) != 1.0:
        argv.extend(["--proxy-reward-weight", str(args.proxy_reward_weight)])
    if getattr(args, "model_hidden_size", None) is not None:
        argv.extend(["--model-hidden-size", str(args.model_hidden_size)])
    if getattr(args, "model_num_layers", None) is not None:
        argv.extend(["--model-num-layers", str(args.model_num_layers)])
    if getattr(args, "separate_actor_critic", False):
        argv.append("--separate-actor-critic")
    if getattr(args, "world_model_path", None):
        argv.extend(["--world-model-path", args.world_model_path])
    if getattr(args, "world_model_feature_mode", None):
        argv.extend(["--world-model-feature-mode", args.world_model_feature_mode])
    if getattr(args, "nonlinearity", None):
        argv.extend(["--nonlinearity", args.nonlinearity])
    if args.episodic_explore_bonus_enabled:
        argv.append("--episodic-explore-bonus-enabled")
    if args.scheduler_model_path:
        argv.extend(["--scheduler-model-path", args.scheduler_model_path])
    if args.bc_init_path:
        argv.extend(["--bc-init-path", args.bc_init_path])
    if getattr(args, "appo_init_checkpoint_path", None):
        argv.extend(["--appo-init-checkpoint-path", args.appo_init_checkpoint_path])
    if args.teacher_bc_path:
        argv.extend(["--teacher-bc-path", args.teacher_bc_path])
    if getattr(args, "teacher_prior_bc_path", None):
        argv.extend(["--teacher-prior-bc-path", args.teacher_prior_bc_path])
    if getattr(args, "teacher_report_path", None):
        argv.extend(["--teacher-report-path", args.teacher_report_path])
    if getattr(args, "teacher_loss_coef", None) is not None:
        argv.extend(["--teacher-loss-coef", str(args.teacher_loss_coef)])
    if args.teacher_loss_type:
        argv.extend(["--teacher-loss-type", args.teacher_loss_type])
    if getattr(args, "teacher_action_boosts", ""):
        argv.extend(["--teacher-action-boosts", args.teacher_action_boosts])
    if getattr(args, "teacher_loss_final_coef", None) is not None:
        argv.extend(["--teacher-loss-final-coef", str(args.teacher_loss_final_coef)])
    if getattr(args, "teacher_loss_warmup_env_steps", None) is not None:
        argv.extend(["--teacher-loss-warmup-env-steps", str(args.teacher_loss_warmup_env_steps)])
    if getattr(args, "teacher_loss_decay_env_steps", None) is not None:
        argv.extend(["--teacher-loss-decay-env-steps", str(args.teacher_loss_decay_env_steps)])
    if getattr(args, "teacher_replay_trace_input", None):
        argv.extend(["--teacher-replay-trace-input", args.teacher_replay_trace_input])
    if getattr(args, "teacher_replay_coef", None) is not None:
        argv.extend(["--teacher-replay-coef", str(args.teacher_replay_coef)])
    if getattr(args, "teacher_replay_final_coef", None) is not None:
        argv.extend(["--teacher-replay-final-coef", str(args.teacher_replay_final_coef)])
    if getattr(args, "teacher_replay_warmup_env_steps", None) is not None:
        argv.extend(["--teacher-replay-warmup-env-steps", str(args.teacher_replay_warmup_env_steps)])
    if getattr(args, "teacher_replay_decay_env_steps", None) is not None:
        argv.extend(["--teacher-replay-decay-env-steps", str(args.teacher_replay_decay_env_steps)])
    if getattr(args, "teacher_replay_batch_size", None) is not None:
        argv.extend(["--teacher-replay-batch-size", str(args.teacher_replay_batch_size)])
    if getattr(args, "teacher_replay_priority_power", 1.0) != 1.0:
        argv.extend(["--teacher-replay-priority-power", str(args.teacher_replay_priority_power)])
    if getattr(args, "teacher_replay_source_mode", "uniform") != "uniform":
        argv.extend(["--teacher-replay-source-mode", str(args.teacher_replay_source_mode)])
    if getattr(args, "teacher_replay_action_boosts", ""):
        argv.extend(["--teacher-replay-action-boosts", str(args.teacher_replay_action_boosts)])
    if getattr(args, "teacher_replay_current_disagreement_boost", 1.0) != 1.0:
        argv.extend(
            [
                "--teacher-replay-current-disagreement-boost",
                str(args.teacher_replay_current_disagreement_boost),
            ]
        )
    if getattr(args, "teacher_replay_confusion_pair_boosts", ""):
        argv.extend(["--teacher-replay-confusion-pair-boosts", str(args.teacher_replay_confusion_pair_boosts)])
    if getattr(args, "teacher_replay_confusion_pair_start_env_steps", 0):
        argv.extend(
            [
                "--teacher-replay-confusion-pair-start-env-steps",
                str(args.teacher_replay_confusion_pair_start_env_steps),
            ]
        )
    if getattr(args, "teacher_policy_logit_residual_scale", 1.0) != 1.0:
        argv.extend(["--teacher-policy-logit-residual-scale", str(args.teacher_policy_logit_residual_scale)])
    if getattr(args, "teacher_policy_residual_logit_cap", 0.0) > 0.0:
        argv.extend(["--teacher-policy-residual-logit-cap", str(args.teacher_policy_residual_logit_cap)])
    if getattr(args, "teacher_policy_blend_coef", 0.0) != 0.0:
        argv.extend(["--teacher-policy-blend-coef", str(args.teacher_policy_blend_coef)])
    if getattr(args, "teacher_policy_fallback_confidence", 0.0) != 0.0:
        argv.extend(["--teacher-policy-fallback-confidence", str(args.teacher_policy_fallback_confidence)])
    if getattr(args, "teacher_policy_disagreement_margin", 0.0) != 0.0:
        argv.extend(["--teacher-policy-disagreement-margin", str(args.teacher_policy_disagreement_margin)])
    if getattr(args, "param_anchor_coef", None) is not None:
        argv.extend(["--param-anchor-coef", str(args.param_anchor_coef)])
    if getattr(args, "actor_loss_scale", None) is not None:
        argv.extend(["--actor-loss-scale", str(args.actor_loss_scale)])
    if getattr(args, "actor_loss_final_scale", None) is not None:
        argv.extend(["--actor-loss-final-scale", str(args.actor_loss_final_scale)])
    if getattr(args, "actor_loss_warmup_env_steps", None) is not None:
        argv.extend(["--actor-loss-warmup-env-steps", str(args.actor_loss_warmup_env_steps)])
    if getattr(args, "actor_loss_decay_env_steps", None) is not None:
        argv.extend(["--actor-loss-decay-env-steps", str(args.actor_loss_decay_env_steps)])
    if args.trace_eval_input:
        argv.extend(["--trace-eval-input", args.trace_eval_input])
    if args.trace_eval_interval_env_steps:
        argv.extend(["--trace-eval-interval-env-steps", str(args.trace_eval_interval_env_steps)])
    if args.write_plan:
        argv.extend(["--write-plan", args.write_plan])
    if getattr(args, "improver_report_output", None):
        argv.extend(["--improver-report-output", args.improver_report_output])
    if args.dry_run:
        argv.append("--dry-run")
    return train_appo_main(argv)


def cmd_rl_evaluate_appo(args):
    from rl.evaluate import evaluate_appo_policy
    from rl.trace_eval import evaluate_trace_policy

    if args.trace_input:
        result = evaluate_trace_policy(
            trace_path=args.trace_input,
            policy="appo",
            appo_experiment=args.experiment,
            appo_train_dir=args.train_dir,
            appo_checkpoint_path=args.checkpoint_path,
            deterministic=not args.stochastic,
        )
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        result = evaluate_appo_policy(
            experiment=args.experiment,
            train_dir=args.train_dir,
            seeds=seeds,
            max_steps=args.max_steps,
            deterministic=not args.stochastic,
            mask_actions=not args.disable_eval_mask,
            compare_baseline=args.compare_baseline,
            checkpoint_path=args.checkpoint_path,
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


def cmd_rl_generate_traces(args):
    from rl.traces import generate_multi_turn_traces

    result = generate_multi_turn_traces(
        output_path=args.output,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed_start=args.seed_start,
        policy=args.policy,
        task=args.task,
        appo_experiment=args.appo_experiment,
        appo_train_dir=args.appo_train_dir,
        bc_model_path=args.bc_model_path,
        forward_model_server_url=args.server_url,
        forward_model_name=args.model_name,
        observation_version=args.observation_version,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_rl_mine_reset_slice(args):
    from rl.traces import mine_reset_teacher_slice

    adjacent_signature = {}
    for token in str(args.adjacent_signature or "").split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid adjacent signature token: {token!r}")
        key, value = token.split("=", 1)
        adjacent_signature[key.strip()] = value.strip()

    result = mine_reset_teacher_slice(
        output_path=args.output,
        seed_start=args.seed_start,
        num_seeds=args.num_seeds,
        task=args.task,
        observation_version=args.observation_version,
        adjacent_signature=adjacent_signature,
        recreate_every=args.recreate_every,
        max_rows=args.max_rows,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_rl_verify_traces(args):
    from rl.traces import verify_trace_file

    result = verify_trace_file(args.input)
    print(json.dumps(result, indent=2))
    return 0


def cmd_rl_train_bc(args):
    from rl.train_bc import main as train_bc_main

    argv = [
        "--input", args.input,
        "--output", args.output,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--hidden-size", str(args.hidden_size),
        "--num-layers", str(args.num_layers),
        "--observation-version", args.observation_version,
    ]
    if args.world_model_path:
        argv.extend(["--world-model-path", args.world_model_path])
    if args.world_model_feature_mode:
        argv.extend(["--world-model-feature-mode", args.world_model_feature_mode])
    if args.distill_teacher_bc_path:
        argv.extend(["--distill-teacher-bc-path", args.distill_teacher_bc_path])
        argv.extend(["--distill-loss-coef", str(args.distill_loss_coef)])
        argv.extend(["--distill-temperature", str(args.distill_temperature)])
    if args.distill_teacher_bc_paths:
        argv.extend(["--distill-teacher-bc-paths", *args.distill_teacher_bc_paths])
        if not args.distill_teacher_bc_path:
            argv.extend(["--distill-loss-coef", str(args.distill_loss_coef)])
            argv.extend(["--distill-temperature", str(args.distill_temperature)])
    if float(getattr(args, "supervised_loss_coef", 1.0)) != 1.0:
        argv.extend(["--supervised-loss-coef", str(args.supervised_loss_coef)])
    if getattr(args, "action_weight_boosts", None):
        argv.extend(["--action-weight-boosts", args.action_weight_boosts])
    if getattr(args, "text_encoder_backend", "none") != "none":
        argv.extend(["--text-encoder-backend", args.text_encoder_backend])
    if int(getattr(args, "text_vocab_size", 4096)) != 4096:
        argv.extend(["--text-vocab-size", str(args.text_vocab_size)])
    if int(getattr(args, "text_embedding_dim", 128)) != 128:
        argv.extend(["--text-embedding-dim", str(args.text_embedding_dim)])
    if getattr(args, "text_model_name", None):
        argv.extend(["--text-model-name", args.text_model_name])
    if int(getattr(args, "text_max_length", 128)) != 128:
        argv.extend(["--text-max-length", str(args.text_max_length)])
    if bool(getattr(args, "text_trainable", False)):
        argv.append("--text-trainable")
    if getattr(args, "device", "auto") != "auto":
        argv.extend(["--device", args.device])
    if bool(getattr(args, "select_by_heldout", False)):
        argv.append("--select-by-heldout")
    if args.heldout_input:
        argv.extend(["--heldout-input", args.heldout_input])
    if args.teacher_report_output:
        argv.extend(["--teacher-report-output", args.teacher_report_output])
    if args.weak_action_input:
        argv.extend(["--weak-action-input", args.weak_action_input])
    return train_bc_main(argv)


def cmd_rl_evaluate_bc(args):
    from rl.evaluate_bc import evaluate_bc_policy
    from rl.trace_eval import evaluate_trace_policy

    if args.trace_input:
        result = evaluate_trace_policy(
            trace_path=args.trace_input,
            policy="bc",
            bc_model_path=args.model,
        )
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        result = evaluate_bc_policy(
            model_path=args.model,
            task=args.task,
            seeds=seeds,
            max_steps=args.max_steps,
            compare_baseline=args.compare_baseline,
        )
    print(json.dumps(result, indent=2))
    return 0


def cmd_rl_train_world_model(args):
    from rl.train_bc import load_trace_rows
    from rl.train_world_model import train_world_model
    from rl.io_utils import atomic_write_json

    rows = load_trace_rows(args.input)
    result = train_world_model(
        rows,
        args.output,
        horizon=args.horizon,
        epochs=args.epochs,
        lr=args.lr,
        hidden_size=args.hidden_size,
        latent_dim=args.latent_dim,
        observation_version=args.observation_version,
        reward_loss_coef=args.reward_loss_coef,
        done_loss_coef=args.done_loss_coef,
        reconstruction_loss_coef=args.reconstruction_loss_coef,
        action_loss_coef=args.action_loss_coef,
        action_class_balance=args.action_class_balance,
        action_class_balance_power=args.action_class_balance_power,
        text_encoder_backend=args.text_encoder_backend,
        text_model_name=args.text_model_name,
        text_max_length=args.text_max_length,
        text_trainable=args.text_trainable,
        text_embedding_dim=args.text_embedding_dim,
    )
    if args.report_output:
        atomic_write_json(args.report_output, result)
    print(json.dumps(result, indent=2))
    return 0


def cmd_rl_evaluate_world_model(args):
    from rl.world_model_eval import evaluate_world_model
    from rl.io_utils import atomic_write_json

    result = evaluate_world_model(
        args.model,
        args.input,
        horizon=args.horizon,
        observation_version=args.observation_version,
        downstream_train_trace_path=args.downstream_train_input,
        downstream_heldout_trace_path=args.downstream_heldout_input,
        downstream_mode=args.downstream_mode,
        downstream_epochs=args.downstream_epochs,
        downstream_lr=args.downstream_lr,
        downstream_hidden_size=args.downstream_hidden_size,
    )
    if args.report_output:
        atomic_write_json(args.report_output, result)
    print(json.dumps(result, indent=2))
    return 0


def cmd_rl_relabel_traces_bc(args):
    from rl.relabel_traces import relabel_trace_actions

    result = relabel_trace_actions(
        args.input,
        args.output,
        bc_model_path=args.bc_model_path,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_rl_transform_traces_world_model(args):
    from rl.world_model_features import transform_trace_with_world_model

    result = transform_trace_with_world_model(
        args.input,
        args.output,
        args.model,
        observation_version_suffix=args.version_suffix,
        mode=args.mode,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_rl_build_proxy_dataset(args):
    from rl.proxy_dataset import build_proxy_dataset_from_trace_file

    result = build_proxy_dataset_from_trace_file(
        input_path=args.input,
        output_path=args.output,
        horizon=args.horizon,
        task_filter=args.task,
        max_rows=args.max_rows,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_rl_train_proxy(args):
    from rl.train_proxy_model import main as train_proxy_main

    argv = [
        "--input", args.input,
        "--output", args.output,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--hidden-size", str(args.hidden_size),
        "--action-embed-dim", str(args.action_embed_dim),
    ]
    if args.heldout_input:
        argv.extend(["--heldout-input", args.heldout_input])
    if args.report_output:
        argv.extend(["--report-output", args.report_output])
    return train_proxy_main(argv)


def cmd_rl_evaluate_proxy(args):
    from rl.proxy_eval import evaluate_proxy_model
    from rl.proxy_report import build_proxy_report

    result = evaluate_proxy_model(args.model, args.input)
    if args.report_output:
        result["report"] = build_proxy_report(args.model, args.input, top_k=args.top_k)
        with open(args.report_output, "w") as f:
            json.dump(result["report"], f, indent=2)
        result["report_output"] = args.report_output
    print(json.dumps(result, indent=2))
    return 0


def cmd_rl_compare_policies(args):
    from rl.evaluate import evaluate_appo_policy
    from rl.evaluate_bc import evaluate_bc_policy
    from src.task_harness import evaluate_task_policy

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    baseline = evaluate_task_policy(
        task=args.task,
        seeds=seeds,
        max_steps=args.max_steps,
        policy="task_greedy",
    )
    result = {
        "task": args.task,
        "seeds": seeds,
        "max_steps": args.max_steps,
        "task_greedy": {"summary": baseline["summary"]},
    }
    if args.bc_model:
        bc_result = evaluate_bc_policy(
            model_path=args.bc_model,
            task=args.task,
            seeds=seeds,
            max_steps=args.max_steps,
        )
        result["bc"] = {"summary": bc_result["summary"]}
    if args.appo_experiment:
        appo_result = evaluate_appo_policy(
            experiment=args.appo_experiment,
            train_dir=args.appo_train_dir,
            seeds=seeds,
            max_steps=args.max_steps,
            deterministic=True,
            mask_actions=True,
        )
        result["appo"] = {"summary": appo_result["summary"]}
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    print(json.dumps(result, indent=2))
    return 0


def cmd_rl_check_determinism(args):
    from rl.debug_tools import check_policy_determinism

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    result = check_policy_determinism(
        policy=args.policy,
        task=args.task,
        seeds=seeds,
        max_steps=args.max_steps,
        repeats=args.repeats,
        bc_model_path=args.bc_model,
        appo_experiment=args.appo_experiment,
        appo_train_dir=args.appo_train_dir,
        observation_version=args.observation_version,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_rl_compare_actions(args):
    from rl.debug_tools import compare_actions_on_teacher_states
    from rl.trace_eval import compare_trace_policies

    if args.input:
        result = compare_trace_policies(
            trace_path=args.input,
            bc_model_path=args.bc_model,
            appo_experiment=args.appo_experiment,
            appo_train_dir=args.appo_train_dir,
            appo_checkpoint_path=args.appo_checkpoint_path,
            summary_only=args.summary_only,
        )
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        result = compare_actions_on_teacher_states(
            task=args.task,
            seeds=seeds,
            max_steps=args.max_steps,
            bc_model_path=args.bc_model,
            appo_experiment=args.appo_experiment,
            appo_train_dir=args.appo_train_dir,
            observation_version=args.observation_version,
        )
    print(json.dumps(result, indent=2))
    return 0


def cmd_rl_short_benchmark(args):
    from rl.train_bc import load_trace_rows, train_bc_model
    from rl.evaluate_bc import evaluate_bc_policy
    from rl.debug_tools import check_policy_determinism
    from rl.trace_eval import evaluate_trace_policy

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = load_trace_rows(args.input)
    train_result = train_bc_model(
        rows,
        args.output,
        epochs=args.epochs,
        lr=args.lr,
        hidden_size=args.hidden_size,
        observation_version=args.observation_version,
    )
    eval_result = evaluate_bc_policy(
        model_path=args.output,
        task=args.task,
        seeds=seeds,
        max_steps=args.max_steps,
        compare_baseline=args.compare_baseline,
    )
    determinism_result = check_policy_determinism(
        policy="bc",
        task=args.task,
        seeds=seeds,
        max_steps=args.max_steps,
        repeats=args.repeats,
        bc_model_path=args.output,
        observation_version=args.observation_version,
    )
    trace_eval_result = evaluate_trace_policy(
        trace_path=args.input,
        policy="bc",
        bc_model_path=args.output,
    )
    result = {
        "train": train_result,
        "eval": eval_result,
        "determinism": {
            "stable": determinism_result["stable"],
            "mismatches": determinism_result["mismatches"],
        },
        "trace_eval": trace_eval_result["summary"],
    }
    if args.report:
        with open(args.report, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    print(json.dumps(result, indent=2))
    return 0


def cmd_rl_trace_report(args):
    from rl.trace_eval import compare_trace_policies, trace_disagreement_report
    from rl.traces import verify_trace_file

    result = {
        "trace_verify": verify_trace_file(args.input),
        "policy_compare": compare_trace_policies(
            trace_path=args.input,
            bc_model_path=args.bc_model,
            appo_experiment=args.appo_experiment,
            appo_train_dir=args.appo_train_dir,
            appo_checkpoint_path=args.appo_checkpoint_path,
            summary_only=True,
        ),
    }
    if args.detailed:
        result["disagreements"] = trace_disagreement_report(
            trace_path=args.input,
            bc_model_path=args.bc_model,
            appo_experiment=args.appo_experiment,
            appo_train_dir=args.appo_train_dir,
            appo_checkpoint_path=args.appo_checkpoint_path,
            top_k=args.top_k,
        )
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    print(json.dumps(result, indent=2))
    return 0


def cmd_rl_trace_disagreements(args):
    from rl.trace_eval import trace_disagreement_report

    result = trace_disagreement_report(
        trace_path=args.input,
        bc_model_path=args.bc_model,
        appo_experiment=args.appo_experiment,
        appo_train_dir=args.appo_train_dir,
        appo_checkpoint_path=args.appo_checkpoint_path,
        top_k=args.top_k,
    )
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    print(json.dumps(result, indent=2))
    return 0


def cmd_rl_shard_traces(args):
    from rl.traces import shard_trace_file

    seeds = None
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    teacher_actions = None
    if args.teacher_actions:
        teacher_actions = [s.strip() for s in args.teacher_actions.split(",") if s.strip()]
    adjacent_signature = None
    if getattr(args, "adjacent_signature", None):
        adjacent_signature = {}
        for token in args.adjacent_signature.split(","):
            token = token.strip()
            if not token:
                continue
            if "=" not in token:
                raise ValueError(f"Invalid adjacent signature token: {token!r}")
            key, value = token.split("=", 1)
            adjacent_signature[key.strip()] = value.strip()
    result = shard_trace_file(
        input_path=args.input,
        output_path=args.output,
        max_episodes=args.max_episodes,
        max_rows=args.max_rows,
        seeds=seeds,
        teacher_actions=teacher_actions,
        adjacent_signature=adjacent_signature,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_rl_rank_checkpoints(args):
    from rl.checkpoint_tools import (
        materialize_trace_best_checkpoint,
        rank_appo_checkpoints_by_trace,
        write_trace_best_alias,
    )

    result = rank_appo_checkpoints_by_trace(
        experiment=args.experiment,
        train_dir=args.train_dir,
        trace_input=args.trace_input,
        top_k=args.top_k,
    )
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    if args.best_alias_output:
        result["best_alias_output"] = write_trace_best_alias(result, args.best_alias_output)
    if args.materialize_best_trace:
        result["best_trace_checkpoint_alias"] = materialize_trace_best_checkpoint(result)
    print(json.dumps(result, indent=2))
    return 0


def cmd_rl_run_dagger(args):
    from rl.dagger import run_dagger_iteration

    result = run_dagger_iteration(
        base_trace_input=args.input,
        dagger_trace_output=args.dagger_output,
        merged_trace_output=args.merged_output,
        bc_output=args.bc_output,
        student_policy=args.student_policy,
        task=args.task,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed_start=args.seed_start,
        appo_experiment=args.appo_experiment,
        appo_train_dir=args.appo_train_dir,
        appo_checkpoint_path=args.appo_checkpoint_path,
        bc_model_path=args.bc_model_path,
        teacher_bc_model_path=args.teacher_bc_model_path,
        observation_version=args.observation_version,
        merge_ratio=args.merge_ratio,
        merge_policy=args.merge_policy,
        dagger_row_policy=args.dagger_row_policy,
        dagger_keep_match_ratio=args.dagger_keep_match_ratio,
        dagger_confusion_pairs=args.dagger_confusion_pairs,
        epochs=args.epochs,
        lr=args.lr,
        hidden_size=args.hidden_size,
        distill_teacher_bc_path=args.distill_teacher_bc_path,
        distill_loss_coef=args.distill_loss_coef,
        distill_temperature=args.distill_temperature,
        heldout_trace_input=args.heldout_input,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_rl_dagger_iterate(args):
    from rl.dagger import run_dagger_schedule

    result = run_dagger_schedule(
        base_trace_input=args.input,
        output_dir=args.output_dir,
        student_policy=args.student_policy,
        task=args.task,
        iterations=args.iterations,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed_start=args.seed_start,
        appo_experiment=args.appo_experiment,
        appo_train_dir=args.appo_train_dir,
        appo_checkpoint_path=args.appo_checkpoint_path,
        bc_model_path=args.bc_model_path,
        teacher_bc_model_path=args.teacher_bc_model_path,
        observation_version=args.observation_version,
        merge_ratio=args.merge_ratio,
        merge_policy=args.merge_policy,
        dagger_row_policy=args.dagger_row_policy,
        dagger_keep_match_ratio=args.dagger_keep_match_ratio,
        dagger_confusion_pairs=args.dagger_confusion_pairs,
        epochs=args.epochs,
        lr=args.lr,
        hidden_size=args.hidden_size,
        distill_teacher_bc_path=args.distill_teacher_bc_path,
        distill_loss_coef=args.distill_loss_coef,
        distill_temperature=args.distill_temperature,
        heldout_trace_input=args.heldout_input,
        random_seed=args.random_seed,
        stop_on_heldout_regression=args.stop_on_heldout_regression,
        min_improvement=args.min_improvement,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_rl_train_behavior_reg(args):
    from rl.train_behavior_reg import main as train_behavior_reg_main

    argv = [
        "--input", args.input,
        "--output", args.output,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--hidden-size", str(args.hidden_size),
        "--observation-version", args.observation_version,
        "--behavior-coef", str(args.behavior_coef),
        "--temperature", str(args.temperature),
        "--class-balance-power", str(args.class_balance_power),
        "--teacher-action-boost-scale", str(args.teacher_action_boost_scale),
    ]
    if args.heldout_input:
        argv.extend(["--heldout-input", args.heldout_input])
    if args.teacher_action_boost:
        argv.extend(["--teacher-action-boost", args.teacher_action_boost])
    if args.teacher_report_output:
        argv.extend(["--teacher-report-output", args.teacher_report_output])
    if args.weak_action_input:
        argv.extend(["--weak-action-input", args.weak_action_input])
    return train_behavior_reg_main(argv)


def cmd_rl_shard_benchmark(args):
    import tempfile
    from rl.trace_eval import evaluate_trace_policy, trace_disagreement_report
    from rl.train_bc import load_trace_rows, train_bc_model
    from rl.train_behavior_reg import train_behavior_regularized_policy

    with tempfile.TemporaryDirectory() as tmpdir:
        bc_path = os.path.join(tmpdir, "bc.pt")
        behavior_path = os.path.join(tmpdir, "behavior.pt")
        train_rows = load_trace_rows(args.input)
        train_bc_model(
            train_rows,
            bc_path,
            epochs=args.epochs,
            lr=args.lr,
            hidden_size=args.hidden_size,
            observation_version=args.observation_version,
        )
        train_behavior_regularized_policy(
            train_rows,
            behavior_path,
            heldout_trace_path=args.heldout_input,
            epochs=args.epochs,
            lr=args.lr,
            hidden_size=args.hidden_size,
            observation_version=args.observation_version,
            behavior_coef=args.behavior_coef,
            temperature=args.temperature,
            class_balance_power=args.class_balance_power,
            teacher_action_boost=[x.strip() for x in args.teacher_action_boost.split(",") if x.strip()],
            teacher_action_boost_scale=args.teacher_action_boost_scale,
        )
        result = {
            "input": args.input,
            "heldout_input": args.heldout_input,
            "bc": evaluate_trace_policy(args.heldout_input or args.input, "bc", bc_model_path=bc_path, summary_only=True)["summary"],
            "behavior_reg": evaluate_trace_policy(args.heldout_input or args.input, "bc", bc_model_path=behavior_path, summary_only=True)["summary"],
        }
        if args.detailed:
            result["bc_disagreements"] = trace_disagreement_report(args.heldout_input or args.input, bc_model_path=bc_path, top_k=5)["bc"]
            result["behavior_reg_disagreements"] = trace_disagreement_report(args.heldout_input or args.input, bc_model_path=behavior_path, top_k=5)["bc"]
        print(json.dumps(result, indent=2))
    return 0


def cmd_rl_teacher_reg_report(args):
    from pathlib import Path
    from rl.trace_eval import evaluate_trace_policy
    from rl.evaluate import list_checkpoint_paths

    checkpoint_dir = Path(args.train_dir) / args.experiment / "checkpoint_p0"
    latest_checkpoint = str(list_checkpoint_paths(args.experiment, args.train_dir)[-1])
    best_trace_alias = checkpoint_dir / "best_trace_match.pth"
    result = {
        "experiment": args.experiment,
        "trace_input": args.trace_input,
        "bc_teacher": evaluate_trace_policy(args.trace_input, "bc", bc_model_path=args.bc_model, summary_only=True)["summary"],
        "latest_appo": evaluate_trace_policy(
            args.trace_input,
            "appo",
            appo_experiment=args.experiment,
            appo_train_dir=args.train_dir,
            appo_checkpoint_path=latest_checkpoint,
            summary_only=True,
        )["summary"],
    }
    if best_trace_alias.exists():
        result["best_trace_appo"] = evaluate_trace_policy(
            args.trace_input,
            "appo",
            appo_experiment=args.experiment,
            appo_train_dir=args.train_dir,
            appo_checkpoint_path=str(best_trace_alias),
            summary_only=True,
        )["summary"]
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    print(json.dumps(result, indent=2))
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
    p_rl.add_argument('--learning-rate', type=float, default=3e-4,
                      help='APPO optimizer learning rate')
    p_rl.add_argument('--gamma', type=float, default=0.999,
                      help='Discount factor for return/value targets')
    p_rl.add_argument('--gae-lambda', type=float, default=0.95,
                      help='GAE lambda used by APPO')
    p_rl.add_argument('--value-loss-coeff', type=float, default=0.5,
                      help='Value loss coefficient; lowering this can stabilize long-horizon teacher-constrained runs')
    p_rl.add_argument('--reward-scale', type=float, default=0.1,
                      help='Sample Factory reward scale applied before value learning')
    p_rl.add_argument('--entropy-coeff', type=float, default=0.01,
                      help='Entropy regularization coefficient')
    p_rl.add_argument('--ppo-clip-ratio', type=float, default=0.1,
                      help='PPO clip ratio')
    p_rl.add_argument('--train-for-env-steps', type=int, default=50000000,
                      help='Target environment steps')
    p_rl.add_argument('--scheduler', type=str, default='rule_based',
                      help='Skill scheduler')
    p_rl.add_argument('--reward-source', type=str, default='hand_shaped',
                      help='Reward source')
    p_rl.add_argument('--learned-reward-path', type=str, default=None,
                      help='Path to learned reward model when using --reward-source learned')
    p_rl.add_argument('--proxy-reward-path', type=str, default=None,
                      help='Path to teacher-derived proxy model when using --reward-source proxy or mixed_proxy')
    p_rl.add_argument('--proxy-reward-weight', type=float, default=1.0,
                      help='Scaling weight for the proxy reward when using --reward-source mixed_proxy')
    p_rl.add_argument('--episodic-explore-bonus-enabled', action='store_true',
                      help='Enable an explore-only episodic novelty bonus')
    p_rl.add_argument('--episodic-explore-bonus-scale', type=float, default=0.0,
                      help='Scale for the explore-only episodic novelty bonus')
    p_rl.add_argument('--episodic-explore-bonus-mode', type=str, default='state_hash',
                      choices=['state_hash', 'tile'],
                      help='Counting mode for the explore-only episodic novelty bonus')
    p_rl.add_argument('--scheduler-model-path', type=str, default=None,
                      help='Path to learned scheduler model when using --scheduler learned')
    p_rl.add_argument('--enabled-skills', type=str, default='explore,survive,combat,descend,resource',
                      help='Comma-separated skill list')
    p_rl.add_argument('--observation-version', type=str, default='v2',
                      help='Observation encoder version (v1, v2, or v3)')
    p_rl.add_argument('--world-model-path', type=str, default=None,
                      help='Optional trained world-model checkpoint used to augment online observations')
    p_rl.add_argument('--world-model-feature-mode', type=str, default=None,
                      choices=['replace', 'concat', 'concat_aux'],
                      help='How to augment online observations with the world model')
    p_rl.add_argument('--env-max-episode-steps', type=int, default=5000,
                      help='Episode horizon for RL training/evaluation envs')
    p_rl.add_argument('--model-hidden-size', type=int, default=None,
                      help='Optional actor MLP width; defaults to teacher/BC hidden size when warm-starting')
    p_rl.add_argument('--model-num-layers', type=int, default=None,
                      help='Optional actor MLP depth; defaults to teacher/BC depth when warm-starting')
    p_rl.add_argument('--separate-actor-critic', action='store_true',
                      help='Use separate actor and critic encoders so value updates do not perturb the actor backbone directly')
    p_rl.add_argument('--disable-input-normalization', action='store_true',
                      help='Disable Sample Factory input normalization; BC warm-start defaults to this off path')
    p_rl.add_argument('--nonlinearity', type=str, default=None, choices=['elu', 'relu', 'tanh'],
                      help='Actor MLP nonlinearity; BC warm-start defaults to relu')
    p_rl.add_argument('--bc-init-path', type=str, default=None,
                      help='Optional BC checkpoint used to warm start APPO')
    p_rl.add_argument('--appo-init-checkpoint-path', type=str, default=None,
                      help='Optional APPO checkpoint used to warm start a fresh APPO experiment')
    p_rl.add_argument('--teacher-bc-path', type=str, default=None,
                      help='Optional frozen BC teacher checkpoint used for auxiliary teacher regularization')
    p_rl.add_argument('--teacher-prior-bc-path', type=str, default=None,
                      help='Optional BC teacher checkpoint used only for rollout-time teacher prior / fallback')
    p_rl.add_argument('--teacher-report-path', type=str, default=None,
                      help='Optional canonical teacher report JSON used to link this improver run back to its teacher artifact')
    p_rl.add_argument('--teacher-loss-coef', type=float, default=0.01,
                      help='Auxiliary teacher loss coefficient; teacher-reg baseline uses 0.01')
    p_rl.add_argument('--teacher-loss-type', type=str, default='ce', choices=['ce', 'kl'],
                      help='Teacher regularization loss type')
    p_rl.add_argument('--teacher-action-boosts', type=str, default='',
                      help='Optional comma-separated action=multiplier boosts for teacher CE/KL, e.g. west=2.0,south=1.5')
    p_rl.add_argument('--teacher-loss-final-coef', type=float, default=0.0,
                      help='Optional final teacher coefficient after scheduled decay; 0 keeps static teacher loss')
    p_rl.add_argument('--teacher-loss-warmup-env-steps', type=int, default=0,
                      help='Optional env-step warmup before teacher-loss decay starts')
    p_rl.add_argument('--teacher-loss-decay-env-steps', type=int, default=0,
                      help='Optional env-step linear decay duration from teacher-loss-coef to teacher-loss-final-coef')
    p_rl.add_argument('--teacher-replay-trace-input', type=str, default=None,
                      help='Optional trusted trace JSONL used for supervised teacher replay during RL')
    p_rl.add_argument('--teacher-replay-coef', type=float, default=0.0,
                      help='Coefficient for supervised teacher replay loss')
    p_rl.add_argument('--teacher-replay-final-coef', type=float, default=0.0,
                      help='Optional final replay coefficient after scheduled decay; 0 keeps static replay loss')
    p_rl.add_argument('--teacher-replay-warmup-env-steps', type=int, default=0,
                      help='Optional env-step warmup before teacher replay decay starts')
    p_rl.add_argument('--teacher-replay-decay-env-steps', type=int, default=0,
                      help='Optional env-step linear decay duration from teacher-replay-coef to teacher-replay-final-coef')
    p_rl.add_argument('--teacher-replay-batch-size', type=int, default=128,
                      help='Replay minibatch size drawn from the trusted trace set')
    p_rl.add_argument('--teacher-replay-priority-power', type=float, default=1.0,
                      help='Reserved replay priority exponent; 1.0 keeps uniform weighting until prioritized replay is enabled')
    p_rl.add_argument('--teacher-replay-source-mode', type=str, default='uniform',
                      help='Reserved replay source mode for prioritized replay experiments')
    p_rl.add_argument('--teacher-replay-action-boosts', type=str, default='',
                      help='Optional comma-separated teacher replay action=multiplier boosts, e.g. east=2.0,south=2.0')
    p_rl.add_argument('--teacher-replay-current-disagreement-boost', type=float, default=1.0,
                      help='Optional multiplier applied to replay CE rows where the current student still disagrees with the replay teacher action')
    p_rl.add_argument('--teacher-replay-confusion-pair-boosts', type=str, default='',
                      help='Optional comma-separated student->teacher=multiplier replay boosts, e.g. east->south=2.0,south->east=2.0')
    p_rl.add_argument('--teacher-replay-confusion-pair-start-env-steps', type=int, default=0,
                      help='Optional env-step threshold before exact replay confusion-pair boosts become active')
    p_rl.add_argument('--teacher-policy-logit-residual-scale', type=float, default=1.0,
                      help='Logit-space residual scale in teacher + s*(student-teacher); 1.0 keeps the raw student policy')
    p_rl.add_argument('--teacher-policy-residual-logit-cap', type=float, default=0.0,
                      help='If >0, clip each action logit residual around the teacher prior during rollout-time action selection')
    p_rl.add_argument('--teacher-policy-blend-coef', type=float, default=0.0,
                      help='Blend coefficient for a teacher policy prior applied at action selection time')
    p_rl.add_argument('--teacher-policy-fallback-confidence', type=float, default=0.0,
                      help='If >0, replace low-confidence student decisions with the teacher prior when max_prob falls below this threshold')
    p_rl.add_argument('--teacher-policy-disagreement-margin', type=float, default=0.0,
                      help='If >0, require this minimum student probability advantage over the teacher-preferred action before a disagreement override is allowed')
    p_rl.add_argument('--param-anchor-coef', type=float, default=0.0,
                      help='L2 anchor coefficient on warm-started encoder/policy parameters')
    p_rl.add_argument('--actor-loss-scale', type=float, default=1.0,
                      help='Scale multiplier applied to the PPO actor loss')
    p_rl.add_argument('--actor-loss-final-scale', type=float, default=1.0,
                      help='Optional final actor-loss scale after scheduled decay')
    p_rl.add_argument('--actor-loss-warmup-env-steps', type=int, default=0,
                      help='Env-step warmup before actor-loss scale decay starts')
    p_rl.add_argument('--actor-loss-decay-env-steps', type=int, default=0,
                      help='Env-step linear decay duration from actor-loss-scale to actor-loss-final-scale')
    p_rl.add_argument('--trace-eval-input', type=str, default=None,
                      help='Optional trusted trace JSONL used for in-training checkpoint selection')
    p_rl.add_argument('--trace-eval-interval-env-steps', type=int, default=0,
                      help='If >0, periodically evaluate new checkpoints on the trusted trace set during training')
    p_rl.add_argument('--trace-eval-top-k', type=int, default=5,
                      help='Trace ranking top-k metadata depth during training')
    p_rl.add_argument('--save-every-sec', type=int, default=120,
                      help='Checkpoint save cadence in seconds')
    p_rl.add_argument('--save-best-every-sec', type=int, default=5,
                      help='Best-checkpoint save cadence in seconds')
    p_rl.add_argument('--no-rnn', action='store_true',
                      help='Disable the GRU core. Teacher-reg baseline uses the non-RNN path')
    p_rl.add_argument('--use-rnn', action='store_true',
                      help='Explicitly enable the GRU core; teacher-reg baseline defaults to non-RNN')
    p_rl.add_argument('--disable-action-mask', action='store_true',
                      help='Disable env-side invalid action clamping')
    p_rl.add_argument('--write-plan', type=str, default=None,
                      help='Optional JSON output path for the resolved training plan')
    p_rl.add_argument('--improver-report-output', type=str, default=None,
                      help='Optional JSON output path for the post-run improver report artifact')
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
    p_rl_eval.add_argument('--trace-input', type=str, default=None,
                           help='Optional trace JSONL for deterministic policy evaluation')
    p_rl_eval.add_argument('--checkpoint-path', type=str, default=None,
                           help='Optional explicit APPO checkpoint path')

    p_rl_cmp = subparsers.add_parser('rl-compare-policies', help='Compare task_greedy, BC, and APPO on a fixed seed suite')
    p_rl_cmp.add_argument('--task', type=str, default='explore',
                          choices=['explore', 'survive', 'combat', 'descend', 'resource'])
    p_rl_cmp.add_argument('--seeds', type=str, default='42,43,44')
    p_rl_cmp.add_argument('--max-steps', type=int, default=50)
    p_rl_cmp.add_argument('--bc-model', type=str, default=None)
    p_rl_cmp.add_argument('--appo-experiment', type=str, default=None)
    p_rl_cmp.add_argument('--appo-train-dir', type=str, default='train_dir/rl')
    p_rl_cmp.add_argument('--output', type=str, default=None)

    p_rl_det = subparsers.add_parser('rl-check-determinism', help='Repeat the same evaluation and diff action traces')
    p_rl_det.add_argument('--policy', type=str, required=True,
                          choices=['task_greedy', 'wall_avoidance', 'bc', 'appo'])
    p_rl_det.add_argument('--task', type=str, default='explore',
                          choices=['explore', 'survive', 'combat', 'descend', 'resource'])
    p_rl_det.add_argument('--seeds', type=str, default='42,43,44')
    p_rl_det.add_argument('--max-steps', type=int, default=20)
    p_rl_det.add_argument('--repeats', type=int, default=3)
    p_rl_det.add_argument('--bc-model', type=str, default=None)
    p_rl_det.add_argument('--appo-experiment', type=str, default=None)
    p_rl_det.add_argument('--appo-train-dir', type=str, default='train_dir/rl')
    p_rl_det.add_argument('--observation-version', type=str, default='v1')

    p_rl_actions = subparsers.add_parser('rl-compare-actions', help='Compare teacher, BC, and APPO on teacher-induced states')
    p_rl_actions.add_argument('--task', type=str, default='explore',
                              choices=['explore', 'survive', 'combat', 'descend', 'resource'])
    p_rl_actions.add_argument('--input', type=str, default=None,
                              help='Optional trace JSONL for deterministic teacher-state comparison')
    p_rl_actions.add_argument('--seeds', type=str, default='42,43,44')
    p_rl_actions.add_argument('--max-steps', type=int, default=20)
    p_rl_actions.add_argument('--bc-model', type=str, default=None)
    p_rl_actions.add_argument('--appo-experiment', type=str, default=None)
    p_rl_actions.add_argument('--appo-train-dir', type=str, default='train_dir/rl')
    p_rl_actions.add_argument('--appo-checkpoint-path', type=str, default=None,
                              help='Optional explicit APPO checkpoint path')
    p_rl_actions.add_argument('--observation-version', type=str, default='v1')
    p_rl_actions.add_argument('--summary-only', action='store_true',
                              help='Only print summaries for trace-based comparisons')

    p_rl_short = subparsers.add_parser('rl-short-benchmark', help='Train BC on a trace shard and run the fast debug checks')
    p_rl_short.add_argument('--input', type=str, required=True)
    p_rl_short.add_argument('--output', type=str, required=True)
    p_rl_short.add_argument('--task', type=str, default='explore',
                            choices=['explore', 'survive', 'combat', 'descend', 'resource'])
    p_rl_short.add_argument('--epochs', type=int, default=10)
    p_rl_short.add_argument('--lr', type=float, default=1e-3)
    p_rl_short.add_argument('--hidden-size', type=int, default=256)
    p_rl_short.add_argument('--observation-version', type=str, default='v1')
    p_rl_short.add_argument('--seeds', type=str, default='42,43,44')
    p_rl_short.add_argument('--max-steps', type=int, default=20)
    p_rl_short.add_argument('--repeats', type=int, default=2)
    p_rl_short.add_argument('--compare-baseline', action='store_true')
    p_rl_short.add_argument('--report', type=str, default=None)

    p_rl_trace = subparsers.add_parser('rl-trace-report', help='Compact deterministic regression report for a trace dataset')
    p_rl_trace.add_argument('--input', type=str, required=True)
    p_rl_trace.add_argument('--bc-model', type=str, default=None)
    p_rl_trace.add_argument('--appo-experiment', type=str, default=None)
    p_rl_trace.add_argument('--appo-train-dir', type=str, default='train_dir/rl')
    p_rl_trace.add_argument('--appo-checkpoint-path', type=str, default=None)
    p_rl_trace.add_argument('--detailed', action='store_true',
                            help='Include per-action disagreement analysis')
    p_rl_trace.add_argument('--top-k', type=int, default=10,
                            help='Number of mismatch/prediction entries to include in detailed output')
    p_rl_trace.add_argument('--output', type=str, default=None)

    p_rl_trace_dis = subparsers.add_parser('rl-trace-disagreements', help='Detailed deterministic disagreement report for a trace dataset')
    p_rl_trace_dis.add_argument('--input', type=str, required=True)
    p_rl_trace_dis.add_argument('--bc-model', type=str, default=None)
    p_rl_trace_dis.add_argument('--appo-experiment', type=str, default=None)
    p_rl_trace_dis.add_argument('--appo-train-dir', type=str, default='train_dir/rl')
    p_rl_trace_dis.add_argument('--appo-checkpoint-path', type=str, default=None)
    p_rl_trace_dis.add_argument('--top-k', type=int, default=10)
    p_rl_trace_dis.add_argument('--output', type=str, default=None)

    p_rl_trace_shard = subparsers.add_parser('rl-shard-traces', help='Write a smaller deterministic shard of a trace dataset')
    p_rl_trace_shard.add_argument('--input', type=str, required=True)
    p_rl_trace_shard.add_argument('--output', type=str, required=True)
    p_rl_trace_shard.add_argument('--max-episodes', type=int, default=None)
    p_rl_trace_shard.add_argument('--max-rows', type=int, default=None)
    p_rl_trace_shard.add_argument('--seeds', type=str, default=None,
                                  help='Optional comma-separated seed filter')
    p_rl_trace_shard.add_argument('--teacher-actions', type=str, default=None,
                                  help='Optional comma-separated teacher-action filter; keeps episodes containing these actions')
    p_rl_trace_shard.add_argument('--adjacent-signature', type=str, default=None,
                                  help='Optional comma-separated local geometry filter, e.g. north=monster_*,south=floor,east=monster_*,west=floor')

    p_rl_rank = subparsers.add_parser('rl-rank-checkpoints', help='Rank APPO checkpoints by deterministic trace match rate')
    p_rl_rank.add_argument('--experiment', type=str, required=True)
    p_rl_rank.add_argument('--train-dir', type=str, default='train_dir/rl')
    p_rl_rank.add_argument('--trace-input', type=str, required=True)
    p_rl_rank.add_argument('--top-k', type=int, default=5)
    p_rl_rank.add_argument('--output', type=str, default=None)
    p_rl_rank.add_argument('--best-alias-output', type=str, default=None,
                           help='Optional JSON path recording the best-by-trace checkpoint path')
    p_rl_rank.add_argument('--materialize-best-trace', action='store_true',
                           help='Copy the current best-by-trace checkpoint to checkpoint_p0/best_trace_match.pth')

    p_rl_teacher = subparsers.add_parser('rl-teacher-reg-report', help='Compare BC teacher, latest APPO, and best-trace APPO on a trusted trace set')
    p_rl_teacher.add_argument('--experiment', type=str, required=True)
    p_rl_teacher.add_argument('--train-dir', type=str, default='train_dir/rl')
    p_rl_teacher.add_argument('--trace-input', type=str, required=True)
    p_rl_teacher.add_argument('--bc-model', type=str, required=True)
    p_rl_teacher.add_argument('--output', type=str, default=None)

    p_rl_dagger = subparsers.add_parser('rl-run-dagger', help='Run one DAgger-lite iteration: relabel student rollouts with the teacher and retrain BC')
    p_rl_dagger.add_argument('--input', type=str, required=True,
                             help='Base trace JSONL used as the starting BC dataset')
    p_rl_dagger.add_argument('--dagger-output', type=str, required=True,
                             help='Output JSONL for newly relabeled student rollouts')
    p_rl_dagger.add_argument('--bc-output', type=str, required=True,
                             help='Output path for the retrained BC checkpoint')
    p_rl_dagger.add_argument('--merged-output', type=str, default=None,
                             help='Optional JSONL path to save the merged BC dataset')
    p_rl_dagger.add_argument('--student-policy', type=str, required=True,
                             choices=['bc', 'appo', 'wall_avoidance'])
    p_rl_dagger.add_argument('--task', type=str, default='explore',
                             choices=['explore', 'survive', 'combat', 'descend', 'resource'])
    p_rl_dagger.add_argument('--num-episodes', type=int, default=8)
    p_rl_dagger.add_argument('--max-steps', type=int, default=20)
    p_rl_dagger.add_argument('--seed-start', type=int, default=42)
    p_rl_dagger.add_argument('--appo-experiment', type=str, default=None)
    p_rl_dagger.add_argument('--appo-train-dir', type=str, default='train_dir/rl')
    p_rl_dagger.add_argument('--appo-checkpoint-path', type=str, default=None)
    p_rl_dagger.add_argument('--bc-model-path', type=str, default=None)
    p_rl_dagger.add_argument('--teacher-bc-model-path', type=str, default=None,
                             help='Optional BC teacher used to relabel student-induced states instead of task_greedy')
    p_rl_dagger.add_argument('--observation-version', type=str, default='v1')
    p_rl_dagger.add_argument('--merge-ratio', type=float, default=0.5,
                             help='Fraction of the base trace dataset to keep when merging with new relabeled traces')
    p_rl_dagger.add_argument('--merge-policy', type=str, default='uniform_merge',
                             choices=['base_only', 'uniform_merge', 'weighted_recent'])
    p_rl_dagger.add_argument('--dagger-row-policy', type=str, default='all',
                             choices=['all', 'disagreement', 'loop_risk', 'failure_slice', 'weak_action', 'hard_only'])
    p_rl_dagger.add_argument('--dagger-keep-match-ratio', type=float, default=0.0,
                             help='Keep a small fraction of on-support DAgger rows as anchors after filtering')
    p_rl_dagger.add_argument('--dagger-confusion-pairs', type=str, default='',
                             help='Optional comma-separated behavior->teacher filters, e.g. east->south,south->east')
    p_rl_dagger.add_argument('--heldout-input', type=str, default=None,
                             help='Optional held-out trace JSONL for post-iteration evaluation')
    p_rl_dagger.add_argument('--epochs', type=int, default=20)
    p_rl_dagger.add_argument('--lr', type=float, default=1e-3)
    p_rl_dagger.add_argument('--hidden-size', type=int, default=256)
    p_rl_dagger.add_argument('--distill-teacher-bc-path', type=str, default=None,
                             help='Optional BC teacher for retraining distillation; defaults to --teacher-bc-model-path when set')
    p_rl_dagger.add_argument('--distill-loss-coef', type=float, default=0.0)
    p_rl_dagger.add_argument('--distill-temperature', type=float, default=1.0)

    p_rl_dagger_sched = subparsers.add_parser('rl-dagger-iterate', help='Run an iterative DAgger schedule with BC retraining and trace-gated reports')
    p_rl_dagger_sched.add_argument('--input', type=str, required=True)
    p_rl_dagger_sched.add_argument('--output-dir', type=str, required=True)
    p_rl_dagger_sched.add_argument('--student-policy', type=str, required=True,
                                   choices=['bc', 'appo', 'wall_avoidance'])
    p_rl_dagger_sched.add_argument('--task', type=str, default='explore',
                                   choices=['explore', 'survive', 'combat', 'descend', 'resource'])
    p_rl_dagger_sched.add_argument('--iterations', type=int, default=3)
    p_rl_dagger_sched.add_argument('--num-episodes', type=int, default=8)
    p_rl_dagger_sched.add_argument('--max-steps', type=int, default=20)
    p_rl_dagger_sched.add_argument('--seed-start', type=int, default=42)
    p_rl_dagger_sched.add_argument('--appo-experiment', type=str, default=None)
    p_rl_dagger_sched.add_argument('--appo-train-dir', type=str, default='train_dir/rl')
    p_rl_dagger_sched.add_argument('--appo-checkpoint-path', type=str, default=None)
    p_rl_dagger_sched.add_argument('--bc-model-path', type=str, default=None)
    p_rl_dagger_sched.add_argument('--teacher-bc-model-path', type=str, default=None)
    p_rl_dagger_sched.add_argument('--observation-version', type=str, default='v1')
    p_rl_dagger_sched.add_argument('--merge-ratio', type=float, default=0.5)
    p_rl_dagger_sched.add_argument('--merge-policy', type=str, default='uniform_merge',
                                   choices=['base_only', 'uniform_merge', 'weighted_recent'])
    p_rl_dagger_sched.add_argument('--dagger-row-policy', type=str, default='all',
                                   choices=['all', 'disagreement', 'loop_risk', 'failure_slice', 'weak_action', 'hard_only'])
    p_rl_dagger_sched.add_argument('--dagger-keep-match-ratio', type=float, default=0.0)
    p_rl_dagger_sched.add_argument('--dagger-confusion-pairs', type=str, default='')
    p_rl_dagger_sched.add_argument('--heldout-input', type=str, default=None)
    p_rl_dagger_sched.add_argument('--epochs', type=int, default=20)
    p_rl_dagger_sched.add_argument('--lr', type=float, default=1e-3)
    p_rl_dagger_sched.add_argument('--hidden-size', type=int, default=256)
    p_rl_dagger_sched.add_argument('--distill-teacher-bc-path', type=str, default=None)
    p_rl_dagger_sched.add_argument('--distill-loss-coef', type=float, default=0.0)
    p_rl_dagger_sched.add_argument('--distill-temperature', type=float, default=1.0)
    p_rl_dagger_sched.add_argument('--random-seed', type=int, default=123)
    p_rl_dagger_sched.add_argument('--stop-on-heldout-regression', action='store_true')
    p_rl_dagger_sched.add_argument('--min-improvement', type=float, default=0.0)

    p_rl_breg = subparsers.add_parser('rl-train-behavior-reg', help='Train an experimental behavior-regularized improver on trace data')
    p_rl_breg.add_argument('--input', type=str, required=True)
    p_rl_breg.add_argument('--output', type=str, required=True)
    p_rl_breg.add_argument('--heldout-input', type=str, default=None)
    p_rl_breg.add_argument('--epochs', type=int, default=20)
    p_rl_breg.add_argument('--lr', type=float, default=1e-3)
    p_rl_breg.add_argument('--hidden-size', type=int, default=256)
    p_rl_breg.add_argument('--observation-version', type=str, default='v1')
    p_rl_breg.add_argument('--behavior-coef', type=float, default=0.1)
    p_rl_breg.add_argument('--temperature', type=float, default=1.0)
    p_rl_breg.add_argument('--class-balance-power', type=float, default=0.0)
    p_rl_breg.add_argument('--teacher-action-boost', type=str, default='')
    p_rl_breg.add_argument('--teacher-action-boost-scale', type=float, default=1.0)
    p_rl_breg.add_argument('--teacher-report-output', type=str, default=None)
    p_rl_breg.add_argument('--weak-action-input', type=str, default=None)

    p_rl_shard_bench = subparsers.add_parser('rl-shard-benchmark', help='Compare BC and behavior-regularized training on a focused trace shard')
    p_rl_shard_bench.add_argument('--input', type=str, required=True)
    p_rl_shard_bench.add_argument('--heldout-input', type=str, default=None)
    p_rl_shard_bench.add_argument('--epochs', type=int, default=20)
    p_rl_shard_bench.add_argument('--lr', type=float, default=1e-3)
    p_rl_shard_bench.add_argument('--hidden-size', type=int, default=256)
    p_rl_shard_bench.add_argument('--observation-version', type=str, default='v1')
    p_rl_shard_bench.add_argument('--behavior-coef', type=float, default=0.1)
    p_rl_shard_bench.add_argument('--temperature', type=float, default=1.0)
    p_rl_shard_bench.add_argument('--class-balance-power', type=float, default=0.0)
    p_rl_shard_bench.add_argument('--teacher-action-boost', type=str, default='')
    p_rl_shard_bench.add_argument('--teacher-action-boost-scale', type=float, default=1.0)
    p_rl_shard_bench.add_argument('--detailed', action='store_true')

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

    p_trace = subparsers.add_parser('rl-generate-traces', help='Generate explicit multi-turn traces')
    p_trace.add_argument('--output', type=str, required=True)
    p_trace.add_argument('--num-episodes', type=int, default=10)
    p_trace.add_argument('--max-steps', type=int, default=30)
    p_trace.add_argument('--seed-start', type=int, default=42)
    p_trace.add_argument('--task', type=str, default='explore',
                         choices=['explore', 'survive', 'combat', 'descend', 'resource'])
    p_trace.add_argument('--policy', type=str, default='task_greedy',
                         choices=['task_greedy', 'wall_avoidance', 'forward_model', 'bc', 'appo'])
    p_trace.add_argument('--appo-experiment', type=str, default=None)
    p_trace.add_argument('--appo-train-dir', type=str, default='train_dir/rl')
    p_trace.add_argument('--bc-model-path', type=str, default=None)
    p_trace.add_argument('--server-url', type=str, default='http://127.0.0.1:8765',
                         help='Forward-model server URL for policy=forward_model')
    p_trace.add_argument('--model-name', type=str, default='llama-server',
                         help='Served forward model name for policy=forward_model')
    p_trace.add_argument('--observation-version', type=str, default='v1',
                         help='Feature encoder version stored in trace rows')

    p_mine = subparsers.add_parser('rl-mine-reset-slice', help='Mine step-0 task_greedy traces from reset states matching a local geometry')
    p_mine.add_argument('--output', type=str, required=True)
    p_mine.add_argument('--seed-start', type=int, default=42)
    p_mine.add_argument('--num-seeds', type=int, default=1000)
    p_mine.add_argument('--task', type=str, default='explore',
                        choices=['explore', 'survive', 'combat', 'descend', 'resource'])
    p_mine.add_argument('--observation-version', type=str, default='v1')
    p_mine.add_argument('--adjacent-signature', type=str, required=True,
                        help='Comma-separated local geometry filter, e.g. north=monster_*,south=floor,east=monster_*,west=floor')
    p_mine.add_argument('--recreate-every', type=int, default=250,
                        help='Recreate the raw NLE env every N resets to reduce reset-instability')
    p_mine.add_argument('--max-rows', type=int, default=None,
                        help='Optional early stop after collecting this many matching rows')

    p_trace_verify = subparsers.add_parser('rl-verify-traces', help='Verify a trace file is multi-turn')
    p_trace_verify.add_argument('--input', type=str, required=True)

    p_bc = subparsers.add_parser('rl-train-bc', help='Train a behavior cloning policy from traces')
    p_bc.add_argument('--input', type=str, required=True)
    p_bc.add_argument('--output', type=str, required=True)
    p_bc.add_argument('--epochs', type=int, default=20)
    p_bc.add_argument('--lr', type=float, default=1e-3)
    p_bc.add_argument('--hidden-size', type=int, default=256)
    p_bc.add_argument('--num-layers', type=int, default=2)
    p_bc.add_argument('--observation-version', type=str, default='v1')
    p_bc.add_argument('--world-model-path', type=str, default=None)
    p_bc.add_argument('--world-model-feature-mode', type=str, default=None,
                      choices=['replace', 'concat', 'concat_aux'])
    p_bc.add_argument('--distill-teacher-bc-path', type=str, default=None)
    p_bc.add_argument('--distill-teacher-bc-paths', type=str, nargs='*', default=None)
    p_bc.add_argument('--distill-loss-coef', type=float, default=0.0)
    p_bc.add_argument('--distill-temperature', type=float, default=1.0)
    p_bc.add_argument('--supervised-loss-coef', type=float, default=1.0)
    p_bc.add_argument('--action-weight-boosts', type=str, default=None)
    p_bc.add_argument('--text-encoder-backend', type=str, default='none', choices=['none', 'hash', 'transformer'])
    p_bc.add_argument('--text-vocab-size', type=int, default=4096)
    p_bc.add_argument('--text-embedding-dim', type=int, default=128)
    p_bc.add_argument('--text-model-name', type=str, default=None)
    p_bc.add_argument('--text-max-length', type=int, default=128)
    p_bc.add_argument('--text-trainable', action='store_true')
    p_bc.add_argument('--device', type=str, default='auto')
    p_bc.add_argument('--select-by-heldout', action='store_true')
    p_bc.add_argument('--heldout-input', type=str, default=None)
    p_bc.add_argument('--teacher-report-output', type=str, default=None)
    p_bc.add_argument('--weak-action-input', type=str, default=None)

    p_bc_eval = subparsers.add_parser('rl-evaluate-bc', help='Evaluate a behavior cloning policy')
    p_bc_eval.add_argument('--model', type=str, required=True)
    p_bc_eval.add_argument('--task', type=str, default='explore',
                           choices=['explore', 'survive', 'combat', 'descend', 'resource'])
    p_bc_eval.add_argument('--seeds', type=str, default='42,43,44')
    p_bc_eval.add_argument('--max-steps', type=int, default=50)
    p_bc_eval.add_argument('--compare-baseline', action='store_true')
    p_bc_eval.add_argument('--trace-input', type=str, default=None,
                           help='Optional trace JSONL for deterministic policy evaluation')

    p_wm = subparsers.add_parser('rl-train-world-model', help='Train a short-horizon latent world model on traces')
    p_wm.add_argument('--input', type=str, required=True)
    p_wm.add_argument('--output', type=str, required=True)
    p_wm.add_argument('--horizon', type=int, default=8)
    p_wm.add_argument('--epochs', type=int, default=20)
    p_wm.add_argument('--lr', type=float, default=1e-3)
    p_wm.add_argument('--hidden-size', type=int, default=256)
    p_wm.add_argument('--latent-dim', type=int, default=128)
    p_wm.add_argument('--observation-version', type=str, default=None)
    p_wm.add_argument('--reward-loss-coef', type=float, default=1.0)
    p_wm.add_argument('--done-loss-coef', type=float, default=0.5)
    p_wm.add_argument('--reconstruction-loss-coef', type=float, default=1.0)
    p_wm.add_argument('--action-loss-coef', type=float, default=0.25)
    p_wm.add_argument('--action-class-balance', action='store_true')
    p_wm.add_argument('--action-class-balance-power', type=float, default=0.5)
    p_wm.add_argument('--text-encoder-backend', type=str, default='none', choices=['none', 'hash', 'transformer'])
    p_wm.add_argument('--text-model-name', type=str, default=None)
    p_wm.add_argument('--text-max-length', type=int, default=128)
    p_wm.add_argument('--text-trainable', action='store_true')
    p_wm.add_argument('--text-embedding-dim', type=int, default=128)
    p_wm.add_argument('--report-output', type=str, default=None)

    p_wm_eval = subparsers.add_parser('rl-evaluate-world-model', help='Evaluate a short-horizon latent world model')
    p_wm_eval.add_argument('--model', type=str, required=True)
    p_wm_eval.add_argument('--input', type=str, required=True)
    p_wm_eval.add_argument('--horizon', type=int, default=8)
    p_wm_eval.add_argument('--observation-version', type=str, default=None)
    p_wm_eval.add_argument('--downstream-train-input', type=str, default=None)
    p_wm_eval.add_argument('--downstream-heldout-input', type=str, default=None)
    p_wm_eval.add_argument('--downstream-mode', type=str, default='concat_aux', choices=['replace', 'concat', 'concat_aux'])
    p_wm_eval.add_argument('--downstream-epochs', type=int, default=20)
    p_wm_eval.add_argument('--downstream-lr', type=float, default=1e-3)
    p_wm_eval.add_argument('--downstream-hidden-size', type=int, default=256)
    p_wm_eval.add_argument('--report-output', type=str, default=None)

    p_wm_xform = subparsers.add_parser('rl-transform-traces-world-model', help='Rewrite trace features using a trained world model encoder')
    p_wm_xform.add_argument('--model', type=str, required=True)
    p_wm_xform.add_argument('--input', type=str, required=True)
    p_wm_xform.add_argument('--output', type=str, required=True)
    p_wm_xform.add_argument('--version-suffix', type=str, default='wm')
    p_wm_xform.add_argument('--mode', type=str, default='replace', choices=['replace', 'concat', 'concat_aux'])

    p_relabel = subparsers.add_parser('rl-relabel-traces-bc', help='Relabel trace actions with a BC teacher')
    p_relabel.add_argument('--input', type=str, required=True)
    p_relabel.add_argument('--output', type=str, required=True)
    p_relabel.add_argument('--bc-model-path', type=str, required=True)

    p_proxy_build = subparsers.add_parser('rl-build-proxy-dataset', help='Build teacher-derived proxy labels from traces')
    p_proxy_build.add_argument('--input', type=str, required=True)
    p_proxy_build.add_argument('--output', type=str, required=True)
    p_proxy_build.add_argument('--horizon', type=int, default=8)
    p_proxy_build.add_argument('--task', type=str, default=None)
    p_proxy_build.add_argument('--max-rows', type=int, default=None)

    p_proxy_train = subparsers.add_parser('rl-train-proxy', help='Train a teacher-derived proxy reward model')
    p_proxy_train.add_argument('--input', type=str, required=True)
    p_proxy_train.add_argument('--heldout-input', type=str, default=None)
    p_proxy_train.add_argument('--output', type=str, required=True)
    p_proxy_train.add_argument('--epochs', type=int, default=40)
    p_proxy_train.add_argument('--lr', type=float, default=1e-3)
    p_proxy_train.add_argument('--hidden-size', type=int, default=256)
    p_proxy_train.add_argument('--action-embed-dim', type=int, default=32)
    p_proxy_train.add_argument('--report-output', type=str, default=None)

    p_proxy_eval = subparsers.add_parser('rl-evaluate-proxy', help='Evaluate a teacher-derived proxy reward model')
    p_proxy_eval.add_argument('--model', type=str, required=True)
    p_proxy_eval.add_argument('--input', type=str, required=True)
    p_proxy_eval.add_argument('--report-output', type=str, default=None)
    p_proxy_eval.add_argument('--top-k', type=int, default=5)

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
        elif args.command == 'rl-compare-policies':
            return cmd_rl_compare_policies(args)
        elif args.command == 'rl-check-determinism':
            return cmd_rl_check_determinism(args)
        elif args.command == 'rl-compare-actions':
            return cmd_rl_compare_actions(args)
        elif args.command == 'rl-short-benchmark':
            return cmd_rl_short_benchmark(args)
        elif args.command == 'rl-trace-report':
            return cmd_rl_trace_report(args)
        elif args.command == 'rl-trace-disagreements':
            return cmd_rl_trace_disagreements(args)
        elif args.command == 'rl-shard-traces':
            return cmd_rl_shard_traces(args)
        elif args.command == 'rl-rank-checkpoints':
            return cmd_rl_rank_checkpoints(args)
        elif args.command == 'rl-teacher-reg-report':
            return cmd_rl_teacher_reg_report(args)
        elif args.command == 'rl-run-dagger':
            return cmd_rl_run_dagger(args)
        elif args.command == 'rl-dagger-iterate':
            return cmd_rl_dagger_iterate(args)
        elif args.command == 'rl-train-behavior-reg':
            return cmd_rl_train_behavior_reg(args)
        elif args.command == 'rl-shard-benchmark':
            return cmd_rl_shard_benchmark(args)
        elif args.command == 'rl-train-reward':
            return cmd_rl_train_reward(args)
        elif args.command == 'rl-train-scheduler':
            return cmd_rl_train_scheduler(args)
        elif args.command == 'rl-generate-traces':
            return cmd_rl_generate_traces(args)
        elif args.command == 'rl-mine-reset-slice':
            return cmd_rl_mine_reset_slice(args)
        elif args.command == 'rl-verify-traces':
            return cmd_rl_verify_traces(args)
        elif args.command == 'rl-train-bc':
            return cmd_rl_train_bc(args)
        elif args.command == 'rl-evaluate-bc':
            return cmd_rl_evaluate_bc(args)
        elif args.command == 'rl-train-world-model':
            return cmd_rl_train_world_model(args)
        elif args.command == 'rl-evaluate-world-model':
            return cmd_rl_evaluate_world_model(args)
        elif args.command == 'rl-transform-traces-world-model':
            return cmd_rl_transform_traces_world_model(args)
        elif args.command == 'rl-relabel-traces-bc':
            return cmd_rl_relabel_traces_bc(args)
        elif args.command == 'rl-build-proxy-dataset':
            return cmd_rl_build_proxy_dataset(args)
        elif args.command == 'rl-train-proxy':
            return cmd_rl_train_proxy(args)
        elif args.command == 'rl-evaluate-proxy':
            return cmd_rl_evaluate_proxy(args)
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
