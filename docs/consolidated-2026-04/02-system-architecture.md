# System Architecture

## High-Level Stack

By 2026-04-06 the repo had five interacting systems:

1. forward-model SFT
2. counterfactual branching and task-harness control
3. explicit trace generation and offline teacher training
4. world-model feature learning and representation augmentation
5. Sample Factory APPO online improvement

This decomposition is explained in:

- [CURRENT-RL-SYSTEM.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/CURRENT-RL-SYSTEM.md)
- [RL-APPO-HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/RL-APPO-HANDOFF.md)
- [README.md](/home/luc/rl-nethack-worktree-20260416/README.md)

## Path A: Forward-Model SFT

This path lives primarily in:

- [train.py](/home/luc/rl-nethack-worktree-20260416/train.py)
- [src/state_encoder.py](/home/luc/rl-nethack-worktree-20260416/src/state_encoder.py)
- [src/data_generator.py](/home/luc/rl-nethack-worktree-20260416/src/data_generator.py)
- [scripts/generate_training_data.py](/home/luc/rl-nethack-worktree-20260416/scripts/generate_training_data.py)

What it trains:

- a language model fine-tuned with Unsloth LoRA and `trl.SFTTrainer`
- tasked with predicting next-step action-conditioned deltas
- trained on ShareGPT-style JSONL conversations

What the inputs look like:

- compact current-state text plus a proposed action in the simple path
- or memory-augmented prompt plus viewport plus action in the richer LLM-policy generation path

What the targets look like:

- state delta descriptions, not full next-state reconstruction

Why it exists:

- it was the original path for learning NetHack dynamics cheaply and locally

Primary docs:

- [IMPLEMENTATION-PLAN.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/IMPLEMENTATION-PLAN.md)
- [PLAN.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/PLAN.md)
- [README.md](/home/luc/rl-nethack-worktree-20260416/README.md)
- [HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/HANDOFF.md)

## Path B: Counterfactual Data And Task Harness

This path grew from one-step what-if control into a meaningful teacher scaffold.

Core components:

- [scripts/generate_counterfactual_data.py](/home/luc/rl-nethack-worktree-20260416/scripts/generate_counterfactual_data.py)
- [src/task_harness.py](/home/luc/rl-nethack-worktree-20260416/src/task_harness.py)
- [src/task_rewards.py](/home/luc/rl-nethack-worktree-20260416/src/task_rewards.py)

What it does:

- defines shaped task rewards for `explore`, `survive`, `combat`, `descend`, and `resource`
- supports one-step branching over candidate actions
- supports a rule-based `task_greedy` teacher/controller

Why it matters:

- it was the first trustworthy local control/evaluation loop before mature learned RL
- it also provided teacher signals and preference-style data for later branches

Primary docs:

- [RL-HARNESS-TASKS.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/misc/RL-HARNESS-TASKS.md)
- [HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/HANDOFF.md)
- [RL-APPO-HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/RL-APPO-HANDOFF.md)

## Path C: Trace Generation And Offline Teachers

This became the main teacher-building substrate.

Core files:

- [rl/traces.py](/home/luc/rl-nethack-worktree-20260416/rl/traces.py)
- [rl/train_bc.py](/home/luc/rl-nethack-worktree-20260416/rl/train_bc.py)
- [rl/train_behavior_reg.py](/home/luc/rl-nethack-worktree-20260416/rl/train_behavior_reg.py)
- [rl/relabel_traces.py](/home/luc/rl-nethack-worktree-20260416/rl/relabel_traces.py)
- [rl/teacher_report.py](/home/luc/rl-nethack-worktree-20260416/rl/teacher_report.py)

Key concepts:

- explicit multi-turn traces
- BC teachers trained from trace rows
- behavior-regularized teachers
- teacher relabeling and teacher distillation
- prompt-conditioned teacher variants

Why it matters:

- this is where the repo’s strongest offline policies came from
- later online work is meaningful only because this teacher pipeline became strong enough

Primary docs:

- [BEHAVIOR-REG-IMPROVEMENT.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/BEHAVIOR-REG-IMPROVEMENT.md)
- [REPORT-BEHAVIOR-REG-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-BEHAVIOR-REG-2026-04-06.md)
- [PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md)

## Path D: World Model And Representation Learning

This path is a short-horizon predictive model over trace rows. It is not a Dreamer-style end-to-end planner.

Core files:

- [rl/world_model.py](/home/luc/rl-nethack-worktree-20260416/rl/world_model.py)
- [rl/world_model_dataset.py](/home/luc/rl-nethack-worktree-20260416/rl/world_model_dataset.py)
- [rl/train_world_model.py](/home/luc/rl-nethack-worktree-20260416/rl/train_world_model.py)
- [rl/world_model_eval.py](/home/luc/rl-nethack-worktree-20260416/rl/world_model_eval.py)
- [rl/world_model_features.py](/home/luc/rl-nethack-worktree-20260416/rl/world_model_features.py)

What it learns from each training example:

- current feature vector
- current action
- current task or skill
- future feature vector after a short horizon
- cumulative reward over that horizon
- done-within-horizon flag

What changed later:

- text-conditioned world-model support
- optional frozen text encoders over trace prompt text
- transformed traces via `replace`, `concat`, or `concat_aux`

The crucial repo-local conclusion is:

- the world model became useful as a teacher-builder and representation path
- it did not by itself fix online RL drift

Primary docs:

- [WORLD-MODEL-PLAN-FOR-RL-NETHACK.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/WORLD-MODEL-PLAN-FOR-RL-NETHACK.md)
- [REPORT-WORLD-MODEL-VALIDATION-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-WORLD-MODEL-VALIDATION-2026-04-06.md)
- [REPORT-WORLD-MODEL-SCALING-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-WORLD-MODEL-SCALING-2026-04-06.md)
- [REPORT-WORLD-MODEL-ENSEMBLE-DISTILL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-WORLD-MODEL-ENSEMBLE-DISTILL-2026-04-06.md)
- [REPORT-LLM-WORLD-MODEL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-LLM-WORLD-MODEL-2026-04-06.md)
- [LESSONS-WORLD-MODEL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/LESSONS-WORLD-MODEL-2026-04-06.md)

## Path E: APPO Online Improvement

This is the real RL path added under `rl/`.

Core files:

- [rl/sf_env.py](/home/luc/rl-nethack-worktree-20260416/rl/sf_env.py)
- [rl/env_adapter.py](/home/luc/rl-nethack-worktree-20260416/rl/env_adapter.py)
- [rl/feature_encoder.py](/home/luc/rl-nethack-worktree-20260416/rl/feature_encoder.py)
- [rl/rewards.py](/home/luc/rl-nethack-worktree-20260416/rl/rewards.py)
- [rl/trainer.py](/home/luc/rl-nethack-worktree-20260416/rl/trainer.py)
- [rl/train_appo.py](/home/luc/rl-nethack-worktree-20260416/rl/train_appo.py)
- [rl/teacher_reg.py](/home/luc/rl-nethack-worktree-20260416/rl/teacher_reg.py)

What this path introduced:

- a real Sample Factory APPO backend
- actor-critic training with recurrence
- task-conditioned environment wrappers
- teacher regularization during online improvement
- later teacher fallback and replay-heavy variants

Why it matters:

- it closed the “can this repo run real RL?” infrastructure gap
- it also exposed the central scientific problem: online drift away from a strong offline teacher

Primary docs:

- [APPO-OPTIONS-OVERHAUL.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/APPO-OPTIONS-OVERHAUL.md)
- [TEACHER-REG-APPO-PLAN.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/TEACHER-REG-APPO-PLAN.md)
- [RL-APPO-HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/RL-APPO-HANDOFF.md)
- [ALIGNMENT-IMPROVEMENT-PLAN.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/ALIGNMENT-IMPROVEMENT-PLAN.md)
- [ALIGNMENT-IMPLEMENTATION-REPORT-2026-04-05.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/misc/ALIGNMENT-IMPLEMENTATION-REPORT-2026-04-05.md)

## Cross-Cutting Infrastructure

Several supporting components became essential to the research loop:

- deterministic trace evaluation and checkpoint ranking
- best-trace checkpoint aliasing
- report generation and disagreement inspection
- test scaffolding for new RL/teacher/world-model paths
- atomic writes and lightweight experiment management

This infrastructure is explained across:

- [CURRENT-RL-SYSTEM.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/CURRENT-RL-SYSTEM.md)
- [PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md)
- [FAST-DEBUG-LOOP-REPORT-2026-04-05.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/FAST-DEBUG-LOOP-REPORT-2026-04-05.md)

