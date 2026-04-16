# Experimental Timeline

## Phase 1: Initial Forward-Model Pipeline

The earliest committed work established:

- structured state encoding,
- delta-style targets,
- random-play and LLM-play data generation,
- Unsloth LoRA training,
- evaluator and manifest generation,
- replay/report tooling.

The core claim in this phase was that predicting action-conditioned deltas is a better training target than reconstructing entire next states.

Primary sources:

- [PLAN.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/PLAN.md)
- [IMPLEMENTATION-PLAN.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/IMPLEMENTATION-PLAN.md)
- [README.md](/home/luc/rl-nethack-worktree-20260416/README.md)
- [HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/HANDOFF.md)

## Phase 2: Local Policy Generation And Throughput Scaling

Once the forward-model path existed, the next constraint was practical data generation throughput.

Key changes:

- local vLLM policy serving
- better fallback action sanitization
- replica-based serving
- in-process vLLM batching experiments

What held up:

- local generation throughput became good enough
- `Qwen2.5-0.5B-Instruct` was fast but low quality
- `Qwen2.5-3B-Instruct` with frontier-biased fallback was the first path with acceptable action balance

Primary sources:

- [HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/HANDOFF.md)
- [README.md](/home/luc/rl-nethack-worktree-20260416/README.md)

## Phase 3: Task Harness And Counterfactual Control

The repo then built:

- task-shaped reward definitions
- one-step counterfactual control
- closed-loop benchmarks for `explore`, `survive`, and later other skills

This phase mattered because it provided the first local, meaningful control signal and a stronger baseline than random or wall-avoidance behavior.

What held up:

- the task harness was useful
- task-greedy control was a meaningful teacher
- but it remained expensive and myopic

Primary sources:

- [RL-HARNESS-TASKS.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/misc/RL-HARNESS-TASKS.md)
- [HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/HANDOFF.md)
- [RL-APPO-HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/RL-APPO-HANDOFF.md)

## Phase 4: APPO Infrastructure Closure

This was the point where the repo genuinely became an RL repo in the narrow technical sense.

What was added:

- Sample Factory APPO backend
- custom NetHack environment wrappers
- action/feature encoding for RL
- scheduler and options scaffolding
- bootstrap logic for dependency conflicts

What held up:

- the infrastructure gap was closed
- APPO training, checkpointing, and evaluation worked
- but policy quality lagged well behind the heuristic teacher

Primary sources:

- [APPO-OPTIONS-OVERHAUL.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/APPO-OPTIONS-OVERHAUL.md)
- [RL-APPO-HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/RL-APPO-HANDOFF.md)
- [CURRENT-RL-SYSTEM.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/CURRENT-RL-SYSTEM.md)

Representative run docs from this stage:

- [RUN-REPORT-2026-04-05-V2-BC-APPO.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/run-reports/RUN-REPORT-2026-04-05-V2-BC-APPO.md)
- [RUN-REPORT-2026-04-05-X10-VALIDATE.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/run-reports/RUN-REPORT-2026-04-05-X10-VALIDATE.md)
- [RUN-REPORT-2026-04-05-X100-NORNN.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/run-reports/RUN-REPORT-2026-04-05-X100-NORNN.md)
- [RUN-REPORT-2026-04-05-X100-TEACHER-REG-COMPARISON.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/run-reports/RUN-REPORT-2026-04-05-X100-TEACHER-REG-COMPARISON.md)

## Phase 5: Evaluation Correction And Fast Debug Loops

This was the most important methodological turning point.

What changed:

- live seeded evaluation was found to be too nondeterministic for promotion decisions
- deterministic trace evaluation was introduced
- short loop validation and disagreement analysis became the main workflow

What held up:

- benchmark quality mattered more than another large run
- deterministic trace match repeatedly contradicted apparent online reward gains
- promotion decisions became much more defensible

Primary sources:

- [FAST-DEBUG-LOOP-REPORT-2026-04-05.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/FAST-DEBUG-LOOP-REPORT-2026-04-05.md)
- [FAST-RL-EXECUTION-PLAN-2026-04-05.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/FAST-RL-EXECUTION-PLAN-2026-04-05.md)
- [POSTMORTEM-NEXT-STEPS-2026-04-05.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/POSTMORTEM-NEXT-STEPS-2026-04-05.md)
- [ROLLING-RESEARCH-THESIS.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/ROLLING-RESEARCH-THESIS.md)

## Phase 6: Teacher-Centric Alignment Work

Once trace evaluation and warm-start correctness were in place, the focus shifted to preserving the teacher under online learning.

Important changes:

- teacher-regularized APPO
- BC warm starts
- better bridge correctness between BC and APPO
- prompt-conditioned teachers
- teacher loss shaping and action-specific weighting
- replay-heavy and DAgger-style probes

What held up:

- the warm-start bridge had been broken and later fixed
- step-0 teacher preservation became possible
- online RL still degraded the teacher after learning began
- targeted teacher-loss shaping helped more than broad anchoring

Primary sources:

- [ALIGNMENT-IMPROVEMENT-PLAN.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/ALIGNMENT-IMPROVEMENT-PLAN.md)
- [ALIGNMENT-IMPLEMENTATION-REPORT-2026-04-05.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/misc/ALIGNMENT-IMPLEMENTATION-REPORT-2026-04-05.md)
- [PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md)
- [REPORT-TEACHER-FALLBACK-HYPOTHESIS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-TEACHER-FALLBACK-HYPOTHESIS-2026-04-06.md)

Representative run reports:

- [RUN-REPORT-2026-04-06-mask-aware-teacher-reg.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/run-reports/RUN-REPORT-2026-04-06-mask-aware-teacher-reg.md)
- [RUN-REPORT-2026-04-06-teacher-fallback-prior-probe.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/run-reports/RUN-REPORT-2026-04-06-teacher-fallback-prior-probe.md)
- [RUN-REPORT-2026-04-06-split-teacher-base-and-supervision.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/run-reports/RUN-REPORT-2026-04-06-split-teacher-base-and-supervision.md)
- [RUN-REPORT-2026-04-06-bounded-residual-teacher-base-probe.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/run-reports/RUN-REPORT-2026-04-06-bounded-residual-teacher-base-probe.md)

## Phase 7: World-Model Feature Path

The world-model branch matured from a representation experiment into a credible teacher-construction path.

Important stages:

- short-horizon latent feature prediction
- trace transformation via world-model latents
- downstream BC probes
- scaling and validation
- text-conditioned world model with frozen text backbone

What held up:

- direct predictive metrics alone were not enough
- world-model augmented traces could improve downstream teacher quality
- text-conditioned world models improved the offline teacher to about `0.9625`
- medium online RL still drifted back toward the old plateau

Primary sources:

- [WORLD-MODEL-PLAN-FOR-RL-NETHACK.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/WORLD-MODEL-PLAN-FOR-RL-NETHACK.md)
- [REPORT-WORLD-MODEL-VALIDATION-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-WORLD-MODEL-VALIDATION-2026-04-06.md)
- [REPORT-WORLD-MODEL-SCALING-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-WORLD-MODEL-SCALING-2026-04-06.md)
- [REPORT-WORLD-MODEL-ENSEMBLE-DISTILL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-WORLD-MODEL-ENSEMBLE-DISTILL-2026-04-06.md)
- [REPORT-LLM-WORLD-MODEL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-LLM-WORLD-MODEL-2026-04-06.md)
- [LESSONS-WORLD-MODEL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/LESSONS-WORLD-MODEL-2026-04-06.md)

## Phase 8: Behavior-Reg And Proxy-Reward Branches

These branches represent the repo trying alternatives to raw APPO schedule tuning.

Behavior-reg branch:

- real and implemented
- stable after the masked-prior fix
- still weaker than the current best teacher

Proxy-reward branch:

- full offline pipeline implemented
- integrated into the reward path cleanly
- did not beat the best short teacher-replay branch

Primary sources:

- [REPORT-BEHAVIOR-REG-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-BEHAVIOR-REG-2026-04-06.md)
- [PLAN-PROXY-REWARD-OVERHAUL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/PLAN-PROXY-REWARD-OVERHAUL-2026-04-06.md)
- [REPORT-PROXY-REWARD-OVERHAUL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-PROXY-REWARD-OVERHAUL-2026-04-06.md)
- [REPORT-TWO-OPTIONS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-TWO-OPTIONS-2026-04-06.md)

Representative run reports:

- [RUN-REPORT-2026-04-06-prompt-conditioned-bc-probe.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/run-reports/RUN-REPORT-2026-04-06-prompt-conditioned-bc-probe.md)
- [RUN-REPORT-2026-04-06-bc-scaling-and-heldout-selection.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/run-reports/RUN-REPORT-2026-04-06-bc-scaling-and-heldout-selection.md)
- [RUN-REPORT-2026-04-06-replay-current-disagreement-boost-probe.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/run-reports/RUN-REPORT-2026-04-06-replay-current-disagreement-boost-probe.md)
- [RUN-REPORT-2026-04-06-exact-confusion-pair-dagger-probe.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/run-reports/RUN-REPORT-2026-04-06-exact-confusion-pair-dagger-probe.md)

## Durable Conclusion From The Timeline

By the end of the committed markdown trail, the repo had converged on a narrow but clear diagnosis:

- the strong offline teacher is real,
- deterministic trace evaluation is the trusted gate,
- representation work can improve the teacher,
- but the online improver still fails to beat the teacher consistently.

The dominant blocker is no longer missing infrastructure. It is teacher-constrained online improvement.

