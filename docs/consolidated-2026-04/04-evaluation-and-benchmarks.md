# Evaluation And Benchmarks

## Core Principle

The most important evaluation lesson in the repo is that weak metrics created confusion, while deterministic trace evaluation created progress.

This point is repeated in:

- [FAST-DEBUG-LOOP-REPORT-2026-04-05.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/FAST-DEBUG-LOOP-REPORT-2026-04-05.md)
- [POSTMORTEM-NEXT-STEPS-2026-04-05.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/POSTMORTEM-NEXT-STEPS-2026-04-05.md)
- [ROLLING-RESEARCH-THESIS.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/ROLLING-RESEARCH-THESIS.md)
- [PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md)

## Benchmark Families

The markdown trail describes several benchmark families. They should not be conflated.

### 1. Throughput And Data-Generation Benchmarks

These answer:

- how fast can we generate examples locally,
- how many examples per model/server setup,
- and whether action quality is reasonable enough to keep scaling.

These are important engineering signals, but they are not promotion metrics for control quality.

Primary docs:

- [HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/HANDOFF.md)
- [README.md](/home/luc/rl-nethack-worktree-20260416/README.md)

### 2. Task-Harness Closed-Loop Metrics

These answer:

- does a controller achieve shaped exploration, survival, combat, descent, or resource goals better than simple baselines,
- and do the task rewards produce useful local behavior?

These were the first behavior metrics that mattered before online RL matured.

Primary docs:

- [RL-HARNESS-TASKS.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/misc/RL-HARNESS-TASKS.md)
- [HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/HANDOFF.md)

### 3. Live Seeded Policy Evaluation

This was used early, then demoted.

Why it was demoted:

- repeated runs on the same seeds produced materially different summaries,
- raw NLE seeded resets were not trustworthy enough for hard regression gating.

Current status in the research narrative:

- diagnostic only
- useful for smoke checking
- not a promotion gate

Primary docs:

- [FAST-DEBUG-LOOP-REPORT-2026-04-05.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/FAST-DEBUG-LOOP-REPORT-2026-04-05.md)
- [ROLLING-RESEARCH-THESIS.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/ROLLING-RESEARCH-THESIS.md)

### 4. Deterministic Trace-Match Evaluation

This became the trusted benchmark family.

What it measures:

- action agreement with a fixed held-out teacher trace set,
- often as top-line trace match for BC, behavior-reg, or APPO checkpoints,
- plus disagreement reports and checkpoint ranking.

Why it matters:

- it is stable enough to gate short experiments,
- it aligns with the teacher-building thesis of the repo,
- it repeatedly exposed drift that reward curves hid.

Primary docs:

- [PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md)
- [ROLLING-RESEARCH-THESIS.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/ROLLING-RESEARCH-THESIS.md)
- [RL-APPO-HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/RL-APPO-HANDOFF.md)

### 5. World-Model Direct Metrics

These include:

- feature cosine similarity,
- reconstruction error,
- action accuracy and top-k action accuracy,
- reward error,
- latent dead-fraction,
- downstream BC probe results from transformed traces.

The key lesson from the docs is that direct world-model metrics are useful but not sufficient. Downstream teacher quality matters more.

Primary docs:

- [REPORT-WORLD-MODEL-VALIDATION-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-WORLD-MODEL-VALIDATION-2026-04-06.md)
- [REPORT-LLM-WORLD-MODEL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-LLM-WORLD-MODEL-2026-04-06.md)
- [LESSONS-WORLD-MODEL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/LESSONS-WORLD-MODEL-2026-04-06.md)

### 6. Proxy-Reward Offline Metrics

These answer:

- whether the proxy model can rank actions on held-out trace slices,
- whether the reward decomposition is sane,
- and whether the reward can be safely injected into online RL.

Primary docs:

- [PLAN-PROXY-REWARD-OVERHAUL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/PLAN-PROXY-REWARD-OVERHAUL-2026-04-06.md)
- [REPORT-PROXY-REWARD-OVERHAUL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-PROXY-REWARD-OVERHAUL-2026-04-06.md)

## Benchmark Lineage And Comparison Safety

The markdown trail warns repeatedly that numbers from different benchmark regimes are not directly interchangeable.

Important benchmark breaks include:

- the demotion of live seeded evaluation,
- the transition from older `v2` regimes to corrected `tracefix_v2`,
- later `v3` and `v4 + wm_*` teacher lines,
- warm-start bridge fixes that changed what step-0 means,
- feature representation changes that altered the effective policy family.

So any comparison must match:

- observation version,
- trace dataset version,
- teacher artifact,
- world-model augmentation mode,
- warm-start implementation,
- and whether the branch is offline, short online, or medium online.

The most explicit warnings are in:

- [ROLLING-RESEARCH-THESIS.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/ROLLING-RESEARCH-THESIS.md)
- [CURRENT-RL-SYSTEM.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/CURRENT-RL-SYSTEM.md)

## Tests And Validation Practices

The markdown trail emphasizes both code tests and short-loop experimental validation.

### Code-Level Validation

Examples:

- broad early unit coverage for state encoding, data generation, evaluator, reporter, manifest, and training entrypoints
- `tests/test_rl_scaffold.py` used repeatedly as the fast integration check for RL/world-model/proxy branches

Examples explicitly cited in the docs:

- [REPORT-LLM-WORLD-MODEL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-LLM-WORLD-MODEL-2026-04-06.md)
- [REPORT-PROXY-REWARD-OVERHAUL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-PROXY-REWARD-OVERHAUL-2026-04-06.md)

### Experiment-Level Validation

The recurring good pattern is:

1. run a short or bounded experiment,
2. score on a fixed held-out trace set,
3. inspect disagreement and trace metadata,
4. only then scale.

This was the major methodological upgrade over the earlier “run longer and hope” approach.

Primary docs:

- [FAST-ITERATION-PLAN.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/FAST-ITERATION-PLAN.md)
- [FAST-RL-UPGRADE-PLAN-2026-04-05.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/FAST-RL-UPGRADE-PLAN-2026-04-05.md)
- [PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md)

## Promotion Gates That Actually Matter

By the end of the committed markdown trail, the real gates are:

- does the branch beat or preserve the strongest trusted teacher on held-out deterministic traces,
- does it survive the short online bridge without immediate collapse,
- does it improve disagreement patterns in a meaningful way,
- and does the result hold under the current benchmark regime rather than an obsolete one.

This is the best concise expression of repo-local evaluation policy:

- short trusted loop first
- held-out trace match second
- scale only after a real short-loop win

