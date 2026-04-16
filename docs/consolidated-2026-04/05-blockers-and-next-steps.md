# Blockers And Next Steps

## Current Blocker

The repo’s main blocker is no longer missing infrastructure.

The recurring conclusion across the latest docs is:

- the offline teacher pipeline is strong,
- the warm-start bridge can preserve the teacher at step 0,
- deterministic trace evaluation is trustworthy enough to gate work,
- but online RL still drifts away from teacher-aligned behavior once learning begins.

This is stated most clearly in:

- [PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md)
- [REPORT-TWO-OPTIONS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-TWO-OPTIONS-2026-04-06.md)
- [REPORT-TEACHER-FALLBACK-HYPOTHESIS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-TEACHER-FALLBACK-HYPOTHESIS-2026-04-06.md)

## Blockers By Area

### Online Improvement Objective

Main issue:

- APPO-style online updates still push the student off the teacher manifold faster than they create trusted improvement.

What the repo evidence says:

- teacher regularization helps
- targeted teacher-action shaping helps
- fallback to the trusted teacher improves stability
- but none of these by themselves produce a teacher-beating learned checkpoint

Primary sources:

- [ALIGNMENT-IMPLEMENTATION-REPORT-2026-04-05.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/misc/ALIGNMENT-IMPLEMENTATION-REPORT-2026-04-05.md)
- [REPORT-TEACHER-FALLBACK-HYPOTHESIS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-TEACHER-FALLBACK-HYPOTHESIS-2026-04-06.md)
- [REPORT-TWO-OPTIONS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-TWO-OPTIONS-2026-04-06.md)

### Behavior-Reg Branch

Main issue:

- the branch is real and stable after fixes, but weaker than the best current teacher.

Primary source:

- [REPORT-BEHAVIOR-REG-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-BEHAVIOR-REG-2026-04-06.md)

### Proxy-Reward Branch

Main issue:

- infrastructure works, but the dataset is too small and the learned signal is not yet strong enough to beat the best teacher-replay branch.

Primary source:

- [REPORT-PROXY-REWARD-OVERHAUL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-PROXY-REWARD-OVERHAUL-2026-04-06.md)

### World-Model Branch

Main issue:

- world-model features improve offline teacher quality more reliably than they improve medium-horizon online RL.

Primary sources:

- [REPORT-LLM-WORLD-MODEL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-LLM-WORLD-MODEL-2026-04-06.md)
- [LESSONS-WORLD-MODEL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/LESSONS-WORLD-MODEL-2026-04-06.md)

### Input Representation And History

A quieter but important blocker is representational:

- most mainline policy paths still rely on current-state-heavy inputs,
- hand-engineered summaries and short local features help,
- but the stack still lacks a strong explicit multi-turn history representation in the main BC/RL loop.

This issue is spread across the code and docs, especially:

- [CURRENT-RL-SYSTEM.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/CURRENT-RL-SYSTEM.md)
- [README.md](/home/luc/rl-nethack-worktree-20260416/README.md)
- [HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/HANDOFF.md)

## Recommended Next Steps

### 1. Preserve The Fast Trusted Loop

This should remain the default project workflow:

1. define the held-out trace gate,
2. run a short bounded experiment,
3. rank by deterministic trace match,
4. inspect disagreement patterns,
5. scale only short-loop winners.

This recommendation is supported across:

- [FAST-ITERATION-PLAN.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/FAST-ITERATION-PLAN.md)
- [PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md)

### 2. Continue Teacher-Constrained Improver Work

The current evidence supports keeping the improver tightly coupled to the teacher rather than reverting to plain APPO.

The most plausible near-term directions are:

- stronger replay and on-policy distillation on student states,
- tighter teacher-aware regularization,
- more controlled DAgger or relabeling loops,
- teacher-as-base mechanisms that stabilize without freezing progress.

Primary docs:

- [REPORT-TWO-OPTIONS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-TWO-OPTIONS-2026-04-06.md)
- [REPORT-TEACHER-FALLBACK-HYPOTHESIS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-TEACHER-FALLBACK-HYPOTHESIS-2026-04-06.md)

### 3. Keep World-Model Work Focused On Teacher Construction

The best supported role for the world-model path is:

- representation learning,
- trace transformation,
- stronger offline teacher construction.

It should not be oversold as the direct online RL solution.

Primary docs:

- [REPORT-LLM-WORLD-MODEL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-LLM-WORLD-MODEL-2026-04-06.md)
- [LESSONS-WORLD-MODEL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/LESSONS-WORLD-MODEL-2026-04-06.md)

### 4. Improve Offline Data And Teacher Coverage Before Scaling New Objectives

This applies especially to:

- proxy-reward work,
- behavior-reg variants,
- rare-action or rare-state trace slices,
- hard-case mining for early-step or confusion states.

The run-report trail strongly suggests that better offline data quality is still one of the best levers in the repo.

Relevant sources:

- [RUN-REPORT-2026-04-06-broader-reset-slice-mining.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/run-reports/RUN-REPORT-2026-04-06-broader-reset-slice-mining.md)
- [RUN-REPORT-2026-04-06-exact-step0-hardcase-mining.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/run-reports/RUN-REPORT-2026-04-06-exact-step0-hardcase-mining.md)
- [RUN-REPORT-2026-04-06-bc-scaling-and-heldout-selection.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/run-reports/RUN-REPORT-2026-04-06-bc-scaling-and-heldout-selection.md)

### 5. Keep The Source Trail Auditable

A practical next step for future doc hygiene is to maintain this consolidated folder as the durable narrative and treat short run reports as append-only evidence.

That means:

- keep short reports small and factual,
- summarize durable conclusions here,
- and update the source index as new branches land.

## Bottom-Line Recommendation

The repo should currently be treated as:

- a strong teacher-building system,
- a credible deterministic evaluation system,
- an implemented but still underperforming online improver stack.

The best next work is not a larger undirected APPO run. It is teacher-constrained improvement under a trusted short-loop gate.

