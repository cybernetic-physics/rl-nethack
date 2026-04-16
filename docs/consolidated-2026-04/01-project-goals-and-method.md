# Project Goals And Method

## Purpose

The repo started with a forward-model thesis and then shifted into a teacher-centric online-improvement thesis.

The original thesis was:

- generate NetHack trajectories,
- train a model to predict action-conditioned state changes,
- use structured state descriptions and artifact manifests rather than raw end-to-end imitation alone,
- then turn those learned representations into better control.

The later thesis became:

- build a strong offline teacher,
- define a trustworthy benchmark,
- and learn an online improver that can beat the teacher without drifting away from teacher-aligned behavior.

This transition is explicit in:

- [HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/HANDOFF.md)
- [CURRENT-RL-SYSTEM.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/CURRENT-RL-SYSTEM.md)
- [ROLLING-RESEARCH-THESIS.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/ROLLING-RESEARCH-THESIS.md)
- [PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md)

## Why NetHack Is Hard Here

The recurring repo-local diagnosis is stable across the markdown trail:

- NetHack is long-horizon and partially observed.
- Raw environment reward is sparse and poorly aligned with the short-horizon skills the repo cares about.
- One-shot imitation is insufficient because the learner visits its own states and drifts.
- Online PPO-style updates can destroy a good offline policy before they improve it.
- Evaluation quality matters as much as algorithm choice because weak evaluation creates false positives.

These assumptions recur in:

- [RL-HARNESS-TASKS.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/misc/RL-HARNESS-TASKS.md)
- [RL-APPO-HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/RL-APPO-HANDOFF.md)
- [REPORT-TWO-OPTIONS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-TWO-OPTIONS-2026-04-06.md)
- [REPORT-TEACHER-FALLBACK-HYPOTHESIS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-TEACHER-FALLBACK-HYPOTHESIS-2026-04-06.md)

## Methodology That Survived

The project tried multiple surface forms, but a few methodological choices clearly survived:

- Prefer structured state encodings and explicit task definitions over raw reward chasing.
- Build offline teachers first, then judge online methods by whether they preserve or improve teacher-aligned behavior.
- Use short debug loops and trusted held-out traces instead of scaling weak ideas.
- Treat representation work, teacher-building, and online improvement as separate problems.
- Preserve the historical trail of experiments so later changes can be interpreted against the correct benchmark regime.

That methodological shift is visible in:

- [FAST-ITERATION-PLAN.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/FAST-ITERATION-PLAN.md)
- [FAST-DEBUG-LOOP-REPORT-2026-04-05.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/FAST-DEBUG-LOOP-REPORT-2026-04-05.md)
- [POSTMORTEM-NEXT-STEPS-2026-04-05.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/POSTMORTEM-NEXT-STEPS-2026-04-05.md)
- [PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md)

## Main Research Questions

Across the document trail, the project repeatedly asks versions of the same questions:

1. Can we cheaply generate action-conditioned NetHack data with enough quality to train useful models?
2. Can a forward model or world model improve teacher quality or control quality?
3. Can the repo support real RL infrastructure rather than only SFT and one-step control?
4. Once the infrastructure works, can online learning beat a strong offline teacher without drifting?

The answer progression by early April 2026 was:

- data generation: yes
- forward-model SFT infrastructure: yes
- RL infrastructure: yes
- strong offline teacher: yes
- teacher-beating online improver: not yet

This conclusion is most clearly stated in:

- [README.md](/home/luc/rl-nethack-worktree-20260416/README.md)
- [CURRENT-RL-SYSTEM.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/CURRENT-RL-SYSTEM.md)
- [RL-APPO-HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/RL-APPO-HANDOFF.md)
- [PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md)

## What The Team Learned To Optimize For

The repo started by optimizing throughput, then shifted toward optimizing for trustworthy signals.

The sequence was roughly:

- get NLE data generation and LoRA training working,
- scale local LLM-policy generation throughput with vLLM,
- introduce task rewards and one-step control,
- add APPO and learned online RL,
- discover that reward and live eval are too noisy,
- adopt deterministic trace match as the main gate,
- optimize for better teachers and teacher-preserving improvers.

The resulting project norm should be read as:

- trusted benchmark first,
- teacher quality second,
- online improvement only when it wins on the trusted benchmark.

That norm is argued in:

- [ROLLING-RESEARCH-THESIS.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/ROLLING-RESEARCH-THESIS.md)
- [PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md)
- [LESSONS-WORLD-MODEL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/LESSONS-WORLD-MODEL-2026-04-06.md)

## Stable Constraints

Several operational realities recur across the docs and shape the methodology:

- the host is a `4x NVIDIA H200` machine,
- local serving commonly uses GPUs `0,1`,
- training commonly uses GPUs `2,3` or all GPUs for short SFT runs,
- APPO depends on Sample Factory bootstrapping rather than a fully locked dependency path,
- deterministic seeded live evaluation through raw NLE is not trusted enough for promotion gating.

These are documented in:

- [README.md](/home/luc/rl-nethack-worktree-20260416/README.md)
- [HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/HANDOFF.md)
- [RL-APPO-HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/RL-APPO-HANDOFF.md)

