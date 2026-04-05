# Alignment Improvement Plan

Date: `2026-04-05`

## Purpose

This document is the next concrete execution plan for improving the RL stack in this repo.

The core lesson from the latest runs is:

- generic novelty bonus is **not** the main bottleneck
- teacher alignment is the bottleneck
- the current repo should be shaped around **teacher-anchored improvement**, not more blind APPO scale

Current best trusted numbers:

- BC teacher on [data/tracefix_v2_explore_traces.jsonl](/home/luc/rl-nethack/data/tracefix_v2_explore_traces.jsonl): `0.6395`
- best teacher-regularized APPO checkpoint: `0.6318`
- best teacher-reg + bonus checkpoint: `0.6085`

That means the current RL path is close, but still below the teacher. The next work should be aimed at pushing past that gap.

## Principles

1. Deterministic trace match is the source of truth.
2. BC is the teacher-aligned anchor.
3. RL is an improvement stage over BC, not the main behavior source.
4. New changes should be debugged in short loops before long runs.
5. Reward shaping is auxiliary; alignment is primary.

## TODO

### 1. Build a real DAgger schedule

Goal:

- make teacher supervision iterative instead of one-shot
- keep the student aligned on the states it actually visits

Action items:

- [ ] add a top-level DAgger runner command in [cli.py](/home/luc/rl-nethack/cli.py)
  - proposed name: `rl-dagger-iterate`
- [ ] implement a reusable schedule in [rl/dagger.py](/home/luc/rl-nethack/rl/dagger.py)
  - inputs:
    - base trace dataset
    - student policy type (`bc`, later `appo`)
    - student checkpoint/model path
    - teacher task
    - iteration count
    - episodes per iteration
    - max steps per episode
    - merge ratio / sampling weights
- [ ] add ratio-controlled merge logic
  - preserve the original teacher trace set
  - merge student-state relabeled traces into a new derived dataset
  - support:
    - `base_only`
    - `uniform_merge`
    - `weighted_recent`
- [ ] automatically retrain BC after each DAgger merge
- [ ] emit a compact report after each iteration:
  - BC train accuracy
  - deterministic trace match on base trace set
  - deterministic trace match on a held-out shard
- [ ] add tests for:
  - deterministic iteration outputs
  - merge ratio correctness
  - iteration report contents

Success criteria:

- DAgger loop runs end to end without manual intervention
- at least one short DAgger schedule improves BC over `0.6395` or improves directional recall on weak shards

### 2. Move trace-match checkpoint selection into the training loop

Goal:

- stop relying on post-hoc ranking only
- preserve the actual best checkpoint by trusted metric during RL

Action items:

- [ ] extend [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py) to support periodic trace evaluation during APPO training
- [ ] add config/CLI options in [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py) and [cli.py](/home/luc/rl-nethack/cli.py):
  - `--trace-eval-input`
  - `--trace-eval-interval-env-steps`
  - `--trace-eval-top-k`
- [ ] on each interval:
  - load the latest checkpoint
  - run deterministic trace eval
  - update a stable best alias if trace match improved
- [ ] write machine-readable metadata next to the alias:
  - `best_trace_match.json`
  - fields:
    - checkpoint path
    - env steps
    - trace match
    - action counts
- [ ] add a small integration test for periodic trace selection

Success criteria:

- long RL runs automatically preserve their best trusted checkpoint
- no manual ranking step is required to recover the best policy

### 3. Improve directional and frontier representation

Goal:

- reduce the persistent directional bias
- improve teacher agreement on `east`, `south`, and `search`

Action items:

- [ ] inspect and extend [rl/feature_encoder.py](/home/luc/rl-nethack/rl/feature_encoder.py)
- [ ] add richer directional features, likely:
  - local map patch around the agent
  - corridor / dead-end indicators
  - frontier counts by direction
  - nearest unexplored tile direction
  - recent action history
  - repeated-state count and recent loop stats
- [ ] version this cleanly as `v3` in:
  - [rl/feature_encoder.py](/home/luc/rl-nethack/rl/feature_encoder.py)
  - [rl/config.py](/home/luc/rl-nethack/rl/config.py)
  - [rl/traces.py](/home/luc/rl-nethack/rl/traces.py)
  - [rl/evaluate_bc.py](/home/luc/rl-nethack/rl/evaluate_bc.py)
  - [rl/evaluate.py](/home/luc/rl-nethack/rl/evaluate.py)
- [ ] generate focused trace shards for weak teacher actions using [rl/traces.py](/home/luc/rl-nethack/rl/traces.py)
  - especially `east`, `south`, and `search`
- [ ] rerun BC first on:
  - full base trace set
  - `east/south` shard
  - `search` shard if large enough
- [ ] add per-action recall/precision checks as regression tests

Success criteria:

- BC with richer features improves on weak-action shards
- directional confusion decreases before any new RL run

### 4. Keep teacher-reg APPO as the baseline RL improver

Goal:

- stabilize the current best RL line
- avoid fragmenting the repo around weaker objectives

Action items:

- [ ] treat teacher-reg APPO as the default RL path in docs and examples
- [ ] keep the current default teacher settings:
  - `teacher_loss_coef = 0.01`
  - `teacher_loss_type = ce`
  - `--no-rnn`
- [ ] add a short-run regression command that always compares:
  - BC teacher
  - latest teacher-reg APPO
  - best teacher-reg APPO
- [ ] make sure all new RL experiments report:
  - deterministic trace match
  - action histogram
  - invalid-action rate
  - repeated-action rate

Success criteria:

- no future RL experiment is merged or trusted without trace-gated comparison against the current best teacher-reg run

### 5. Prepare the next algorithmic step: behavior-regularized improvement

Goal:

- if APPO + DAgger + better features still stall below BC, move to a more appropriate improvement method

Action items:

- [ ] create a design note for AWAC-like or BRAC-like improvement in this repo
- [ ] identify where to plug this in:
  - new module under `rl/`
  - likely separate from Sample Factory mainline
- [ ] define the minimum viable offline-to-online dataset format using current traces
- [ ] decide whether to start with:
  - KL-regularized actor update
  - advantage-weighted BC loss
  - offline replay mixed with short online rollouts
- [ ] add a small experimental training entrypoint once the design is clear

Success criteria:

- the repo is ready for a constrained improvement algorithm if teacher-reg APPO plateaus

## Short Execution Order

Implement in this order:

1. DAgger schedule
2. in-training trace checkpoint selection
3. richer `v3` directional/frontier features
4. BC reruns on focused shards
5. new teacher-reg APPO runs with improved BC + features
6. behavior-regularized algorithm design if still stalled

## Fast Debug Loop

Do not use long runs as the main development loop.

Preferred loop:

1. modify feature or DAgger logic
2. run BC on a small or focused trace set
3. check deterministic trace match and per-action recall
4. if BC improves, run short teacher-reg APPO
5. only then run larger APPO

Use these tools:

- [rl-trace-report](/home/luc/rl-nethack/cli.py)
- [rl-trace-disagreements](/home/luc/rl-nethack/cli.py)
- [rl-rank-checkpoints](/home/luc/rl-nethack/cli.py)
- [rl-run-dagger](/home/luc/rl-nethack/cli.py)

## What Not To Do

- do not spend more sweep budget on the current generic episodic bonus
- do not select checkpoints by APPO reward
- do not add more complexity to the scheduler before low-level alignment improves
- do not reintroduce RNNs into the mainline until the non-RNN teacher path is clearly surpassed

## Bottom Line

The next repo shape should be:

- teacher traces
- iterative DAgger refresh
- BC teacher
- teacher-regularized RL improvement
- deterministic trace-gated checkpointing

That is the most direct path past the current alignment bottleneck.
