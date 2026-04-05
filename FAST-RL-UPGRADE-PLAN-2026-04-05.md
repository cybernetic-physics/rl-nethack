# Fast RL Upgrade Plan (2026-04-05)

This document defines the next upgrade plan for the repo after the latest
APPO rerun and the deterministic trace-debugging work.

It is designed around one principle:

- do not use long APPO runs as the primary debugging tool

Instead, each likely failure mode should be exposed by the cheapest loop that
can reveal it.

This plan is grounded in the current code and reports:

- [RUN-REPORT-2026-04-05-X10-RERUN.md](/home/luc/rl-nethack/RUN-REPORT-2026-04-05-X10-RERUN.md)
- [FAST-DEBUG-LOOP-REPORT-2026-04-05.md](/home/luc/rl-nethack/FAST-DEBUG-LOOP-REPORT-2026-04-05.md)
- [POSTMORTEM-NEXT-STEPS-2026-04-05.md](/home/luc/rl-nethack/POSTMORTEM-NEXT-STEPS-2026-04-05.md)


## 1. Current Diagnosis

The repo now has:

- deterministic trace evaluation
- trace sharding
- BC training and evaluation
- BC warm-start APPO
- a real APPO backend

The latest result is clear:

- the pipeline runs
- BC remains stronger than APPO on the trusted trace benchmark
- APPO improves training reward but drifts away from teacher behavior

So the current blocker is not infrastructure.

The blocker is objective mismatch:

- the student is not staying aligned with the teacher while learning on its own
  state distribution


## 2. Main Failure Modes To Target

There are four likely failure classes.

### A. Teacher-distribution shift

Symptoms:

- BC is decent on teacher traces
- APPO gets worse on teacher-state regression after RL fine-tuning

Most likely fix:

- DAgger-style aggregation
- teacher-regularized RL

### B. Observation / representation weakness

Symptoms:

- stable directional confusions, especially `east` / `south`
- APPO and BC both over-predict one direction

Most likely fix:

- richer local geometry features
- stronger directional history

### C. Reward / objective misalignment

Symptoms:

- training reward improves
- trusted trace benchmark gets worse
- “best training reward” checkpoint is not actually best policy

Most likely fix:

- teacher regularization during RL
- checkpoint selection by trace metric

### D. Teacher path brittleness

Symptoms:

- `fork()` warning in the teacher counterfactual path
- higher risk once teacher relabeling becomes central

Most likely fix:

- remove or replace the `os.fork()` path in
  [src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py)


## 3. Fast Debug Ladder

The next work should follow this ladder.

### Layer 0: Deterministic Trace Gating

Trusted metric:

- deterministic trace match rate on fixed trace files

Commands already available:

```bash
uv run python cli.py rl-trace-report --input ...
uv run python cli.py rl-trace-disagreements --input ...
uv run python cli.py rl-shard-traces --input ...
```

Rule:

- no model/config change is “better” unless it improves the deterministic trace
  benchmark or a defined slice benchmark

Do not gate on:

- live seeded eval
- training reward alone

### Layer 1: Directional Slice Regression

Use small trace shards to isolate the current failure:

- full trace set
- `east/south` slice
- future `search` slice

Required metrics:

- overall match rate
- per-action precision / recall / f1
- common confusion pairs

Use this loop to debug feature changes before any RL run.

### Layer 2: BC-Only Debugging

BC should remain the main cheap proxy for policy quality.

Why:

- fast
- deterministic enough
- directly tied to features and trace data
- much cheaper than APPO

Any observation change should first go through:

1. train BC on fixed trace set
2. run trace report on full set
3. run trace report on direction-focused shards

Only if BC improves should it proceed to RL.

### Layer 3: Teacher-Student Drift Tests

The repo needs a cheaper way to detect RL drift than “wait for a long run and
compare the end checkpoint.”

Add a short-cycle evaluation:

1. start from BC
2. run APPO for `10k`, `25k`, `50k` frames
3. evaluate each checkpoint on the same fixed trace set
4. stop when trace match starts dropping

This will answer:

- how quickly does APPO drift?
- does a regularizer slow or stop that drift?

### Layer 4: Short RL Sweep, Not Big RL Sweep

If a change survives Layers 0-3, then run:

- a small sweep over a few settings
- not a single big x10 run

The first sweep should be over:

- teacher-regularization weight
- trace-regularization schedule
- action-mask strictness if changed

This is a targeted sweep, not exploratory brute force.


## 4. What To Build Next

### Workstream 1: Teacher-Regularized APPO

This is the highest-priority model change.

Goal:

- keep the student close to the BC teacher while training on student rollouts

Implementation idea:

- on student observations, compute teacher action distribution from BC
- add a KL or cross-entropy loss between student policy and BC policy
- anneal the weight over training

New code likely needed:

- `rl/teacher_regularization.py`
- updates to [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- possibly updates to [rl/model.py](/home/luc/rl-nethack/rl/model.py)

Fast loop for this:

1. run `10k/25k/50k` APPO steps
2. compare trace match against BC and against unregularized APPO

Success criterion:

- trace match should not immediately collapse below BC warm start

### Workstream 2: DAgger-Lite

This is the highest-priority data upgrade.

Goal:

- expose the learner to states it actually visits

Implementation idea:

1. roll out current student on fixed seeds
2. record student-visited states
3. relabel them with `task_greedy`
4. merge them into the trace dataset
5. retrain BC

New code likely needed:

- `rl/run_dagger_iteration.py`
- or equivalent CLI additions

Fast loop for this:

1. generate a small student trace shard
2. relabel it
3. retrain BC only
4. compare against baseline BC on the trusted trace set

Success criterion:

- BC improves on student-induced states without regressing badly on teacher traces

### Workstream 3: Checkpoint Selection By Trace Metric

Current problem:

- best training reward checkpoint is often not best trace behavior checkpoint

Fix:

- after every save interval, run a cheap deterministic trace eval on a fixed shard
- save:
  - best-by-training-reward
  - best-by-trace-match

Fast loop benefit:

- avoids wasting analysis time on the wrong checkpoint

### Workstream 4: Directional Feature Upgrade

Current encoder:

- [rl/feature_encoder.py](/home/luc/rl-nethack/rl/feature_encoder.py)

Current likely weakness:

- not enough local geometry to distinguish symmetric movement choices robustly

Candidate additions:

- a larger local patch around the player
- explicit frontier distances by direction
- short directional displacement history
- “did the local neighborhood actually change?” features
- recent turn deltas for exploration progress

Fast loop for this:

1. implement one feature change
2. BC train on full trace set
3. run full trace report
4. run `east/south` slice report

Success criterion:

- BC directional recall improves, especially `east` / `south`

### Workstream 5: Remove `fork()` From Teacher Path

Current smell:

- [src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py) uses `os.fork()`

Why this matters:

- teacher relabeling and DAgger will rely on this path more heavily
- the current warning suggests it is brittle in a multi-threaded process

Fix options:

1. subprocess-based isolated scoring
2. copyable env wrapper / save-restore state path if possible
3. a batched teacher-action service if needed

This is not the very first scientific priority, but it is the most obvious
technical debt in the harness.


## 5. Parameter Sweep Plan

Parameter sweeps should be narrow and hypothesis-driven.

Do not sweep large APPO runs first.

### Sweep A: Teacher-Regularization Weight

Sweep on short APPO runs:

- `0.01`
- `0.05`
- `0.1`
- `0.2`

Metric:

- best deterministic trace match over `10k/25k/50k` frames

### Sweep B: Teacher-Regularization Schedule

Compare:

- constant
- linear decay
- cosine decay

Metric:

- trace match retention vs BC baseline

### Sweep C: BC dataset mix ratio after one DAgger iteration

Compare:

- old traces only
- `75/25` old/new
- `50/50` old/new

Metric:

- full trace match
- student-state trace match
- `east/south` slice match

### Sweep D: Directional feature ablations

Compare:

- current `v2`
- `v2 + local_patch`
- `v2 + displacement_history`
- `v2 + frontier_distance`

Metric:

- BC on trace slices


## 6. Weekly Execution Order

### Phase 1: Fastest wins

1. best-by-trace checkpoint selection
2. DAgger-lite trace relabel loop
3. teacher-regularized APPO scaffold

### Phase 2: Representation

4. directional feature upgrade
5. BC ablation loop on focused shards

### Phase 3: RL re-entry

6. short teacher-regularized APPO sweeps
7. only then another x10-style run if short runs show real trace gains


## 7. Success Criteria

The repo should consider the next stage successful only if all of these happen:

1. BC improves on deterministic trace slices after DAgger-lite or feature work
2. teacher-regularized APPO no longer collapses below BC immediately
3. best-by-trace checkpoint selection consistently picks a different checkpoint
   than best-by-training-reward when drift occurs
4. a short APPO run shows better trace behavior than BC or at least matches BC
   while improving task reward

Only after that should the repo spend more GPU time on larger RL runs.


## 8. Bottom Line

The next progress in this repo will not come from another large unguided APPO
run.

It will come from:

- keeping the student aligned with the teacher while it learns
- aggregating student-induced states back into training
- improving the observation signal for directional decisions
- and using deterministic trace slices as the fast regression loop

That is the most direct path to converting the current working harness into a
useful learning system.
