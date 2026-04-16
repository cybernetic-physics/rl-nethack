# Fast Iteration Plan

This document defines a faster upgrade and debugging loop for the current
repo. It is grounded in the code and run results as of 2026-04-05.

The goal is simple:

- stop using large end-to-end training runs as the primary debugging tool,
- isolate failures earlier and cheaper,
- make policy comparisons trustworthy,
- and only spend GPU time on runs that answer a specific question.


## 1. Current Diagnosis

The repo now has a real hybrid stack:

- forward-model SFT in [train.py](/home/luc/rl-nethack/train.py)
- heuristic teacher and task evaluator in
  [src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py)
- multi-turn trace generation in [rl/traces.py](/home/luc/rl-nethack/rl/traces.py)
- behavior cloning in [rl/train_bc.py](/home/luc/rl-nethack/rl/train_bc.py)
- APPO RL in [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py) and
  [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py)
- learned reward and scheduler side paths in
  [rl/train_reward_model.py](/home/luc/rl-nethack/rl/train_reward_model.py) and
  [rl/train_scheduler.py](/home/luc/rl-nethack/rl/train_scheduler.py)

The latest validation run in
[RUN-REPORT-2026-04-05-V2-BC-APPO.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/run-reports/RUN-REPORT-2026-04-05-V2-BC-APPO.md)
showed:

- the `v2` trace -> BC -> APPO warm-start pipeline works,
- BC warm-start improves early APPO training dynamics,
- final APPO policy still collapses into repetitive behavior,
- and evaluation is currently nondeterministic.

That last point changes the priorities.

Right now the most expensive thing in the repo is not training. It is drawing
conclusions from unstable experiments.


## 2. Main Principle

The repo needs a layered debug loop.

Do not debug the system by repeatedly running:

1. trace generation,
2. BC training,
3. reward training,
4. scheduler training,
5. APPO training,
6. full evaluation.

That is too slow and too confounded.

Instead, every issue should be assigned to the cheapest layer that can expose
it.


## 3. Fast Debug Ladder

### Layer 0: Deterministic Environment And Evaluation

This is the first blocker.

Current code paths:

- [rl/evaluate.py](/home/luc/rl-nethack/rl/evaluate.py)
- [rl/evaluate_bc.py](/home/luc/rl-nethack/rl/evaluate_bc.py)
- [rl/env_adapter.py](/home/luc/rl-nethack/rl/env_adapter.py)
- [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py)

Problem:

- repeated eval of the same checkpoint on the same seeds gives different
  results.

Until this is fixed:

- no small gain is trustworthy,
- no sweep is trustworthy,
- no APPO vs BC comparison is trustworthy.

Required work:

1. add a deterministic regression command
2. run the same evaluation `N=5` times
3. fail if summaries differ outside exact or near-exact tolerance
4. log:
   - seed
   - action sequence
   - allowed-action set
   - env reward
   - shaped reward
   - position
   - state hash

Likely root causes to check:

1. NLE reset/seed path differs between codepaths
2. hidden stochasticity in policy inference
3. memory state is not initialized or updated identically
4. allowed-action reconstruction is order-dependent
5. actor-critic recurrent state or normalization path differs between runs

Deliverable:

- `rl-check-determinism` CLI command
- deterministic eval test in `tests/`


### Layer 1: Single-Episode Replay

Current relevant code:

- [src/closed_loop_debug.py](/home/luc/rl-nethack/src/closed_loop_debug.py)
- [rl/traces.py](/home/luc/rl-nethack/rl/traces.py)

Purpose:

- verify one policy on one known episode, step by step

This is cheaper and more informative than running aggregate metrics over many
seeds.

Required work:

1. generate one short `explore` golden trace
2. add replay for:
   - `task_greedy`
   - BC
   - APPO
3. compare at each step:
   - action chosen
   - state hash
   - allowed actions
   - shaped reward
   - position

This should answer:

- where does BC first diverge from the teacher?
- where does APPO first diverge from BC?
- is the divergence caused by observation loss, bad reward, or recurrent drift?

Deliverable:

- `rl-replay-trace` CLI command


### Layer 2: Oracle Agreement Tests

Current relevant code:

- [src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py)
- [rl/train_bc.py](/home/luc/rl-nethack/rl/train_bc.py)
- [rl/evaluate_bc.py](/home/luc/rl-nethack/rl/evaluate_bc.py)

Purpose:

- debug representation and imitation quality without RL noise

We already know BC is better than APPO-from-scratch. So BC should be the main
debug target before any RL tuning.

Required work:

1. add a command that samples states from traces and measures:
   - teacher action
   - BC action
   - APPO action
2. report disagreement rates by action type
3. save failure cases where:
   - teacher picks movement but model repeats
   - teacher picks `search` but model moves
   - model chooses action outside top-k teacher alternatives

Deliverable:

- `rl-compare-actions` CLI command


### Layer 3: Small BC Experiments

BC should become the cheap proxy for policy quality.

Why:

- fast to train
- deterministic enough to debug once eval is fixed
- directly tied to trace quality and feature quality
- much cheaper than APPO

Required experiment loop:

1. modify observation encoder
2. train BC for `5-10` minutes or less
3. run deterministic replay and short eval
4. keep only changes that improve BC

This means most policy representation work should happen in:

- [rl/feature_encoder.py](/home/luc/rl-nethack/rl/feature_encoder.py)
- [rl/bc_model.py](/home/luc/rl-nethack/rl/bc_model.py)

not directly in APPO.


### Layer 4: Short APPO Fine-Tunes

APPO should be used only after BC passes the cheap tests.

Required experiment form:

- warm-start from BC
- `10k-30k` env steps
- one skill only
- deterministic eval on a fixed seed set
- compare against the pre-RL BC checkpoint

Question to answer:

- did RL improve the BC policy, or destroy it?

If APPO is worse than BC after a short fine-tune, the run should be considered
failed immediately. Do not scale it.


## 4. What To Sweep

The repo does need sweeps, but only after determinism is fixed.

The first sweeps should be narrow and cheap.

### Sweep A: Observation Version

File:

- [rl/feature_encoder.py](/home/luc/rl-nethack/rl/feature_encoder.py)

Candidates:

- `v1`
- current `v2`
- `v3_local_patch`
- `v3_local_patch_plus_history`

Measure first with BC:

- train accuracy
- teacher agreement
- short deterministic closed-loop reward
- repeated-action rate


### Sweep B: Action Space Per Skill

Files:

- [rl/options.py](/home/luc/rl-nethack/rl/options.py)
- [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py)

Candidates for `explore`:

- full current action set
- strict movement-only plus `search`
- strict movement-only plus `search` and `pickup`

Measure with BC first, APPO second.

This is likely high leverage because the current failure mode is directional
collapse and wasted actions.


### Sweep C: BC Warm-Start Strength

Files:

- [rl/train_bc.py](/home/luc/rl-nethack/rl/train_bc.py)
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)

Candidates:

- small BC checkpoint
- medium BC checkpoint
- higher-accuracy BC checkpoint

Measure:

- APPO performance delta from pre-RL BC
- how quickly APPO degrades or improves


### Sweep D: Reward Source

Files:

- [rl/rewards.py](/home/luc/rl-nethack/rl/rewards.py)
- [rl/train_reward_model.py](/home/luc/rl-nethack/rl/train_reward_model.py)

Candidates:

- hand-shaped only
- learned only
- mixed hand-shaped + learned

But do this only after the policy and eval path are stable.


## 5. What Not To Sweep Yet

Do not spend time sweeping:

- many APPO hyperparameters at once
- scheduler variants
- reward model architectures
- multi-skill training
- long SFT runs

Reason:

- the current bottleneck is not optimizer quality,
- it is evaluation trustworthiness and policy collapse.

Large sweeps now would generate noise, not knowledge.


## 6. Faster Data Strategy

The repo should stop treating trace generation as one big offline step.

Instead use a gold/silver loop.

### Gold traces

Teacher:

- `task_greedy`

Use for:

- regression traces
- BC bootstrap
- action agreement analysis

These are expensive and should stay small and high-quality.


### Silver traces

Teacher:

- current BC or BC+APPO student

Use for:

- student rollout collection
- teacher relabeling on visited states

This is the DAgger-style path.

Process:

1. roll out current student
2. relabel sampled states with `task_greedy`
3. merge into dataset
4. retrain BC
5. repeat

This is probably the highest-value next algorithmic change in the repo.


## 7. Concrete New Commands To Add

The current CLI is good, but the repo needs explicit fast-debug commands.

Add:

1. `rl-check-determinism`
   - run same eval repeatedly
   - compare summaries and action sequences

2. `rl-replay-trace`
   - replay one trace with a chosen policy
   - emit step-by-step diff

3. `rl-compare-actions`
   - compare teacher vs BC vs APPO on fixed states

4. `rl-dagger-round`
   - roll out student
   - relabel visited states with teacher
   - append to dataset
   - optionally retrain BC

5. `rl-short-benchmark`
   - one command for the standard quick experiment:
     - train BC
     - short APPO warm-start
     - deterministic compare


## 8. Standard Iteration Loop

This should become the default workflow.

### Loop A: Representation Debug

1. change feature encoder or action mask
2. train BC on a fixed trace shard
3. run `rl-check-determinism`
4. run `rl-compare-actions`
5. run short deterministic BC eval

Expected runtime:

- minutes, not an hour


### Loop B: RL Improvement Check

1. start from a fixed BC checkpoint
2. run short APPO fine-tune
3. run deterministic BC vs APPO comparison
4. accept only if APPO beats BC on fixed seeds

Expected runtime:

- `10-20` minutes


### Loop C: Teacher Distribution Repair

1. roll out current student on fixed seeds
2. relabel visited states with `task_greedy`
3. retrain BC
4. compare teacher agreement and closed-loop reward

Expected runtime:

- bounded and cheap relative to full RL


## 9. Longer-Term Upgrade Path

Once the fast iteration loop is in place:

1. fix deterministic evaluation
2. add DAgger-style aggregation
3. improve observation encoder through BC-first sweeps
4. make APPO a policy-improvement stage over BC
5. only then revisit:
   - learned reward scaling
   - multi-skill APPO
   - hierarchical scheduler learning
   - world-model integration


## 10. Immediate Next Steps

In order:

1. implement `rl-check-determinism`
2. debug deterministic eval until repeated runs match
3. implement `rl-compare-actions`
4. add one `explore` golden replay case
5. implement one DAgger round for `explore`
6. rerun BC-first evaluation before any more large APPO jobs


## 11. Decision Rule

The repo should use the following rule for the next phase:

- if a change cannot beat the current baseline in the fast loop,
  it does not earn a larger RL run.

Current baselines to beat:

- teacher agreement on trace states
- BC closed-loop `explore` performance
- deterministic short-horizon `task_greedy` gap reduction

This is the right way to make the harness easier to debug while still moving
toward a stronger RL system.
