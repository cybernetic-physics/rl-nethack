# Teacher-Regularized APPO Plan

## Goal

Improve the RL stack without relying on larger blind APPO runs.

The next iteration should directly address the current failure mode:

- BC teacher remains stronger on the trusted trace benchmark
- APPO improves training reward but drifts away from teacher-aligned behavior
- the current shaped reward is too weak and too incomplete to be the sole objective

So the next plan is:

1. add teacher-regularized APPO
2. add an `explore`-only episodic exploration bonus
3. validate both with the fast deterministic loop before any long run

## Current Diagnosis

The current repo has three important facts:

1. The repaired non-RNN BC warm-start path is operationally sound.
2. Deterministic trace evaluation is now the trusted regression harness.
3. The remaining bottleneck is objective alignment, not infrastructure.

What this means in practice:

- APPO is currently trained against hand-shaped scalar reward in [src/task_rewards.py](/home/luc/rl-nethack/src/task_rewards.py) via [rl/rewards.py](/home/luc/rl-nethack/rl/rewards.py).
- The metric we actually trust is deterministic teacher trace match in [rl/trace_eval.py](/home/luc/rl-nethack/rl/trace_eval.py).
- Those two objectives are not the same.

That is why runs can show:

- rising training reward
- worse teacher alignment

This is a textbook proxy-objective problem.

## Literature-Grounded Direction

The literature points to a hybrid solution:

- **Kickstarting**: keep teacher supervision active during RL, not just at initialization.
- **DAgger**: correct sequential distribution shift by relabeling student-induced states.
- **DQfD / R2D3**: demonstrations should remain part of the learning process, not only pretraining.
- **NGU / Agent57**: sparse hard exploration benefits from explicit novelty signals.
- **SkillHack / MaestroMotif**: long-horizon NetHack-like tasks benefit from skill structure and richer feedback than raw environment reward.

For this repo, the two most leverageful changes are:

1. teacher-regularized APPO
2. episodic exploration bonus for `explore`

These are the smallest changes that directly improve learning signal quality without rebuilding the whole stack again.

## Part 1: Teacher-Regularized APPO

### Intended Behavior

During APPO training, the policy should not be optimized only for RL return.

It should also be softly constrained toward the teacher policy on the states that the student actually visits.

This changes the optimization target from:

- maximize shaped return

to:

- maximize shaped return
- while staying behaviorally close to the BC teacher

### Why This Matters

BC warm start helps the initial policy.
But once APPO starts updating, that information is currently free to drift away.

Teacher regularization makes the teacher signal persistent instead of one-shot.

This is the most direct fix for the observed pattern:

- early APPO improvement
- later drift into `west`-biased or otherwise teacher-misaligned behavior

### Implementation Plan

#### 1. Add teacher-policy loading in the RL trainer path

Files likely involved:

- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
- [rl/bc_model.py](/home/luc/rl-nethack/rl/bc_model.py)

Add config/options:

- `--teacher-bc-path`
- `--teacher-loss-coef`
- `--teacher-loss-type` with initial choices:
  - `ce`
  - `kl`

The teacher should be loaded as a frozen BC policy.

#### 2. Compute teacher targets on student states

The regularization must be applied on actual student rollouts, not offline traces only.

For each sampled batch:

- encode the same observation vector used by the student
- run the frozen BC teacher
- compute teacher action distribution or argmax target

#### 3. Add teacher loss to learner update

Initial choice:

- cross-entropy from student logits to teacher argmax action

Later option:

- KL divergence to teacher distribution

Initial total loss:

`loss_total = loss_appo + alpha * loss_teacher`

with `alpha = teacher_loss_coef`

#### 4. Keep this non-RNN-only at first

Do not mix this with recurrent policy work yet.

The repaired non-RNN path is the current stable base.

#### 5. Save teacher-regularized run metadata

Every run report and plan file should record:

- teacher checkpoint path
- teacher loss type
- teacher loss coefficient

### Fast Validation Loop

Do not start with long runs.

Run short ladders:

- `10k` env steps
- `25k` env steps
- `50k` env steps

For each:

1. rank checkpoints by trace match
2. compare against BC baseline
3. inspect action histogram and directional recall

If teacher regularization works, expected signals are:

- slower drift from BC
- better sustained trace match
- less directional collapse

### First Sweep

Narrow sweep only:

- `teacher_loss_coef`: `0.05`, `0.1`, `0.25`, `0.5`
- `teacher_loss_type`: start with `ce` only
- keep all other APPO settings fixed

Success criterion:

- beat current repaired APPO line on deterministic trace match
- ideally beat BC or at least match BC longer before drift

## Part 2: Explore-Only Episodic Bonus

### Intended Behavior

The current shaped `explore` reward gives some novelty signal, but it is still too local and too brittle.

We should add a lightweight episodic exploration bonus that rewards visiting less-frequent states or tiles within an episode.

This is not meant to replace the shaped reward.
It is meant to improve gradient quality for exploration behavior.

### Why This Matters

NetHack exploration is sparse and long-horizon.

A policy can get stuck in directional habits even when the proxy reward looks acceptable.
An episodic novelty bonus should help preserve frontier-seeking behavior and reduce shallow repetitive moves.

### Design Constraints

Keep it simple and local to the `explore` skill.

Do not add a general intrinsic-reward framework yet.

The bonus should:

- apply only when active skill is `explore`
- be cheap to compute
- be easy to disable
- integrate cleanly with current reward accounting

### Initial Bonus Proposal

Use existing state hashing from [src/task_rewards.py](/home/luc/rl-nethack/src/task_rewards.py) or timestep/path tracking from [rl/env_adapter.py](/home/luc/rl-nethack/rl/env_adapter.py).

For each episode:

- maintain count of visited hashed states or positions
- compute bonus like:
  - `bonus = 1 / sqrt(count)`
  - or `bonus = 1.0 if first_visit else 0.0`

Start conservative.

Candidate initial version:

- state-hash or tile-visit episodic bonus
- only for `explore`
- zero for other skills

### Implementation Plan

Files likely involved:

- [rl/env_adapter.py](/home/luc/rl-nethack/rl/env_adapter.py)
- [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py)
- [rl/config.py](/home/luc/rl-nethack/rl/config.py)
- possibly [rl/rewards.py](/home/luc/rl-nethack/rl/rewards.py)

Add config/options:

- `episodic_explore_bonus_enabled`
- `episodic_explore_bonus_scale`
- `episodic_explore_bonus_mode`
  - `state_hash`
  - `tile`

### Fast Validation Loop

Before APPO:

1. run short deterministic rollouts with the current teacher and BC policy
2. log episodic bonus values
3. verify it actually rewards frontier progress rather than random jitter

Then in short APPO ladders:

- compare with and without episodic bonus
- keep teacher regularization fixed

## Implementation Status (2026-04-05)

Both planned pieces are now implemented in the repo:

- teacher-regularized APPO
- `explore`-only episodic exploration bonus

The teacher-regularization path is live through:

- [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py)
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
- [cli.py](/home/luc/rl-nethack/cli.py)

The episodic bonus path is live through:

- [rl/env_adapter.py](/home/luc/rl-nethack/rl/env_adapter.py)
- [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py)
- [rl/config.py](/home/luc/rl-nethack/rl/config.py)
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
- [cli.py](/home/luc/rl-nethack/cli.py)

Current results on the trusted deterministic trace benchmark:

- BC teacher on [data/tracefix_v2_explore_traces.jsonl](/home/luc/rl-nethack/data/tracefix_v2_explore_traces.jsonl): `0.6395`
- teacher regularization only, `alpha=0.01`, `10k`: `0.6085`
- teacher regularization only, `alpha=0.01`, `20k`: `0.6124`
- teacher regularization + state-hash bonus `0.05`, `10k`: `0.5698`
- teacher regularization + state-hash bonus `0.01`, `10k`: `0.6085`
- teacher regularization + tile bonus `0.01`, `10k`: `0.5969`
- teacher regularization + state-hash bonus `0.005`, `10k`: `0.5891`

Current conclusion:

- Part 1 is functional and mildly promising only in the very small-loss regime.
- Part 2 is implemented and debugged, but the initial episodic bonus variants do not improve the trusted regression metric yet.
- The bonus path should remain available but disabled by default while the next experiments focus on better alignment and feature quality.

Expected positive signs:

- less repeated action bias
- better `east`/`south` recall on teacher slices
- higher sustained trace match at later checkpoints

## Combined Experiment Matrix

Keep the matrix small.

### Stage A: Teacher Regularization Only

Runs:

- base repaired APPO
- `alpha=0.05`
- `alpha=0.1`
- `alpha=0.25`
- `alpha=0.5`

Budget:

- `10k`, `25k`, `50k` only

Gate:

- deterministic trace match

### Stage B: Add Episodic Bonus to Best Teacher-Reg Setting

Runs:

- best Stage A config
- best Stage A + novelty scale `0.05`
- best Stage A + novelty scale `0.1`
- best Stage A + novelty scale `0.25`

Budget:

- `25k`, `50k`

Gate:

- deterministic trace match
- directional recall on `east/south` trace slice

### Stage C: Longer Validation

Only after short runs show a real gain:

- run one x10 validation
- preserve `best_trace_match.pth`
- stop early if trace score regresses clearly

## Fast Debug Loop

This work should use the fast loop, not blind long training.

### Required Commands

Deterministic trace benchmark:

```bash
uv run python cli.py rl-trace-report \
  --input data/tracefix_v2_explore_traces.jsonl \
  --bc-model output/tracefix_v2_explore_bc.pt \
  --appo-experiment <experiment>
```

Checkpoint ranking:

```bash
uv run python cli.py rl-rank-checkpoints \
  --experiment <experiment> \
  --trace-input data/tracefix_v2_explore_traces.jsonl \
  --materialize-best-trace
```

Focused directional slice:

```bash
uv run python cli.py rl-shard-traces \
  --input data/tracefix_v2_explore_traces.jsonl \
  --output /tmp/east_south.jsonl \
  --teacher-actions east,south \
  --max-episodes 6
```

Detailed disagreement audit:

```bash
uv run python cli.py rl-trace-disagreements \
  --input /tmp/east_south.jsonl \
  --bc-model output/tracefix_v2_explore_bc.pt \
  --appo-experiment <experiment>
```

### Required Regression Checks

For every experiment:

1. aggregate trace match
2. per-action recall
3. action histogram
4. invalid-action rate
5. whether best-trace checkpoint is preserved

## Success Criteria

The next iteration counts as a success if:

1. teacher-regularized APPO beats the current repaired APPO baseline on deterministic trace match
2. the improvement is sustained beyond the earliest checkpoint
3. directional collapse is reduced
4. the best-trace checkpoint is preserved automatically

Stretch goal:

- match or beat BC on the trusted trace benchmark

## Out of Scope For This Iteration

Do not mix these in yet:

- recurrent policy revival
- large scheduler changes
- full learned reward-model replacement
- Go-Explore-style archive methods
- world-model redesign

Those may be good later, but they are not the highest-leverage next step.

## Recommended Execution Order

1. implement teacher BC loading and teacher loss in APPO learner
2. run short teacher-loss sweeps
3. materialize best-trace checkpoint on every ranking pass
4. add explore-only episodic bonus
5. run short combined sweeps
6. only then run one longer validation
