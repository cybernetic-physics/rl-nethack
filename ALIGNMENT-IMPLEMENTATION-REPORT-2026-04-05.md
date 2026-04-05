# Alignment Plan Implementation Report (2026-04-05)

This document records the implementation status and observed behavior for the five steps in
[ALIGNMENT-IMPROVEMENT-PLAN.md](./ALIGNMENT-IMPROVEMENT-PLAN.md).

The main outcome is:

- all five planned steps are now implemented in code
- steps 1-4 are wired into the active RL/BC workflow
- step 5 is implemented as a working experimental baseline, not yet the mainline improver
- the fast debug loop is materially stronger than before

## Step 1: Real DAgger Schedule

Status: implemented and debugged

Code:
- [rl/dagger.py](./rl/dagger.py)
- [rl/traces.py](./rl/traces.py)
- [cli.py](./cli.py)

What changed:
- added `run_dagger_schedule(...)`
- added iterative command `rl-dagger-iterate`
- added controlled merge policies:
  - `base_only`
  - `uniform_merge`
  - `weighted_recent`
- added optional held-out trace evaluation after each iteration

Validated behavior:
- the schedule runs end to end
- each iteration writes:
  - relabeled DAgger traces
  - merged trace set
  - retrained BC model
  - deterministic base/heldout trace evaluations

Observed result on a cheap `v3` validation:
- base `v3` BC held-out trace match: `0.6500`
- DAgger schedule with `weighted_recent`, `merge_ratio=0.75`, 2 iterations:
  - iteration 0 held-out trace match: `0.5750`
  - iteration 1 held-out trace match: `0.5125`

Conclusion:
- the iterative DAgger loop is mechanically correct
- naive small-schedule DAgger is currently harmful
- the next use of this path should be schedule tuning, not immediate promotion to default

## Step 2: Trace-Match Checkpoint Selection Inside Training

Status: implemented and validated

Code:
- [rl/checkpoint_tools.py](./rl/checkpoint_tools.py)
- [rl/trainer.py](./rl/trainer.py)
- [rl/train_appo.py](./rl/train_appo.py)
- [rl/evaluate.py](./rl/evaluate.py)
- [cli.py](./cli.py)

What changed:
- APPO training can now monitor checkpoints against a trusted trace dataset during training
- new config and CLI support:
  - `--trace-eval-input`
  - `--trace-eval-interval-env-steps`
  - `--trace-eval-top-k`
- best-by-trace alias is written during training:
  - `checkpoint_p0/best_trace_match.pth`
  - `checkpoint_p0/best_trace_match.json`

Validated behavior:
- smoke experiment `appo_trace_eval_smoke` wrote:
  - `train_dir/rl/appo_trace_eval_smoke/checkpoint_p0/best_trace_match.pth`
  - `train_dir/rl/appo_trace_eval_smoke/checkpoint_p0/best_trace_match.json`
- recorded best trace checkpoint:
  - env steps: `2176`
  - match rate: `0.6202`

Conclusion:
- best-trace checkpoint selection is no longer post-hoc only
- this closes a real gap in the previous training loop

## Step 3: Richer Directional / Frontier Representation (`v3`)

Status: implemented, debugged, and validated on real traces

Code:
- [rl/feature_encoder.py](./rl/feature_encoder.py)
- [rl/timestep.py](./rl/timestep.py)
- [rl/traces.py](./rl/traces.py)
- [tests/test_rl_scaffold.py](./tests/test_rl_scaffold.py)

What changed:
- added `v3` observation representation
- `v3` expands the encoder from `160` dims to `244` dims
- adds:
  - 3x3 local map patch categories
  - directional ray features
  - richer local geometric cues

Important bug fixed during validation:
- the trace generator failed for non-APPO policies because `rnn_states` was uninitialized
- the trace row writer also depended on `active_skill` and `allowed_actions` without defining them in that frame
- both issues are fixed in [rl/traces.py](./rl/traces.py)

Validated behavior:
- `task_greedy` trace generation now works with `v3`
- held-out `v3` teacher traces were generated cleanly:
  - train set: `200` rows
  - held-out set: `80` rows
  - all multi-turn
  - all `v3`
  - feature dim `244`

Cheap `v3` BC validation:
- train accuracy: `0.8800`
- held-out deterministic trace match: `0.6500`

Focused held-out disagreement slice:
- strong `east` recall: `0.8889`
- weak `south` recall: `0.3636`
- zero `west` recall on the small slice

Conclusion:
- the `v3` path is real and operational
- representation is improved enough to validate on held-out traces
- the remaining directional weakness is now clearer: `south` and `west` still need work

## Step 4: Teacher-Regularized APPO as the Default RL Improver

Status: implemented earlier, kept as the baseline, now integrated with in-training trace gating

Code:
- [rl/teacher_reg.py](./rl/teacher_reg.py)
- [rl/train_appo.py](./rl/train_appo.py)
- [rl/trainer.py](./rl/trainer.py)
- [rl/evaluate.py](./rl/evaluate.py)
- [cli.py](./cli.py)

What changed in this pass:
- teacher-reg remains wired as the default RL-improvement path
- `rl-train-appo` now defaults to:
  - non-RNN mainline unless `--use-rnn` is set
  - teacher path inferred from `--bc-init-path` when `--teacher-bc-path` is not provided
- added `rl-teacher-reg-report` for fast comparison of:
  - BC teacher
  - latest APPO checkpoint
  - best-trace APPO checkpoint

Validated behavior:
- report command works on real experiments
- current real comparison for `appo_teacher_reg_001_x100` on the trusted trace set:
  - BC teacher: `0.6395`
  - latest APPO: `0.6318`
  - best-trace APPO: `0.6318`

Conclusion:
- teacher-reg APPO remains the strongest RL improvement path in the repo
- it is still below BC, but now close enough that representation and data-alignment changes are the main levers

## Step 5: Behavior-Regularized Improvement

Status: implemented as an experimental path and validated on held-out traces

Code:
- [rl/train_behavior_reg.py](./rl/train_behavior_reg.py)
- [cli.py](./cli.py)
- [BEHAVIOR-REG-IMPROVEMENT.md](./BEHAVIOR-REG-IMPROVEMENT.md)

What changed:
- added `rl-train-behavior-reg`
- current implementation trains a conservative policy on trace data with:
  - imitation loss
  - KL regularization toward empirical behavior prior

Validated behavior on cheap `v3` run:
- train accuracy: `0.8750`
- base trace match: `0.8750`
- held-out trace match: `0.6375`

Interpretation:
- this simple behavior-regularized baseline is already competitive with the current BC teacher line
- it should still be treated as experimental:
  - it is not yet an AWAC/BRAC-style full offline-to-online improver
  - it has not yet been integrated into the APPO improvement stage

Conclusion:
- step 5 is implemented enough to support fast experimentation
- it is the right next algorithmic branch if teacher-reg APPO still stalls

## Fast-Loop Outcome

The repo now has a substantially better debug loop than before:

1. modify traces / features / DAgger / reward code
2. run deterministic BC or behavior-reg trace eval
3. run held-out trace disagreements
4. only then run short teacher-reg APPO
5. rely on in-training best-trace checkpoint selection instead of reward-best checkpoints

This is the main structural improvement from the implementation pass.

## What Worked

- trace-gated checkpointing works
- `v3` traces work
- `v3` BC works on held-out traces
- DAgger schedule mechanics work
- behavior-regularized training path works

## What Still Does Not Work

- naive DAgger schedule degrades the policy
- directional confusion is still concentrated in `south` and `west`
- teacher-reg APPO still does not clearly beat BC
- the `fork()`-based teacher path in [src/task_harness.py](./src/task_harness.py) remains a structural smell

## Recommended Next Moves

1. tune DAgger schedule on held-out `v3` traces before using it in large RL runs
2. improve `v3` directional cues specifically for `south` / `west`
3. compare `v3` BC against behavior-regularized training on the same fixed held-out shards
4. keep teacher-reg APPO as the online improver
5. only promote behavior-regularized improvement to the mainline if it consistently beats BC on held-out traces
