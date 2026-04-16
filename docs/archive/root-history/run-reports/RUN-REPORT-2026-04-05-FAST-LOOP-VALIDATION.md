# Run Report: Fast Loop Validation

Date: 2026-04-05

This report documents a fresh bounded validation run after adding the new
fast-iteration tooling:

- [FAST-ITERATION-PLAN.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/FAST-ITERATION-PLAN.md)
- [rl/debug_tools.py](/home/luc/rl-nethack/rl/debug_tools.py)
- new CLI commands in [cli.py](/home/luc/rl-nethack/cli.py):
  - `rl-check-determinism`
  - `rl-compare-actions`
  - `rl-short-benchmark`

The goal of this run was not only to train another RL checkpoint, but to
validate that:

1. the new fast-debug loop is operational,
2. the current BC/APPO path still runs end to end,
3. the new tooling gives actionable diagnostics on the current failure modes.


## Goal

Run a fresh bounded `v2` pipeline:

1. generate fresh multi-turn `explore` traces from `task_greedy`
2. train a fresh `v2` BC policy
3. run the new short benchmark and debug commands
4. run a larger BC-warm-start APPO training job
5. evaluate BC and APPO with the new comparison tools
6. document what improved and what still failed


## Fresh Artifacts

Trace shard:

- [data/validate_v2_explore_traces.jsonl](/home/luc/rl-nethack/data/validate_v2_explore_traces.jsonl)

Fresh BC checkpoint:

- [output/validate_v2_explore_bc.pt](/home/luc/rl-nethack/output/validate_v2_explore_bc.pt)

Fresh APPO experiment:

- [train_dir/rl/appo_validate_v2_bc_large](/home/luc/rl-nethack/train_dir/rl/appo_validate_v2_bc_large)
- final checkpoint:
  [train_dir/rl/appo_validate_v2_bc_large/checkpoint_p0/checkpoint_000000050_51200.pth](/home/luc/rl-nethack/train_dir/rl/appo_validate_v2_bc_large/checkpoint_p0/checkpoint_000000050_51200.pth)

Comparison snapshot:

- [output/validate_v2_compare.json](/home/luc/rl-nethack/output/validate_v2_compare.json)


## Run Commands

### 1. Generate fresh bounded `v2` traces

Started with:

```bash
uv run python cli.py rl-generate-traces \
  --output data/validate_v2_explore_traces.jsonl \
  --num-episodes 20 \
  --max-steps 20 \
  --task explore \
  --policy task_greedy \
  --observation-version v2
```

Because `task_greedy` is still the slow fork-per-action teacher, I cut the
run once the shard was large enough for a meaningful BC/APPO pass.

Verified shard:

```bash
uv run python cli.py rl-verify-traces \
  --input data/validate_v2_explore_traces.jsonl
```

Verified stats:

- episodes: `6`
- rows: `119`
- max steps in episode: `20`
- avg steps in episode: `19.83`
- all episodes multi-turn: `true`


### 2. Train fresh `v2` BC checkpoint

```bash
uv run python cli.py rl-train-bc \
  --input data/validate_v2_explore_traces.jsonl \
  --output output/validate_v2_explore_bc.pt \
  --epochs 40 \
  --lr 0.001 \
  --hidden-size 512 \
  --observation-version v2
```

Result:

- examples: `119`
- input dim: `160`
- hidden size: `512`
- final loss: `0.5915`
- train accuracy: `0.7647`


### 3. Validate the new short loop

```bash
uv run python cli.py rl-short-benchmark \
  --input data/validate_v2_explore_traces.jsonl \
  --output /tmp/validate_short_bc.pt \
  --task explore \
  --epochs 5 \
  --lr 0.001 \
  --hidden-size 256 \
  --observation-version v2 \
  --seeds 42 \
  --max-steps 5 \
  --repeats 2
```

Result:

- the command worked end to end
- it trained BC
- it evaluated BC
- it ran repeated determinism checks
- it ran teacher-action comparison

This validates the new fast loop as a working tool, not just a plan.


### 4. Large BC-warm-start APPO run

Run on GPU `2` only:

```bash
CUDA_VISIBLE_DEVICES=2 uv run python cli.py rl-train-appo \
  --experiment appo_validate_v2_bc_large \
  --num-workers 4 \
  --num-envs-per-worker 8 \
  --rollout-length 32 \
  --recurrence 16 \
  --batch-size 1024 \
  --num-batches-per-epoch 1 \
  --ppo-epochs 1 \
  --train-for-env-steps 50000 \
  --enabled-skills explore \
  --observation-version v2 \
  --bc-init-path output/validate_v2_explore_bc.pt
```

Result:

- run completed successfully
- final checkpoint:
  [train_dir/rl/appo_validate_v2_bc_large/checkpoint_p0/checkpoint_000000050_51200.pth](/home/luc/rl-nethack/train_dir/rl/appo_validate_v2_bc_large/checkpoint_p0/checkpoint_000000050_51200.pth)
- collected frames: `51,200`
- final reported FPS: `190.7`

Training dynamics:

- warm-start loaded correctly
- early return still fell negative quickly
- later in training, mean shaped return partially recovered from the worst
  collapse
- final average episode reward in the training log was still negative:
  around `-104.266`

So this run was healthier than the earlier bad APPO run, but still not a good
policy.


## Post-Run Evaluation

### Fresh BC evaluation

Command:

```bash
uv run python cli.py rl-evaluate-bc \
  --model output/validate_v2_explore_bc.pt \
  --task explore \
  --seeds 42,43,44 \
  --max-steps 20
```

Result:

- avg unique tiles: `47.67`
- avg rooms discovered: `1.0`
- avg env reward: `0.0`
- action mix:
  - `east 35`
  - `south 18`
  - `north 4`
  - `west 3`

Interpretation:

- BC is still movement-biased
- but it is not completely collapsed to a single action


### Fresh APPO evaluation

Command:

```bash
uv run python cli.py rl-evaluate-appo \
  --experiment appo_validate_v2_bc_large \
  --seeds 42,43,44 \
  --max-steps 20
```

Result:

- avg task reward: `-0.55`
- avg unique tiles: `84.33`
- avg rooms discovered: `1.0`
- repeated action rate: `0.7167`
- invalid action rate: `0.0`
- action mix:
  - `west 37`
  - `north 12`
  - `south 11`

Interpretation:

- this is much better than the previous strongly negative APPO run
- strict action masking is still working
- APPO still has a serious repetition problem
- APPO is still not a trustworthy replacement for the teacher


### Joint policy comparison

Command:

```bash
uv run python cli.py rl-compare-policies \
  --task explore \
  --seeds 42,43,44 \
  --max-steps 20 \
  --bc-model output/validate_v2_explore_bc.pt \
  --appo-experiment appo_validate_v2_bc_large \
  --output output/validate_v2_compare.json
```

Snapshot result:

- `task_greedy`
  - avg task reward: `25.9667`
  - avg unique tiles: `36.0`
  - repeated action rate: `0.1333`
- `bc`
  - avg unique tiles: `49.0`
  - avg env reward: `0.6667`
- `appo`
  - avg task reward: `-1.45`
  - avg unique tiles: `58.0`
  - repeated action rate: `0.7833`

Interpretation:

- the teacher still wins clearly on the repo’s shaped reward
- APPO covers more tiles than BC in this snapshot, but does it in a much more
  repetitive and less teacher-aligned way
- APPO is no longer catastrophically worse than before, but it is still not
  actually good


### Teacher-state action comparison

Command:

```bash
uv run python cli.py rl-compare-actions \
  --task explore \
  --seeds 42,43,44 \
  --max-steps 20 \
  --bc-model output/validate_v2_explore_bc.pt \
  --appo-experiment appo_validate_v2_bc_large \
  --appo-train-dir train_dir/rl \
  --observation-version v2
```

Result:

- teacher vs BC match rate: `0.2667`
- teacher vs APPO match rate: `0.2333`

Interpretation:

- APPO did not improve teacher-state agreement over BC
- this is the strongest evidence from this run that RL is still not improving
  the policy in the right way


### Determinism checks

BC:

```bash
uv run python cli.py rl-check-determinism \
  --policy bc \
  --task explore \
  --seeds 42,43,44 \
  --max-steps 20 \
  --repeats 2 \
  --bc-model output/validate_v2_explore_bc.pt \
  --observation-version v2
```

APPO:

```bash
uv run python cli.py rl-check-determinism \
  --policy appo \
  --task explore \
  --seeds 42 \
  --max-steps 10 \
  --repeats 2 \
  --appo-experiment appo_validate_v2_bc_large \
  --appo-train-dir train_dir/rl
```

Result:

- both were still unstable
- BC repeated runs diverged from step `0`
- APPO repeated runs diverged by step `6`
- the same seed can still produce different initial states and trajectories in
  raw NLE eval paths

This confirms the same top-priority issue found earlier:

- evaluation nondeterminism is still real
- the new debug tooling correctly exposes it


## What Worked

1. The new fast-debug commands are real and useful.
2. A fresh `v2` trace -> BC -> APPO run completed successfully.
3. BC warm-start still works correctly.
4. APPO throughput on GPU `2` is good enough for fast iteration.
5. Strict invalid-action masking is still working.
6. This APPO run was materially better than the earlier very bad APPO run on
   shaped reward.


## What Did Not Work

1. Evaluation determinism is still broken.
2. APPO still does not beat the teacher.
3. APPO does not improve teacher-state agreement over BC.
4. Repetition/collapse is still the main low-level control failure mode.
5. The repo is still better at producing a working supervised bootstrap than at
   converting that bootstrap into a better RL policy.


## Main Conclusion

This run validated the new tooling and the current diagnosis.

The repo is now in a better engineering state:

- fast iteration is possible
- full pipeline runs are operational
- the failure mode is clearer

But the core scientific result did not change:

- BC is a usable bootstrap
- APPO from that bootstrap is not yet a policy improvement stage
- the next highest-value work is still:
  - fix deterministic evaluation
  - improve the representation / action-space constraints
  - add DAgger-style teacher relabeling


## Recommended Next Step

Do **not** spend the next cycle on a larger APPO hyperparameter sweep.

Do this instead:

1. use `rl-check-determinism` to isolate the NLE reset/eval instability
2. implement one DAgger round for `explore`
3. rerun BC and teacher-state agreement before any more large RL jobs

That is the most defensible next move from the evidence in this run.
