# X10 Rerun Validation Report

Date: 2026-04-05

## Purpose

Run another large APPO validation after the recent fast-debug-loop changes and
deterministic trace tooling improvements.

This run was meant to answer one question:

- does another x10-scale BC-warm-start APPO run materially improve the
  trusted deterministic trace benchmark?

## Summary

The full pipeline ran without code changes or crash fixes during the run.

That is good news operationally:

- trace verification passed
- BC artifact was still valid
- APPO launched cleanly
- APPO trained stably for `230,400` frames
- checkpointing worked
- deterministic trace evaluation worked on both latest and best checkpoints

But the scientific result was negative.

This rerun did **not** improve the trusted regression target:

- BC trace match rate stayed at `0.6395`
- APPO latest checkpoint trace match rate was only `0.4651`
- APPO best-by-training-reward checkpoint trace match rate was only `0.3837`

So in this rerun:

- scale alone did not help the trusted benchmark
- the training reward and live diagnostic reward improved
- but the policy became worse on teacher-state imitation

That is strong evidence that the RL objective is currently drifting away from
the teacher regression target rather than improving it.

## Commands Run

### 1. Revalidate the fast loop artifacts

```bash
uv run pytest -q tests/test_rl_scaffold.py

uv run python cli.py rl-verify-traces \
  --input data/tracefix_v2_explore_traces.jsonl

uv run python cli.py rl-evaluate-bc \
  --model output/tracefix_v2_explore_bc.pt \
  --trace-input data/tracefix_v2_explore_traces.jsonl
```

### 2. Launch the new x10-scale APPO rerun

```bash
CUDA_VISIBLE_DEVICES=2 uv run python cli.py rl-train-appo \
  --experiment appo_tracefix_v2_bc_x10_rerun \
  --num-workers 4 \
  --num-envs-per-worker 8 \
  --rollout-length 32 \
  --recurrence 16 \
  --batch-size 1024 \
  --num-batches-per-epoch 1 \
  --ppo-epochs 1 \
  --train-for-env-steps 500000 \
  --enabled-skills explore \
  --observation-version v2 \
  --bc-init-path output/tracefix_v2_explore_bc.pt
```

The run was stopped manually after `230,400` frames once it had:

- comfortably passed the previous stopping point,
- stayed in a positive-reward regime for a sustained period,
- and produced enough checkpoints for deterministic comparison.

### 3. Evaluate the latest APPO checkpoint on the trusted trace set

```bash
uv run python cli.py rl-trace-report \
  --input data/tracefix_v2_explore_traces.jsonl \
  --bc-model output/tracefix_v2_explore_bc.pt \
  --appo-experiment appo_tracefix_v2_bc_x10_rerun \
  --detailed \
  --top-k 8
```

### 4. Evaluate the focused east/south slice

```bash
uv run python cli.py rl-trace-disagreements \
  --input /tmp/tracefix_east_south.jsonl \
  --bc-model output/tracefix_v2_explore_bc.pt \
  --appo-experiment appo_tracefix_v2_bc_x10_rerun \
  --top-k 6
```

### 5. Evaluate the saved best training checkpoint explicitly

```bash
uv run python cli.py rl-trace-report \
  --input data/tracefix_v2_explore_traces.jsonl \
  --bc-model output/tracefix_v2_explore_bc.pt \
  --appo-experiment appo_tracefix_v2_bc_x10_rerun \
  --appo-checkpoint-path train_dir/rl/appo_tracefix_v2_bc_x10_rerun/checkpoint_p0/best_000000161_164864_reward_87.032.pth \
  --detailed \
  --top-k 8

uv run python cli.py rl-trace-disagreements \
  --input /tmp/tracefix_east_south.jsonl \
  --bc-model output/tracefix_v2_explore_bc.pt \
  --appo-experiment appo_tracefix_v2_bc_x10_rerun \
  --appo-checkpoint-path train_dir/rl/appo_tracefix_v2_bc_x10_rerun/checkpoint_p0/best_000000161_164864_reward_87.032.pth \
  --top-k 6
```

### 6. Live diagnostic evaluation

```bash
uv run python cli.py rl-evaluate-appo \
  --experiment appo_tracefix_v2_bc_x10_rerun \
  --seeds 42,43,44 \
  --max-steps 50

uv run python cli.py rl-evaluate-appo \
  --experiment appo_tracefix_v2_bc_x10_rerun \
  --checkpoint-path train_dir/rl/appo_tracefix_v2_bc_x10_rerun/checkpoint_p0/best_000000161_164864_reward_87.032.pth \
  --seeds 42,43,44 \
  --max-steps 50
```

## Artifacts

APPO experiment directory:

- [train_dir/rl/appo_tracefix_v2_bc_x10_rerun](/home/luc/rl-nethack/train_dir/rl/appo_tracefix_v2_bc_x10_rerun)

Checkpoint directory:

- [train_dir/rl/appo_tracefix_v2_bc_x10_rerun/checkpoint_p0](/home/luc/rl-nethack/train_dir/rl/appo_tracefix_v2_bc_x10_rerun/checkpoint_p0)

Relevant checkpoints:

- latest at stop time:
  [checkpoint_000000225_230400.pth](/home/luc/rl-nethack/train_dir/rl/appo_tracefix_v2_bc_x10_rerun/checkpoint_p0/checkpoint_000000225_230400.pth)
- earlier saved checkpoint:
  [checkpoint_000000209_214016.pth](/home/luc/rl-nethack/train_dir/rl/appo_tracefix_v2_bc_x10_rerun/checkpoint_p0/checkpoint_000000209_214016.pth)
- best-by-training-reward checkpoint:
  [best_000000161_164864_reward_87.032.pth](/home/luc/rl-nethack/train_dir/rl/appo_tracefix_v2_bc_x10_rerun/checkpoint_p0/best_000000161_164864_reward_87.032.pth)

Reference trace / BC artifacts:

- [data/tracefix_v2_explore_traces.jsonl](/home/luc/rl-nethack/data/tracefix_v2_explore_traces.jsonl)
- [output/tracefix_v2_explore_bc.pt](/home/luc/rl-nethack/output/tracefix_v2_explore_bc.pt)

## Results

### Baseline BC

Trusted deterministic trace eval on the corrected trace set remained:

- match rate: `0.6395`
- invalid action rate: `0.0`

This is the reference target for the APPO rerun.

### APPO training behavior

The rerun was stable and substantially positive by training reward.

Important points during training:

- early phase was strongly negative, as expected
- the run crossed into positive reward around `111k-113k` frames
- best observed training reward during the run:
  - `87.032` at `164,864` frames
- run stopped manually at:
  - `230,400` frames

This was a clean operational run. No code fixes were required mid-run.

### Latest APPO checkpoint on the trusted trace set

Latest checkpoint:

- `checkpoint_000000225_230400.pth`

Trusted deterministic trace results:

- match rate: `0.4651`
- invalid action rate: `0.0`
- predicted action counts:
  - `north: 60`
  - `east: 63`
  - `west: 130`
  - `south: 5`

This is materially worse than BC:

- BC: `0.6395`
- APPO latest: `0.4651`

The main failure mode is clear:

- APPO drifted hard toward `west`
- `south` recall collapsed to `0.0816`
- `west` recall rose to `0.92`, but precision dropped to `0.3538`

### Best-by-training-reward checkpoint on the trusted trace set

Best checkpoint:

- `best_000000161_164864_reward_87.032.pth`

Trusted deterministic trace results:

- match rate: `0.3837`
- invalid action rate: `0.0`
- predicted action counts:
  - `north: 111`
  - `west: 144`
  - `east: 1`
  - `south: 2`

This is even worse than the latest checkpoint on the trace benchmark.

Important implication:

- best training reward is **not** best trace behavior
- the current APPO objective is not aligned with teacher agreement

### Focused east/south slice

Slice file:

- `/tmp/tracefix_east_south.jsonl`

BC on the slice:

- match rate: `0.7083`

APPO latest on the slice:

- match rate: `0.55`
- `east` recall: `0.4444`
- `south` recall: `0.0417`
- `west` recall: `0.9375`

APPO best checkpoint on the slice:

- match rate: `0.4833`
- `east` recall: `0.0`
- `south` recall: `0.0`
- `west` recall: `0.9062`

This confirms the latest regression very clearly:

- the rerun over-optimized toward `west`
- and largely lost the ability to reproduce teacher `east` / `south`

### Live diagnostic evaluation

Latest checkpoint live eval on seeds `42,43,44`:

- avg task reward: `-23.45`
- avg unique tiles: `45.0`
- repeated action rate: `0.86`
- dominant action: `east`

Best checkpoint live eval on the same seeds:

- avg task reward: `42.9`
- avg unique tiles: `115.67`
- repeated action rate: `0.5`
- action mix:
  - `north: 72`
  - `west: 48`
  - `south: 25`
  - `east: 5`

These numbers look better than the deterministic trace benchmark, but live eval
is still diagnostic-only because seeded NLE reset is not reliable enough for
regression gating.

## Main Takeaways

1. The pipeline is operationally solid.
   - no crash
   - no checkpointing failure
   - no evaluation failure

2. The fast debug loop paid off.
   - the trace tools immediately showed that this rerun regressed on the
     trusted benchmark, despite good-looking training reward

3. More APPO scale alone is not the answer right now.
   - this rerun did not beat BC
   - and in fact produced worse teacher-state behavior than the prior repaired
     x10 run

4. The current RL objective is misaligned with the teacher target.
   - best training reward checkpoint was among the worst trace checkpoints
   - this is the strongest current signal in the project

5. The dominant failure mode is directional collapse.
   - latest checkpoint collapses toward `west`
   - earlier runs had `north` or `east` collapse
   - this suggests the policy is still exploiting a shallow movement bias rather
     than learning stable directional control

## Recommended Next Step

Do **not** launch another large APPO run immediately.

The next correct move is:

1. keep using the deterministic trace loop as the source of truth
2. improve the observation / action representation for directional control
3. add more teacher-guided constraints or DAgger-style aggregation before more
   RL scale
4. replace or eliminate the `fork()`-based counterfactual teacher path in
   [src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py)

The key lesson from this rerun is simple:

- the harness is now good enough to tell us when scaling fails
- and this scaling attempt failed on the trusted regression target
