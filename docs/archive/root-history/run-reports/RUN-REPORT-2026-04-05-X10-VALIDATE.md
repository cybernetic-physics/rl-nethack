# Run Report: 2026-04-05 X10 Validation

## Goal

Validate the repaired non-RNN BC-warm-start APPO path with an x10-scale run and use the deterministic trace benchmark as the source of truth.

Primary gates:

- BC teacher trace match on the trusted trace set: `0.6395`
- Prior repaired APPO best on the trusted trace set: `0.6240`

Trusted trace dataset:

- [data/tracefix_v2_explore_traces.jsonl](/home/luc/rl-nethack/data/tracefix_v2_explore_traces.jsonl)

## Command

```bash
CUDA_VISIBLE_DEVICES=2 uv run python cli.py rl-train-appo \
  --experiment appo_tracefix_v2_bc_nornn_x10_validate \
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
  --bc-init-path output/tracefix_v2_explore_bc.pt \
  --no-rnn
```

Training was intentionally stopped early once the deterministic trace metric clearly peaked and then regressed.

Actual stop point:

- `101,376` env frames

## Validation Commands

Checkpoint ranking:

```bash
uv run python cli.py rl-rank-checkpoints \
  --experiment appo_tracefix_v2_bc_nornn_x10_validate \
  --trace-input data/tracefix_v2_explore_traces.jsonl \
  --top-k 10
```

Compact deterministic report:

```bash
uv run python cli.py rl-trace-report \
  --input data/tracefix_v2_explore_traces.jsonl \
  --bc-model output/tracefix_v2_explore_bc.pt \
  --appo-experiment appo_tracefix_v2_bc_nornn_x10_validate
```

Best-trace materialization fix applied after the run:

```bash
uv run python cli.py rl-rank-checkpoints \
  --experiment appo_tracefix_v2_bc_nornn_x10_validate \
  --trace-input data/tracefix_v2_explore_traces.jsonl \
  --top-k 5 \
  --materialize-best-trace
```

## Artifacts

Experiment:

- [train_dir/rl/appo_tracefix_v2_bc_nornn_x10_validate](/home/luc/rl-nethack/train_dir/rl/appo_tracefix_v2_bc_nornn_x10_validate)

Key checkpoints still present after retention:

- [checkpoint_000000096_98304.pth](/home/luc/rl-nethack/train_dir/rl/appo_tracefix_v2_bc_nornn_x10_validate/checkpoint_p0/checkpoint_000000096_98304.pth)
- [checkpoint_000000099_101376.pth](/home/luc/rl-nethack/train_dir/rl/appo_tracefix_v2_bc_nornn_x10_validate/checkpoint_p0/checkpoint_000000099_101376.pth)
- [best_000000098_100352_reward_14.984.pth](/home/luc/rl-nethack/train_dir/rl/appo_tracefix_v2_bc_nornn_x10_validate/checkpoint_p0/best_000000098_100352_reward_14.984.pth)

Best-trace alias created after fixing the retention hole:

- [best_trace_match.pth](/home/luc/rl-nethack/train_dir/rl/appo_tracefix_v2_bc_nornn_x10_validate/checkpoint_p0/best_trace_match.pth)
- [best_trace_match.json](/home/luc/rl-nethack/train_dir/rl/appo_tracefix_v2_bc_nornn_x10_validate/checkpoint_p0/best_trace_match.json)

Teacher BC checkpoint:

- [output/tracefix_v2_explore_bc.pt](/home/luc/rl-nethack/output/tracefix_v2_explore_bc.pt)

## Results

### Baseline

From the trusted trace report:

- BC trace match: `0.6395`

### APPO Trace Ranking During the Run

Early checkpoints:

- `checkpoint_000000000_0.pth`: `0.6047`
- `checkpoint_000000020_20480.pth`: `0.6085`
- `checkpoint_000000045_46080.pth`: `0.6318`

This `46k`-frame checkpoint was the best deterministic result observed in the run and was:

- better than the repaired warm start
- better than the prior repaired APPO best (`0.6240`)
- still below the BC teacher (`0.6395`)

Later checkpoints regressed while training reward kept rising:

- `checkpoint_000000071_72704.pth`: `0.6240`
- `checkpoint_000000096_98304.pth`: `0.6085`
- `checkpoint_000000099_101376.pth`: `0.6085`
- `best_000000098_100352_reward_14.984.pth`: `0.6124`

### Final Trusted Summary

From the final deterministic trace report, using the default retained APPO checkpoint:

- BC trace match: `0.6395`
- APPO trace match: `0.6085`

So the run validated the repaired path but did not surpass the teacher.

## What Worked

- The non-RNN BC warm-start path trained cleanly.
- The deterministic trace ranking caught the actual best part of the run.
- The run improved over the previous repaired APPO line early in training.
- GPU isolation on GPU `2` worked cleanly.
- The repaired fast loop let us stop early once the trusted metric drifted.

## What Did Not Work

- Training reward improvement did not align with trace-match improvement.
- The deterministic metric peaked early and then regressed.
- The run still did not beat the BC teacher.
- Live seed-based evaluation remains unsuitable for regression gating.

## Bugs Found During This Run

### 1. Best-by-trace checkpoint was not preserved

Problem:

- The best deterministic checkpoint in this run was `checkpoint_000000045_46080.pth`.
- Sample Factory checkpoint retention later deleted it.
- That meant the best checkpoint by the trusted metric could not be reloaded after the run.

Fix:

- Added stable best-trace checkpoint materialization in the ranking path.
- New alias files:
  - `checkpoint_p0/best_trace_match.pth`
  - `checkpoint_p0/best_trace_match.json`

Important caveat:

- This fix was applied after the run, so the deleted `46k` checkpoint itself could not be recovered.
- The current alias preserves the best surviving checkpoint, not the historical peak from this run.

### 2. Live eval path remains unreliable for fast gating

Problem:

- The live seed-based APPO eval path still behaves like an old diagnostic path rather than a clean regression harness.

Decision:

- Continue using trace-based deterministic evaluation as the gating path.

## Conclusion

This was a successful validation run in the engineering sense:

- the repaired non-RNN BC warm-start APPO path trains reliably
- the fast trace-based debug loop works
- the run improved over earlier repaired APPO checkpoints at its peak

But it was not a scientific win yet:

- APPO still did not beat BC
- the trusted metric peaked at `0.6318` and then drifted down to about `0.61`

The current conclusion is unchanged but sharper:

- the harness is finally good enough to trust
- the remaining problem is objective alignment, not infrastructure
- more raw scale alone is still not enough

## Recommended Next Step

Do not launch a larger blind APPO run next.

Do this instead:

1. keep the non-RNN BC warm-start path
2. add teacher-regularized APPO
3. select checkpoints by deterministic trace match
4. preserve `best_trace_match.pth` during every run

