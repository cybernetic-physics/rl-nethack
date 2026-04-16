# Fast Debug Loop Report (2026-04-05)

This note captures the current state of the RL fast-iteration tooling after the
trace-eval and trace-diagnostics fixes.

## What Changed

The fast debug loop now has two new deterministic tools:

- `rl-shard-traces`
  - writes a small, fixed subset of an existing trace dataset
  - preserves full episodes instead of cutting rows mid-episode
- `rl-trace-disagreements`
  - evaluates BC and/or APPO on a fixed trace dataset
  - reports:
    - overall match rate
    - invalid-action rate
    - teacher action histogram
    - predicted action histogram
    - common confusion pairs
    - per-teacher-action match rates

`rl-trace-report` can also now include the same disagreement breakdown via
`--detailed`.

## Why This Matters

Before this patch, the deterministic trace loop told us only that:

- BC was better than APPO on the trusted trace dataset
- APPO improved with scale but still trailed the teacher

That was useful, but still too coarse. The new disagreement report makes the
remaining policy gap diagnosable instead of just measurable.

## Commands Used

### Full corrected trace disagreement report

```bash
uv run python cli.py rl-trace-disagreements \
  --input data/tracefix_v2_explore_traces.jsonl \
  --bc-model output/tracefix_v2_explore_bc.pt \
  --appo-experiment appo_tracefix_v2_bc_x10 \
  --top-k 8
```

### Small reproducible shard

```bash
uv run python cli.py rl-shard-traces \
  --input data/tracefix_v2_explore_traces.jsonl \
  --output /tmp/tracefix_shard_ep3.jsonl \
  --max-episodes 3
```

### Compact shard report with detailed disagreements

```bash
uv run python cli.py rl-trace-report \
  --input /tmp/tracefix_shard_ep3.jsonl \
  --bc-model output/tracefix_v2_explore_bc.pt \
  --appo-experiment appo_tracefix_v2_bc_x10 \
  --detailed \
  --top-k 5
```

## Key Results

### Full corrected trace set

Trace file:

- `data/tracefix_v2_explore_traces.jsonl`

Trusted deterministic results:

- BC match rate: `0.6395`
- APPO match rate: `0.5853`
- both invalid-action rates: `0.0`

Teacher action histogram:

- `north: 75`
- `east: 82`
- `south: 49`
- `west: 50`
- `search: 2`

### BC confusion profile

BC is strongest on:

- `north`: `0.7867`
- `west`: `0.58`

BC is weaker on:

- `east`: `0.6220`
- `south`: `0.5306`
- `search`: `0.0` on two rare examples

Most common BC confusions:

- `east -> north`: `22`
- `west -> north`: `12`
- `south -> east`: `11`
- `south -> north`: `10`
- `north -> east`: `9`

### APPO confusion profile

APPO is strongest on:

- `north`: `0.6933`
- `east`: `0.6463`

APPO is weakest on:

- `south`: `0.3673`
- `west`: `0.56`
- `search`: `0.0` on two rare examples

Most common APPO confusions:

- `south -> east`: `18`
- `east -> north`: `17`
- `north -> east`: `14`
- `west -> north`: `12`
- `south -> north`: `10`

## What We Learned

The remaining gap is not uniform.

Both policies are directionally biased, but in different ways:

- BC still over-predicts `north`
- APPO has a particularly bad `south` problem
- both policies are materially worse on `east` and `south` than on `north`
- `search` is effectively not learned at all yet in the current trace set

That means the next debugging target should not be “general policy collapse.”
It should be:

1. directional ambiguity in the observation/features
2. whether recent-position / repeated-state features are sufficient to encode
   heading and local geometry
3. whether teacher traces underrepresent the conditions that make `east` and
   `south` distinct from `north`

## Small-Shard Result

Shard file:

- `/tmp/tracefix_shard_ep3.jsonl`

This small shard produced a stable, quick regression loop:

- BC match rate: `0.7333`
- APPO match rate: `0.7000`

The shard reproduces the same main failure mode:

- `east` remains the weakest directional label for both models

This is valuable because it means we can debug the issue on a 3-episode shard
instead of rerunning a large end-to-end pipeline.

## Recommended Next Steps

1. Add a direction-focused regression command over trace shards.
   - specifically report `north/east/south/west/search` precision and recall

2. Audit the `v2` observation encoder for missing orientation/local-map cues.
   - the current confusion pattern suggests the policy still lacks enough signal
     to separate symmetric movement choices reliably

3. Create targeted trace shards where the teacher strongly prefers `east` or
   `south`.
   - use those as cheap BC and APPO regression tests

4. Keep using deterministic trace eval as the source of truth.
   - live seeded eval remains diagnostic-only because raw NLE reset is not
     stable enough for regression gating


## Direction-Focused Slice

To turn the confusion pattern into a faster regression target, the harness can
now slice a trace file by teacher action.

### Commands

```bash
uv run python cli.py rl-shard-traces \
  --input data/tracefix_v2_explore_traces.jsonl \
  --output /tmp/tracefix_east_south.jsonl \
  --teacher-actions east,south \
  --max-episodes 6

uv run python cli.py rl-trace-disagreements \
  --input /tmp/tracefix_east_south.jsonl \
  --bc-model output/tracefix_v2_explore_bc.pt \
  --appo-experiment appo_tracefix_v2_bc_x10 \
  --top-k 6
```

### Slice Result

Slice file summary:

- episodes: `6`
- rows: `120`
- all multi-turn: `true`
- observation versions: `["v2"]`
- feature dims: `[160]`

BC on the slice:

- overall match rate: `0.7083`
- per-action recall:
  - `north: 0.8649`
  - `east: 0.5926`
  - `south: 0.6250`
  - `west: 0.6875`

APPO on the slice:

- overall match rate: `0.6833`
- per-action recall:
  - `north: 0.8649`
  - `east: 0.6296`
  - `south: 0.5417`
  - `west: 0.6250`

### Updated Reading

This slice sharpens the earlier conclusion:

- APPO is not uniformly worse than BC.
- APPO slightly improves `east` recall on this slice.
- APPO is still worse on `south` and `west`.
- both models over-predict `north`.

So the next likely bottleneck is not just optimization. It is directional state
representation: the current `v2` observation still seems to make some movement
choices too symmetric.
