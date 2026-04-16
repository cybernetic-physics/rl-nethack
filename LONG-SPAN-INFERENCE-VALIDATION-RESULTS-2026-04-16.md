# Long-Span Inference Validation Results

Date: `2026-04-16`

## Goal

Validate the newly trained rebuilt-canonical long-context adapter at inference time with a larger closed-loop live run than the earlier `4 x 32` check, while keeping the prompt format aligned with the actual long-history inference path.

This note follows the same general direction as:

- [LONG-BOOTSTRAP-SMOKE-RESULTS-2026-04-16.md](/home/luc/rl-nethack-worktree-20260416/LONG-BOOTSTRAP-SMOKE-RESULTS-2026-04-16.md)
- [LONG-CONTEXT-NLD-TRAINING-RESULTS-2026-04-16.md](/home/luc/rl-nethack-worktree-20260416/LONG-CONTEXT-NLD-TRAINING-RESULTS-2026-04-16.md)
- [LONG-CONTEXT-QWEN-1M-PLAN-2026-04-16.md](/home/luc/rl-nethack-worktree-20260416/LONG-CONTEXT-QWEN-1M-PLAN-2026-04-16.md)

but uses the rebuilt canonical dataset format that matches the policy interface we now expect to use at inference.


## Model And Artifacts

Validated adapter:

- [output/qwen14b_rebuilt_canonical_nld_32k_fullsize_validation](/home/luc/rl-nethack-worktree-20260416/output/qwen14b_rebuilt_canonical_nld_32k_fullsize_validation)
- training metadata: [training_meta.json](/home/luc/rl-nethack-worktree-20260416/output/qwen14b_rebuilt_canonical_nld_32k_fullsize_validation/training_meta.json)

Serving path used for eval:

- [scripts/start_transformers_chat_server.py](/home/luc/rl-nethack-worktree-20260416/scripts/start_transformers_chat_server.py)

Inference harness:

- [src/long_sequence_live_eval.py](/home/luc/rl-nethack-worktree-20260416/src/long_sequence_live_eval.py)
- [cli.py](/home/luc/rl-nethack-worktree-20260416/cli.py)


## Commands Run

Serve the trained adapter:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/start_transformers_chat_server.py \
  --model Qwen/Qwen2.5-14B-Instruct-1M \
  --adapter output/qwen14b_rebuilt_canonical_nld_32k_fullsize_validation \
  --served-model-name qwen14b-rebuilt-canonical-nld-32k-fullsize \
  --port 8008
```

Attempted longer x10 validations:

```bash
uv run python cli.py live-evaluate-long-sequences \
  --seeds 42,43,44,45,46,47,48,49,50,51 \
  --max-steps 128 \
  --server-url http://127.0.0.1:8008 \
  --model-name qwen14b-rebuilt-canonical-nld-32k-fullsize \
  --max-context-tokens 32768 \
  --output output/rebuilt/qwen14b_rebuilt_canonical_nld_32k_longspan_live_eval_10x128.json
```

```bash
uv run python cli.py live-evaluate-long-sequences \
  --seeds 42,43,44,45,46,47,48,49,50,51 \
  --max-steps 64 \
  --server-url http://127.0.0.1:8008 \
  --model-name qwen14b-rebuilt-canonical-nld-32k-fullsize \
  --max-context-tokens 32768 \
  --output output/rebuilt/qwen14b_rebuilt_canonical_nld_32k_longspan_live_eval_10x64.json
```

Completed x10 validation run:

```bash
uv run python cli.py live-evaluate-long-sequences \
  --seeds 42,43,44,45,46,47,48,49,50,51 \
  --max-steps 32 \
  --server-url http://127.0.0.1:8008 \
  --model-name qwen14b-rebuilt-canonical-nld-32k-fullsize \
  --max-context-tokens 32768 \
  --output output/rebuilt/qwen14b_rebuilt_canonical_nld_32k_longspan_live_eval_10x32.json
```


## Practical Throughput Limit

The `10 x 128` and `10 x 64` runs stayed healthy, but they were too slow to use as practical validation artifacts on this serving setup.

Observed behavior:

- the adapter server remained healthy throughout
- requests continued returning `200 OK`
- GPU utilization stayed pinned on the serving GPU
- the eval harness does not emit partial JSON; it only writes the report at the end

Operational conclusion:

- on a single served `14B` adapter with growing long-history prompts, `10 x 64` and `10 x 128` are currently too inference-bound for a fast local validation loop
- `10 x 32` is the longest x10 sweep that completed cleanly in a reasonable wall-clock window on this setup

This is a serving-throughput constraint, not evidence that the policy path itself is broken.


## Completed Result: `10 x 32`

Primary output:

- [output/rebuilt/qwen14b_rebuilt_canonical_nld_32k_longspan_live_eval_10x32.json](/home/luc/rl-nethack-worktree-20260416/output/rebuilt/qwen14b_rebuilt_canonical_nld_32k_longspan_live_eval_10x32.json)

Summary metrics:

- episodes: `10`
- total steps: `320`
- avg steps: `32.0`
- avg final depth: `1.0`
- avg final HP: `13.7`
- invalid / odd action rate: `0.015625`

Total action mix:

- `wear`: `115`
- `north`: `109`
- `fire`: `64`
- `wait`: `32`

Per-seed outcomes:

- seed `42`: `32` steps, final depth `1`, final HP `14`, actions `fire 6 / wear 14 / north 12`
- seed `43`: `32` steps, final depth `1`, final HP `14`, actions `fire 6 / wear 9 / north 5 / wait 12`
- seed `44`: `32` steps, final depth `1`, final HP `14`, actions `fire 7 / wear 13 / north 12`
- seed `45`: `32` steps, final depth `1`, final HP `14`, actions `fire 6 / wear 12 / north 11 / wait 3`
- seed `46`: `32` steps, final depth `1`, final HP `14`, actions `fire 6 / wear 13 / north 13`
- seed `47`: `32` steps, final depth `1`, final HP `11`, actions `fire 6 / wear 10 / north 10 / wait 6`
- seed `48`: `32` steps, final depth `1`, final HP `14`, actions `fire 6 / wear 11 / north 13 / wait 2`
- seed `49`: `32` steps, final depth `1`, final HP `14`, actions `fire 8 / wear 12 / north 12`
- seed `50`: `32` steps, final depth `1`, final HP `14`, actions `fire 9 / wear 12 / north 11`
- seed `51`: `32` steps, final depth `1`, final HP `14`, actions `fire 4 / wear 9 / north 10 / wait 9`


## Comparison To The Earlier `4 x 32` Validation

Earlier result:

- [output/rebuilt/qwen14b_rebuilt_canonical_nld_32k_fullsize_live_eval_4x32.json](/home/luc/rl-nethack-worktree-20260416/output/rebuilt/qwen14b_rebuilt_canonical_nld_32k_fullsize_live_eval_4x32.json)

Earlier summary:

- episodes: `4`
- total steps: `128`
- avg final depth: `1.0`
- avg final HP: `14.0`
- invalid / odd action rate: `0.0234375`

What changed in the larger x10 pass:

- coverage increased from `4` seeds to `10`
- the policy stayed stable over the larger sample
- invalid / odd action rate improved from `0.0234375` to `0.015625`
- action mix stayed in the same narrow family: mostly `wear`, `north`, `fire`, with some `wait`
- final HP dipped slightly from `14.0` to `13.7`
- depth progress did not improve; all runs remained at dungeon depth `1`


## Interpretation

What the run supports:

1. The rebuilt canonical long-context training path transfers cleanly to the real inference path.
2. The trained adapter is stable across `10` deterministic live rollouts, not just `4`.
3. Invalid / odd action behavior is low and slightly better than the earlier smaller live check.

What it does not support:

1. The model is not yet showing meaningful long-horizon task progress.
2. The action distribution is still narrow and NLD-shaped.
3. The current policy still fails to convert long-context formatting improvements into actual dungeon advancement.

Bottom line:

The rebuilt canonical dataset format appears to have fixed prompt-format alignment and action-validity issues better than the earlier paths, but the trained adapter still does not show substantive gameplay competence in closed loop. The next bottleneck is policy quality, not long-context serialization correctness.


## Next Steps

Priority order:

1. Improve the live harness so longer-than-`32` x10 sweeps can checkpoint partial progress instead of only writing at the end.
2. Run held-out offline benchmark evaluation for the same adapter on rebuilt canonical benchmark data if not already cached.
3. Train another rebuilt-canonical adapter with a stronger action mix target, because the current policy remains dominated by `wear` / `north` / `fire`.
4. Add evaluation metrics beyond final HP and depth `1`, especially inventory-change, position-change, and illegal-action-family tracking.
