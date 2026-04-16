# Long Bootstrap Smoke Results

Date: `2026-04-16`

## Goal

Run a real smoke pass on the new long-history pipeline using genuinely long episodes, then check whether the resulting corpus is already good enough for:

- long-sequence next-action SFT
- selective negative / KTO mining
- preference-style training

This was meant to answer two practical questions:

1. does the long-history dataset format actually work at larger horizon?
2. do the generated long rollouts already contain the labels needed for preference mining?


## Commands Run

Generate a genuinely long rollout shard:

```bash
uv run python cli.py generate-long-sequences \
  --num-games 6 \
  --max-steps 1024 \
  --seed-start 2000 \
  --output data/long_bootstrap/long_sequence_train.jsonl \
  --eval-output data/long_bootstrap/long_sequence_eval.jsonl \
  --eval-fraction 0.2 \
  --max-context-tokens 65536 \
  --board-mode tokenized \
  --persist-dual-views \
  --source long_bootstrap_nle
```

Build a deterministic benchmark shard:

```bash
uv run python cli.py build-long-sequence-benchmark \
  --input data/long_bootstrap/long_sequence_eval.jsonl \
  --output data/long_bootstrap/long_sequence_benchmark.jsonl \
  --per-bucket 256 \
  --per-phase 256 \
  --per-action-family 256
```

Attempt KTO-style preference mining:

```bash
uv run python scripts/build_long_sequence_kto_dataset.py \
  --input data/long_bootstrap/long_sequence_train.jsonl \
  --output data/long_bootstrap/preferences/long_sequence_kto_train.jsonl \
  --output-format kto
```

Run a capped long-history LoRA smoke train:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python train.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data data/long_bootstrap/long_sequence_train.jsonl \
  --eval-data data/long_bootstrap/long_sequence_eval.jsonl \
  --output output/long_bootstrap_qwen_0_5b_smoke256 \
  --max-seq-length 8192 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --epochs 1 \
  --max-steps 10 \
  --logging-steps 1 \
  --save-steps 5 \
  --save-total-limit 1 \
  --warmup-steps 2 \
  --dataset-num-proc 1 \
  --dataloader-num-workers 0 \
  --gradient-checkpointing \
  --max-train-examples 256 \
  --max-eval-examples 64
```


## Artifacts

- Train shard: [data/long_bootstrap/long_sequence_train.jsonl](/home/luc/rl-nethack-worktree-20260416/data/long_bootstrap/long_sequence_train.jsonl)
- Eval shard: [data/long_bootstrap/long_sequence_eval.jsonl](/home/luc/rl-nethack-worktree-20260416/data/long_bootstrap/long_sequence_eval.jsonl)
- Benchmark shard: [data/long_bootstrap/long_sequence_benchmark.jsonl](/home/luc/rl-nethack-worktree-20260416/data/long_bootstrap/long_sequence_benchmark.jsonl)
- KTO mining output: [data/long_bootstrap/preferences/long_sequence_kto_train.jsonl](/home/luc/rl-nethack-worktree-20260416/data/long_bootstrap/preferences/long_sequence_kto_train.jsonl)
- Smoke adapter: [output/long_bootstrap_qwen_0_5b_smoke256](/home/luc/rl-nethack-worktree-20260416/output/long_bootstrap_qwen_0_5b_smoke256)
- Smoke training metadata: [output/long_bootstrap_qwen_0_5b_smoke256/training_meta.json](/home/luc/rl-nethack-worktree-20260416/output/long_bootstrap_qwen_0_5b_smoke256/training_meta.json)


## Results

### Long rollout generation

The long rollout shard completed successfully:

- games: `6`
- total examples: `6144`
- train examples: `5120`
- eval examples: `1024`
- benchmark examples: `256`
- target context budget: `65536`

Observed metadata statistics on the training shard:

- mean estimated context length: `52495.29` tokens
- max estimated context length: `65506` tokens
- mean included history steps: `299.77`
- max included history steps: `443`
- mean current-state token estimate: `175.81`
- max current-state token estimate: `258`

Interpretation:

- this is no longer a short-horizon corpus
- the builder is successfully packing hundreds of prior turns into one example
- the board serialization remains compact enough that the state snapshot itself is not the bottleneck
- the dominant token cost is the rolling history


### Raw sample quality

The generated rows are structurally correct for long-history next-action SFT:

- `system` prompt framing the task as next-action prediction
- `user` turn containing:
  - episode id
  - target step
  - rolling history
  - current stats
  - message text
  - full tokenized board
- `assistant` turn containing exactly one next action
- persisted dual views:
  - exact ASCII board
  - tokenized board

That means the new serialization format is working as intended.


### Preference mining

KTO mining produced:

- rows: `0`
- positives: `0`
- negatives: `0`

This is not a miner bug. The corpus labels explain it.

Inspection of the generated long shard showed:

- `outcome = unknown` for all `5120` train rows
- `game_phase = early` for all `5120` train rows

Interpretation:

- long rollouts alone are not enough to support selective negative mining
- the current live NLE-generated corpus does not yet carry useful terminal outcome labels for these episodes
- without win/loss resolution, teacher action metadata, or explicit failure labels, the KTO / pairwise / selective-negative path has nothing reliable to mine

This is the main blocker on preference training from locally generated long rollouts.


### Long-history SFT smoke train

The capped smoke train completed successfully on the long shard.

Training result from [training_meta.json](/home/luc/rl-nethack-worktree-20260416/output/long_bootstrap_qwen_0_5b_smoke256/training_meta.json):

- base model: `Qwen/Qwen2.5-0.5B-Instruct`
- trainer stack: repo-native `transformers + peft` fallback
- LoRA rank: `16`
- LoRA alpha: `32`
- max sequence length: `8192`
- max steps: `10`
- train examples used: `256`
- eval examples used: `64`
- final loss: `0.1066`
- adapter hash: `b3d2ed7f1f3c6970fc8f97a7cf3bcdbe2fc19d6bfbbba17a9089a238fbda4f58`

Interpretation:

- the long-history SFT path is operational
- the no-`trl` fallback training path is operational
- the long-sequence format is trainable with LoRA
- the main remaining problems are data semantics and scaling, not basic software correctness


## Operational Notes

An uncapped run on the full `5120`-example training shard was started first, but it was not a good smoke-test configuration. The dominant cost became dataset preprocessing and tokenization rather than actual training.

That is why the completed smoke run used:

- `--max-train-examples 256`
- `--max-eval-examples 64`

This is the correct operational pattern for early iteration:

- use the long-horizon shard
- cap the example count aggressively for smoke tests
- only scale the row count after the end-to-end path is stable


## Main Takeaways

1. The long-history dataset format is working.
The examples are genuinely long, preserve full-board information, and can include hundreds of prior steps.

2. The long-history SFT path is working.
We can already train a LoRA adapter on the long-history corpus without `trl`.

3. Preference mining is not ready from this corpus alone.
The current live-generated long shard lacks the outcome or teacher labels needed for KTO / pairwise mining.

4. The immediate bottleneck has shifted.
The main blocker is now label quality and corpus provenance, not sequence serialization.


## What To Do Next

Priority order:

1. Import or generate long traces with real terminal outcomes.
Best options remain:
- NLD / ttyrec-backed winning games
- AutoAscend long traces with episode completion labels
- teacher-annotated traces with preferred alternatives

2. Fix outcome propagation for locally generated long episodes.
If the rollout ends in a real terminal state, that outcome should be propagated back onto all rows for that episode instead of leaving them as `unknown`.

3. Re-run preference mining on a corpus with real labels.
That should produce nonzero KTO / pairwise rows if the data source is actually useful.

4. Keep smoke-training on capped subsets before scaling.
For long-sequence debugging, capped row counts are a better control surface than shrinking the history window.

5. Only after that, move to `Qwen/Qwen2.5-14B-Instruct-1M`.
The repo is ready for that path, but the next meaningful gain will come from better long-span winning data, not from swapping the base model first.

