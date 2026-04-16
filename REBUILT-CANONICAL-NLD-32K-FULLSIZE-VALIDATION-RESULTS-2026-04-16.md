# Rebuilt Canonical NLD 32k Fullsize Validation Results

Date: `2026-04-16`

## Goal

Run a real larger-scale validation pass on the rebuilt canonical long-context policy format, using data that is much closer to the runtime inference surface than the older mixed-format NLD runs.

This run follows the same overall shape as:

- [LONG-BOOTSTRAP-SMOKE-RESULTS-2026-04-16.md](/home/luc/rl-nethack-worktree-20260416/LONG-BOOTSTRAP-SMOKE-RESULTS-2026-04-16.md)
- [LONG-CONTEXT-NLD-TRAINING-RESULTS-2026-04-16.md](/home/luc/rl-nethack-worktree-20260416/LONG-CONTEXT-NLD-TRAINING-RESULTS-2026-04-16.md)
- [LONG-CONTEXT-QWEN-1M-PLAN-2026-04-16.md](/home/luc/rl-nethack-worktree-20260416/LONG-CONTEXT-QWEN-1M-PLAN-2026-04-16.md)

but swaps in the rebuilt canonical replay corpus rather than the older imported mixed-format slice.

## Corpus Used

Canonical rebuilt raw-source NLD smoke corpus:

- [data/rebuilt/nld_hf_taster/long_sequences_smoke_canonical.jsonl](/home/luc/rl-nethack-worktree-20260416/data/rebuilt/nld_hf_taster/long_sequences_smoke_canonical.jsonl)
- [data/rebuilt/nld_hf_taster/long_sequences_smoke_canonical.jsonl.validation.json](/home/luc/rl-nethack-worktree-20260416/data/rebuilt/nld_hf_taster/long_sequences_smoke_canonical.jsonl.validation.json)

Validation split used for the fullsize run:

- [data/rebuilt/nld_hf_taster/fullsize_validation/train.jsonl](/home/luc/rl-nethack-worktree-20260416/data/rebuilt/nld_hf_taster/fullsize_validation/train.jsonl)
- [data/rebuilt/nld_hf_taster/fullsize_validation/eval_tail_256.jsonl](/home/luc/rl-nethack-worktree-20260416/data/rebuilt/nld_hf_taster/fullsize_validation/eval_tail_256.jsonl)
- [data/rebuilt/nld_hf_taster/fullsize_validation/benchmark_eval_tail_256.jsonl](/home/luc/rl-nethack-worktree-20260416/data/rebuilt/nld_hf_taster/fullsize_validation/benchmark_eval_tail_256.jsonl)

Split counts:

- total rows: `1349`
- train rows: `1093`
- eval rows: `256`
- benchmark rows: `256`

## Training Command

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc_per_node=4 train.py \
  --model Qwen/Qwen2.5-14B-Instruct-1M \
  --data data/rebuilt/nld_hf_taster/fullsize_validation/train.jsonl \
  --eval-data data/rebuilt/nld_hf_taster/fullsize_validation/eval_tail_256.jsonl \
  --output output/qwen14b_rebuilt_canonical_nld_32k_fullsize_validation \
  --max-seq-length 32768 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --lr 2e-4 \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 2 \
  --dataset-num-proc 4 \
  --dataloader-num-workers 2 \
  --logging-steps 5 \
  --save-steps 25 \
  --save-total-limit 1 \
  --warmup-steps 5 \
  --gradient-checkpointing
```

## Training Issues Encountered

Training did not succeed on the first attempt.

Two real trainer-compatibility issues surfaced:

1. `FastLanguageModel.from_pretrained(...)` returned `tokenizer=None` on this distributed path.
2. The installed `trl` / Unsloth trainer stack expected `processing_class` instead of `tokenizer` in the trainer constructor signature.

These were fixed in [train.py](/home/luc/rl-nethack-worktree-20260416/train.py):

- add an explicit `AutoTokenizer` fallback when Unsloth returns `None`
- set a pad token if missing
- map `tokenizer -> processing_class` automatically inside the trainer init-kwargs shim when required by the installed trainer version

Those fixes were committed in:

- `d96bdb5` `Fix long-context trainer tokenizer compatibility`

After the fix, the full 4-GPU run completed successfully.

## Training Result

Output:

- [output/qwen14b_rebuilt_canonical_nld_32k_fullsize_validation](/home/luc/rl-nethack-worktree-20260416/output/qwen14b_rebuilt_canonical_nld_32k_fullsize_validation)
- [output/qwen14b_rebuilt_canonical_nld_32k_fullsize_validation/training_meta.json](/home/luc/rl-nethack-worktree-20260416/output/qwen14b_rebuilt_canonical_nld_32k_fullsize_validation/training_meta.json)

Configuration:

- model: `Qwen/Qwen2.5-14B-Instruct-1M`
- context: `32768`
- GPUs: `4x H200`
- LoRA rank: `32`
- LoRA alpha: `64`
- per-device batch size: `1`
- gradient accumulation: `2`
- global batch: `8`
- epochs: `1`
- full train split used: `1093` rows
- full eval split used: `256` rows

Final training result from [training_meta.json](/home/luc/rl-nethack-worktree-20260416/output/qwen14b_rebuilt_canonical_nld_32k_fullsize_validation/training_meta.json):

- final loss: `0.07646430849376386`
- global steps: `137`
- adapter hash: `e140194d6bde4aaef8bbb67b863c15ff526467bfd22d1d9bf010f277e67c972f`

Interpretation:

- the rebuilt canonical corpus now supports a real `32k` long-context `14B` run on `4x H200`
- this was a complete full-split validation run for the rebuilt canonical smoke corpus, not a tiny capped smoke subset
- the main failure mode encountered was trainer/runtime compatibility, not data corruption or OOM

## Live Validation

Served adapter:

- model: `Qwen/Qwen2.5-14B-Instruct-1M`
- adapter: [output/qwen14b_rebuilt_canonical_nld_32k_fullsize_validation](/home/luc/rl-nethack-worktree-20260416/output/qwen14b_rebuilt_canonical_nld_32k_fullsize_validation)

Live eval output:

- [output/rebuilt/qwen14b_rebuilt_canonical_nld_32k_fullsize_live_eval_4x32.json](/home/luc/rl-nethack-worktree-20260416/output/rebuilt/qwen14b_rebuilt_canonical_nld_32k_fullsize_live_eval_4x32.json)

Evaluation regime:

- seeds: `42,43,44,45`
- horizon: `32`

Summary:

- episodes: `4`
- total steps: `128`
- avg final depth: `1.0`
- avg final HP: `14.0`
- invalid/odd action rate: `0.0234375`

Action distribution:

- `wear`: `40`
- `north`: `37`
- `wait`: `29`
- `fire`: `22`

Interpretation:

- the adapter is producing almost entirely valid canonical actions
- the odd-action rate is far lower than the earlier bad smoke adapters
- but the action policy is still visibly shaped by the rebuilt NLD distribution and is over-emitting things like `wear` and `fire`
- online behavior is cleaner, but still not translating into actual depth progress in the short harness

## Held-out Offline Benchmark

Target benchmark output path:

- [output/rebuilt/qwen14b_rebuilt_canonical_nld_32k_fullsize_benchmark_eval.json](/home/luc/rl-nethack-worktree-20260416/output/rebuilt/qwen14b_rebuilt_canonical_nld_32k_fullsize_benchmark_eval.json)

Status at the time of writing:

- the full `256`-example offline benchmark pass was launched against the served adapter
- the adapter server remained healthy and responded continuously
- the run was inference-bound and substantially slower than the live probe
- the benchmark JSON had not finished writing yet at the time this note was created

So the current validation package is:

- fullsize `32k` training: complete
- live `4 x 32` online validation: complete
- held-out `256`-example offline benchmark: launched and in progress during this write-up

## Bottom Line

1. The rebuilt canonical corpus is trainable at real long-context scale.
2. The first full `Qwen 14B` `32k` validation run on that corpus completed successfully on `4x H200`.
3. We did hit real trainer integration issues, but they were fixed in-repo and the run was restarted from scratch successfully.
4. The resulting adapter has much better action validity than the earlier odd-action-heavy smoke models.
5. The next bottleneck is policy quality, not serialization:
   - action distribution is still skewed toward NLD-like behaviors
   - short online rollouts still do not show actual dungeon progress

## Follow-on Work

Priority order:

1. Finish and record the held-out offline benchmark report once the long eval completes.
2. Compare this rebuilt canonical run directly against:
   - the older mixed-format NLD run
   - the medium bootstrap adapter
   - the rebuilt bootstrap adapter
3. Reduce `wear` / `fire` overproduction by tightening action filtering or rebalancing the rebuilt corpus.
4. Extend online validation beyond the short `4 x 32` harness so long-horizon gains can actually show up if they exist.
