# Operator Quickstart

## Purpose

This is the shortest practical path for someone operating the current repo. It is not a full theory document. It is the minimum working guide for:

- validating the environment,
- finding the current best documentation,
- running the trusted test/eval loop,
- and avoiding the benchmark mistakes that caused confusion earlier.

For the broader research narrative, use:

- [README.md](/home/luc/rl-nethack-worktree-20260416/docs/consolidated-2026-04/README.md)
- [04-evaluation-and-benchmarks.md](/home/luc/rl-nethack-worktree-20260416/docs/consolidated-2026-04/04-evaluation-and-benchmarks.md)
- [05-blockers-and-next-steps.md](/home/luc/rl-nethack-worktree-20260416/docs/consolidated-2026-04/05-blockers-and-next-steps.md)

## First Rule

Do not use live seeded evaluation as the main promotion gate.

The trusted gate is:

- deterministic held-out trace match

This is the main conclusion of:

- [PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md)
- [ROLLING-RESEARCH-THESIS.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/ROLLING-RESEARCH-THESIS.md)

## Environment Bring-Up

Install the main env:

```bash
uv sync --extra train --extra test --extra serve
```

Smoke check:

```bash
uv run python cli.py smoke-test
uv run pytest -q tests/test_rl_scaffold.py
```

Why these two:

- `smoke-test` checks the baseline repo path
- `tests/test_rl_scaffold.py` is the fast regression suite repeatedly used by the RL/world-model/proxy branches

## The Current Best Documentation To Read First

Read in this order:

1. [docs/consolidated-2026-04/README.md](/home/luc/rl-nethack-worktree-20260416/docs/consolidated-2026-04/README.md)
2. [docs/consolidated-2026-04/02-system-architecture.md](/home/luc/rl-nethack-worktree-20260416/docs/consolidated-2026-04/02-system-architecture.md)
3. [docs/consolidated-2026-04/04-evaluation-and-benchmarks.md](/home/luc/rl-nethack-worktree-20260416/docs/consolidated-2026-04/04-evaluation-and-benchmarks.md)
4. [docs/consolidated-2026-04/05-blockers-and-next-steps.md](/home/luc/rl-nethack-worktree-20260416/docs/consolidated-2026-04/05-blockers-and-next-steps.md)

If you need the raw historical trail after that, use:

- [docs/consolidated-2026-04/99-source-index.md](/home/luc/rl-nethack-worktree-20260416/docs/consolidated-2026-04/99-source-index.md)

## Which “Best” Result Matters

There is not one global number across the whole repo history. Benchmark regimes changed.

The important current reference points from the markdown trail are:

- strong short scheduled-replay APPO branch on the trusted short trace regime:
  - about `0.95` trace match
  - source discussion: [REPORT-PROXY-REWARD-OVERHAUL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-PROXY-REWARD-OVERHAUL-2026-04-06.md)
- stronger offline text-conditioned world-model teacher on transformed traces:
  - `0.9625` held-out trace match
  - source: [REPORT-LLM-WORLD-MODEL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-LLM-WORLD-MODEL-2026-04-06.md)

These are not interchangeable because they come from different representation / trace regimes.

## Fast Trusted Loop

This is the default operator loop the docs now support:

1. choose the exact held-out trace set
2. run a short offline or short online experiment
3. rank by deterministic trace match
4. inspect disagreement or trace metadata
5. scale only if the short run wins

That is the core project norm.

## Common Operator Tasks

### 1. Generate forward-model SFT data

```bash
uv run python cli.py generate \
  --num-games 300 \
  --max-steps 30 \
  --output data/pipeline_train.jsonl \
  --eval-output data/pipeline_eval.jsonl \
  --eval-fraction 0.2
```

Reference:

- [RL-APPO-HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/RL-APPO-HANDOFF.md)

### 2. Train the forward model

```bash
CUDA_VISIBLE_DEVICES=0,1,2 uv run torchrun --standalone --nproc_per_node=3 \
  train.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data data/pipeline_train.jsonl \
  --eval-data data/pipeline_eval.jsonl \
  --output output/pipeline_adapter \
  --max-seq-length 768 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lr 2e-4 \
  --batch-size 2 \
  --gradient-accumulation-steps 2 \
  --max-steps 60 \
  --dataset-num-proc 4 \
  --dataloader-num-workers 2 \
  --logging-steps 10 \
  --save-steps 30
```

Reference:

- [RL-APPO-HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/RL-APPO-HANDOFF.md:634)

### 3. Generate and verify traces

```bash
uv run python cli.py rl-generate-traces \
  --output data/pipeline_explore_traces_clean.jsonl \
  --num-episodes 40 \
  --max-steps 24 \
  --task explore \
  --policy task_greedy

uv run python cli.py rl-verify-traces \
  --input data/pipeline_explore_traces_clean.jsonl
```

This is important because:

- the repo depends on explicit multi-turn trace files,
- and bad trace assumptions caused earlier confusion.

### 4. Train a BC teacher from traces

```bash
uv run python cli.py rl-train-bc \
  --input data/pipeline_explore_traces_clean.jsonl \
  --output output/pipeline_explore_bc.pt \
  --epochs 25 \
  --lr 0.001
```

Then evaluate with the repo’s trace-aware tools rather than only with live seeded rollout checks.

### 5. Run short APPO experiments, not long blind ones

Use short debug branches and trace ranking first.

Do not start with:

- large undirected APPO runs,
- long reward-only sweeps,
- or benchmark comparisons across incompatible trace regimes.

## What To Compare Against

When reading or running experiments, check all of the following before comparing numbers:

- observation version
- trace dataset version
- whether traces were world-model transformed
- teacher artifact
- warm-start path
- offline vs short online vs medium online run type

If these do not match, the numbers are probably not directly comparable.

## Current Tactical Recommendation

If you are operating the current repo today:

- use the consolidated docs first,
- use deterministic trace match as the gate,
- treat the repo as a strong teacher-building system with a still-unsolved online improver problem,
- and only scale branches that win the short trusted loop.

