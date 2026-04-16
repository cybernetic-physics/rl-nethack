# rl-nethack

NetHack research repo spanning three active directions:

- offline teacher building from explicit trace data
- APPO-style online improvement with teacher constraints
- long-context next-action SFT on rolling game histories

As of 2026-04-16, this is no longer just a small forward-model project. The repo contains a real RL stack under `rl/`, a long-sequence data and evaluation path under `src/`, and the earlier forward-model tooling remains available as a secondary path.

## Problem

The repo is trying to solve a specific NetHack research problem:

- build strong local teachers from traces, long-context supervision, and auxiliary models
- use those teachers to improve a learned online policy
- do that under a benchmark that is stable enough to trust

What is solved well enough to work with:

- offline teacher construction
- trace generation and deterministic trace evaluation
- BC, world-model, reward-model, scheduler-model, and APPO plumbing
- long-history next-action dataset building and LoRA training

What is still not solved:

- online improvement that beats the strongest teacher without drifting off the teacher manifold

## Start Here

Read these first if you want the current state of the project rather than the historical root markdown trail:

- [docs/consolidated-2026-04/README.md](docs/consolidated-2026-04/README.md)
- [docs/consolidated-2026-04/02-system-architecture.md](docs/consolidated-2026-04/02-system-architecture.md)
- [docs/consolidated-2026-04/04-evaluation-and-benchmarks.md](docs/consolidated-2026-04/04-evaluation-and-benchmarks.md)
- [docs/consolidated-2026-04/05-blockers-and-next-steps.md](docs/consolidated-2026-04/05-blockers-and-next-steps.md)
- [docs/consolidated-2026-04/07-operator-quickstart.md](docs/consolidated-2026-04/07-operator-quickstart.md)

Two recent repo-local notes matter for the current long-context branch:

- [LONG-CONTEXT-QWEN-1M-PLAN-2026-04-16.md](LONG-CONTEXT-QWEN-1M-PLAN-2026-04-16.md)
- [LONG-CONTEXT-NLD-TRAINING-RESULTS-2026-04-16.md](LONG-CONTEXT-NLD-TRAINING-RESULTS-2026-04-16.md)

## Current Status

Current high-level picture:

- the repo has working offline teacher, world-model, proxy-reward, and APPO infrastructure
- deterministic trace evaluation is the trusted short-loop gate
- the main unsolved problem is teacher-constrained online improvement without drift
- the newest text-policy direction is long-history next-action prediction, not short-context delta prediction

Current long-context status from committed notes on 2026-04-16:

- the repo can build and train on long-sequence next-action corpora
- a `Qwen/Qwen2.5-14B-Instruct-1M` LoRA run completed successfully at `32k` context on `4x H200`
- the same trainer path did not fit `64k` context in that configuration
- the larger NLD-trained adapter changed online action behavior, but did not yet beat the earlier medium bootstrap adapter on the short live harness

## Evaluation Rule

Do not use live seeded evaluation as the main promotion gate.

The trusted gate in this repo is:

- deterministic held-out trace match

That is the main evaluation policy reflected in the consolidated docs and the RL tooling.

## Which Files Matter

If you are changing code, these are the files most likely to matter:

- [cli.py](/home/luc/rl-nethack-worktree-20260416/cli.py): main operator surface and subcommand wiring
- [README.md](/home/luc/rl-nethack-worktree-20260416/README.md): top-level repo orientation
- [pyproject.toml](/home/luc/rl-nethack-worktree-20260416/pyproject.toml): dependency groups and Python version
- [docs/consolidated-2026-04/07-operator-quickstart.md](/home/luc/rl-nethack-worktree-20260416/docs/consolidated-2026-04/07-operator-quickstart.md): shortest current workflow doc

For long-context data and text-policy work:

- [src/long_sequence_dataset.py](/home/luc/rl-nethack-worktree-20260416/src/long_sequence_dataset.py)
- [src/long_sequence_corpus.py](/home/luc/rl-nethack-worktree-20260416/src/long_sequence_corpus.py)
- [src/long_sequence_eval.py](/home/luc/rl-nethack-worktree-20260416/src/long_sequence_eval.py)
- [src/long_sequence_benchmark.py](/home/luc/rl-nethack-worktree-20260416/src/long_sequence_benchmark.py)
- [src/nld_long_sequence_import.py](/home/luc/rl-nethack-worktree-20260416/src/nld_long_sequence_import.py)
- [train.py](/home/luc/rl-nethack-worktree-20260416/train.py)
- [train_preferences.py](/home/luc/rl-nethack-worktree-20260416/train_preferences.py)
- [train_kto.py](/home/luc/rl-nethack-worktree-20260416/train_kto.py)
- [scripts/start_vllm_qwen_1m_server.sh](/home/luc/rl-nethack-worktree-20260416/scripts/start_vllm_qwen_1m_server.sh)

For traces, BC teachers, and trace evaluation:

- [rl/traces.py](/home/luc/rl-nethack-worktree-20260416/rl/traces.py)
- [rl/train_bc.py](/home/luc/rl-nethack-worktree-20260416/rl/train_bc.py)
- [rl/evaluate_bc.py](/home/luc/rl-nethack-worktree-20260416/rl/evaluate_bc.py)
- [rl/trace_eval.py](/home/luc/rl-nethack-worktree-20260416/rl/trace_eval.py)
- [rl/debug_tools.py](/home/luc/rl-nethack-worktree-20260416/rl/debug_tools.py)
- [rl/teacher_report.py](/home/luc/rl-nethack-worktree-20260416/rl/teacher_report.py)

For online RL:

- [rl/train_appo.py](/home/luc/rl-nethack-worktree-20260416/rl/train_appo.py)
- [rl/trainer.py](/home/luc/rl-nethack-worktree-20260416/rl/trainer.py)
- [rl/config.py](/home/luc/rl-nethack-worktree-20260416/rl/config.py)
- [rl/sf_env.py](/home/luc/rl-nethack-worktree-20260416/rl/sf_env.py)
- [rl/env_adapter.py](/home/luc/rl-nethack-worktree-20260416/rl/env_adapter.py)
- [rl/teacher_reg.py](/home/luc/rl-nethack-worktree-20260416/rl/teacher_reg.py)
- [rl/evaluate.py](/home/luc/rl-nethack-worktree-20260416/rl/evaluate.py)
- [rl/bootstrap.py](/home/luc/rl-nethack-worktree-20260416/rl/bootstrap.py)

For world-model, reward, proxy, and scheduler branches:

- [rl/train_world_model.py](/home/luc/rl-nethack-worktree-20260416/rl/train_world_model.py)
- [rl/world_model.py](/home/luc/rl-nethack-worktree-20260416/rl/world_model.py)
- [rl/world_model_eval.py](/home/luc/rl-nethack-worktree-20260416/rl/world_model_eval.py)
- [rl/proxy_dataset.py](/home/luc/rl-nethack-worktree-20260416/rl/proxy_dataset.py)
- [rl/train_proxy_model.py](/home/luc/rl-nethack-worktree-20260416/rl/train_proxy_model.py)
- [rl/proxy_eval.py](/home/luc/rl-nethack-worktree-20260416/rl/proxy_eval.py)
- [rl/train_reward_model.py](/home/luc/rl-nethack-worktree-20260416/rl/train_reward_model.py)
- [rl/train_scheduler.py](/home/luc/rl-nethack-worktree-20260416/rl/train_scheduler.py)

For the older forward-model/reporting path:

- [src/state_encoder.py](/home/luc/rl-nethack-worktree-20260416/src/state_encoder.py)
- [src/data_generator.py](/home/luc/rl-nethack-worktree-20260416/src/data_generator.py)
- [src/evaluator.py](/home/luc/rl-nethack-worktree-20260416/src/evaluator.py)
- [src/reporter.py](/home/luc/rl-nethack-worktree-20260416/src/reporter.py)
- [generate_demo_report.py](/home/luc/rl-nethack-worktree-20260416/generate_demo_report.py)
- [smoke_test.py](/home/luc/rl-nethack-worktree-20260416/smoke_test.py)

For AutoAscend expert capture:

- [autoascend_traces/run_with_trace.py](/home/luc/rl-nethack-worktree-20260416/autoascend_traces/run_with_trace.py)
- [autoascend_traces/trace_recorder.py](/home/luc/rl-nethack-worktree-20260416/autoascend_traces/trace_recorder.py)
- [autoascend_traces/patch_nle.py](/home/luc/rl-nethack-worktree-20260416/autoascend_traces/patch_nle.py)
- [autoascend_traces/Dockerfile.light](/home/luc/rl-nethack-worktree-20260416/autoascend_traces/Dockerfile.light)
- [references/README.md](/home/luc/rl-nethack-worktree-20260416/references/README.md)

## Main Code Paths

### 1. Long-context next-action SFT

This is the newest main text-policy direction.

- `src/long_sequence_dataset.py`: builds rolling-history ShareGPT-style next-action data
- `src/long_sequence_corpus.py`: compiles token-budgeted mixed corpora
- `src/long_sequence_eval.py`: exact next-action evaluation on long-sequence JSONL
- `src/long_sequence_benchmark.py`: deterministic benchmark shard builder
- `train.py`: LoRA SFT, including metadata filtering, weighted SFT, and native curriculum support
- `train_preferences.py`: pairwise preference LoRA training
- `train_kto.py`: KTO-style LoRA training

The training target here is:

- long serialized game history -> next action

not:

- state + action -> delta

### 2. Explicit traces and offline teacher training

This is still the strongest policy-building substrate in the repo.

- `rl/traces.py`: multi-turn trace generation
- `rl/train_bc.py`: behavior cloning from traces
- `rl/relabel_traces.py`: relabel traces with a BC teacher
- `rl/trace_eval.py`: deterministic trace-match evaluation
- `rl/teacher_report.py`: teacher and checkpoint reports

### 3. APPO online improvement

This is the real RL stack under `rl/`.

- `rl/train_appo.py`: APPO entrypoint
- `rl/trainer.py`: trainer scaffold
- `rl/sf_env.py`, `rl/env_adapter.py`: environment and Sample Factory integration
- `rl/teacher_reg.py`: teacher-aware online losses and replay
- `rl/evaluate.py`: live policy evaluation

### 4. World-model and proxy branches

These are implemented and used as support paths rather than the main answer by themselves.

- `rl/train_world_model.py`, `rl/world_model.py`, `rl/world_model_eval.py`
- `rl/proxy_dataset.py`, `rl/train_proxy_model.py`, `rl/proxy_eval.py`
- `rl/train_reward_model.py`, `rl/train_scheduler.py`

### 5. Earlier forward-model path

The original short-context forward-model tooling is still present.

- `src/state_encoder.py`
- `src/data_generator.py`
- `src/evaluator.py`
- `src/reporter.py`

This path is still usable, but it is no longer the best description of the repo.

## CLI Surface

The main operator entrypoint is `cli.py`.

Notable command families:

- long-sequence data and eval: `generate-long-sequences`, `convert-long-sequences`, `import-nld-long-sequences`, `build-long-sequence-corpus`, `evaluate-long-sequences`, `build-long-sequence-benchmark`, `compare-long-sequence-evals`
- RL and traces: `rl-generate-traces`, `rl-verify-traces`, `rl-train-bc`, `rl-evaluate-bc`, `rl-train-appo`, `rl-evaluate-appo`, `rl-rank-checkpoints`, `rl-teacher-reg-report`, `rl-run-dagger`, `rl-dagger-iterate`
- world-model and proxy: `rl-train-world-model`, `rl-evaluate-world-model`, `rl-build-proxy-dataset`, `rl-train-proxy`, `rl-evaluate-proxy`
- older baseline utilities: `generate`, `report`, `evaluate`, `manifest`, `golden-generate`, `golden-evaluate`, `smoke-test`

## Repo Layout

```text
autoascend_traces/          AutoAscend trace capture and NLE compatibility patches
docs/consolidated-2026-04/  Current durable docs and operator guidance
docs/archive/root-history/  Historical reports, plans, and handoffs
nle_agent/                  HTTP/text-policy agent code
references/                 Papers and external-repo notes
rl/                         APPO, BC, world-model, proxy-reward, trace-eval stack
scripts/                    Launchers and dataset preparation scripts
src/                        Long-sequence builders, forward-model utilities, reporting
tests/                      472 collected tests in this worktree
cli.py                      Main CLI entrypoint
train.py                    LoRA SFT training
train_preferences.py        Pairwise preference LoRA training
train_kto.py                KTO-style LoRA training
```

## Install

Requirements:

- Python `>=3.10,<3.13`
- `uv`
- `nle==1.2.0`
- CUDA GPUs for training or vLLM serving
- Docker if you want the AutoAscend trace-capture path

Install only tests:

```bash
uv sync --extra test
```

Install the usual full local stack:

```bash
uv sync --extra train --extra test --extra serve
```

The APPO backend is bootstrapped on demand by `rl/bootstrap.py` because upstream `sample-factory` packaging does not align cleanly with the working `nle==1.2.0` environment.

If you want the containerized legacy forward-model path instead of a local env:

```bash
docker compose up
```

That uses [docker-compose.yml](/home/luc/rl-nethack-worktree-20260416/docker-compose.yml) to generate baseline data, run `train.py`, and emit a manifest.

## First Validation

```bash
uv run python cli.py smoke-test
uv run pytest -q tests/test_rl_scaffold.py
```

Those are the fastest useful sanity checks for the baseline repo path and the RL/world-model/proxy scaffold.

You can also run the standalone reporter smoke script directly:

```bash
uv run python smoke_test.py
```

And generate the interactive HTML replay demo with:

```bash
uv run python generate_demo_report.py
```

## Quick Starts

### Long-sequence smoke corpus from live NLE rollouts

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

Evaluate a served model on that benchmark:

```bash
uv run python cli.py evaluate-long-sequences \
  --input data/long_bootstrap/long_sequence_benchmark.jsonl \
  --server-url http://127.0.0.1:8000/v1 \
  --model-name Qwen/Qwen2.5-14B-Instruct-1M
```

### Import NLD/ttyrec data into long-sequence format

Register the extracted dataset root with the helper script:

```bash
uv run python scripts/prepare_nld_dataset.py \
  --zip data/nld-aa/nld-aa-dir-aa.zip \
  --extract-dir data/nld-aa/extracted \
  --dataset-name nld-aa-local \
  --register
```

Then import a bounded shard:

```bash
uv run python cli.py import-nld-long-sequences \
  --dataset-name nld-aa-local \
  --output data/nld-aa_long_sequences_smoke.jsonl \
  --dbfilename ttyrecs.db \
  --max-games 64 \
  --min-turns 1000 \
  --min-maxlvl 5 \
  --max-context-tokens 65536 \
  --source nld-aa-local
```

### Long-context LoRA smoke train

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

For the current 1M-context serving branch, launch vLLM with the repo script:

```bash
./scripts/start_vllm_qwen_1m_server.sh
```

That script reads optional settings from [scripts/qwen_1m_vllm.env.example](/home/luc/rl-nethack-worktree-20260416/scripts/qwen_1m_vllm.env.example) if you copy it to `scripts/qwen_1m_vllm.env`.

For the older short-context policy-generation server:

```bash
./scripts/start_vllm_policy_server.sh Qwen/Qwen2.5-1.5B-Instruct
```

### Trace generation and BC teacher training

```bash
uv run python cli.py rl-generate-traces \
  --output data/pipeline_explore_traces_clean.jsonl \
  --num-episodes 40 \
  --max-steps 24 \
  --task explore \
  --policy task_greedy

uv run python cli.py rl-verify-traces \
  --input data/pipeline_explore_traces_clean.jsonl

uv run python cli.py rl-train-bc \
  --input data/pipeline_explore_traces_clean.jsonl \
  --output output/pipeline_explore_bc.pt \
  --epochs 25 \
  --lr 0.001
```

### APPO run with the repo scaffold

```bash
uv run python cli.py rl-train-appo \
  --experiment appo_debug \
  --train-dir train_dir/rl \
  --train-for-env-steps 200000 \
  --trace-eval-input data/pipeline_explore_traces_clean.jsonl \
  --trace-eval-interval-env-steps 50000 \
  --teacher-bc-path output/pipeline_explore_bc.pt
```

Use short trace-gated runs first. Do not start with long blind reward sweeps.

### Older forward-model path

Generate delta-prediction training data:

```bash
uv run python cli.py generate \
  --num-games 200 \
  --max-steps 50 \
  --output data/train.jsonl \
  --eval-output data/eval.jsonl \
  --eval-fraction 0.2
```

Train the forward model:

```bash
uv run torchrun --standalone --nproc_per_node=4 train.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --data data/train.jsonl \
  --eval-data data/eval.jsonl \
  --output output/adapter
```

Run a local replay/report:

```bash
uv run python cli.py report --seed 42 --max-steps 30 --output-dir output/report_seed_42
```

Evaluate against a served model:

```bash
uv run python cli.py evaluate \
  --seeds 42,43,44 \
  --max-steps 20 \
  --server-url http://127.0.0.1:8000
```

## Artifacts And Storage

This source repo is intended to stay relatively small.

Canonical remotes already used by the repo:

- dataset artifacts: `https://huggingface.co/datasets/lmc7150/rl-nethack-data`
- model artifacts: `https://huggingface.co/lmc7150/rl-nethack-models`

Helper scripts:

- `scripts/push_hf_data.sh`
- `scripts/push_hf_models.sh`

Keep code, configs, and docs here. Push larger datasets, checkpoints, manifests, and reports to the artifact remotes instead of growing the git repo.

## Historical Material

The old root markdown trail has been moved under:

- [docs/archive/root-history/](docs/archive/root-history/)

Use the source map here if you need to audit the historical reasoning trail:

- [docs/consolidated-2026-04/99-source-index.md](docs/consolidated-2026-04/99-source-index.md)

## References

External paper and repo notes live in:

- [references/README.md](references/README.md)

AutoAscend trace capture lives in:

- [autoascend_traces/](autoascend_traces/)

Build and run the lightweight capture container with:

```bash
cd autoascend_traces
docker build -f Dockerfile.light -t autoascend .
docker run --rm -v "$(pwd)/output:/output" autoascend
```

## Bottom Line

Treat this repo as:

- a strong teacher-building system
- a credible deterministic evaluation system
- an implemented but still unsolved online improver stack
- an active long-context next-action training branch centered on Qwen 1M models
