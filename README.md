# rl-nethack

NetHack RL research: LLM agents, expert trace capture, and LoRA fine-tuning for local GPU machines.

## Current Orientation

If you are trying to understand the repo as it exists now, start with the consolidated docs:

- [docs/consolidated-2026-04/README.md](/home/luc/rl-nethack-worktree-20260416/docs/consolidated-2026-04/README.md)
- [docs/consolidated-2026-04/07-operator-quickstart.md](/home/luc/rl-nethack-worktree-20260416/docs/consolidated-2026-04/07-operator-quickstart.md)

Those docs summarize the committed markdown trail and preserve citations back to the original reports, plans, handoffs, and research notes.

Important current status:

- the repo now has a real APPO backend, a strong offline teacher path, a world-model representation path, and a trusted deterministic trace benchmark
- the main open problem is no longer infrastructure
- the main open problem is teacher-constrained online improvement without drift

The strongest broad conclusions are synthesized in:

- [docs/consolidated-2026-04/03-experimental-timeline.md](/home/luc/rl-nethack-worktree-20260416/docs/consolidated-2026-04/03-experimental-timeline.md)
- [docs/consolidated-2026-04/04-evaluation-and-benchmarks.md](/home/luc/rl-nethack-worktree-20260416/docs/consolidated-2026-04/04-evaluation-and-benchmarks.md)
- [docs/consolidated-2026-04/05-blockers-and-next-steps.md](/home/luc/rl-nethack-worktree-20260416/docs/consolidated-2026-04/05-blockers-and-next-steps.md)

Important benchmark warning:

- do not use live seeded evaluation as the main promotion gate
- use deterministic held-out trace match instead
- do not compare numbers across different trace / representation regimes unless the benchmark setup matches

## What's Here

### AutoAscend Expert Trace Capture (`autoascend_traces/`)

Capture step-by-step gameplay traces from [AutoAscend](https://github.com/maciej-sypetkowski/autoascend), a classical NetHack bot that explores dungeons, fights monsters, and descends stairs. These traces serve as expert demonstrations for training or evaluating LLM agents.

```bash
cd autoascend_traces/

# Build the Docker image (Ubuntu 20.04, NLE v0.7.3, no GPU needed)
docker build -f Dockerfile.light -t autoascend .

# Run 3 games x 2000 steps
docker run --rm -v $(pwd)/output:/output autoascend

# Output: output/autoascend_traces.json
```

The captured traces include per-step: ASCII dungeon map, HP/max_hp, depth, position, action taken, game messages, kill events. Convert to the game viewer format and serve:

```bash
# See scripts/ for conversion and viewer generation
python3 -m http.server 8080 -d output/
# Open autoascend_viewer.html in browser
```

**Compatibility fixes applied** (see `fix_env_wrapper_gym_compat.patch`):
- Legacy AutoAscend NetHackChallenge wrapper patched for deterministic seeding (`patch_nle.py`)
- Core repo note: plain `nle.env.NLE().reset(seed=...)` is still not reproducible enough for trusted policy regression
- gym>=0.21 `env._actions` -> `env.unwrapped._actions`
- `seed()` takes 1 arg, not 2

### LLM Agent (`nle_agent/`)

An HTTP-based LLM agent that plays NetHack via NLE. Sends structured game state (HP, position, adjacent tiles, monsters, items) to a language model and receives action commands back.

Supports multiple backends:
- Local vLLM / OpenAI-compatible server
- Local llama-server
- OpenRouter API
- Any OpenAI-compatible endpoint

### Forward Model Training Pipeline

Train a LoRA adapter to predict state deltas in NetHack -- not to play, but to learn game physics. A model that can predict what changes after an action has internalized the rules of the world.

**Core insight: delta prediction.** Instead of predicting the full next state (90%+ identical to input), predict only what changed:

```
pos:(-1,0) | hp:-2 | gold:same | depth:same | alive:yes | msg:The newt bites!
```

This produces denser training signal, shorter sequences, and cleaner gradients.

### Evaluation + Manifest

The pipeline can generate training data, fine-tune a LoRA adapter, evaluate a model on held-out seeds, and build a manifest that records the model, dataset, adapter, and scores used for a run.

### Current Local Benchmarks

- Machine: 4x NVIDIA H200
- Random data generation via `cli.py generate`: 1,000 examples in about 1.35s
- LLM-policy data generation via vLLM on GPUs `0,1`: 5,000 examples in about 30s with `Qwen/Qwen2.5-0.5B-Instruct`
- Improved local policy generation with `Qwen/Qwen2.5-3B-Instruct` and frontier-biased fallback:
  - `1,000` samples in `9.04s` on one TP=2 vLLM server
  - `1,000` samples in `8.09s` on two 1-GPU vLLM replicas
  - `10,000` samples in `78.76s` on two replicas
- Experimental in-process `vllm-batch` backend:
  - `1,000` samples in `43.30s` end-to-end on 2 GPUs
  - `10,000` samples in `111.20s` end-to-end on 2 GPUs
  - Quality is acceptable, but on current settings it is slower than the replica-server path once startup is included
- The earlier 5k `0.5B` dataset is still only a throughput baseline; the newer `3B` replica path is the first one with a reasonably balanced action mix.

## Project Structure

```
autoascend_traces/        Expert bot trace capture (Docker + trace runner)
  Dockerfile.light        Lightweight image, no GPU
  run_with_trace.py       Monkey-patches agent.step() to record observations
  trace_recorder.py       Writes JSON traces
  patch_nle.py            Seeds the legacy NetHackChallenge wrapper deterministically
  requirements.light.txt  Pinned deps (gym, NLE, torch, etc.)

nle_agent/
  agent_http.py           LLM agent with action map (direction names -> NLE indices)

scripts/
  generate_counterfactual_data.py   What-if analysis at combat moments
  generate_training_data.py         LLM policy data generation -> ShareGPT JSONL
  start_vllm_policy_server.sh       Start local vLLM policy server on GPUs 0,1
  start_vllm_policy_replicas.sh     Start two 1-GPU vLLM replicas on GPUs 0 and 1

src/
  state_encoder.py        NLE obs -> structured features + delta encoding
  data_generator.py       Random play -> training pairs
  evaluator.py            Prediction accuracy scoring
  reporter.py             HTML + text gameplay replays
  manifest.py             Manifest builder (SHA256 hashes)
  memory_tracker.py       Memory-augmented forward model training pairs

tests/                     295 tests across 7 test files
cli.py                     CLI: generate, report, evaluate, manifest, smoke-test
train.py                   Unsloth LoRA training (GPU required)
pyproject.toml             uv project definition and dependency groups
uv.lock                    Locked dependency resolution for reproducible setup
.gitattributes             Git LFS tracking rules for datasets
docker-compose.yml         Local Docker Compose training job with GPU access
docs/consolidated-2026-04/ Consolidated research, architecture, eval, lit review, and operator docs
docs/archive/root-history/ Historical markdown trail moved out of the project root
```

## Quick Start

### Requirements

- Python 3.10+
- `uv`
- `git-lfs` if you want dataset files to round-trip cleanly through Git
- [NLE](https://github.com/heuritech/nle) (NetHack Learning Environment)
- Docker (for AutoAscend traces)
- For training: CUDA GPU, [Unsloth](https://github.com/unslothai/unsloth), TRL, PEFT

### Install

```bash
uv sync --extra test

# For local GPU training:
uv sync --extra train --extra test

# For local GPU policy serving with vLLM:
uv sync --extra serve

# If you want training + serving tools in one env:
uv sync --extra train --extra test --extra serve
```

### Smoke Test (no GPU needed)

```bash
uv run python cli.py smoke-test
```

### Generate Random Forward-Model Data

```bash
uv run python cli.py generate --num-games 200 --max-steps 50 --output data/train.jsonl
```

This generates one supervised training example per environment step for the
forward model. These are single-step examples, but they are collected from
multi-turn episodes.

### Generate LLM-Policy Data at High Throughput

Preferred path: serve two 1-GPU replicas on GPUs `0,1`:

```bash
./scripts/start_vllm_policy_replicas.sh Qwen/Qwen2.5-3B-Instruct
```

Then run concurrent local policy generation against both replicas:

```bash
uv run python scripts/generate_training_data.py \
  --backend vllm \
  --model Qwen/Qwen2.5-3B-Instruct \
  --server-url http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1 \
  --num-games 200 \
  --max-steps 50 \
  --workers 64 \
  --cooldown 0
```

If you want the older single-server path instead, serve one TP=2 instance on GPUs `0,1`:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./scripts/start_vllm_policy_server.sh Qwen/Qwen2.5-1.5B-Instruct
```

Then point generation at one server URL:

```bash
uv run python scripts/generate_training_data.py \
  --backend vllm \
  --model Qwen/Qwen2.5-3B-Instruct \
  --server-url http://127.0.0.1:8000/v1 \
  --num-games 200 \
  --max-steps 50 \
  --workers 64 \
  --cooldown 0
```

This setup keeps GPUs `2,3` available for other work, including training.

### Download And Process NLD-AA

For the new long-context next-action training path, the best local supervised source is currently `NLD-AA`.

The extracted `NLD-AA` layout is treated as `nle_data/...`, not `altorg`, so the processing flow is:

1. download shard zips
2. extract and register the local `nle_data` root
3. import into the repo’s long-sequence JSONL format
4. train a LoRA adapter on that imported corpus

Download all `NLD-AA` shards:

```bash
mkdir -p data/nld-aa
for shard in aa ab ac ad ae af ag ah ai aj ak al am an ao ap; do
  curl -L -C - -o "data/nld-aa/nld-aa-dir-${shard}.zip" \
    "https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-${shard}.zip"
done
```

Extract and register the local dataset root:

```bash
ZIP_ARGS=()
for shard in aa ab ac ad ae af ag ah ai aj ak al am an ao ap; do
  ZIP_ARGS+=(--zip "data/nld-aa/nld-aa-dir-${shard}.zip")
done

uv run python scripts/prepare_nld_dataset.py \
  "${ZIP_ARGS[@]}" \
  --extract-dir data/nld-aa/extracted \
  --dataset-name nld-aa-local \
  --register
```

That creates a local `ttyrecs.db` registration for:

- extracted root: `data/nld-aa/extracted/nle_data`
- dataset name: `nld-aa-local`

Import a bounded smoke shard first:

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

Then scale to a larger training shard:

```bash
uv run python cli.py import-nld-long-sequences \
  --dataset-name nld-aa-local \
  --output data/nld-aa_long_sequences_train.jsonl \
  --dbfilename ttyrecs.db \
  --max-games 2048 \
  --min-turns 1000 \
  --min-maxlvl 5 \
  --max-context-tokens 65536 \
  --source nld-aa-local
```

If you want to mix that imported shard into the token-budgeted long-context corpus builder:

```bash
uv run python cli.py build-long-sequence-corpus \
  --input data/nld-aa_long_sequences_train.jsonl \
  --output data/nld-aa_long_sequences_train_mixed.jsonl \
  --manifest-output data/nld-aa_long_sequences_train_mixed.manifest.json \
  --target-tokens 1000000000
```

### Train On The New Long-Sequence Data

The long-context path is next-action prediction on rolling histories, not delta prediction.

Start with a small smoke run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python train.py \
  --model Qwen/Qwen2.5-14B-Instruct-1M \
  --data data/nld-aa_long_sequences_smoke.jsonl \
  --output output/qwen14b_nldaa_smoke \
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
  --gradient-checkpointing
```

Then move to a larger single-node LoRA run:

```bash
MODEL=Qwen/Qwen2.5-14B-Instruct-1M \
TRAIN_DATA=data/nld-aa_long_sequences_train.jsonl \
EVAL_DATA=data/nld-aa_long_sequences_smoke.jsonl \
OUTPUT=output/qwen14b_nldaa_long_lora \
MAX_SEQ_LENGTH=65536 \
GRAD_ACCUM=16 \
DATASET_NUM_PROC=4 \
DATALOADER_NUM_WORKERS=2 \
bash scripts/train_qwen_1m_long_lora.sh
```

If you want the native in-code curriculum instead of a flat run:

```bash
MODEL=Qwen/Qwen2.5-14B-Instruct-1M \
TRAIN_DATA=data/nld-aa_long_sequences_train.jsonl \
EVAL_DATA=data/nld-aa_long_sequences_smoke.jsonl \
OUTPUT=output/qwen14b_nldaa_curriculum \
bash scripts/train_qwen_1m_native_curriculum.sh
```

Practical notes:

- `ttyrecs.db` must exist before `add_nledata_directory(...)` can register the root; the prep script handles this for you.
- `NLD-AA` metadata uses hex-like strings such as `0x0` for some fields; the importer now handles that.
- Run a bounded smoke import and smoke train first before scaling to thousands of games.

## Full Pipeline

This section is the real operator guide.

If you want to go from:

1. data generation
2. forward-model SFT
3. trace generation
4. reward / scheduler training
5. behavior cloning
6. APPO RL
7. evaluation

these are the commands to run, in order.

There are now **three distinct training/data tracks** in this repo:

- forward-model SFT
- trace-based policy training / BC
- APPO RL

They are related, but they are not the same thing.

### Mental Model

Before the exact commands, here is the right way to think about the system.

#### Track A: forward model

This is trained by [train.py](/home/luc/rl-nethack/train.py).

Input:

- current state
- chosen action

Target:

- predicted delta after the action

This is the SFT path.

It does **not** directly produce an RL policy. It produces a model that can
predict what will happen next.

#### Track B: traces

This is the bridge between planning/SFT-style supervision and policy training.

A trace file is a **multi-turn** episode export.

Each row in a trace file contains:

- `episode_id`
- `step`
- `task`
- `action`
- `allowed_actions`
- `feature_vector`
- `delta`
- `reward`
- `done`
- hashes and planner metadata

Important:

- trace files are explicitly **multi-turn**
- there is now a verifier command that checks this

#### Track C: RL / policy training

This is the APPO path under [rl/](/home/luc/rl-nethack/rl).

This is now real learned RL:

- rollout workers
- recurrence
- policy/value training
- checkpoints

The current best way to bootstrap that policy is:

- generate good traces
- optionally train BC from those traces
- train reward / scheduler models
- run APPO with masking and learned components


## Stage 0: Environment Setup

### Minimal install for docs/tests

```bash
uv sync --extra test
```

### Full install for training + serving + RL

```bash
uv sync --extra train --extra test --extra serve
```

What this gives you:

- `train.py` dependencies for LoRA SFT
- test dependencies
- vLLM serving dependencies
- the project CLI

The APPO backend itself is auto-bootstrapped on demand the first time you run:

```bash
uv run python cli.py rl-train-appo ...
```

because upstream `sample-factory` metadata conflicts with current `nle`
packaging.


## Stage 1: Generate Forward-Model Training Data

This is the simplest path and does not require any model server.

```bash
uv run python cli.py generate \
  --num-games 200 \
  --max-steps 50 \
  --output data/train.jsonl \
  --eval-output data/eval.jsonl \
  --eval-fraction 0.2
```

What this does:

- plays `200` NetHack episodes
- each episode is up to `50` steps
- writes one JSONL row per step
- each row is a ShareGPT-style conversation:
  - system prompt
  - user prompt with state + action
  - assistant target with the delta

This dataset is for SFT of the forward model.

It is not a trace file for BC/RL.


## Stage 2: Train The Forward Model With SFT

Use all 4 H200s:

```bash
uv run torchrun --standalone --nproc_per_node=4 train.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --data data/train.jsonl \
  --eval-data data/eval.jsonl \
  --output output/adapter \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lr 2e-4 \
  --epochs 1 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --dataset-num-proc 8 \
  --dataloader-num-workers 8
```

Notes:

- this is distributed LoRA training
- default path is bf16 LoRA, not 4-bit, because this box has enough memory
- output is a LoRA adapter directory, typically:
  - `output/adapter`

This SFT model is a **forward model**, not a policy.

That means:

- it can be evaluated on next-step prediction
- it can be used as a teacher/planner for trace generation
- it cannot be directly loaded into the APPO actor-critic as weights


## Stage 3: Serve The Forward Model

If you want to use the forward model during trace generation, you need to serve
it.

The repo’s evaluation and forward-model trace path expect an OpenAI-compatible
chat endpoint.

Example with your own server:

```bash
# Example only: use whatever OpenAI-compatible server you prefer
# and point it at your trained adapter / merged model.
```

The CLI assumes:

- server URL like `http://127.0.0.1:8765`
- chat endpoint at `/v1/chat/completions`

You will use that server in:

- `cli.py evaluate`
- `cli.py golden-evaluate`
- `cli.py rl-generate-traces --policy forward_model`


## Stage 4: Evaluate The Forward Model

Basic held-out evaluation:

```bash
uv run python cli.py evaluate \
  --seeds 500,501,502,503,504 \
  --max-steps 20 \
  --server-url http://127.0.0.1:8765
```

Golden debug evaluation:

```bash
uv run python cli.py golden-generate \
  --seed 42 \
  --max-steps 10 \
  --output data/golden_episode.jsonl

uv run python cli.py golden-evaluate \
  --input data/golden_episode.jsonl \
  --server-url http://127.0.0.1:8765
```

Use the golden path before trusting larger evaluations. It catches train/eval
format mismatches fast.


## Stage 5: Generate Multi-Turn Traces

This is the new path you asked for explicitly.

These traces are **definitely multi-turn** now.

You can verify them with a dedicated command.

### Option A: Generate traces from `task_greedy`

```bash
uv run python cli.py rl-generate-traces \
  --output data/explore_task_greedy_traces.jsonl \
  --num-episodes 100 \
  --max-steps 30 \
  --task explore \
  --policy task_greedy
```

### Option B: Generate traces from the served forward model

This is the main way to use the SFT model in the RL workflow.

```bash
uv run python cli.py rl-generate-traces \
  --output data/explore_forward_model_traces.jsonl \
  --num-episodes 100 \
  --max-steps 30 \
  --task explore \
  --policy forward_model \
  --server-url http://127.0.0.1:8765 \
  --model-name llama-server
```

How this works:

- for each state
- for each allowed action
- the forward model predicts the delta
- the trace generator scores the predicted outcome
- it picks the best action
- then it rolls the real env forward

So this is a true **multi-turn teacher-in-the-loop trace generator** using the
SFT forward model.

### Option C: Generate traces from a trained APPO policy

```bash
uv run python cli.py rl-generate-traces \
  --output data/explore_appo_traces.jsonl \
  --num-episodes 100 \
  --max-steps 30 \
  --task explore \
  --policy appo \
  --appo-experiment appo_explore_masked
```

### Option D: Generate traces from a BC policy

```bash
uv run python cli.py rl-generate-traces \
  --output data/explore_bc_traces.jsonl \
  --num-episodes 100 \
  --max-steps 30 \
  --task explore \
  --policy bc \
  --bc-model-path output/explore_bc.pt
```

### Verify that traces are actually multi-turn

```bash
uv run python cli.py rl-verify-traces \
  --input data/explore_task_greedy_traces.jsonl
```

Expected output includes:

- `episodes`
- `rows`
- `max_steps_in_episode`
- `avg_steps_in_episode`
- `multi_turn_episodes`
- `all_multi_turn`

If `all_multi_turn` is `true`, the trace file is a real multi-turn dataset.


## Stage 6: Train A Behavior Cloning Policy From Traces

Once you have a trace file, you can train a policy directly on it.

Example:

```bash
uv run python cli.py rl-train-bc \
  --input data/explore_task_greedy_traces.jsonl \
  --output output/explore_bc.pt \
  --epochs 20 \
  --lr 1e-3
```

This trains a compact policy network on:

- `feature_vector`
- action labels
- allowed-action masks

This is the cleanest direct bridge from:

- teacher traces
- to a trainable policy

without going straight into RL.

Evaluate the BC policy:

```bash
uv run python cli.py rl-evaluate-bc \
  --model output/explore_bc.pt \
  --task explore \
  --seeds 42,43,44 \
  --max-steps 50 \
  --compare-baseline
```

Use BC as:

- a bootstrap policy,
- a control baseline,
- or a future initialization source for RL-related policy work.


## Stage 7: Train Learned Reward Models

Reward models are now trainable from task-harness preference pairs.

Example:

```bash
uv run python cli.py rl-train-reward \
  --task explore \
  --seeds 42,43,44,45,46,47 \
  --max-steps 30 \
  --dataset-output data/explore_reward_prefs.jsonl \
  --output output/explore_reward.pt \
  --epochs 20 \
  --lr 1e-3
```

This uses task-harness counterfactual branches to build pairwise preferences
and then trains a Bradley-Terry-style reward model.

You can then use that model in APPO.


## Stage 8: Train A Learned Scheduler

The scheduler path is also trainable now.

Example:

```bash
uv run python cli.py rl-train-scheduler \
  --seeds 42,43,44,45,46,47 \
  --max-steps 30 \
  --dataset-output data/scheduler_rows.jsonl \
  --output output/scheduler.pt \
  --epochs 20 \
  --lr 1e-3
```

This trains a small classifier to imitate the current rule-based scheduler.

That gives you a real learned high-level scheduler artifact for the APPO env.


## Stage 9: Train APPO RL

There are now several useful APPO modes.

### Baseline APPO with hand-shaped reward

```bash
uv run python cli.py rl-train-appo \
  --experiment appo_explore \
  --num-workers 4 \
  --num-envs-per-worker 8 \
  --rollout-length 32 \
  --recurrence 16 \
  --batch-size 1024 \
  --num-batches-per-epoch 1 \
  --ppo-epochs 1 \
  --train-for-env-steps 20000 \
  --enabled-skills explore
```

Important:

- env-side invalid action clamping is on by default now
- invalid requests are penalized
- this is a real RL run

### APPO with learned reward

```bash
uv run python cli.py rl-train-appo \
  --experiment appo_explore_learned_reward \
  --num-workers 4 \
  --num-envs-per-worker 8 \
  --rollout-length 32 \
  --recurrence 16 \
  --batch-size 1024 \
  --num-batches-per-epoch 1 \
  --ppo-epochs 1 \
  --train-for-env-steps 20000 \
  --enabled-skills explore \
  --reward-source learned \
  --learned-reward-path output/explore_reward.pt
```

### APPO with learned scheduler

```bash
uv run python cli.py rl-train-appo \
  --experiment appo_learned_scheduler \
  --num-workers 4 \
  --num-envs-per-worker 8 \
  --rollout-length 32 \
  --recurrence 16 \
  --batch-size 1024 \
  --num-batches-per-epoch 1 \
  --ppo-epochs 1 \
  --train-for-env-steps 20000 \
  --scheduler learned \
  --scheduler-model-path output/scheduler.pt
```

### APPO with both learned reward and learned scheduler

```bash
uv run python cli.py rl-train-appo \
  --experiment appo_full_stack \
  --num-workers 4 \
  --num-envs-per-worker 8 \
  --rollout-length 32 \
  --recurrence 16 \
  --batch-size 1024 \
  --num-batches-per-epoch 1 \
  --ppo-epochs 1 \
  --train-for-env-steps 20000 \
  --reward-source learned \
  --learned-reward-path output/explore_reward.pt \
  --scheduler learned \
  --scheduler-model-path output/scheduler.pt
```


## Stage 10: Evaluate APPO

Use the built-in evaluator:

```bash
uv run python cli.py rl-evaluate-appo \
  --experiment appo_explore \
  --seeds 42,43,44 \
  --max-steps 50 \
  --compare-baseline
```

This reports:

- avg task reward
- avg unique tiles
- avg rooms discovered
- repeated action rate
- invalid action rate
- action counts

If `--compare-baseline` is set and the run is single-skill, it also runs
`task_greedy` so you can compare directly.


## Recommended End-To-End Flows

### Flow A: simplest useful forward-model path

```bash
uv sync --extra train --extra test --extra serve

uv run python cli.py generate \
  --num-games 200 \
  --max-steps 50 \
  --output data/train.jsonl \
  --eval-output data/eval.jsonl

uv run torchrun --standalone --nproc_per_node=4 train.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --data data/train.jsonl \
  --eval-data data/eval.jsonl \
  --output output/adapter
```

### Flow B: teacher traces -> BC

```bash
uv run python cli.py rl-generate-traces \
  --output data/explore_task_greedy_traces.jsonl \
  --num-episodes 100 \
  --max-steps 30 \
  --task explore \
  --policy task_greedy

uv run python cli.py rl-verify-traces \
  --input data/explore_task_greedy_traces.jsonl

uv run python cli.py rl-train-bc \
  --input data/explore_task_greedy_traces.jsonl \
  --output output/explore_bc.pt

uv run python cli.py rl-evaluate-bc \
  --model output/explore_bc.pt \
  --task explore \
  --seeds 42,43,44 \
  --max-steps 50
```

### Flow C: SFT forward model -> teacher traces -> BC -> RL

```bash
# 1. Generate SFT data
uv run python cli.py generate \
  --num-games 200 \
  --max-steps 50 \
  --output data/train.jsonl \
  --eval-output data/eval.jsonl

# 2. Train forward model
uv run torchrun --standalone --nproc_per_node=4 train.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --data data/train.jsonl \
  --eval-data data/eval.jsonl \
  --output output/adapter

# 3. Serve that forward model using your OpenAI-compatible server

# 4. Generate multi-turn traces with the forward model in the loop
uv run python cli.py rl-generate-traces \
  --output data/explore_forward_model_traces.jsonl \
  --num-episodes 100 \
  --max-steps 30 \
  --task explore \
  --policy forward_model \
  --server-url http://127.0.0.1:8765 \
  --model-name llama-server

# 5. Verify they are multi-turn
uv run python cli.py rl-verify-traces \
  --input data/explore_forward_model_traces.jsonl

# 6. Train BC from those traces
uv run python cli.py rl-train-bc \
  --input data/explore_forward_model_traces.jsonl \
  --output output/explore_bc.pt

# 7. Train reward model
uv run python cli.py rl-train-reward \
  --task explore \
  --output output/explore_reward.pt

# 8. Train scheduler
uv run python cli.py rl-train-scheduler \
  --output output/scheduler.pt

# 9. Run APPO with learned reward + learned scheduler
uv run python cli.py rl-train-appo \
  --experiment appo_full_stack \
  --num-workers 4 \
  --num-envs-per-worker 8 \
  --rollout-length 32 \
  --recurrence 16 \
  --batch-size 1024 \
  --num-batches-per-epoch 1 \
  --ppo-epochs 1 \
  --train-for-env-steps 20000 \
  --reward-source learned \
  --learned-reward-path output/explore_reward.pt \
  --scheduler learned \
  --scheduler-model-path output/scheduler.pt

# 10. Evaluate APPO against baseline
uv run python cli.py rl-evaluate-appo \
  --experiment appo_full_stack \
  --seeds 42,43,44 \
  --max-steps 50 \
  --compare-baseline
```


## What “Use The SFT Model” Means In This Repo

This is important because it is easy to misunderstand.

Right now, “use the SFT model” means:

- train the forward model with `train.py`
- serve it behind an OpenAI-compatible endpoint
- use it in `rl-generate-traces --policy forward_model`

It does **not** currently mean:

- directly loading LoRA weights into the APPO actor-critic

because those architectures are different.

The correct bridge today is:

- SFT forward model
- multi-turn trace generation
- BC and/or RL training from those traces


## Current State Of The Stack

Today the repo can do all of the following:

- generate forward-model SFT data
- train a distributed LoRA forward model
- evaluate the forward model
- generate explicit multi-turn traces
- verify that traces are multi-turn
- train BC from traces
- train learned reward models
- train learned schedulers
- train APPO RL with masking
- evaluate APPO checkpoints against baselines

The main thing that is still not true is:

- APPO does not yet beat `task_greedy` reliably

So the stack is now functionally complete enough to iterate on policy quality,
which is the correct next bottleneck.

## Current Priorities

The repo is no longer blocked on local compute. The bottleneck has moved to policy-data quality and how efficiently requests are fed to the inference server.

Recommended next moves:
- Keep `Qwen/Qwen2.5-3B-Instruct` as the local policy baseline and use the replica topology on GPUs `0,1`
- Keep the new `vllm-batch` backend as an experiment, but do not switch default generation to it yet
- Generate a filtered local corpus at 50k-200k examples now that the action distribution looks materially better
- Use all 4 H200s for forward-model LoRA runs on at least Qwen 2.5 3B, and likely 7B once the dataset is no longer tiny

### Train Fast on 4x H200 (GPU required)

```bash
uv run torchrun --standalone --nproc_per_node=4 train.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --data data/train.jsonl \
  --eval-data data/eval.jsonl \
  --output output/adapter \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lr 2e-4 \
  --epochs 1 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --dataset-num-proc 8 \
  --dataloader-num-workers 8
```

For this host, the training script is set up for distributed `torchrun` and defaults to bf16 LoRA instead of 4-bit loading, because H200s have enough memory and bf16 is the faster path.

### Evaluate & Build Manifest

```bash
uv run python cli.py evaluate --seeds 500,501,502,503,504 --max-steps 20

uv run python cli.py manifest \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --training-data data/train.jsonl \
  --adapter output/adapter \
  --baseline-scores '{"field_accuracy": 0.32}' \
  --post-scores '{"field_accuracy": 0.71}' \
  --output output/manifest.json
```

### Golden Debug Harness

```bash
uv run python cli.py golden-generate --seed 42 --max-steps 10 --output data/golden_episode.jsonl
uv run python cli.py golden-evaluate --input data/golden_episode.jsonl --server-url http://127.0.0.1:8000
```

Use this before trusting larger runs. The goal is to catch prompt-format, parsing, and evaluator mismatches on a tiny saved episode.

### Run with Docker Compose on This Machine

```bash
docker compose up
```

The compose job installs `uv`, syncs the `train` extra from [pyproject.toml](/home/luc/rl-nethack/pyproject.toml), mounts the repo into the container, and launches distributed training with `torchrun --nproc_per_node=4`. On this 4x H200 host it exposes `CUDA_VISIBLE_DEVICES=0,1,2,3` by default.

## How the Forward Model Works

1. **Collect data**: Play NetHack with wall-avoidance random policy. Record (obs_before, action, obs_after).
2. **Extract features**: Convert raw NLE observations into structured text.
3. **Compute deltas**: What changed between observations.
4. **Train**: LoRA fine-tune to predict deltas from (state, action).
5. **Evaluate**: On unseen held-out seeds, measure per-field accuracy.

## License

See repository for license information.
