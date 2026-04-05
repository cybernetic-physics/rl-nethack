# rl-nethack

NetHack RL research: LLM agents, expert trace capture, and LoRA fine-tuning for local GPU machines.

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
- NLE v0.7.3 patched for deterministic seeding (`patch_nle.py`)
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
- Caveat: the 5k vLLM policy dataset is high-throughput but low-quality. It is dominated by `wait` and other weak actions, so it should be treated as a throughput baseline, not a final training corpus.

## Project Structure

```
autoascend_traces/        Expert bot trace capture (Docker + trace runner)
  Dockerfile.light        Lightweight image, no GPU
  run_with_trace.py       Monkey-patches agent.step() to record observations
  trace_recorder.py       Writes JSON traces
  patch_nle.py            Seeds NetHackChallenge deterministically
  requirements.light.txt  Pinned deps (gym, NLE, torch, etc.)

nle_agent/
  agent_http.py           LLM agent with action map (direction names -> NLE indices)

scripts/
  generate_counterfactual_data.py   What-if analysis at combat moments
  generate_training_data.py         LLM policy data generation -> ShareGPT JSONL
  start_vllm_policy_server.sh       Start local vLLM policy server on GPUs 0,1

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
PLAN.md                    Full architecture document
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

### Generate Training Data

```bash
uv run python cli.py generate --num-games 200 --max-steps 50 --output data/train.jsonl
```

### Generate LLM-Policy Data at High Throughput

Serve the policy model with vLLM on GPUs `0,1`:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./scripts/start_vllm_policy_server.sh Qwen/Qwen2.5-1.5B-Instruct
```

Then run concurrent local policy generation against that server:

```bash
uv run python scripts/generate_training_data.py \
  --backend vllm \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --server-url http://127.0.0.1:8000/v1 \
  --num-games 200 \
  --max-steps 50 \
  --workers 64 \
  --cooldown 0
```

This setup keeps GPUs `2,3` available for other work, including training.

## Current Priorities

The repo is no longer blocked on local compute. The bottleneck has moved to policy-data quality and how efficiently requests are fed to the inference server.

Recommended next moves:
- Upgrade local policy generation from `Qwen/Qwen2.5-0.5B-Instruct` to `Qwen/Qwen2.5-1.5B-Instruct` or `Qwen/Qwen2.5-3B-Instruct`
- Move from thread-per-request generation to a true batched/offline vLLM path
- Generate a filtered local corpus at 50k-200k examples once action quality looks sane
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
