# rl-nethack

NetHack RL research: LLM agents, expert trace capture, and LoRA fine-tuning with attested virgin benchmarks inside Phala TEE CVMs.

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
- Local llama-server (Qwen 2.5 3B)
- OpenRouter API
- Any OpenAI-compatible endpoint

### Forward Model Training Pipeline

Train a LoRA adapter to predict state deltas in NetHack -- not to play, but to learn game physics. A model that can predict what changes after an action has internalized the rules of the world.

**Core insight: delta prediction.** Instead of predicting the full next state (90%+ identical to input), predict only what changed:

```
pos:(-1,0) | hp:-2 | gold:same | depth:same | alive:yes | msg:The newt bites!
```

This produces denser training signal, shorter sequences, and cleaner gradients.

### Virgin Benchmark

The entire pipeline runs inside a Phala TEE CVM with GPU access, producing an attested manifest that cryptographically links training data, code, adapter weights, and evaluation scores. Evaluation uses "virgin" dungeon seeds generated inside the TEE that were never seen during training -- proving no test contamination.

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
  generate_training_data.py         Random play -> ShareGPT JSONL

src/
  state_encoder.py        NLE obs -> structured features + delta encoding
  data_generator.py       Random play -> training pairs
  evaluator.py            Prediction accuracy scoring
  reporter.py             HTML + text gameplay replays
  manifest.py             Attested manifest builder (SHA256 hashes)
  memory_tracker.py       Memory-augmented forward model training pairs

tests/                     295 tests across 7 test files
cli.py                     CLI: generate, report, evaluate, manifest, smoke-test
train.py                   Unsloth LoRA training (GPU required)
docker-compose.yml         Phala CVM definition with GPU
PLAN.md                    Full architecture document
```

## Quick Start

### Requirements

- Python 3.10+
- [NLE](https://github.com/heuritech/nle) (NetHack Learning Environment)
- Docker (for AutoAscend traces)
- For training: CUDA GPU, [Unsloth](https://github.com/unslothai/unsloth), TRL, PEFT

### Install

```bash
pip install nle pytest
# For GPU training:
pip install unsloth trl peft transformers datasets
```

### Smoke Test (no GPU needed)

```bash
python3 cli.py smoke-test
```

### Generate Training Data

```bash
python3 cli.py generate --num-games 200 --max-steps 50 --output data/train.jsonl
```

### Train (GPU required)

```bash
python3 train.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --data data/train.jsonl \
  --eval-data data/eval.jsonl \
  --output output/adapter \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lr 2e-4 \
  --epochs 1 \
  --batch-size 4
```

### Evaluate & Build Manifest

```bash
python3 cli.py evaluate --seeds 500,501,502,503,504 --max-steps 20

python3 cli.py manifest \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --training-data data/train.jsonl \
  --adapter output/adapter \
  --baseline-scores '{"field_accuracy": 0.32}' \
  --post-scores '{"field_accuracy": 0.71}' \
  --output output/manifest.json
```

### Deploy to Phala CVM

```bash
cp .env.example .env
# Edit .env with HF_TOKEN and WANDB_API_KEY
docker compose up
```

Runs: generate -> train -> evaluate -> manifest, all inside the TEE.

## How the Forward Model Works

1. **Collect data**: Play NetHack with wall-avoidance random policy. Record (obs_before, action, obs_after).
2. **Extract features**: Convert raw NLE observations into structured text.
3. **Compute deltas**: What changed between observations.
4. **Train**: LoRA fine-tune to predict deltas from (state, action).
5. **Evaluate**: On unseen virgin seeds, measure per-field accuracy.

## License

See repository for license information.
