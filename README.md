# dstack-lora

LoRA fine-tuning of a language model to learn NetHack game physics inside a Phala TEE Confidential VM (CVM) with attested virgin benchmarks.

## What It Does

This project trains a small language model (via LoRA) to predict what happens after an action in NetHack -- not to play the game, but to learn its physics. A model that can accurately predict state transitions has internalized the rules of the game world: movement, combat, items, exploration, death.

The entire pipeline runs inside a Phala TEE CVM with GPU access, producing an attested manifest that cryptographically links training data, code, adapter weights, and before/after evaluation scores. Evaluation uses "virgin" dungeon seeds generated inside the TEE that were never seen during training.

## Why Delta Prediction

This is the core design insight. Rather than asking the model to predict the full next state (position, HP, gold, depth, message, all tiles), we ask it to predict only what *changed*:

```
pos:(-1,0) | hp:-2 | gold:same | depth:same | alive:yes | msg:The newt bites!
```

Delta prediction produces vastly denser training signal:

- **Most actions produce no change** in most fields. A full-state target would be 90%+ identical to the input. The model would learn to copy, not predict.
- **Deltas are sparse and informative**. When something changes, the model must learn exactly what caused it and by how much.
- **Compact targets** mean shorter sequences, faster training, and cleaner gradient signal per token.
- **The model still learns everything** -- to predict `hp:-2` it must understand combat, to predict `pos:(0,1)` it must understand movement and walls.

## Architecture

```
Raw NLE Observation (chars 21x79, blstats[24], message[256])
        |
  StateEncoder.encode_full()
        |
  Structured Features (position, HP, adjacent tiles, monsters, items)
        |
  format_prompt()  +  Action  -->  User message
        |
        v
  [LoRA-tuned LLM]
        |
        v
  "pos:(-1,0) | hp:same | gold:+1 | depth:same | alive:yes | msg:You see here a scroll."
        |
  parse_prediction() --> structured delta dict
        |
  compute_accuracy() vs ground truth from encode_delta()
```

### Structured Features

Instead of feeding raw ASCII art (21x79 grid) to the model, we extract compact structured text:

```
HP:13/13 AC:7 Str:15 Dex:14
Pos:(4,8) Gold:0 Depth:1 Turn:45
Adjacent: north=floor south=corridor east=wall west=floor
Monsters: d@(6,9)
Items: scroll@(3,7)
Action: south
```

This gives the model just enough context to predict outcomes without drowning in irrelevant tile data.

## Project Structure

```
cli.py                 CLI: generate, report, evaluate, manifest, smoke-test
train.py               Unsloth LoRA training (GPU required)
src/
  state_encoder.py     NLE obs -> structured features + delta encoding
  data_generator.py    Random play -> ShareGPT JSONL (forward model data)
  evaluator.py         Prediction accuracy scoring (field-level)
  reporter.py          HTML + text gameplay replays
  manifest.py          Attested manifest builder (SHA256 hashes)
nle_agent/
  agent_http.py        Action map (direction names -> NLE action indices)
docker-compose.yml     Phala CVM definition with GPU
tests/                 295 tests across 7 test files
```

## Quick Start

### Requirements

- Python 3.10+
- [NLE](https://github.com/heuritech/nle) (NetHack Learning Environment)
- For training: CUDA GPU, [Unsloth](https://github.com/unslothai/unsloth), TRL, PEFT

### Install

```bash
pip install nle pytest
# For GPU training:
pip install unsloth trl peft transformers datasets
```

### Smoke Test (no GPU needed)

Validates the full data generation -> JSONL verification -> manifest pipeline:

```bash
python3 cli.py smoke-test
```

### Generate Training Data

Play NetHack randomly to collect (state, action) -> delta examples:

```bash
python3 cli.py generate --num-games 200 --max-steps 50 --output data/train.jsonl
```

With train/eval split:

```bash
python3 cli.py generate \
  --num-games 200 \
  --max-steps 50 \
  --output data/train.jsonl \
  --eval-output data/eval.jsonl \
  --eval-fraction 0.2
```

This produces ShareGPT-formatted JSONL where each line is:

```json
{
  "conversations": [
    {"role": "system", "content": "Predict the outcome of a NetHack action..."},
    {"role": "user", "content": "HP:13/13 AC:7 Str:15 Dex:14\nPos:(4,8) Gold:0 Depth:1 Turn:45\nAdjacent: north=floor south=corridor east=wall west=floor\nMonsters: none\nItems: scroll@(3,7)\nAction: south"},
    {"role": "assistant", "content": "pos:(0,1) | hp:same | gold:same | depth:same | alive:yes | msg:"}
  ]
}
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

Training uses:
- 4-bit quantized base model via Unsloth
- LoRA rank 16, alpha 32, targeting all linear layers
- RSLoRA (rank-stabilized) for better convergence
- Gradient accumulation steps of 4
- 10 warmup steps

### Generate a Game Report

```bash
python3 cli.py report --seed 42 --max-steps 30 --output-dir output/reports
```

Produces:
- `output/reports/game_seed_42.html` -- scrollable dark-themed replay with HP bars, ASCII map, color-coded events (gold=green, damage=red, death=dark red, explore=blue)
- `output/reports/game_seed_42.txt` -- compact text replay
- Terminal output with step-by-step log

### Evaluate Predictions

Requires a running llama-server with the trained adapter:

```bash
# Start server (example)
llama-server -m model.gguf --port 8765 -c 2048

# Evaluate on unseen seeds
python3 cli.py evaluate --seeds 500,501,502,503,504 --max-steps 20
```

Output:

```
Evaluation Results:
  Examples evaluated: 87
  Exact match rate:   45.2%
  Position accuracy:  78.1%
  HP accuracy:        82.4%
  Gold accuracy:      91.3%
  Depth accuracy:     96.6%
  Survived accuracy:  94.8%
```

### Build Attested Manifest

```bash
python3 cli.py manifest \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --training-data data/train.jsonl \
  --adapter output/adapter \
  --baseline-scores '{"field_accuracy": 0.32}' \
  --post-scores '{"field_accuracy": 0.71}' \
  --output output/manifest.json
```

The manifest contains:
- SHA256 of training data file
- SHA256 of adapter weights (adapter_model.safetensors)
- Baseline and post-training evaluation scores
- Per-metric improvement deltas
- A self-hash: SHA256 of the entire manifest (excluding itself) for tamper detection

## Deploy to Phala CVM

The `docker-compose.yml` defines a single-shot training pipeline for Phala TEE deployment:

```bash
# Configure environment
cp .env.example .env
# Edit .env with your HF_TOKEN and WANDB_API_KEY

# Run the full pipeline inside CVM:
# generate -> train -> evaluate -> manifest
docker compose up
```

The container runs the complete pipeline:

1. **Generate** training data from random NLE gameplay (200 games, 50 steps each)
2. **Train** LoRA adapter on GPU
3. **Evaluate** on virgin seeds (500-504) that were never used in training
4. **Build** attested manifest with all hashes and scores

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `BASE_MODEL` | Qwen/Qwen2.5-3B-Instruct | HuggingFace model identifier |
| `NUM_GAMES` | 200 | Games to play for training data |
| `MAX_STEPS` | 50 | Max steps per game |
| `LORA_RANK` | 16 | LoRA adapter rank |
| `LORA_ALPHA` | 32 | LoRA alpha |
| `LR` | 2e-4 | Learning rate |
| `EPOCHS` | 1 | Training epochs |
| `BATCH_SIZE` | 4 | Per-device batch size |
| `HF_TOKEN` | -- | HuggingFace access token |
| `WANDB_API_KEY` | -- | Weights & Biases key |

## Running Tests

295 tests across 7 test files covering all modules:

```bash
pytest tests/ -v
```

Test files:

| File | Coverage |
|---|---|
| `test_state_encoder.py` | Full state encoding, delta encoding, tile naming, monster/item detection, prompt/target formatting |
| `test_data_generator.py` | Game generation, JSONL format, wall avoidance policy, train/eval splitting |
| `test_evaluator.py` | Prediction parsing, accuracy computation, model evaluation, test data generation |
| `test_reporter.py` | Text replay, HTML replay, HP bars, event descriptions, summary formatting |
| `test_manifest.py` | File hashing, manifest build/save/load/verify, self-hash integrity |
| `test_train.py` | Argument parsing, conversation formatting, dataset loading, training metadata |
| `test_cli.py` | All CLI subcommands, argument handling, smoke test end-to-end |

## How the Forward Model Works

The model is trained as a **forward model** -- given state S and action A, predict the delta to the next state S'. This is self-supervised from random exploration:

1. **Data collection**: Play NetHack with a wall-avoidance random policy. At each step, record (observation_before, action, observation_after).
2. **Feature extraction**: Convert raw NLE observations into compact structured text (position, HP, adjacent tiles, visible monsters/items).
3. **Delta computation**: Compute what changed between observations (position delta, HP delta, gold delta, depth delta, survival, message).
4. **Training**: Fine-tune a LoRA adapter to predict the delta given the state and action.
5. **Evaluation**: On unseen dungeon seeds, measure per-field prediction accuracy.

A model that achieves high accuracy on these predictions has learned NetHack's transition dynamics: which tiles are walkable, what happens in combat, how stairs work, what messages mean.

## License

See repository for license information.
