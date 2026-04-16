# rl-nethack Training Plan

## Recent Status

- Local training path is now DDP-enabled via `torchrun`
- Repo dependency management is now `uv`-based
- Local policy generation now has a vLLM backend pinned to GPUs 0,1
- A 5k local policy dataset was generated quickly, but its action quality is too weak to treat as final training data

## Concept

Train a LoRA adapter on a local GPU machine and produce a manifest that records:

1. Which base model was used
2. What training data was used
3. What training code ran
4. What the result was
5. Which held-out benchmarks were used before and after training

## Architecture

```
Phase 1: SETUP
  - Pull base model from HuggingFace (verify SHA256 matches expected)
  - Load training data from volume (verify SHA256)
  - Hash all inputs -> record in manifest

Phase 2: BENCHMARK PREPARATION
  - Define a deterministic held-out evaluation set
  - Hash the benchmark inputs -> record in manifest

Phase 3: BASELINE EVALUATION
  - Run base model (no LoRA) against held-out benchmark
  - Record baseline scores -> sealed in manifest

Phase 4: TRAINING
  - Run Unsloth LoRA fine-tuning
  - Record loss curve, hyperparams, final adapter weights
  - Hash the LoRA adapter -> record in manifest

Phase 5: POST-TRAINING EVALUATION
  - Run trained model against the same held-out benchmark
  - Record post-training scores -> sealed in manifest

Phase 6: MANIFEST OUTPUT
  - Compile all hashes, metrics, configs into a JSON manifest
  - Output manifest + virgin benchmark + results
```
## Model Selection

This host has 4x H200s, so VRAM is not the immediate constraint. Start with Qwen 2.5 3B or 7B for fast iteration, then scale sequence length, batch size, or model size as needed.

## Current Recommendation

The old bottleneck was infrastructure. The current bottleneck is data quality.

Recommended order of work:
1. Fix local policy generation quality with a stronger small model and better prompts.
2. Change data generation from request-by-request serving to a true batched/offline pipeline.
3. Regenerate a materially larger corpus.
4. Then spend the extra H200 capacity on bigger forward-model runs.

## Training Task: What's Interesting?

The training task needs to be:
1. Small enough to complete in a reasonable time (hours, not days)
2. Verifiable -- we can clearly see the model learned something
3. Interesting enough to demonstrate the concept

### Option A: Synthetic World Knowledge
Create a fictional world with ~100 facts (characters, places, events). Train the model on these facts. Virgin benchmark asks questions about this world. Clear signal: model knows fictional facts it didn't know before, and we can prove the test questions were sealed.

**Pros**: Very clean signal, easy to verify, fun demo
**Cons**: Not "useful" training, just a proof of concept

### Option B: Code Generation for a Mini-Language
Define a tiny DSL (domain-specific language). Train the model to generate correct programs in it. Virgin benchmark = unseen problem specs. Evaluate by running the generated code.

**Pros**: Executable evaluation (pass/fail is objective), impressive demo
**Cons**: More complex setup

### Option C: Style Transfer / Persona Adoption
Train the model to respond in a specific distinctive style (e.g., pirate speak, a specific author's voice). Virgin benchmark = unseen prompts, evaluated by classifier.

**Pros**: Visually obvious results
**Cons**: Harder to quantify objectively

### Option D: Arithmetic / Logic Puzzles
Generate arithmetic problems or simple logic puzzles at varying difficulty. Train on a subset, test on the rest (virgin set). Evaluation is exact match.

**Pros**: Objectively evaluable, clear before/after signal
**Cons**: Might be too easy for larger models (ceiling effect)

**Recommendation**: Start with **Option A (Synthetic World)** for the MVP demo because it gives the clearest before/after training signal.
## Manifest Structure

```json
{
  "version": "1.0",
  "base_model": {
    "name": "Qwen/Qwen3-4B",
    "revision": "abc123def",
    "sha256": "<hash of model files>",
    "source": "huggingface"
  },
  "training_data": {
    "file": "dataset.jsonl",
    "sha256": "<hash>",
    "size_bytes": 1234567,
    "num_samples": 1000
  },
  "training_code": {
    "file": "train.py",
    "sha256": "<hash>"
  },
  "training_config": {
    "lora_rank": 16,
    "lora_alpha": 32,
    "learning_rate": 2e-4,
    "epochs": 3,
    "batch_size": 4,
    "max_seq_length": 2048
  },
  "training_results": {
    "final_loss": 0.234,
    "loss_curve": [1.2, 0.8, 0.5, 0.3, 0.234],
    "training_duration_seconds": 3600,
    "adapter_sha256": "<hash of LoRA weights>",
    "adapter_size_bytes": 45678901
  },
  "benchmark": {
    "num_questions": 50,
    "benchmark_sha256": "<hash>",
    "baseline_scores": {
      "exact_match": 0.12,
      "contains_answer": 0.20
    },
    "post_training_scores": {
      "exact_match": 0.68,
      "contains_answer": 0.82
    },
    "improvement": {
      "exact_match_delta": 0.56,
      "contains_answer_delta": 0.62
    }
  },
  "artifacts": {
    "base_model_hash": "...",
    "adapter_path": "/data/output/adapter",
    "manifest_hash": "<self-hash after compilation>"
  }
}
```

## Local Machine Target

Target machine: 4x H200 GPUs with enough headroom for larger models, longer sequence lengths, and faster iteration.

## Project Structure

```
rl-nethack/
  docker-compose.yml      # Local Docker Compose job with GPU access
  train.py                # Main training + eval pipeline
  build_manifest.py       # Compile manifest
  dataset.jsonl           # Training data
  .env                    # HF_TOKEN, WANDB_API_KEY (secrets)
  .env.example            # Template
  deploy/
    key.pem               # SSH deploy key (gitignored)
  output/                 # LoRA adapter + manifest (after training)
  PLAN.md                 # This file
  README.md               # How to deploy and run
```

## Local Run Steps

```bash
# 1. Install the local environment
uv sync --extra train --extra test --extra serve

# 2. Optional: start local policy serving on GPUs 0,1
CUDA_VISIBLE_DEVICES=0,1 ./scripts/start_vllm_policy_server.sh Qwen/Qwen2.5-1.5B-Instruct

# 3. Generate data
uv run python scripts/generate_training_data.py --backend vllm --model Qwen/Qwen2.5-1.5B-Instruct --server-url http://127.0.0.1:8000/v1 --num-games 200 --max-steps 50 --workers 64 --cooldown 0

# 4. Train on all 4 GPUs
uv run torchrun --standalone --nproc_per_node=4 train.py ...
```

## Open Questions

1. How much action-quality improvement do we get by moving local policy generation from 0.5B to 1.5B or 3B?
2. Should `scripts/generate_training_data.py` grow a true batch/offline vLLM path instead of issuing one HTTP request per step?
3. Once data quality is fixed, should the forward model move directly to Qwen 2.5 7B on all 4 H200s?
