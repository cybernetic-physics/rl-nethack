# dstack-lora: Attested LoRA Training in TEE

## Concept

Train a LoRA adapter inside a Phala TEE CVM with a GPU, and produce a **cryptographically attested manifest** that proves:

1. Which base model was used (pinned by hash from HuggingFace)
2. What training data was used (hash-verified)
3. What training code ran (hash of the training script)
4. What the result was (LoRA adapter hash + loss metrics)
5. **Virgin benchmark**: test problems generated entirely inside the TEE, never seen by any human, used to evaluate the model before and after training

The virgin benchmark is the key innovation. In standard ML, you can never fully prove your test set wasn't contaminated (leaked into training data). By generating test instances inside the sealed TEE environment, we get a provably uncontaminated evaluation. The TEE's remote attestation proves the code that generated the questions is what we claim it is.

## Architecture

```
Phase 1: SETUP (inside TEE)
  - Pull base model from HuggingFace (verify SHA256 matches expected)
  - Load training data from volume (verify SHA256)
  - Hash all inputs -> record in manifest

Phase 2: VIRGIN BENCHMARK GENERATION (inside TEE, sealed)
  - Use a deterministic procedure to generate N test questions
  - The questions are written ONLY to a sealed file inside the TEE
  - NEVER logged externally, NEVER sent to any API
  - Hash the virgin benchmark -> record in manifest

Phase 3: BASELINE EVALUATION
  - Run base model (no LoRA) against virgin benchmark
  - Record baseline scores -> sealed in manifest

Phase 4: TRAINING
  - Run Unsloth LoRA fine-tuning
  - Record loss curve, hyperparams, final adapter weights
  - Hash the LoRA adapter -> record in manifest

Phase 5: POST-TRAINING EVALUATION
  - Merge LoRA adapter into base model
  - Run trained model against SAME virgin benchmark
  - Record post-training scores -> sealed in manifest

Phase 6: MANIFEST OUTPUT
  - Compile all hashes, metrics, configs into a signed JSON manifest
  - Include TEE attestation quote (proves which code ran)
  - Output manifest + virgin benchmark + results
```

## Model Selection

For a compelling demo that runs on a single GPU:

| Model | Params | VRAM needed (LoRA) | Why |
|-------|--------|---------------------|-----|
| Qwen3-4B | 4B | ~10GB | Sweet spot: capable enough to show real learning, fits on T4/4090 |
| Qwen3-1.7B | 1.7B | ~6GB | Fast iteration, good for testing |
| Llama-3.2-3B | 3B | ~8GB | Original choice, well-known |
| Qwen3-0.6B | 0.6B | ~3GB | Tiny, almost instant training |

**Recommendation**: Qwen3-4B -- big enough to actually learn something interesting, small enough for a single GPU. Falls back to Qwen3-1.7B if VRAM is tight.

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

**Recommendation**: Start with **Option A (Synthetic World)** for the MVP demo -- it's the clearest proof of concept. The fictional facts create an uncontaminatable test set by definition (the facts don't exist outside the TEE). Then we can extend to Option B for a more serious demo.

## Virgin Benchmark Generation

The virgin benchmark generator runs inside the TEE and produces test instances that NO human has ever seen. Strategy:

```
1. Define a seed + procedure (e.g., template + RNG)
2. Generate N question-answer pairs
3. Write to sealed file (never leaves TEE during generation)
4. Hash the file -> part of the attested manifest
5. Only revealed AFTER training is complete (in final manifest output)
```

For the synthetic world approach:
- The TEE generates a fictional world (names, relationships, events) using templates + seeded RNG
- Training data = a subset of facts about this world
- Virgin benchmark = questions about facts NOT in training data but derivable from the world structure
- This tests generalization, not memorization

For a simpler approach:
- Training data = 80% of the generated facts
- Virgin benchmark = the other 20%, sealed during training
- After training, reveal the virgin set and show model performance on it

## Attested Manifest Structure

```json
{
  "version": "1.0",
  "tee_attestation": {
    "quote": "<hex-encoded TEE quote>",
    "measurement": "<hash of CVM image>",
    "timestamp": "2026-04-02T14:30:00Z"
  },
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
    "sha256": "<hash>",
    "virgin_benchmark_generator": "generate_benchmark.py",
    "generator_sha256": "<hash>"
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
  "virgin_benchmark": {
    "num_questions": 50,
    "benchmark_sha256": "<hash of sealed benchmark>",
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

## Instance Type

Need a Phala GPU instance. The compose requests `nvidia` GPU devices. Expected to need:
- Minimum: T4 (16GB VRAM) for Qwen3-4B with LoRA
- Ideal: A100 (40GB) for headroom and faster training
- Disk: 50GB+ for model weights + data

## Project Structure

```
dstack-lora/
  docker-compose.yml      # CVM definition with GPU
  train.py                # Main training + eval pipeline
  generate_benchmark.py   # Virgin benchmark generator (runs in TEE)
  build_manifest.py       # Compile attested manifest
  dataset.jsonl           # Training data (or generated in-TEE)
  .env                    # HF_TOKEN, WANDB_API_KEY (secrets)
  .env.example            # Template
  deploy/
    key.pem               # SSH deploy key (gitignored)
  output/                 # LoRA adapter + manifest (after training)
  PLAN.md                 # This file
  README.md               # How to deploy and run
```

## Deployment Steps

```bash
# 1. Create .env with your tokens
cp .env.example .env
# edit .env with real tokens

# 2. Generate a deploy key
mkdir -p deploy
ssh-keygen -t ed25519 -f deploy/key.pem -N "" -C "dstack-lora-deploy"

# 3. Deploy to Phala
phala deploy --cvm-id dstack-lora \
  -c docker-compose.yml \
  -e .env \
  -t <gpu-instance-type> \
  --disk-size 80G \
  --wait

# 4. Monitor training
phala ssh dstack-lora -- -i deploy/key.pem "docker logs -f dstack-lora-trainer-1"

# 5. Fetch results
phala cp dstack-lora:output/manifest.json ./output/
```

## Open Questions

1. **Phala GPU availability**: Need to confirm which GPU instance types are available and their pricing
2. **Virgin benchmark seal timing**: Should the benchmark be generated before training starts, or should it be pre-generated and injected as a sealed volume?
3. **TEE attestation access**: Can we read the TEE quote from inside the CVM? (via /var/run/dstack.sock or dstack SDK)
4. **Self-hashing manifest**: The manifest can't include its own hash -- need to decide if we hash it after compilation or use a merkle-style approach
5. **Model download verification**: HuggingFace doesn't always provide file-level SHA256 in a standard way -- may need to compute on download
