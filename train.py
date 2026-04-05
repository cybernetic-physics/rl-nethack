#!/usr/bin/env python3
"""
Unsloth LoRA training script for local GPU execution.

Trains a LoRA adapter on ShareGPT-formatted JSONL data using Unsloth + TRL.

Usage:
    python train.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --data /data/train.jsonl \
        --eval-data /data/eval.jsonl \
        --output /data/output/adapter \
        --max-seq-length 1024 \
        --lora-rank 16 \
        --lora-alpha 32 \
        --lr 2e-4 \
        --epochs 1 \
        --batch-size 4 \
        --max-steps -1
"""

import argparse
import json
import os
import sys
import time

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _env_int(name, default):
    """Parse an integer env var with a fallback."""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def get_distributed_context():
    """Return rank/local_rank/world_size from torchrun-style env vars."""
    return {
        "rank": _env_int("RANK", 0),
        "local_rank": _env_int("LOCAL_RANK", 0),
        "world_size": _env_int("WORLD_SIZE", 1),
    }


def is_main_process():
    """Whether this process is global rank 0."""
    return get_distributed_context()["rank"] == 0


def log(message):
    """Print from the main process only."""
    if is_main_process():
        print(message)


def cleanup_distributed():
    """Tear down torch.distributed cleanly when launched with torchrun."""
    try:
        import torch
    except Exception:
        return

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

# ---------------------------------------------------------------------------
# Utility functions (importable without GPU)
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    """Parse CLI arguments and return the namespace."""
    parser = argparse.ArgumentParser(
        description="Unsloth LoRA fine-tuning for NLE game prediction"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model name or path (default: Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument(
        "--data", type=str, default="/data/train.jsonl",
        help="Path to training JSONL data in ShareGPT format",
    )
    parser.add_argument(
        "--eval-data", type=str, default=None,
        help="Path to evaluation JSONL data (optional)",
    )
    parser.add_argument(
        "--output", type=str, default="/data/output/adapter",
        help="Output directory for the LoRA adapter",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=1024,
        help="Maximum sequence length for training (default: 1024)",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=32,
        help="LoRA alpha (default: 32)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Per-device batch size (default: 4)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=4,
        help="Gradient accumulation steps per process (default: 4)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=-1,
        help="Max training steps (-1 = use epochs, default: -1)",
    )
    parser.add_argument(
        "--dataset-num-proc", type=int, default=max(1, min(8, os.cpu_count() or 1)),
        help="Worker processes for dataset preprocessing (default: min(8, cpu_count))",
    )
    parser.add_argument(
        "--dataloader-num-workers", type=int, default=4,
        help="DataLoader workers per process (default: 4)",
    )
    parser.add_argument(
        "--logging-steps", type=int, default=10,
        help="Training log frequency in optimizer steps (default: 10)",
    )
    parser.add_argument(
        "--save-steps", type=int, default=100,
        help="Checkpoint save frequency in optimizer steps (default: 100)",
    )
    parser.add_argument(
        "--save-total-limit", type=int, default=2,
        help="How many checkpoints to keep (default: 2)",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=10,
        help="Warmup steps (default: 10)",
    )
    parser.add_argument(
        "--ddp-find-unused-parameters", action="store_true", default=False,
        help="Enable DDP unused-parameter discovery (default: disabled for speed)",
    )
    parser.add_argument(
        "--gradient-checkpointing", action="store_true", default=False,
        help="Enable gradient checkpointing to save memory at some speed cost",
    )
    parser.add_argument(
        "--packing", action="store_true", default=False,
        help="Enable sequence packing in SFTTrainer if supported",
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true", default=False,
        help="Use 4-bit QLoRA loading. Disabled by default because bf16 LoRA is faster on H100s",
    )
    parser.add_argument(
        "--eval-after-train", action="store_true", default=False,
        help="Run evaluation after training via llama-server",
    )
    parser.add_argument(
        "--eval-seeds", type=str, default="42,43,44",
        help="Comma-separated seeds for post-training evaluation (default: 42,43,44)",
    )
    parser.add_argument(
        "--eval-server", type=str, default="http://127.0.0.1:8765",
        help="llama-server URL for post-training evaluation",
    )
    return parser.parse_args(argv)


def format_conversation_text(conversation):
    """Format a ShareGPT conversation dict into a single text string for training.

    Takes a dict with a 'conversations' key containing a list of
    {"role": ..., "content": ...} messages and produces a ChatML-style
    formatted string.

    Args:
        conversation: dict with 'conversations' key (list of message dicts).

    Returns:
        str: formatted conversation text.
    """
    parts = []
    for msg in conversation["conversations"]:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    # Join with newlines and add a final newline for the assistant generation target
    text = "\n".join(parts) + "\n"
    return text


def format_dataset_conversations(examples):
    """Map function for HuggingFace Dataset: format conversations column to text.

    This is used as the `formatting_func` or dataset map to convert
    the ShareGPT conversations into plain text for SFTTrainer.

    Args:
        examples: batch dict with 'conversations' key (list of list of dicts).

    Returns:
        dict with 'text' key containing formatted strings.
    """
    texts = []
    for conv in examples["conversations"]:
        texts.append(format_conversation_text({"conversations": conv}))
    return {"text": texts}


def load_training_data(data_path):
    """Load JSONL training data as a HuggingFace Dataset.

    Args:
        data_path: path to JSONL file in ShareGPT format.

    Returns:
        HuggingFace Dataset with 'conversations' column.
    """
    from datasets import Dataset

    # Read JSONL manually to handle the format, then create Dataset
    rows = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    dataset = Dataset.from_list(rows)
    return dataset


def prepare_dataset(dataset, num_proc):
    """Pre-format ShareGPT conversations into text to reduce trainer overhead."""
    if "text" in dataset.column_names or "conversations" not in dataset.column_names:
        return dataset

    return dataset.map(
        format_dataset_conversations,
        batched=True,
        num_proc=max(1, num_proc),
    )


def save_training_metadata(
    output_dir,
    base_model,
    data_path,
    data_hash,
    final_loss,
    global_steps,
    args,
    adapter_hash=None,
):
    """Save training metadata to output_dir/training_meta.json.

    Args:
        output_dir: output directory path.
        base_model: base model name.
        data_path: path to training data.
        data_hash: SHA256 of training data file.
        final_loss: final training loss.
        global_steps: number of global training steps.
        args: parsed CLI args (for lora/learning rate config).
        adapter_hash: SHA256 of the saved adapter weights file (optional).

    Returns:
        dict: the metadata that was saved.
    """
    meta = {
        "base_model": base_model,
        "data_path": data_path,
        "data_hash": data_hash,
        "final_loss": final_loss,
        "global_steps": global_steps,
        "config": {
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_target_modules": LORA_TARGET_MODULES,
            "lora_use_rslora": True,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": getattr(args, "gradient_accumulation_steps", 4),
            "max_seq_length": args.max_seq_length,
            "max_steps": args.max_steps,
            "world_size": get_distributed_context()["world_size"],
            "load_in_4bit": getattr(args, "load_in_4bit", False),
            "gradient_checkpointing": getattr(args, "gradient_checkpointing", False),
        },
        "adapter_hash": adapter_hash,
        "timestamp": time.time(),
    }

    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "training_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    return meta


# ---------------------------------------------------------------------------
# Main training function (requires GPU + unsloth)
# ---------------------------------------------------------------------------

def main():
    """Main training entry point. Requires GPU, unsloth, trl, peft."""
    args = parse_args()
    dist = get_distributed_context()

    try:
        # -- Late imports: these require GPU --
        import torch
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from src.manifest import hash_file

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for training")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if dist["world_size"] > 1:
            torch.cuda.set_device(dist["local_rank"])

        bf16_supported = torch.cuda.is_bf16_supported()
        use_bf16 = bf16_supported
        use_fp16 = not use_bf16

        global_batch = args.batch_size * args.gradient_accumulation_steps * dist["world_size"]

        log("=== Unsloth LoRA Training ===")
        log(f"Base model : {args.model}")
        log(f"Train data : {args.data}")
        log(f"Eval data  : {args.eval_data}")
        log(f"Output dir : {args.output}")
        log(f"LoRA rank  : {args.lora_rank}")
        log(f"LoRA alpha : {args.lora_alpha}")
        log(f"LR         : {args.lr}")
        log(f"Epochs     : {args.epochs}")
        log(f"Batch size : {args.batch_size} per GPU")
        log(f"Grad accum : {args.gradient_accumulation_steps}")
        log(f"World size : {dist['world_size']}")
        log(f"Global batch: {global_batch}")
        log(f"Max steps  : {args.max_steps}")
        log(f"Max seq len: {args.max_seq_length}")
        log(f"Precision  : {'bf16' if use_bf16 else 'fp16'}")
        log(f"4-bit load : {args.load_in_4bit}")

        data_hash = hash_file(args.data)
        log(f"Data hash  : {data_hash}")

        model_load_kwargs = {
            "model_name": args.model,
            "max_seq_length": args.max_seq_length,
            "load_in_4bit": args.load_in_4bit,
            "dtype": torch.bfloat16 if use_bf16 else torch.float16,
        }
        if dist["world_size"] > 1:
            model_load_kwargs["device_map"] = {"": torch.cuda.current_device()}

        model, tokenizer = FastLanguageModel.from_pretrained(**model_load_kwargs)

        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            target_modules=LORA_TARGET_MODULES,
            use_rslora=True,
            bias="none",
            use_gradient_checkpointing="unsloth" if args.gradient_checkpointing else False,
        )

        train_dataset = prepare_dataset(load_training_data(args.data), args.dataset_num_proc)
        log(f"Training examples: {len(train_dataset)}")

        eval_dataset = None
        if args.eval_data and os.path.isfile(args.eval_data):
            eval_dataset = prepare_dataset(load_training_data(args.eval_data), args.dataset_num_proc)
            log(f"Eval examples: {len(eval_dataset)}")

        training_args = TrainingArguments(
            output_dir=args.output,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps if args.max_steps > 0 else -1,
            learning_rate=args.lr,
            fp16=use_fp16,
            bf16=use_bf16,
            logging_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            seed=42,
            report_to="none",
            ddp_find_unused_parameters=args.ddp_find_unused_parameters,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_pin_memory=True,
            gradient_checkpointing=args.gradient_checkpointing,
            save_on_each_node=False,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            args=training_args,
            packing=args.packing,
        )

        log("Starting training...")
        train_result = trainer.train()

        final_loss = train_result.training_loss
        global_steps = train_result.global_step
        log(f"Training complete. Final loss: {final_loss:.4f}, Steps: {global_steps}")

        os.makedirs(args.output, exist_ok=True)
        trainer.save_model(args.output)
        tokenizer.save_pretrained(args.output)

        if dist["world_size"] > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        log(f"Adapter saved to: {args.output}")

        adapter_weights_path = os.path.join(args.output, "adapter_model.safetensors")
        adapter_hash = None
        if os.path.isfile(adapter_weights_path):
            adapter_hash = hash_file(adapter_weights_path)
            log(f"Adapter hash: {adapter_hash}")

        if is_main_process():
            save_training_metadata(
                output_dir=args.output,
                base_model=args.model,
                data_path=args.data,
                data_hash=data_hash,
                final_loss=final_loss,
                global_steps=global_steps,
                args=args,
                adapter_hash=adapter_hash,
            )
            print(f"Training metadata saved to: {os.path.join(args.output, 'training_meta.json')}")

        if args.eval_after_train and is_main_process():
            print("Running post-training evaluation...")
            from src.evaluator import run_evaluation
            from src.state_encoder import StateEncoder

            seeds = [int(s.strip()) for s in args.eval_seeds.split(",")]
            encoder = StateEncoder()
            eval_result = run_evaluation(
                seeds=seeds,
                max_steps=10,
                encoder=encoder,
                server_url=args.eval_server,
            )
            print(f"Evaluation results: {json.dumps(eval_result['accuracy'], indent=2)}")

        log("=== Done ===")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
