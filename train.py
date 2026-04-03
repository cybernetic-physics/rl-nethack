#!/usr/bin/env python3
"""
Unsloth LoRA Training Script for GPU CVM Deployment.

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
        "--max-steps", type=int, default=-1,
        help="Max training steps (-1 = use epochs, default: -1)",
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
            "lora_target_modules": "all-linear",
            "lora_use_rslora": True,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_seq_length": args.max_seq_length,
            "max_steps": args.max_steps,
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

    # -- Late imports: these require GPU --
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from src.manifest import hash_file

    print(f"=== Unsloth LoRA Training ===")
    print(f"Base model : {args.model}")
    print(f"Train data : {args.data}")
    print(f"Eval data  : {args.eval_data}")
    print(f"Output dir : {args.output}")
    print(f"LoRA rank  : {args.lora_rank}")
    print(f"LoRA alpha : {args.lora_alpha}")
    print(f"LR         : {args.lr}")
    print(f"Epochs     : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"Max steps  : {args.max_steps}")
    print(f"Max seq len: {args.max_seq_length}")

    # 1. Compute data hash before training
    data_hash = hash_file(args.data)
    print(f"Data hash  : {data_hash}")

    # 2. Load base model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,  # auto-detect
    )

    # 3. Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules="all-linear",
        use_rslora=True,
        bias="none",
    )

    # 4. Load training data
    train_dataset = load_training_data(args.data)
    print(f"Training examples: {len(train_dataset)}")

    # 5. Load eval data if provided
    eval_dataset = None
    if args.eval_data and os.path.isfile(args.eval_data):
        eval_dataset = load_training_data(args.eval_data)
        print(f"Eval examples: {len(eval_dataset)}")

    # 6. Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.lr,
        fp16=not hasattr(model, "is_bf16_supported"),
        bf16=False,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        seed=42,
        report_to="none",
    )

    # 7. Create SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        formatting_func=format_conversation_text if "conversations" in train_dataset.column_names else None,
        max_seq_length=args.max_seq_length,
        args=training_args,
    )

    # If the dataset has a 'conversations' column but no 'text', map it
    if "conversations" in train_dataset.column_names and "text" not in train_dataset.column_names:
        train_dataset = train_dataset.map(
            format_dataset_conversations,
            batched=True,
            remove_columns=["conversations"],
        )
        trainer.train_dataset = train_dataset

    # 8. Train
    print("Starting training...")
    train_result = trainer.train()

    final_loss = train_result.training_loss
    global_steps = train_result.global_step
    print(f"Training complete. Final loss: {final_loss:.4f}, Steps: {global_steps}")

    # 9. Save LoRA adapter
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Adapter saved to: {args.output}")

    # 10. Compute adapter hash
    adapter_weights_path = os.path.join(args.output, "adapter_model.safetensors")
    adapter_hash = None
    if os.path.isfile(adapter_weights_path):
        adapter_hash = hash_file(adapter_weights_path)
        print(f"Adapter hash: {adapter_hash}")

    # 11. Save training metadata
    meta = save_training_metadata(
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

    # 12. Optional post-training evaluation
    if args.eval_after_train:
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

    print("=== Done ===")


if __name__ == "__main__":
    main()
