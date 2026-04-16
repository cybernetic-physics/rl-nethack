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
import inspect
import json
import os
import sys
import time
from collections import defaultdict

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
    parser.add_argument(
        "--metadata-equals", type=str, nargs="*", default=None,
        help="Optional metadata equality filters as key=value, e.g. target_context_bucket=256k outcome=win",
    )
    parser.add_argument(
        "--max-train-examples", type=int, default=None,
        help="Optional cap on the number of training examples after filtering",
    )
    parser.add_argument(
        "--max-eval-examples", type=int, default=None,
        help="Optional cap on the number of eval examples after filtering",
    )
    parser.add_argument(
        "--weighted-sft", action="store_true", default=False,
        help="Enable sample-weighted SFT if the dataset carries sample_weight or label fields",
    )
    parser.add_argument(
        "--preference-positive-weight", type=float, default=1.0,
        help="Default sample weight for label=True preference rows when converting them for weighted SFT",
    )
    parser.add_argument(
        "--preference-negative-weight", type=float, default=-0.25,
        help="Default sample weight for label=False preference rows when converting them for weighted SFT",
    )
    parser.add_argument(
        "--curriculum-buckets", type=str, default="",
        help="Comma-separated context buckets for a native single-run curriculum, e.g. 128k,256k,512k,1M",
    )
    parser.add_argument(
        "--curriculum-stage-repeats", type=str, default="",
        help="Comma-separated repeat counts per curriculum stage, e.g. 2,2,1,1",
    )
    parser.add_argument(
        "--curriculum-metadata-key", type=str, default="target_context_bucket",
        help="Metadata key used to identify curriculum buckets (default: target_context_bucket)",
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


def normalize_training_row(row, positive_weight=1.0, negative_weight=-0.25):
    """Normalize supported row schemas into ShareGPT-style training rows."""
    normalized = dict(row)
    if "conversations" not in normalized and "messages" in normalized and "completion" in normalized:
        normalized["conversations"] = [
            *list(normalized["messages"]),
            {"role": "assistant", "content": normalized["completion"]},
        ]
    if "sample_weight" not in normalized and "label" in normalized:
        normalized["sample_weight"] = positive_weight if bool(normalized["label"]) else negative_weight
    return normalized


def normalize_training_dataset(dataset, positive_weight=1.0, negative_weight=-0.25):
    """Normalize all supported row schemas into one trainable dataset schema."""
    from datasets import Dataset

    rows = [
        normalize_training_row(
            row,
            positive_weight=positive_weight,
            negative_weight=negative_weight,
        )
        for row in dataset
    ]
    return Dataset.from_list(rows)


def dataset_has_sample_weights(dataset):
    return "sample_weight" in getattr(dataset, "column_names", [])


def parse_curriculum_buckets(raw: str) -> list[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def parse_curriculum_stage_repeats(raw: str, stage_count: int) -> list[int]:
    if not raw:
        return [1] * stage_count
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    repeats = [int(part) for part in parts]
    if len(repeats) != stage_count:
        raise ValueError(
            f"curriculum-stage-repeats must have {stage_count} entries, got {len(repeats)}"
        )
    return repeats


def parse_metadata_filters(filters):
    """Parse key=value CLI filters into a dict."""
    parsed = {}
    for item in filters or []:
        if "=" not in item:
            raise ValueError(f"Invalid metadata filter {item!r}; expected key=value")
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def _normalize_metadata_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def filter_dataset_by_metadata(dataset, metadata_filters):
    """Keep only rows whose metadata matches all requested key=value filters."""
    if not metadata_filters:
        return dataset

    def row_matches(row):
        metadata = row.get("metadata") or {}
        for key, expected in metadata_filters.items():
            if _normalize_metadata_value(metadata.get(key)) != expected:
                return False
        return True

    keep_indices = [idx for idx, row in enumerate(dataset) if row_matches(row)]
    return dataset.select(keep_indices)


def build_curriculum_dataset(dataset, buckets, stage_repeats, metadata_key="target_context_bucket"):
    """
    Build a single-run cumulative curriculum dataset.

    Stage i contains all examples whose metadata bucket is in buckets[: i + 1].
    Each stage can be repeated multiple times to bias the schedule toward
    shorter or longer contexts without requiring a multi-run shell wrapper.
    """
    from datasets import concatenate_datasets

    if not buckets:
        return dataset
    if len(stage_repeats) != len(buckets):
        raise ValueError("stage_repeats must match buckets length")

    bucket_rows: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(dataset):
        metadata = row.get("metadata") or {}
        bucket_rows[str(metadata.get(metadata_key, "unknown"))].append(idx)

    stages = []
    for stage_end, repeat in enumerate(stage_repeats):
        if repeat <= 0:
            continue
        indices = []
        for bucket in buckets[: stage_end + 1]:
            indices.extend(bucket_rows.get(bucket, []))
        if not indices:
            continue
        stage_dataset = dataset.select(indices)
        stages.extend([stage_dataset] * repeat)

    if not stages:
        return dataset.select([])
    if len(stages) == 1:
        return stages[0]
    return concatenate_datasets(stages)


def truncate_dataset(dataset, max_examples):
    """Cap dataset length while preserving order."""
    if max_examples is None or len(dataset) <= max_examples:
        return dataset
    return dataset.select(range(max_examples))


def prepare_dataset(dataset, num_proc):
    """Pre-format ShareGPT conversations into text to reduce trainer overhead."""
    if "text" in dataset.column_names or "conversations" not in dataset.column_names:
        return dataset

    return dataset.map(
        format_dataset_conversations,
        batched=True,
        num_proc=max(1, num_proc),
    )


def tokenize_text_dataset(dataset, tokenizer, max_seq_length, num_proc):
    """Tokenize preformatted text rows for plain transformers Trainer fallback."""
    if "input_ids" in dataset.column_names:
        return dataset
    if "text" not in dataset.column_names:
        dataset = prepare_dataset(dataset, num_proc)

    def tokenize_batch(examples):
        encoded = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        encoded["labels"] = [list(ids) for ids in encoded["input_ids"]]
        return encoded

    return dataset.map(
        tokenize_batch,
        batched=True,
        num_proc=max(1, num_proc),
        remove_columns=[col for col in dataset.column_names if col not in {"sample_weight"}],
    )


def weighted_mean_loss(per_example_loss, sample_weight):
    """Compute a signed weighted mean with abs-weight normalization."""
    import torch

    weights = sample_weight.to(per_example_loss.device).to(per_example_loss.dtype)
    denom = weights.abs().sum().clamp_min(torch.finfo(per_example_loss.dtype).eps)
    return (per_example_loss * weights).sum() / denom


def build_training_arguments_kwargs(training_arguments_cls, **kwargs):
    """Filter kwargs to those supported by the installed transformers version."""
    supported = set(inspect.signature(training_arguments_cls.__init__).parameters)
    return {key: value for key, value in kwargs.items() if key in supported}


def build_trainer_init_kwargs(trainer_cls, **kwargs):
    """Filter trainer init kwargs to those supported by the installed version."""
    supported = set(inspect.signature(trainer_cls.__init__).parameters)
    return {key: value for key, value in kwargs.items() if key in supported}


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
            "weighted_sft": getattr(args, "weighted_sft", False),
            "preference_positive_weight": getattr(args, "preference_positive_weight", 1.0),
            "preference_negative_weight": getattr(args, "preference_negative_weight", -0.25),
            "curriculum_buckets": parse_curriculum_buckets(getattr(args, "curriculum_buckets", "")),
            "curriculum_stage_repeats": getattr(args, "curriculum_stage_repeats", ""),
            "curriculum_metadata_key": getattr(args, "curriculum_metadata_key", "target_context_bucket"),
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
        import torch.nn.functional as F
        try:
            from unsloth import FastLanguageModel
        except Exception:
            FastLanguageModel = None
        try:
            from trl import SFTTrainer
        except Exception:
            SFTTrainer = None
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
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
        log(f"Weighted SFT: {args.weighted_sft}")
        using_unsloth = FastLanguageModel is not None and SFTTrainer is not None
        log(f"Trainer stack: {'unsloth+trl' if using_unsloth else 'transformers+peft fallback'}")
        curriculum_buckets = parse_curriculum_buckets(args.curriculum_buckets)
        curriculum_stage_repeats = parse_curriculum_stage_repeats(
            args.curriculum_stage_repeats,
            len(curriculum_buckets),
        )
        if curriculum_buckets:
            log(f"Curriculum buckets: {curriculum_buckets}")
            log(f"Curriculum repeats: {curriculum_stage_repeats}")

        data_hash = hash_file(args.data)
        log(f"Data hash  : {data_hash}")
        metadata_filters = parse_metadata_filters(args.metadata_equals)
        if metadata_filters:
            log(f"Metadata filters: {metadata_filters}")

        if using_unsloth:
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
        else:
            from peft import LoraConfig, get_peft_model

            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            )
            if args.gradient_checkpointing:
                model.gradient_checkpointing_enable()
            peft_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.0,
                bias="none",
                target_modules=LORA_TARGET_MODULES,
                task_type="CAUSAL_LM",
                use_rslora=True,
            )
            model = get_peft_model(model, peft_config)

        train_dataset = load_training_data(args.data)
        train_dataset = normalize_training_dataset(
            train_dataset,
            positive_weight=args.preference_positive_weight,
            negative_weight=args.preference_negative_weight,
        )
        train_dataset = filter_dataset_by_metadata(train_dataset, metadata_filters)
        train_dataset = truncate_dataset(train_dataset, args.max_train_examples)
        train_dataset = build_curriculum_dataset(
            train_dataset,
            curriculum_buckets,
            curriculum_stage_repeats,
            metadata_key=args.curriculum_metadata_key,
        )
        if using_unsloth:
            train_dataset = prepare_dataset(train_dataset, args.dataset_num_proc)
        else:
            train_dataset = tokenize_text_dataset(train_dataset, tokenizer, args.max_seq_length, args.dataset_num_proc)
        use_weighted_sft = bool(args.weighted_sft or dataset_has_sample_weights(train_dataset))
        log(f"Training examples: {len(train_dataset)}")
        if use_weighted_sft:
            log("Using sample-weighted SFT loss")

        eval_dataset = None
        if args.eval_data and os.path.isfile(args.eval_data):
            eval_dataset = load_training_data(args.eval_data)
            eval_dataset = normalize_training_dataset(
                eval_dataset,
                positive_weight=args.preference_positive_weight,
                negative_weight=args.preference_negative_weight,
            )
            eval_dataset = filter_dataset_by_metadata(eval_dataset, metadata_filters)
            eval_dataset = truncate_dataset(eval_dataset, args.max_eval_examples)
            if using_unsloth:
                eval_dataset = prepare_dataset(eval_dataset, args.dataset_num_proc)
            else:
                eval_dataset = tokenize_text_dataset(eval_dataset, tokenizer, args.max_seq_length, args.dataset_num_proc)
            log(f"Eval examples: {len(eval_dataset)}")

        training_args = TrainingArguments(
            **build_training_arguments_kwargs(
                TrainingArguments,
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
                remove_unused_columns=False,
                group_by_length=not bool(curriculum_buckets),
            )
        )

        class SampleWeightDataCollator:
            def __init__(self, base_collator):
                self.base_collator = base_collator

            def __call__(self, features):
                sample_weights = [float(feature.get("sample_weight", 1.0)) for feature in features]
                stripped = []
                for feature in features:
                    copied = dict(feature)
                    copied.pop("sample_weight", None)
                    stripped.append(copied)
                batch = self.base_collator(stripped)
                batch["sample_weight"] = torch.tensor(sample_weights, dtype=torch.float32)
                return batch

        base_trainer_cls = SFTTrainer if using_unsloth else Trainer

        class CurriculumSFTTrainer(base_trainer_cls):
            def _get_train_sampler(self, *args, **kwargs):
                if curriculum_buckets:
                    from torch.utils.data import SequentialSampler

                    return SequentialSampler(self.train_dataset)
                return super()._get_train_sampler(*args, **kwargs)

        class WeightedSFTTrainer(CurriculumSFTTrainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                sample_weight = inputs.pop("sample_weight", None)
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits
                if labels is None:
                    loss = outputs.get("loss") if isinstance(outputs, dict) else outputs.loss
                    return (loss, outputs) if return_outputs else loss

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                token_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view(shift_labels.size(0), shift_labels.size(1))
                valid_mask = shift_labels.ne(-100)
                token_loss = token_loss * valid_mask
                per_example_loss = token_loss.sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1)
                if sample_weight is None:
                    loss = per_example_loss.mean()
                else:
                    loss = weighted_mean_loss(per_example_loss, sample_weight)
                return (loss, outputs) if return_outputs else loss

        trainer_cls = WeightedSFTTrainer if use_weighted_sft else CurriculumSFTTrainer
        trainer_kwargs = {}
        if curriculum_buckets:
            from transformers import TrainerCallback

            class CurriculumProgressCallback(TrainerCallback):
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs is None:
                        return
                    logs.setdefault("curriculum_buckets", ",".join(curriculum_buckets))

            trainer_kwargs["callbacks"] = [CurriculumProgressCallback()]

        if using_unsloth:
            trainer = trainer_cls(
                **build_trainer_init_kwargs(
                    trainer_cls,
                    model=model,
                    tokenizer=tokenizer,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    dataset_text_field="text",
                    max_seq_length=args.max_seq_length,
                    args=training_args,
                    packing=args.packing,
                    **trainer_kwargs,
                )
            )
            if use_weighted_sft:
                trainer.data_collator = SampleWeightDataCollator(trainer.data_collator)
        else:
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            if use_weighted_sft:
                data_collator = SampleWeightDataCollator(data_collator)
            trainer = trainer_cls(
                **build_trainer_init_kwargs(
                    trainer_cls,
                    model=model,
                    tokenizer=tokenizer,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    args=training_args,
                    data_collator=data_collator,
                    **trainer_kwargs,
                )
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
