#!/usr/bin/env python3
"""
KTO-style LoRA training for labeled long-sequence preference rows.

This trains on rows with:
- messages
- completion
- label (True/False) or sample_weight

The objective compares policy completion scores against a frozen reference
model score on the same completion:
- positives push policy above reference
- negatives push policy below reference
"""

from __future__ import annotations

import argparse
import json
import os
import time

from train import (
    LORA_TARGET_MODULES,
    cleanup_distributed,
    get_distributed_context,
    is_main_process,
    log,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="KTO-style LoRA fine-tuning for labeled long-sequence preference data")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct-1M")
    parser.add_argument("--reference-model", type=str, default=None, help="Optional explicit frozen reference model; defaults to --model")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--eval-data", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-seq-length", type=int, default=131072)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--desirable-weight", type=float, default=1.0)
    parser.add_argument("--undesirable-weight", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False)
    parser.add_argument("--load-in-4bit", action="store_true", default=False)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    return parser.parse_args(argv)


def format_messages_prefix(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    return "\n".join(parts) + "\n"


def normalize_kto_row(row: dict) -> dict:
    normalized = dict(row)
    if "messages" not in normalized and "conversations" in normalized:
        normalized["messages"] = normalized["conversations"][:-1]
    if "completion" not in normalized and "conversations" in normalized:
        normalized["completion"] = normalized["conversations"][-1]["content"]
    if "label" not in normalized and "sample_weight" in normalized:
        normalized["label"] = float(normalized["sample_weight"]) > 0.0
    return normalized


def load_kto_rows(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(normalize_kto_row(json.loads(line)))
    return rows


def truncate_rows(rows: list[dict], max_examples: int | None) -> list[dict]:
    if max_examples is None or len(rows) <= max_examples:
        return rows
    return rows[:max_examples]


def build_kto_texts(row: dict) -> tuple[str, str]:
    prompt_text = format_messages_prefix(row["messages"])
    completion_text = prompt_text + f"<|im_start|>assistant\n{row['completion']}<|im_end|>\n"
    return prompt_text, completion_text


def save_kto_training_metadata(output_dir, args, *, base_model, reference_model, data_path, data_hash, final_loss, global_steps, adapter_hash=None):
    meta = {
        "base_model": base_model,
        "reference_model": reference_model,
        "data_path": data_path,
        "data_hash": data_hash,
        "final_loss": final_loss,
        "global_steps": global_steps,
        "timestamp": time.time(),
        "adapter_hash": adapter_hash,
        "training_objective": "kto_style",
        "config": {
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_target_modules": LORA_TARGET_MODULES,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_seq_length": args.max_seq_length,
            "max_steps": args.max_steps,
            "load_in_4bit": args.load_in_4bit,
            "gradient_checkpointing": args.gradient_checkpointing,
            "beta": args.beta,
            "desirable_weight": args.desirable_weight,
            "undesirable_weight": args.undesirable_weight,
            "world_size": get_distributed_context()["world_size"],
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return meta


def main():
    args = parse_args()
    dist = get_distributed_context()

    try:
        import torch
        import torch.nn.functional as F
        from datasets import Dataset
        from torch.nn.utils.rnn import pad_sequence
        from transformers import Trainer, TrainingArguments
        from unsloth import FastLanguageModel
        from src.manifest import hash_file

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for KTO training")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if dist["world_size"] > 1:
            torch.cuda.set_device(dist["local_rank"])

        bf16_supported = torch.cuda.is_bf16_supported()
        use_bf16 = bf16_supported
        use_fp16 = not use_bf16

        reference_model_name = args.reference_model or args.model

        log("=== KTO-Style LoRA Training ===")
        log(f"Base model      : {args.model}")
        log(f"Reference model : {reference_model_name}")
        log(f"Train data      : {args.data}")
        log(f"Eval data       : {args.eval_data}")
        log(f"Output dir      : {args.output}")

        data_hash = hash_file(args.data)

        def load_model(model_name: str, trainable: bool):
            load_kwargs = {
                "model_name": model_name,
                "max_seq_length": args.max_seq_length,
                "load_in_4bit": args.load_in_4bit,
                "dtype": torch.bfloat16 if use_bf16 else torch.float16,
            }
            if dist["world_size"] > 1:
                load_kwargs["device_map"] = {"": torch.cuda.current_device()}
            model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
            if trainable:
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
                model.eval()
                for param in model.parameters():
                    param.requires_grad_(False)
            return model, tokenizer

        policy_model, tokenizer = load_model(args.model, trainable=True)
        reference_model, _ = load_model(reference_model_name, trainable=False)

        def preprocess_rows(rows: list[dict]) -> Dataset:
            processed = []
            for row in rows:
                prompt_text, completion_text = build_kto_texts(row)
                prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                full_ids = tokenizer(
                    completion_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=args.max_seq_length,
                )["input_ids"]
                prompt_len = min(len(prompt_ids), len(full_ids))
                processed.append(
                    {
                        "input_ids": full_ids,
                        "labels": ([-100] * prompt_len) + full_ids[prompt_len:],
                        "label_sign": 1.0 if bool(row["label"]) else -1.0,
                    }
                )
            return Dataset.from_list(processed)

        train_rows = truncate_rows(load_kto_rows(args.data), args.max_train_examples)
        train_dataset = preprocess_rows(train_rows)
        eval_dataset = None
        if args.eval_data and os.path.isfile(args.eval_data):
            eval_rows = truncate_rows(load_kto_rows(args.eval_data), args.max_eval_examples)
            eval_dataset = preprocess_rows(eval_rows)

        class KTODataCollator:
            def __init__(self, tokenizer_obj):
                self.tokenizer = tokenizer_obj

            def __call__(self, features):
                pad_id = self.tokenizer.pad_token_id or 0
                input_ids = pad_sequence(
                    [torch.tensor(feature["input_ids"], dtype=torch.long) for feature in features],
                    batch_first=True,
                    padding_value=pad_id,
                )
                labels = pad_sequence(
                    [torch.tensor(feature["labels"], dtype=torch.long) for feature in features],
                    batch_first=True,
                    padding_value=-100,
                )
                label_sign = torch.tensor([float(feature["label_sign"]) for feature in features], dtype=torch.float32)
                return {
                    "input_ids": input_ids,
                    "attention_mask": input_ids.ne(pad_id).long(),
                    "labels": labels,
                    "label_sign": label_sign,
                }

        def sequence_mean_logprob(logits, labels):
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            token_logps = -F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(shift_labels.size(0), shift_labels.size(1))
            valid = shift_labels.ne(-100)
            token_logps = token_logps * valid
            return token_logps.sum(dim=1) / valid.sum(dim=1).clamp_min(1)

        class KTOTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                labels = inputs["labels"]
                label_sign = inputs["label_sign"].to(labels.device)
                policy_outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                with torch.no_grad():
                    ref_outputs = reference_model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )
                policy_score = sequence_mean_logprob(policy_outputs.logits, labels)
                ref_score = sequence_mean_logprob(ref_outputs.logits, labels)
                margin = policy_score - ref_score
                positive_mask = label_sign.gt(0)
                negative_mask = label_sign.lt(0)
                losses = []
                if positive_mask.any():
                    losses.append(
                        -args.desirable_weight * F.logsigmoid(args.beta * margin[positive_mask]).mean()
                    )
                if negative_mask.any():
                    losses.append(
                        -args.undesirable_weight * F.logsigmoid(-args.beta * margin[negative_mask]).mean()
                    )
                loss = sum(losses) / max(1, len(losses))
                if return_outputs:
                    return loss, {"policy_score": policy_score.detach(), "ref_score": ref_score.detach()}
                return loss

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
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_pin_memory=True,
            gradient_checkpointing=args.gradient_checkpointing,
            save_on_each_node=False,
            remove_unused_columns=False,
        )

        trainer = KTOTrainer(
            model=policy_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=KTODataCollator(tokenizer),
        )

        train_result = trainer.train()
        final_loss = train_result.training_loss
        global_steps = train_result.global_step

        os.makedirs(args.output, exist_ok=True)
        trainer.save_model(args.output)
        tokenizer.save_pretrained(args.output)

        adapter_weights_path = os.path.join(args.output, "adapter_model.safetensors")
        adapter_hash = hash_file(adapter_weights_path) if os.path.isfile(adapter_weights_path) else None

        if is_main_process():
            save_kto_training_metadata(
                args.output,
                args,
                base_model=args.model,
                reference_model=reference_model_name,
                data_path=args.data,
                data_hash=data_hash,
                final_loss=final_loss,
                global_steps=global_steps,
                adapter_hash=adapter_hash,
            )
            log(f"Training metadata saved to: {os.path.join(args.output, 'training_meta.json')}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
