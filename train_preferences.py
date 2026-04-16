#!/usr/bin/env python3
"""
Pairwise preference LoRA training for long-sequence NetHack data.

This script trains a policy to score teacher-preferred actions above rejected
losing actions on the same state history. It uses a simple pairwise logistic
loss over completion log-probabilities and does not require TRL.
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
    parser = argparse.ArgumentParser(description="Pairwise preference LoRA fine-tuning for long-sequence NetHack data")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct-1M")
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
    parser.add_argument("--dataset-num-proc", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--beta", type=float, default=1.0, help="Pairwise logistic scale on chosen vs rejected score margin")
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


def normalize_preference_row(row: dict) -> dict:
    normalized = dict(row)
    if "messages" not in normalized and "conversations" in normalized:
        normalized["messages"] = normalized["conversations"][:-1]
    return normalized


def load_preference_data(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(normalize_preference_row(json.loads(line)))
    return rows


def truncate_rows(rows: list[dict], max_examples: int | None) -> list[dict]:
    if max_examples is None or len(rows) <= max_examples:
        return rows
    return rows[:max_examples]


def build_preference_texts(row: dict) -> tuple[str, str, str]:
    prompt_text = format_messages_prefix(row["messages"])
    chosen_text = prompt_text + f"<|im_start|>assistant\n{row['chosen']}<|im_end|>\n"
    rejected_text = prompt_text + f"<|im_start|>assistant\n{row['rejected']}<|im_end|>\n"
    return prompt_text, chosen_text, rejected_text


def save_preference_training_metadata(output_dir, args, *, base_model, data_path, data_hash, final_loss, global_steps, adapter_hash=None):
    meta = {
        "base_model": base_model,
        "data_path": data_path,
        "data_hash": data_hash,
        "final_loss": final_loss,
        "global_steps": global_steps,
        "timestamp": time.time(),
        "adapter_hash": adapter_hash,
        "training_objective": "pairwise_preference",
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
            raise RuntimeError("CUDA is required for preference training")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if dist["world_size"] > 1:
            torch.cuda.set_device(dist["local_rank"])

        bf16_supported = torch.cuda.is_bf16_supported()
        use_bf16 = bf16_supported
        use_fp16 = not use_bf16

        log("=== Pairwise Preference LoRA Training ===")
        log(f"Base model : {args.model}")
        log(f"Train data : {args.data}")
        log(f"Eval data  : {args.eval_data}")
        log(f"Output dir : {args.output}")
        log(f"Beta       : {args.beta}")
        log(f"Max seq len: {args.max_seq_length}")

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

        def preprocess_rows(rows: list[dict]) -> Dataset:
            processed = []
            for row in rows:
                prompt_text, chosen_text, rejected_text = build_preference_texts(row)
                prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                chosen_ids = tokenizer(chosen_text, add_special_tokens=False, truncation=True, max_length=args.max_seq_length)["input_ids"]
                rejected_ids = tokenizer(rejected_text, add_special_tokens=False, truncation=True, max_length=args.max_seq_length)["input_ids"]
                prompt_len_chosen = min(len(prompt_ids), len(chosen_ids))
                prompt_len_rejected = min(len(prompt_ids), len(rejected_ids))
                processed.append(
                    {
                        "chosen_input_ids": chosen_ids,
                        "chosen_labels": ([-100] * prompt_len_chosen) + chosen_ids[prompt_len_chosen:],
                        "rejected_input_ids": rejected_ids,
                        "rejected_labels": ([-100] * prompt_len_rejected) + rejected_ids[prompt_len_rejected:],
                    }
                )
            return Dataset.from_list(processed)

        train_rows = truncate_rows(load_preference_data(args.data), args.max_train_examples)
        train_dataset = preprocess_rows(train_rows)
        eval_dataset = None
        if args.eval_data and os.path.isfile(args.eval_data):
            eval_rows = truncate_rows(load_preference_data(args.eval_data), args.max_eval_examples)
            eval_dataset = preprocess_rows(eval_rows)

        class PairwiseDataCollator:
            def __init__(self, tokenizer_obj):
                self.tokenizer = tokenizer_obj

            def __call__(self, features):
                def _pad(name, pad_value):
                    tensors = [torch.tensor(feature[name], dtype=torch.long) for feature in features]
                    return pad_sequence(tensors, batch_first=True, padding_value=pad_value)

                chosen_input_ids = _pad("chosen_input_ids", self.tokenizer.pad_token_id or 0)
                chosen_labels = _pad("chosen_labels", -100)
                rejected_input_ids = _pad("rejected_input_ids", self.tokenizer.pad_token_id or 0)
                rejected_labels = _pad("rejected_labels", -100)
                return {
                    "chosen_input_ids": chosen_input_ids,
                    "chosen_attention_mask": chosen_input_ids.ne(self.tokenizer.pad_token_id or 0).long(),
                    "chosen_labels": chosen_labels,
                    "rejected_input_ids": rejected_input_ids,
                    "rejected_attention_mask": rejected_input_ids.ne(self.tokenizer.pad_token_id or 0).long(),
                    "rejected_labels": rejected_labels,
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

        class PairwisePreferenceTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                chosen_outputs = model(
                    input_ids=inputs["chosen_input_ids"],
                    attention_mask=inputs["chosen_attention_mask"],
                )
                rejected_outputs = model(
                    input_ids=inputs["rejected_input_ids"],
                    attention_mask=inputs["rejected_attention_mask"],
                )
                chosen_score = sequence_mean_logprob(chosen_outputs.logits, inputs["chosen_labels"])
                rejected_score = sequence_mean_logprob(rejected_outputs.logits, inputs["rejected_labels"])
                margin = chosen_score - rejected_score
                loss = -F.logsigmoid(args.beta * margin).mean()
                if return_outputs:
                    return loss, {
                        "chosen_score": chosen_score.detach(),
                        "rejected_score": rejected_score.detach(),
                    }
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

        trainer = PairwisePreferenceTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=PairwiseDataCollator(tokenizer),
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
            save_preference_training_metadata(
                args.output,
                args,
                base_model=args.model,
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
