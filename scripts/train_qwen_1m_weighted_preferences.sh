#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct-1M}"
TRAIN_DATA="${TRAIN_DATA:-data/long_sequences_kto_weighted_sft_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-}"
OUTPUT_DIR="${OUTPUT_DIR:-output/qwen_1m_weighted_preferences}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-131072}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
POSITIVE_WEIGHT="${POSITIVE_WEIGHT:-1.0}"
NEGATIVE_WEIGHT="${NEGATIVE_WEIGHT:--0.25}"

CMD=(
  uv run python train.py
  --model "$MODEL"
  --data "$TRAIN_DATA"
  --output "$OUTPUT_DIR"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --lora-rank "$LORA_RANK"
  --lora-alpha "$LORA_ALPHA"
  --lr "$LR"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRAD_ACCUM"
  --warmup-steps "$WARMUP_STEPS"
  --logging-steps "$LOGGING_STEPS"
  --save-steps "$SAVE_STEPS"
  --save-total-limit "$SAVE_TOTAL_LIMIT"
  --gradient-checkpointing
  --weighted-sft
  --preference-positive-weight "$POSITIVE_WEIGHT"
  --preference-negative-weight "$NEGATIVE_WEIGHT"
)

if [[ -n "$EVAL_DATA" ]]; then
  CMD+=(--eval-data "$EVAL_DATA")
fi

printf 'Running weighted preference LoRA training:\n%s\n' "${CMD[*]}"
exec "${CMD[@]}"
