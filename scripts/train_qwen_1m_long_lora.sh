#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct-1M}"
TRAIN_DATA="${TRAIN_DATA:-data/long_sequence_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-data/long_sequence_eval.jsonl}"
OUTPUT="${OUTPUT:-output/qwen_1m_long_lora}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-131072}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LR="${LR:-2e-4}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Training long-context LoRA"
echo "  model:           ${MODEL}"
echo "  train data:      ${TRAIN_DATA}"
echo "  eval data:       ${EVAL_DATA}"
echo "  output:          ${OUTPUT}"
echo "  max seq length:  ${MAX_SEQ_LENGTH}"

exec uv run torchrun --nproc_per_node=4 train.py \
  --model "${MODEL}" \
  --data "${TRAIN_DATA}" \
  --eval-data "${EVAL_DATA}" \
  --output "${OUTPUT}" \
  --max-seq-length "${MAX_SEQ_LENGTH}" \
  --lora-rank "${LORA_RANK}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lr "${LR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRAD_ACCUM}" \
  --dataset-num-proc "${DATASET_NUM_PROC}" \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
  --gradient-checkpointing \
  ${EXTRA_ARGS}
