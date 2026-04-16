#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct-1M}"
TRAIN_DATA="${TRAIN_DATA:-data/long_sequence_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-data/long_sequence_eval.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/qwen_1m_curriculum}"
CONTEXT_BUCKETS="${CONTEXT_BUCKETS:-128k,256k,512k,1M}"
SEQ_LENGTHS="${SEQ_LENGTHS:-131072,262144,524288,1010000}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-}"
COMMON_EXTRA_ARGS="${COMMON_EXTRA_ARGS:-}"

IFS=',' read -r -a BUCKET_ARRAY <<< "${CONTEXT_BUCKETS}"
IFS=',' read -r -a SEQ_ARRAY <<< "${SEQ_LENGTHS}"

if [[ "${#BUCKET_ARRAY[@]}" -ne "${#SEQ_ARRAY[@]}" ]]; then
  echo "CONTEXT_BUCKETS and SEQ_LENGTHS must have the same number of entries" >&2
  exit 1
fi

for idx in "${!BUCKET_ARRAY[@]}"; do
  bucket="${BUCKET_ARRAY[$idx]}"
  seq_len="${SEQ_ARRAY[$idx]}"
  stage_dir="${OUTPUT_ROOT}/stage_${idx}_${bucket}"
  extra_args="${COMMON_EXTRA_ARGS}"
  if [[ -n "${MAX_TRAIN_EXAMPLES}" ]]; then
    extra_args="${extra_args} --max-train-examples ${MAX_TRAIN_EXAMPLES}"
  fi
  if [[ -n "${MAX_EVAL_EXAMPLES}" ]]; then
    extra_args="${extra_args} --max-eval-examples ${MAX_EVAL_EXAMPLES}"
  fi

  echo
  echo "=== Curriculum stage ${idx} (${bucket}) ==="
  echo "  seq len: ${seq_len}"
  echo "  output:  ${stage_dir}"

  uv run torchrun --nproc_per_node=4 train.py \
    --model "${MODEL}" \
    --data "${TRAIN_DATA}" \
    --eval-data "${EVAL_DATA}" \
    --output "${stage_dir}" \
    --max-seq-length "${seq_len}" \
    --metadata-equals "target_context_bucket=${bucket}" \
    --gradient-checkpointing \
    ${extra_args}
done
