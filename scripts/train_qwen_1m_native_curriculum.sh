#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct-1M}"
TRAIN_DATA="${TRAIN_DATA:-data/long_sequence_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-data/long_sequence_eval.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/qwen_1m_native_curriculum}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-262144}"
CURRICULUM_BUCKETS="${CURRICULUM_BUCKETS:-128k,256k,512k,1M}"
CURRICULUM_STAGE_REPEATS="${CURRICULUM_STAGE_REPEATS:-2,2,1,1}"
CURRICULUM_METADATA_KEY="${CURRICULUM_METADATA_KEY:-target_context_bucket}"
COMMON_EXTRA_ARGS="${COMMON_EXTRA_ARGS:-}"

CMD=(
  uv run torchrun --nproc_per_node=4 train.py
  --model "$MODEL"
  --data "$TRAIN_DATA"
  --eval-data "$EVAL_DATA"
  --output "$OUTPUT_DIR"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --gradient-checkpointing
  --curriculum-buckets "$CURRICULUM_BUCKETS"
  --curriculum-stage-repeats "$CURRICULUM_STAGE_REPEATS"
  --curriculum-metadata-key "$CURRICULUM_METADATA_KEY"
)

if [[ -n "$COMMON_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( $COMMON_EXTRA_ARGS )
  CMD+=("${EXTRA_ARGS[@]}")
fi

printf 'Running native single-run curriculum training:\n%s\n' "${CMD[*]}"
exec "${CMD[@]}"
