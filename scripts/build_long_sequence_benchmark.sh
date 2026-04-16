#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INPUT="${INPUT:-data/long_sequence_eval.jsonl}"
OUTPUT="${OUTPUT:-data/long_sequence_benchmark.jsonl}"
PER_BUCKET="${PER_BUCKET:-32}"
PER_PHASE="${PER_PHASE:-32}"
PER_ACTION_FAMILY="${PER_ACTION_FAMILY:-32}"

CMD=(
  uv run python cli.py build-long-sequence-benchmark
  --input "$INPUT"
  --output "$OUTPUT"
  --per-bucket "$PER_BUCKET"
  --per-phase "$PER_PHASE"
  --per-action-family "$PER_ACTION_FAMILY"
)

printf 'Building long-sequence benchmark shard:\n%s\n' "${CMD[*]}"
exec "${CMD[@]}"
