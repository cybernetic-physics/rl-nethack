#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_NAME="${DATASET_NAME:-nld-nao}"
OUTPUT="${OUTPUT:-data/${DATASET_NAME}_gold_wins_long_sequences.jsonl}"
DBFILENAME="${DBFILENAME:-ttyrecs.db}"
ROOT_PATH="${ROOT_PATH:-}"
DATASET_TYPE="${DATASET_TYPE:-altorg}"
MAX_GAMES="${MAX_GAMES:-256}"
MIN_TURNS="${MIN_TURNS:-1000}"
MIN_MAXLVL="${MIN_MAXLVL:-10}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-128000}"
SEQ_LENGTH="${SEQ_LENGTH:-64}"

CMD=(
  uv run python cli.py import-nld-long-sequences
  --dataset-name "$DATASET_NAME"
  --output "$OUTPUT"
  --dbfilename "$DBFILENAME"
  --wins-only
  --max-games "$MAX_GAMES"
  --min-turns "$MIN_TURNS"
  --min-maxlvl "$MIN_MAXLVL"
  --seq-length "$SEQ_LENGTH"
  --max-context-tokens "$MAX_CONTEXT_TOKENS"
)

if [[ -n "$ROOT_PATH" ]]; then
  CMD+=(--root-path "$ROOT_PATH" --dataset-type "$DATASET_TYPE")
fi

printf 'Importing NLD gold wins shard:\n%s\n' "${CMD[*]}"
exec "${CMD[@]}"
