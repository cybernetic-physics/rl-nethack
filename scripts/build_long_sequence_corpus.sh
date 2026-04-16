#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${OUTPUT:-data/corpus/long_sequence_mixed_1b.jsonl}"
MANIFEST_OUTPUT="${MANIFEST_OUTPUT:-${OUTPUT}.manifest.json}"
TARGET_TOKENS="${TARGET_TOKENS:-1000000000}"
FULL_EPISODE_FRACTION="${FULL_EPISODE_FRACTION:-0.40}"
VERY_LONG_FRACTION="${VERY_LONG_FRACTION:-0.35}"
LONG_FRACTION="${LONG_FRACTION:-0.20}"
MEDIUM_FRACTION="${MEDIUM_FRACTION:-0.05}"
FULL_EPISODE_WINNING_SHARE="${FULL_EPISODE_WINNING_SHARE:-0.70}"
VERY_LONG_MIN_TOKENS="${VERY_LONG_MIN_TOKENS:-65536}"
LONG_MIN_TOKENS="${LONG_MIN_TOKENS:-16384}"
MEDIUM_MIN_TOKENS="${MEDIUM_MIN_TOKENS:-4096}"
VERY_LONG_STRIDE="${VERY_LONG_STRIDE:-16}"
LONG_STRIDE="${LONG_STRIDE:-32}"
MEDIUM_STRIDE="${MEDIUM_STRIDE:-64}"
INPUTS="${INPUTS:-}"

if [[ -z "${INPUTS}" ]]; then
  echo "Set INPUTS to a comma-separated list of long-sequence JSONL shards." >&2
  exit 1
fi

declare -a INPUT_ARGS=()
IFS=',' read -r -a INPUT_ARRAY <<< "${INPUTS}"
for path in "${INPUT_ARRAY[@]}"; do
  trimmed="$(echo "${path}" | xargs)"
  if [[ -n "${trimmed}" ]]; then
    INPUT_ARGS+=(--input "${trimmed}")
  fi
done

echo "Building token-budgeted long-sequence corpus"
echo "  output:          ${OUTPUT}"
echo "  manifest:        ${MANIFEST_OUTPUT}"
echo "  target tokens:   ${TARGET_TOKENS}"
echo "  inputs:"
for path in "${INPUT_ARRAY[@]}"; do
  echo "    - $(echo "${path}" | xargs)"
done

exec uv run python cli.py build-long-sequence-corpus \
  "${INPUT_ARGS[@]}" \
  --output "${OUTPUT}" \
  --manifest-output "${MANIFEST_OUTPUT}" \
  --target-tokens "${TARGET_TOKENS}" \
  --full-episode-fraction "${FULL_EPISODE_FRACTION}" \
  --very-long-fraction "${VERY_LONG_FRACTION}" \
  --long-fraction "${LONG_FRACTION}" \
  --medium-fraction "${MEDIUM_FRACTION}" \
  --full-episode-winning-share "${FULL_EPISODE_WINNING_SHARE}" \
  --very-long-min-tokens "${VERY_LONG_MIN_TOKENS}" \
  --long-min-tokens "${LONG_MIN_TOKENS}" \
  --medium-min-tokens "${MEDIUM_MIN_TOKENS}" \
  --very-long-stride "${VERY_LONG_STRIDE}" \
  --long-stride "${LONG_STRIDE}" \
  --medium-stride "${MEDIUM_STRIDE}"
