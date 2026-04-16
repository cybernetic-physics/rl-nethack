#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WEIGHTED_REPORT="${WEIGHTED_REPORT:-output/weighted_eval.json}"
PAIRWISE_REPORT="${PAIRWISE_REPORT:-output/pairwise_eval.json}"
KTO_REPORT="${KTO_REPORT:-output/kto_eval.json}"
OUTPUT="${OUTPUT:-output/preference_method_compare.json}"

CMD=(
  uv run python cli.py compare-long-sequence-evals
  --inputs
  "weighted=${WEIGHTED_REPORT}"
  "pairwise=${PAIRWISE_REPORT}"
  "kto=${KTO_REPORT}"
  --output "$OUTPUT"
)

printf 'Comparing long-sequence preference methods:\n%s\n' "${CMD[*]}"
exec "${CMD[@]}"
