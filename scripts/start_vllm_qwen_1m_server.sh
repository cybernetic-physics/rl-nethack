#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ENV_FILE="${SCRIPT_DIR}/qwen_1m_vllm.env"
if [[ -f "${DEFAULT_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${DEFAULT_ENV_FILE}"
fi

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct-1M}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1010000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-131072}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

ARGS=(
  serve "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --served-model-name "${MODEL}"
)

if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  ARGS+=(--enable-chunked-prefill)
fi

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  ARGS+=(--enforce-eager)
fi

echo "Launching ${MODEL}"
echo "  host:                 ${HOST}:${PORT}"
echo "  gpus:                 ${CUDA_VISIBLE_DEVICES}"
echo "  tensor parallel:      ${TENSOR_PARALLEL_SIZE}"
echo "  max model len:        ${MAX_MODEL_LEN}"
echo "  max num seqs:         ${MAX_NUM_SEQS}"
echo "  max batched tokens:   ${MAX_NUM_BATCHED_TOKENS}"
echo "  gpu mem utilization:  ${GPU_MEMORY_UTILIZATION}"

exec uv run vllm "${ARGS[@]}" ${EXTRA_ARGS}
