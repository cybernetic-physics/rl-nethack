#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-3B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT0="${PORT0:-8000}"
PORT1="${PORT1:-8001}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cleanup() {
  jobs -p | xargs -r kill
  wait || true
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="${GPU0}" uv run vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT0}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --served-model-name "${MODEL}" \
  --enforce-eager ${EXTRA_ARGS} &

CUDA_VISIBLE_DEVICES="${GPU1}" uv run vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT1}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --served-model-name "${MODEL}" \
  --enforce-eager ${EXTRA_ARGS} &

echo "Replicas starting:"
echo "  http://${HOST}:${PORT0}/v1 on GPU ${GPU0}"
echo "  http://${HOST}:${PORT1}/v1 on GPU ${GPU1}"

wait
