#!/usr/bin/env bash
set -euo pipefail

# Adjust these paths if needed
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B-Base}"
LORA_NAME="${LORA_NAME:-hack}"
LORA_PATH="${LORA_PATH:-/home/patrick/cotcontrol/qwen3_8b_sft_reward_hacker_hack/checkpoint-205}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Keep max model len conservative so the server starts reliably
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
DTYPE="${DTYPE:-bfloat16}"

echo "Starting server"
echo "  BASE_MODEL=$BASE_MODEL"
echo "  LORA_NAME=$LORA_NAME"
echo "  LORA_PATH=$LORA_PATH"
echo "  HOST=$HOST PORT=$PORT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
vllm serve "$BASE_MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --trust-remote-code \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --enable-lora \
  --max-lora-rank 64 \
  --lora-modules "${LORA_NAME}=${LORA_PATH}"