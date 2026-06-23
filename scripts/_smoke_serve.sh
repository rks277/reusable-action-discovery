#!/bin/bash
# Bring up the 7B (GPU0) and 14B (GPU1) vLLM servers for the smoke test.
cd "$HOME/reusable-action-discovery" || exit 1
mkdir -p logs
pkill -f vllm.entrypoints 2>/dev/null
sleep 3
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --port 8001 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 256 --no-enable-log-requests \
  > logs/vllm_7b.log 2>&1 &
echo "7B pid $!"
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct --dtype bfloat16 --port 8002 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 192 --no-enable-log-requests \
  > logs/vllm_14b.log 2>&1 &
echo "14B pid $!"
