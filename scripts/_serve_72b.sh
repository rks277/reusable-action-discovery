#!/bin/bash
# Bring up the 72B (FP8, TP=2 across both GPUs) vLLM server for benchmarking.
cd "$HOME/reusable-action-discovery" || exit 1
mkdir -p logs
pkill -f vllm.entrypoints 2>/dev/null
sleep 4
CUDA_VISIBLE_DEVICES=0,1 nohup .venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-72B-Instruct --quantization fp8 --tensor-parallel-size 2 \
  --port 8003 --max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 32 \
  --no-enable-log-requests \
  > logs/vllm_72b.log 2>&1 &
echo "72B pid $!"
