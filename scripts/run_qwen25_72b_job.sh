#!/usr/bin/env bash
# Qwen2.5-72B-Instruct on the oracle env — 2x H100 80GB, BF16, tensor-parallel-size 2.
# BF16 72B (~145GB) fits across 2x80GB (160GB) but KV cache is TIGHT, so: enforce-eager
# (no cudagraph memory), modest max-model-len, low concurrency. Obfuscated tool names,
# hermes parser (Qwen2.5 has no thinking mode).
# Launch:
#   cd ~/reusable-action-discovery && rm -f JOB_DONE.flag \
#   && setsid nohup bash scripts/run_qwen25_72b_job.sh > logs/oracle_72b.out 2>&1 < /dev/null &
set -euo pipefail; trap '' HUP
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$REPO/.venv/bin/python"; LOG="$REPO/logs/oracle_72b.log"
PORT=8000; VLLM_URL="http://127.0.0.1:$PORT/v1"
MODEL="Qwen/Qwen2.5-72B-Instruct"
mkdir -p "$REPO/logs"; exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%H:%M:%S')] $*"; }

pkill -9 -f "vllm serve" 2>/dev/null || true; sleep 4

log "GPUs visible:"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /'

log "Starting vLLM: $MODEL  (TP=2, BF16, enforce-eager)"
setsid nohup "$REPO/.venv/bin/vllm" serve "$MODEL" --port $PORT \
    --tensor-parallel-size 2 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --no-enable-log-requests --enforce-eager \
    --gpu-memory-utilization 0.95 --max-model-len 12288 \
    > "$REPO/logs/vllm_72b.log" 2>&1 < /dev/null &
VLLM_PID=$!
log "vLLM PID=$VLLM_PID -- waiting for /health (TP init + 145GB load can take several min) ..."
healthy=0
for i in $(seq 1 180); do        # up to 15 min
    sleep 5
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        log "healthy after $((i*5))s"; healthy=1; break
    fi
done
if [ "$healthy" -ne 1 ]; then
    log "ERROR: vLLM not healthy in 900s -- check logs/vllm_72b.log (likely OOM; drop --max-model-len to 8192)"
    kill -9 "$VLLM_PID" 2>/dev/null || true; exit 1
fi

log "Running sweep (concurrency=6)"
VLLM_BASE_URL="$VLLM_URL" "$PYTHON" "$REPO/scripts/run_oracle_sweep.py" \
    --qwen25 --model "$MODEL" --concurrency 6 || log "WARN: sweep failed"

log "Stopping vLLM"
kill "$VLLM_PID" 2>/dev/null || true; sleep 5; kill -9 "$VLLM_PID" 2>/dev/null || true
pkill -9 -f "vllm" 2>/dev/null || true
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | while read p; do [ -n "$p" ] && kill -9 "$p" 2>/dev/null || true; done

touch "$REPO/JOB_DONE.flag"
log "======== 72B DONE ========"
