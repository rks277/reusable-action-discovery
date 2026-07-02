#!/usr/bin/env bash
# Clean re-run of 9B and 27B with context-length fix (max-model-len 32768 for 9B,
# 16384 for 27B-GPTQ to stay within 40GB + enforce-eager).
# Launch: cd ~/reusable-action-discovery && rm -f JOB_DONE.flag \
#   && setsid nohup bash scripts/run_qwen35_oracle_9b27b.sh > logs/oracle_9b27b.out 2>&1 < /dev/null &
set -euo pipefail; trap '' HUP
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$REPO/.venv/bin/python"; LOG="$REPO/logs/oracle_9b27b.log"
PORT=8000; VLLM_URL="http://127.0.0.1:$PORT/v1"
mkdir -p "$REPO/logs"; exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%H:%M:%S')] $*"; }

pkill -9 -f "vllm serve" 2>/dev/null || true; sleep 5

start_vllm() {
    local model="$1"; shift
    log "Starting vLLM: $model  flags: $*"
    setsid nohup "$REPO/.venv/bin/vllm" serve "$model" \
        --port $PORT --enable-auto-tool-choice --tool-call-parser qwen3_xml \
        --no-enable-log-requests "$@" \
        > "$REPO/logs/vllm_${model//\//_}_clean.log" 2>&1 < /dev/null &
    VLLM_PID=$!
    log "PID=$VLLM_PID -- waiting for /health ..."
    for i in $(seq 1 90); do
        sleep 5
        curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && log "healthy after $((i*5))s" && return 0
    done
    log "ERROR: not healthy"; kill "$VLLM_PID" 2>/dev/null || true; return 1
}

stop_vllm() {
    kill "$VLLM_PID" 2>/dev/null || true; sleep 5
    kill -9 "$VLLM_PID" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true; sleep 5
}

# 9B — 18GB in BF16, plenty of room for 32k ctx on 40GB A100
log "=== 9B ==="
if start_vllm "Qwen/Qwen3.5-9B" \
    --gpu-memory-utilization 0.92 --max-model-len 32768; then
    VLLM_BASE_URL="$VLLM_URL" "$PYTHON" "$REPO/scripts/run_oracle_sweep.py" \
        --qwen --model "Qwen/Qwen3.5-9B" || log "WARN: 9B sweep failed"
    stop_vllm
fi

# 27B-GPTQ — ~14GB weights; 16k ctx uses ~8GB KV cache, total ~22GB < 40GB
log "=== 27B ==="
if start_vllm "Qwen/Qwen3.5-27B-GPTQ-Int4" \
    --quantization gptq --enforce-eager \
    --gpu-memory-utilization 0.88 --max-model-len 16384; then
    VLLM_BASE_URL="$VLLM_URL" "$PYTHON" "$REPO/scripts/run_oracle_sweep.py" \
        --qwen --model "Qwen/Qwen3.5-27B-GPTQ-Int4" || log "WARN: 27B sweep failed"
    stop_vllm
fi

touch "$REPO/JOB_DONE.flag"
log "=== ALL DONE ==="
