#!/usr/bin/env bash
# Clean re-run of 2B and 4B with the context-length fix (max-model-len 32768 +
# in-code graceful overflow handling). Launch:
#   cd ~/reusable-action-discovery && rm -f JOB_DONE.flag \
#   && setsid nohup bash scripts/run_qwen35_oracle_2b4b.sh > logs/oracle_2b4b.out 2>&1 < /dev/null &
set -euo pipefail; trap '' HUP
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$REPO/.venv/bin/python"; LOG="$REPO/logs/oracle_2b4b.log"
PORT=8000; VLLM_URL="http://127.0.0.1:$PORT/v1"
mkdir -p "$REPO/logs"; exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%H:%M:%S')] $*"; }

start_vllm() {
    local model="$1"
    log "Starting vLLM: $model (max-model-len 32768)"
    setsid nohup "$REPO/.venv/bin/vllm" serve "$model" \
        --port $PORT --enable-auto-tool-choice --tool-call-parser qwen3_xml \
        --no-enable-log-requests --gpu-memory-utilization 0.92 --max-model-len 32768 \
        > "$REPO/logs/vllm_${model//\//_}.log" 2>&1 < /dev/null &
    VLLM_PID=$!
    log "PID=$VLLM_PID -- waiting for /health ..."
    for i in $(seq 1 90); do
        sleep 5
        curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && log "healthy after $((i*5))s" && return 0
    done
    log "ERROR: not healthy in 450s"; kill "$VLLM_PID" 2>/dev/null || true; return 1
}

stop_vllm() {
    kill "$VLLM_PID" 2>/dev/null || true; sleep 5
    kill -9 "$VLLM_PID" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true; sleep 4
}

# make sure nothing is already on the GPU
pkill -9 -f "vllm serve" 2>/dev/null || true; sleep 4

for model in "Qwen/Qwen3.5-2B" "Qwen/Qwen3.5-4B"; do
    log "=== $model ==="
    if start_vllm "$model"; then
        VLLM_BASE_URL="$VLLM_URL" "$PYTHON" "$REPO/scripts/run_oracle_sweep.py" \
            --qwen --model "$model" || log "WARN: sweep failed for $model"
        stop_vllm
    else
        log "SKIP $model (vLLM failed to start)"
    fi
done
log "=== DONE ===" && touch "$REPO/JOB_DONE.flag"
