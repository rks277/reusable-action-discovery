#!/usr/bin/env bash
# Run the remaining work: 27B sweep (with enforce-eager fix) + 2B/4B re-runs (with hallucination fix).
# Launch detached:
#   cd ~/reusable-action-discovery && rm -f JOB_DONE.flag \
#   && setsid nohup bash scripts/run_qwen35_oracle_remaining.sh > logs/oracle_remaining.out 2>&1 < /dev/null &
set -euo pipefail
trap '' HUP

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$REPO/.venv/bin/python"
LOG="$REPO/logs/oracle_remaining.log"
PORT=8000
VLLM_URL="http://127.0.0.1:$PORT/v1"

mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

start_vllm() {
    local model="$1"; shift
    log "Starting vLLM: $model  extra_flags: $*"
    setsid nohup "$REPO/.venv/bin/vllm" serve "$model" \
        --port $PORT \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_xml \
        --no-enable-log-requests \
        "$@" \
        > "$REPO/logs/vllm_${model//\//_}.log" 2>&1 < /dev/null &
    VLLM_PID=$!
    log "vLLM PID=$VLLM_PID -- waiting for /health ..."
    for i in $(seq 1 120); do
        sleep 5
        if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
            log "vLLM healthy after $((i*5))s"; return 0
        fi
    done
    log "ERROR: vLLM did not become healthy in 600s"
    kill "$VLLM_PID" 2>/dev/null || true; return 1
}

stop_vllm() {
    log "Stopping vLLM (PID=$VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true; sleep 5
    kill -9 "$VLLM_PID" 2>/dev/null || true
    pkill -f "vllm serve" 2>/dev/null || true; sleep 3
}

sweep() {
    local model="$1"
    VLLM_BASE_URL="$VLLM_URL" "$PYTHON" "$REPO/scripts/run_oracle_sweep.py" --qwen --model "$model"
}

# ---- 1. 27B sweep (enforce-eager to avoid CUDA graph OOM) --------------------------
log "=== 27B sweep ==="
if start_vllm "Qwen/Qwen3.5-27B-GPTQ-Int4" \
    --quantization gptq \
    --enforce-eager \
    --gpu-memory-utilization 0.82; then
    sweep "Qwen/Qwen3.5-27B-GPTQ-Int4" || log "WARN: 27B sweep failed"
    stop_vllm
else
    log "ERROR: 27B vLLM failed again -- check logs/vllm_Qwen_Qwen3.5-27B-GPTQ-Int4.log"
fi

# ---- 2. 2B re-run (hallucination fix active) ----------------------------------------
log "=== 2B re-run ==="
if start_vllm "Qwen/Qwen3.5-2B" --gpu-memory-utilization 0.92 --max-model-len 8192; then
    sweep "Qwen/Qwen3.5-2B" || log "WARN: 2B re-run failed"
    stop_vllm
fi

# ---- 3. 4B re-run (hallucination fix active) ----------------------------------------
log "=== 4B re-run ==="
if start_vllm "Qwen/Qwen3.5-4B" --gpu-memory-utilization 0.92 --max-model-len 8192; then
    sweep "Qwen/Qwen3.5-4B" || log "WARN: 4B re-run failed"
    stop_vllm
fi

touch "$REPO/JOB_DONE.flag"
log "=== ALL REMAINING DONE ==="
log "Sweeps: $(ls runs/oracle/oracle_sweep_* -d 2>/dev/null | wc -l) total dirs"
