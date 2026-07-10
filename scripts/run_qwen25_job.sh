#!/usr/bin/env bash
# Qwen2.5 dense instruct size-sweep on the oracle env, H100 80GB via vLLM.
# Models: 0.5B, 1.5B, 3B, 7B, 14B, 32B (all BF16). Obfuscated (letter) tool names.
# Qwen2.5 has NO thinking mode (no /no_think). Tool parser: hermes.
# Launch:
#   cd ~/reusable-action-discovery && rm -f JOB_DONE.flag \
#   && setsid nohup bash scripts/run_qwen25_job.sh > logs/oracle_qwen25.out 2>&1 < /dev/null &
set -euo pipefail; trap '' HUP
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$REPO/.venv/bin/python"; LOG="$REPO/logs/oracle_qwen25.log"
PORT=8000; VLLM_URL="http://127.0.0.1:$PORT/v1"
mkdir -p "$REPO/logs"; exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Qwen2.5 uses Hermes-style tool calls. If tool calls come back as text (mass noops),
# switch to "qwen2_5" or "hermes" variants. TEST 0.5B FIRST.
TOOL_PARSER="hermes"

pkill -9 -f "vllm serve" 2>/dev/null || true; sleep 4

start_vllm() {
    local model="$1"; shift
    log "Starting vLLM: $model  flags: $*"
    setsid nohup "$REPO/.venv/bin/vllm" serve "$model" \
        --port $PORT --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" \
        --no-enable-log-requests --gpu-memory-utilization 0.90 --enforce-eager \
        "$@" > "$REPO/logs/vllm_${model//\//_}.log" 2>&1 < /dev/null &
    VLLM_PID=$!
    log "PID=$VLLM_PID -- waiting for /health ..."
    for i in $(seq 1 120); do
        sleep 5
        curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && log "healthy after $((i*5))s" && return 0
    done
    log "ERROR: not healthy in 600s"; kill "$VLLM_PID" 2>/dev/null || true; return 1
}

stop_vllm() {
    kill "$VLLM_PID" 2>/dev/null || true; sleep 5
    kill -9 "$VLLM_PID" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true; sleep 3
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | while read p; do [ -n "$p" ] && kill -9 "$p" 2>/dev/null || true; done
    for i in $(seq 1 30); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        [ "${used:-9999}" -lt 2000 ] && break
        sleep 3
    done
}

# id | vllm flags | episode concurrency. Small -> high concurrency; 32B BF16 (~64GB)
# is memory-bound -> smaller context + low concurrency.
MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct|--max-model-len 32768|32"
    "Qwen/Qwen2.5-1.5B-Instruct|--max-model-len 32768|32"
    "Qwen/Qwen2.5-3B-Instruct|--max-model-len 32768|32"
    "Qwen/Qwen2.5-7B-Instruct|--max-model-len 32768|32"
    "Qwen/Qwen2.5-14B-Instruct|--max-model-len 32768|24"
    "Qwen/Qwen2.5-32B-Instruct|--max-model-len 16384 --gpu-memory-utilization 0.92|8"
)

for entry in "${MODELS[@]}"; do
    IFS='|' read -r HF FLAGS CONC <<< "$entry"
    SHORT="${HF##*/}"
    log "======== $SHORT  (concurrency=$CONC) ========"
    if start_vllm "$HF" $FLAGS; then
        VLLM_BASE_URL="$VLLM_URL" "$PYTHON" "$REPO/scripts/run_oracle_sweep.py" \
            --qwen25 --model "$HF" --concurrency "$CONC" || log "WARN: sweep failed for $SHORT"
        stop_vllm
    else
        log "SKIP $SHORT (vLLM failed to start)"
    fi
done

touch "$REPO/JOB_DONE.flag"
log "======== ALL DONE ========"
