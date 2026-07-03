#!/usr/bin/env bash
# Qwen3 (NOT 3.5) DENSE size-sweep on the oracle env, H100 80GB via vLLM.
# Models: 0.6B, 1.7B, 4B, 8B, 14B, 32B (all BF16). Obfuscated (letter) tool names,
# same REPS_BY_N as oracle_sweep. One model at a time.
# Launch:
#   cd ~/reusable-action-discovery && rm -f JOB_DONE.flag \
#   && setsid nohup bash scripts/run_qwen3_dense_job.sh > logs/oracle_qwen3d.out 2>&1 < /dev/null &
set -euo pipefail; trap '' HUP
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$REPO/.venv/bin/python"; LOG="$REPO/logs/oracle_qwen3d.log"
PORT=8000; VLLM_URL="http://127.0.0.1:$PORT/v1"
mkdir -p "$REPO/logs"; exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Qwen3 dense are standard transformers (no GDN/mamba), so CUDA graphs are fine --
# but --enforce-eager is kept for safety/parity with the 3.5 job; drop it for ~speed.
# TOOL PARSER: Qwen3 (original) uses Hermes-style tool calls -> "hermes". If tool calls
# come back as text (noops), switch to "qwen3_xml". TEST 0.6B FIRST to confirm.
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

# id | vllm serve flags | episode concurrency. Small models have ample KV cache on the
# H100 -> high concurrency. 32B BF16 (~64GB) is memory-bound: shrink context AND drop
# concurrency so its small KV cache isn't over-subscribed (thrashing).
MODELS=(
    "Qwen/Qwen3-0.6B|--max-model-len 32768|32"
    "Qwen/Qwen3-1.7B|--max-model-len 32768|32"
    "Qwen/Qwen3-4B|--max-model-len 32768|32"
    "Qwen/Qwen3-8B|--max-model-len 32768|32"
    "Qwen/Qwen3-14B|--max-model-len 32768|24"
    "Qwen/Qwen3-32B|--max-model-len 16384 --gpu-memory-utilization 0.92|8"
)

for entry in "${MODELS[@]}"; do
    IFS='|' read -r HF FLAGS CONC <<< "$entry"
    SHORT="${HF##*/}"
    log "======== $SHORT  (concurrency=$CONC) ========"
    if start_vllm "$HF" $FLAGS; then
        VLLM_BASE_URL="$VLLM_URL" "$PYTHON" "$REPO/scripts/run_oracle_sweep.py" \
            --qwen3dense --model "$HF" --concurrency "$CONC" || log "WARN: sweep failed for $SHORT"
        stop_vllm
    else
        log "SKIP $SHORT (vLLM failed to start)"
    fi
done

touch "$REPO/JOB_DONE.flag"
log "======== ALL DONE ========"
