#!/usr/bin/env bash
# Full Qwen3.5 oracle sweep on an H100 80GB via vLLM: all 4 models, obfuscated
# (letter) tool names, 3x reps. One model served at a time (fits individually with
# lots of headroom on 80GB). Context-length fix baked in (max-model-len 32768).
# Launch detached:
#   cd ~/reusable-action-discovery && rm -f JOB_DONE.flag \
#   && setsid nohup bash scripts/run_qwen35_oracle_h100.sh > logs/oracle_h100.out 2>&1 < /dev/null &
set -euo pipefail; trap '' HUP
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$REPO/.venv/bin/python"; LOG="$REPO/logs/oracle_h100.log"
PORT=8000; VLLM_URL="http://127.0.0.1:$PORT/v1"
mkdir -p "$REPO/logs"; exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%H:%M:%S')] $*"; }

pkill -9 -f "vllm serve" 2>/dev/null || true; sleep 4

start_vllm() {
    local model="$1"; shift
    log "Starting vLLM: $model  flags: $*"
    # --enforce-eager is REQUIRED: Qwen3.5's hybrid (GDN/mamba + multimodal) arch hangs
    # during CUDA-graph capture on vLLM 0.24; eager mode skips capture and starts cleanly.
    setsid nohup "$REPO/.venv/bin/vllm" serve "$model" \
        --port $PORT --enable-auto-tool-choice --tool-call-parser qwen3_xml \
        --no-enable-log-requests --gpu-memory-utilization 0.90 --max-model-len 32768 \
        --enforce-eager \
        "$@" > "$REPO/logs/vllm_${model//\//_}_h100.log" 2>&1 < /dev/null &
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
    pkill -9 -f "vllm serve" 2>/dev/null || true; sleep 3
    # pkill misses the detached EngineCore child; kill whatever still holds the GPU.
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | while read p; do [ -n "$p" ] && kill -9 "$p" 2>/dev/null || true; done
    # wait until GPU is actually free before serving the next model
    for i in $(seq 1 30); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        [ "${used:-9999}" -lt 2000 ] && break
        sleep 3
    done
}

# HF_MODEL:extra vllm flags. 27B is full BF16 (~54GB) -- fits on H100 80GB with room
# for a 32k KV cache; no quantization, no --enforce-eager needed.
MODELS=(
    "Qwen/Qwen3.5-2B:"
    "Qwen/Qwen3.5-4B:"
    "Qwen/Qwen3.5-9B:"
    "Qwen/Qwen3.5-27B:"
)

for entry in "${MODELS[@]}"; do
    HF="${entry%%:*}"; FLAGS="${entry#*:}"; SHORT="${HF##*/}"
    log "======== $SHORT ========"
    if start_vllm "$HF" $FLAGS; then
        VLLM_BASE_URL="$VLLM_URL" "$PYTHON" "$REPO/scripts/run_oracle_sweep.py" \
            --qwen --model "$HF" || log "WARN: sweep failed for $SHORT"
        stop_vllm
    else
        log "SKIP $SHORT (vLLM failed to start)"
    fi
done

touch "$REPO/JOB_DONE.flag"
log "======== ALL DONE ========"
log "$(ls -d runs/oracle/oracle_sweep_* 2>/dev/null | tail -4)"
