#!/usr/bin/env bash
# Qwen3.5-2B and -4B on toolworld v3 (grid_v3), DENSE mode, via vLLM on one H100 80GB.
# Completes the qwen3.5 size roster (9B + 27B already run under this same code/config).
# Text-based (no tool-call flags). --enforce-eager avoids the Qwen3.5 hybrid-arch
# CUDA-graph hang. enable_thinking=False is injected via extra_body in raw_chat.py for
# qwen3* on vLLM. Dense = DENSE_N[9,10,11] x DENSE_T[2,3,4] x REPS(20) = 180 eps/model
# -> runs/grid_v3/dense/grid_sweep_v3_Qwen3-5-{2,4}B_<ts>/.
# Launch:
#   cd ~/reusable-action-discovery && rm -f JOB_DONE.flag \
#   && setsid nohup bash scripts/run_grid_v3_qwen35_2b_4b_dense.sh > logs/grid_2b4b.out 2>&1 < /dev/null &
set -euo pipefail; trap '' HUP
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$REPO/.venv/bin/python"; LOG="$REPO/logs/grid_2b4b.log"
PORT=8000; VLLM_URL="http://127.0.0.1:$PORT/v1"
CONC=128
mkdir -p "$REPO/logs"; exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%H:%M:%S')] $*"; }

MODELS=("Qwen/Qwen3.5-2B" "Qwen/Qwen3.5-4B")

stop_vllm() {
    kill "$VLLM_PID" 2>/dev/null || true; sleep 5; kill -9 "$VLLM_PID" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true; sleep 3
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | while read p; do [ -n "$p" ] && kill -9 "$p" 2>/dev/null || true; done
    for i in $(seq 1 30); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        [ "${used:-9999}" -lt 2000 ] && break; sleep 3
    done
}

pkill -9 -f "vllm serve" 2>/dev/null || true; sleep 4

for MODEL in "${MODELS[@]}"; do
    ID="${MODEL//\//_}"
    log "======== $MODEL ========"
    log "Starting vLLM (text mode, single GPU): $MODEL"
    setsid nohup "$REPO/.venv/bin/vllm" serve "$MODEL" --port $PORT \
        --no-enable-log-requests --enforce-eager \
        --gpu-memory-utilization 0.90 --max-model-len 32768 \
        > "$REPO/logs/vllm_grid_${ID}.log" 2>&1 < /dev/null &
    VLLM_PID=$!
    log "vLLM PID=$VLLM_PID -- waiting for /health ..."
    healthy=0
    for i in $(seq 1 180); do
        sleep 5
        curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && { log "healthy after $((i*5))s"; healthy=1; break; }
    done
    if [ "$healthy" -ne 1 ]; then log "ERROR: $MODEL not healthy; skipping"; kill -9 "$VLLM_PID" 2>/dev/null||true; stop_vllm; continue; fi

    log "Running grid_v3 DENSE sweep for $MODEL (VLLM_CONCURRENCY=$CONC)"
    cd "$REPO"
    VLLM_BASE_URL="$VLLM_URL" VLLM_CONCURRENCY=$CONC PYTHONPATH="$REPO" \
        "$PYTHON" -m scripts.run_grid_sweep_v3 --model "$MODEL" --dense || log "WARN: sweep failed for $MODEL"

    log "Stopping vLLM for $MODEL"
    stop_vllm
done

touch "$REPO/JOB_DONE.flag"
log "======== GRID 2B+4B DENSE DONE ========"
