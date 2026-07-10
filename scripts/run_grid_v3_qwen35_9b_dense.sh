#!/usr/bin/env bash
# Qwen3.5-9B on toolworld v3 (grid_v3), DENSE mode, via vLLM on one H100.
# Toolworld is TEXT-based (RawChat.chat), so vLLM is served WITHOUT tool-calling
# flags. --enforce-eager avoids the Qwen3.5 hybrid-arch CUDA-graph hang. /no_think
# is injected by toolworld_v3.run for qwen* models (vLLM won't auto-suppress it).
# Dense = DENSE_N[9,10,11] x DENSE_T[2,3,4] x REPS(20) = 180 episodes -> runs/grid_v3/dense/.
# Launch:
#   cd ~/reusable-action-discovery && rm -f JOB_DONE.flag \
#   && setsid nohup bash scripts/run_grid_v3_qwen35_9b_dense.sh > logs/grid_9b.out 2>&1 < /dev/null &
set -euo pipefail; trap '' HUP
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$REPO/.venv/bin/python"; LOG="$REPO/logs/grid_9b.log"
PORT=8000; VLLM_URL="http://127.0.0.1:$PORT/v1"
MODEL="Qwen/Qwen3.5-9B"
mkdir -p "$REPO/logs"; exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%H:%M:%S')] $*"; }

pkill -9 -f "vllm serve" 2>/dev/null || true; sleep 4

log "Starting vLLM (text mode, single GPU): $MODEL"
setsid nohup "$REPO/.venv/bin/vllm" serve "$MODEL" --port $PORT \
    --no-enable-log-requests --enforce-eager \
    --gpu-memory-utilization 0.90 --max-model-len 32768 \
    > "$REPO/logs/vllm_grid_9b.log" 2>&1 < /dev/null &
VLLM_PID=$!
log "vLLM PID=$VLLM_PID -- waiting for /health ..."
healthy=0
for i in $(seq 1 150); do
    sleep 5
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && { log "healthy after $((i*5))s"; healthy=1; break; }
done
[ "$healthy" -ne 1 ] && { log "ERROR: not healthy"; kill -9 "$VLLM_PID" 2>/dev/null||true; exit 1; }

log "Running grid_v3 DENSE sweep for $MODEL (VLLM_CONCURRENCY=128)"
cd "$REPO"
VLLM_BASE_URL="$VLLM_URL" VLLM_CONCURRENCY=128 PYTHONPATH="$REPO" \
    "$PYTHON" -m scripts.run_grid_sweep_v3 --model "$MODEL" --dense || log "WARN: sweep failed"

log "Stopping vLLM"
kill "$VLLM_PID" 2>/dev/null || true; sleep 5; kill -9 "$VLLM_PID" 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | while read p; do [ -n "$p" ] && kill -9 "$p" 2>/dev/null || true; done
touch "$REPO/JOB_DONE.flag"
log "======== GRID 9B DENSE DONE ========"
