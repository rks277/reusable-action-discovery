#!/usr/bin/env bash
# Qwen3.5-4B ONLY on toolworld v3 (grid_v3), DENSE mode, via vLLM on one H100 80GB.
# Speed variant: higher episode concurrency (256). NOTE: prefix caching was tried and
# REVERTED -- it deadlocks vLLM 0.24.0 on Qwen3.5's hybrid Mamba arch ("align" cache
# mode; hung with 153 reqs at 0 tok/s). Concurrency 256 is metric-neutral vs the
# 2B/9B/27B runs (serving opts don't change outputs; concurrency already varied there).
# Text-based (no tool-call flags). --enforce-eager avoids the Qwen3.5 hybrid-arch hang.
# Launch (after 2B has finished + its server is down):
#   cd ~/reusable-action-discovery \
#   && setsid nohup bash scripts/run_grid_v3_qwen35_4b_fast.sh > logs/grid_4b.out 2>&1 < /dev/null &
set -euo pipefail; trap '' HUP
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$REPO/.venv/bin/python"; LOG="$REPO/logs/grid_4b.log"
PORT=8000; VLLM_URL="http://127.0.0.1:$PORT/v1"
MODEL="Qwen/Qwen3.5-4B"; CONC=256
mkdir -p "$REPO/logs"; exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%H:%M:%S')] $*"; }

pkill -9 -f "vllm serve" 2>/dev/null || true; sleep 4

log "Starting vLLM (text mode, no prefix caching): $MODEL"
setsid nohup "$REPO/.venv/bin/vllm" serve "$MODEL" --port $PORT \
    --no-enable-log-requests --enforce-eager \
    --gpu-memory-utilization 0.90 --max-model-len 32768 \
    > "$REPO/logs/vllm_grid_4b_fast.log" 2>&1 < /dev/null &
VLLM_PID=$!
log "vLLM PID=$VLLM_PID -- waiting for /health (Qwen3.5 warmup ~6-7 min) ..."
healthy=0
for i in $(seq 1 180); do
    sleep 5
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && { log "healthy after $((i*5))s"; healthy=1; break; }
done
[ "$healthy" -ne 1 ] && { log "ERROR: not healthy"; kill -9 "$VLLM_PID" 2>/dev/null||true; exit 1; }

log "Running grid_v3 DENSE sweep for $MODEL (VLLM_CONCURRENCY=$CONC, no prefix caching)"
cd "$REPO"
VLLM_BASE_URL="$VLLM_URL" VLLM_CONCURRENCY=$CONC PYTHONPATH="$REPO" \
    "$PYTHON" -m scripts.run_grid_sweep_v3 --model "$MODEL" --dense || log "WARN: sweep failed"

log "Stopping vLLM"
kill "$VLLM_PID" 2>/dev/null || true; sleep 5; kill -9 "$VLLM_PID" 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | while read p; do [ -n "$p" ] && kill -9 "$p" 2>/dev/null || true; done
touch "$REPO/JOB_DONE.flag"
log "======== GRID 4B FAST DONE ========"
