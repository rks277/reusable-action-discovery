#!/usr/bin/env bash
# Oracle-discovery sweep for Qwen3.5 on a single A100 40GB via vLLM.
# Serves each model on port 8000 one at a time, runs pilot then sweep, tears down.
# Launch detached (AFK-safe):
#   cd ~/reusable-action-discovery && mkdir -p logs && rm -f JOB_DONE.flag \
#   && setsid nohup bash scripts/run_qwen35_oracle_job.sh > logs/oracle_job.out 2>&1 < /dev/null &
set -euo pipefail
trap '' HUP

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$REPO/.venv/bin/python"
LOG="$REPO/logs/oracle_job.log"
PORT=8000
VLLM_URL="http://127.0.0.1:$PORT/v1"
CONCURRENCY=6   # concurrent episodes against the local vLLM server

mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---- vLLM helpers ------------------------------------------------------------------

start_vllm() {
    local model="$1" quant_flag="${2:-}"
    log "Starting vLLM: $model"
    VLLM_BASE_URL="$VLLM_URL" \
    setsid nohup "$REPO/.venv/bin/vllm" serve "$model" \
        --port $PORT \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_xml \
        --no-enable-log-requests \
        --gpu-memory-utilization 0.92 \
        --max-model-len 8192 \
        $quant_flag \
        > "$REPO/logs/vllm_${model//\//_}.log" 2>&1 < /dev/null &
    VLLM_PID=$!
    log "vLLM PID=$VLLM_PID -- waiting for /health ..."
    for i in $(seq 1 120); do
        sleep 5
        if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
            log "vLLM healthy after $((i*5))s"
            return 0
        fi
    done
    log "ERROR: vLLM did not become healthy in 600s"
    kill "$VLLM_PID" 2>/dev/null || true
    return 1
}

stop_vllm() {
    log "Stopping vLLM (PID=$VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true
    sleep 5
    kill -9 "$VLLM_PID" 2>/dev/null || true
    # also kill any orphan vllm processes
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 3
}

run_phase() {
    local script="$1"; shift
    VLLM_BASE_URL="$VLLM_URL" \
    "$PYTHON" "$REPO/scripts/$script" --qwen "$@"
}

# ---- model roster ------------------------------------------------------------------
# Format: "HF_MODEL_ID[:quant_flag]"  (quant_flag is passed to vllm serve)
MODELS=(
    "Qwen/Qwen3.5-2B:"
    "Qwen/Qwen3.5-4B:"
    "Qwen/Qwen3.5-9B:"
    "Qwen/Qwen3.5-27B-GPTQ-Int4:--quantization gptq --enforce-eager --gpu-memory-utilization 0.82"
)

# ---- Phase 1: pilot (difficulty calibration, tools OFF) ----------------------------
log "======== PHASE 1: PILOT (tools off) ========"
for entry in "${MODELS[@]}"; do
    HF_MODEL="${entry%%:*}"
    QUANT="${entry#*:}"
    SHORT="${HF_MODEL##*/}"

    if start_vllm "$HF_MODEL" "$QUANT"; then
        log "Pilot: $SHORT"
        run_phase oracle_pilot.py --model "$HF_MODEL" || log "WARN: pilot failed for $SHORT"
        stop_vllm
    else
        log "SKIP pilot for $SHORT (vLLM failed to start)"
    fi
done
log "Pilot complete. Check runs/oracle/oracle_pilot_*/episodes.jsonl for calibration data."

# ---- Phase 2: full sweep -----------------------------------------------------------
log "======== PHASE 2: FULL SWEEP ========"
for entry in "${MODELS[@]}"; do
    HF_MODEL="${entry%%:*}"
    QUANT="${entry#*:}"
    SHORT="${HF_MODEL##*/}"

    if start_vllm "$HF_MODEL" "$QUANT"; then
        log "Sweep: $SHORT"
        run_phase run_oracle_sweep.py --model "$HF_MODEL" || log "WARN: sweep failed for $SHORT"
        stop_vllm
    else
        log "SKIP sweep for $SHORT (vLLM failed to start)"
    fi
done

# ---- Phase 3: re-run 2B and 4B with fixed code ------------------------------------
log "======== PHASE 3: RE-RUN 2B + 4B (with hallucination fix) ========"
RERUN_MODELS=(
    "Qwen/Qwen3.5-2B:"
    "Qwen/Qwen3.5-4B:"
)
for entry in "${RERUN_MODELS[@]}"; do
    HF_MODEL="${entry%%:*}"
    QUANT="${entry#*:}"
    SHORT="${HF_MODEL##*/}"

    if start_vllm "$HF_MODEL" "$QUANT"; then
        log "Re-run sweep: $SHORT"
        run_phase run_oracle_sweep.py --model "$HF_MODEL" || log "WARN: re-run sweep failed for $SHORT"
        stop_vllm
    else
        log "SKIP re-run for $SHORT (vLLM failed to start)"
    fi
done

# ---- Done --------------------------------------------------------------------------
touch "$REPO/JOB_DONE.flag"
log "======== ALL DONE ========"
log "Results in runs/oracle/oracle_sweep_*"
log "Score with: python scripts/oracle_counterfactual.py runs/oracle/<sweep>/episodes.jsonl"
