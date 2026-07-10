#!/usr/bin/env bash
# Oracle-discovery size-sweep across Qwen2.5 / Qwen3 (dense) / Qwen3.5, on 2x H100 80GB.
# Two PARALLEL single-GPU lanes (NO tensor-parallel): GPU0 on port 8000, GPU1 on port 8001.
# Each lane loops its models: serve one model on its GPU -> run that model's oracle sweep
# (sequential + letter-obfuscated tool names, the run_oracle_sweep.py defaults) -> tear down.
# Every model fits a single 80GB H100 (largest is 32B BF16 ~64GB), so no TP=2 needed.
#
# LANE-SAFE teardown: kill by PID + wait for THAT GPU's memory to drain (nvidia-smi -i $GPU).
# NEVER a global `pkill -f vllm` inside a lane -- that would kill the other lane's server.
#
# Resumable: a per-model flag (logs/done_<id>.flag) lets a restart skip finished models.
#
# Launch:
#   cd ~/reusable-action-discovery && rm -f JOB_DONE.flag \
#   && setsid nohup bash scripts/run_oracle_2lane_job.sh > logs/oracle_2lane.out 2>&1 < /dev/null &
set -euo pipefail; trap '' HUP
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$REPO/.venv/bin/python"; VLLM="$REPO/.venv/bin/vllm"
mkdir -p "$REPO/logs"
log() { echo "[$(date '+%H:%M:%S')] [$1] ${*:2}"; }

TOOL_PARSER="${TOOL_PARSER:-hermes}"   # Qwen2.5/3/3.5 all use hermes-style tool calls in vLLM

# Roster split across the two GPUs, balanced so each lane carries one of the biggest models.
# Fields: family_flag | HF_id | extra_vllm_flags | episode_concurrency
LANE0=(
  "--qwen25|Qwen/Qwen2.5-32B-Instruct|--max-model-len 16384 --gpu-memory-utilization 0.92|8"
  "--qwen|Qwen/Qwen3.5-27B|--max-model-len 16384 --gpu-memory-utilization 0.92|10"
  "--qwen3dense|Qwen/Qwen3-14B|--max-model-len 32768|24"
  "--qwen3dense|Qwen/Qwen3-8B|--max-model-len 32768|32"
  "--qwen3dense|Qwen/Qwen3-4B|--max-model-len 32768|32"
  "--qwen25|Qwen/Qwen2.5-3B-Instruct|--max-model-len 32768|32"
  "--qwen3dense|Qwen/Qwen3-1.7B|--max-model-len 32768|32"
  "--qwen3dense|Qwen/Qwen3-0.6B|--max-model-len 32768|32"
)
LANE1=(
  "--qwen3dense|Qwen/Qwen3-32B|--max-model-len 16384 --gpu-memory-utilization 0.92|8"
  "--qwen25|Qwen/Qwen2.5-14B-Instruct|--max-model-len 32768|24"
  "--qwen|Qwen/Qwen3.5-9B|--max-model-len 32768|32"
  "--qwen25|Qwen/Qwen2.5-7B-Instruct|--max-model-len 32768|32"
  "--qwen|Qwen/Qwen3.5-4B|--max-model-len 32768|32"
  "--qwen|Qwen/Qwen3.5-2B|--max-model-len 32768|32"
  "--qwen25|Qwen/Qwen2.5-1.5B-Instruct|--max-model-len 32768|32"
  "--qwen25|Qwen/Qwen2.5-0.5B-Instruct|--max-model-len 32768|32"
)

run_lane() {
    local GPU="$1"; local PORT="$2"; shift 2
    local ENTRIES=("$@")
    local URL="http://127.0.0.1:$PORT/v1"
    for entry in "${ENTRIES[@]}"; do
        IFS='|' read -r FAM HF FLAGS CONC <<< "$entry"
        local ID="${HF//\//_}"
        local DONE="$REPO/logs/done_${ID}.flag"
        if [ -f "$DONE" ]; then log "gpu$GPU" "SKIP $HF (done flag present)"; continue; fi
        log "gpu$GPU" "======== $HF  (conc=$CONC, parser=$TOOL_PARSER) ========"

        # --- serve on this GPU only ---
        CUDA_VISIBLE_DEVICES="$GPU" setsid nohup "$VLLM" serve "$HF" \
            --port "$PORT" --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" \
            --no-enable-log-requests --enforce-eager --gpu-memory-utilization 0.90 \
            $FLAGS > "$REPO/logs/vllm_${ID}.log" 2>&1 < /dev/null &
        local PID=$!
        log "gpu$GPU" "vLLM PID=$PID -- waiting for /health ..."
        local healthy=0
        for i in $(seq 1 180); do
            sleep 5
            curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && { log "gpu$GPU" "healthy after $((i*5))s"; healthy=1; break; }
        done
        if [ "$healthy" -ne 1 ]; then
            log "gpu$GPU" "ERROR: $HF not healthy in 900s -- see logs/vllm_${ID}.log; skipping"
            kill -9 "$PID" 2>/dev/null || true
            _drain "$GPU" "$PID"; continue
        fi

        # --- run this model's oracle sweep against THIS lane's endpoint ---
        VLLM_BASE_URL="$URL" "$PYTHON" "$REPO/scripts/run_oracle_sweep.py" \
            "$FAM" --model "$HF" --concurrency "$CONC" \
            && touch "$DONE" || log "gpu$GPU" "WARN: sweep failed for $HF"

        _drain "$GPU" "$PID"
    done
    log "gpu$GPU" "======== LANE $GPU DONE ========"
}

# Lane-safe teardown: kill the specific server + wait for THIS GPU's memory to drain.
_drain() {
    local GPU="$1"; local PID="$2"
    kill "$PID" 2>/dev/null || true; sleep 4
    kill -9 "$PID" 2>/dev/null || true
    # kill any compute app still resident on THIS GPU (by pid), never a global pkill.
    nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | while read -r p; do [ -n "$p" ] && kill -9 "$p" 2>/dev/null || true; done
    for i in $(seq 1 40); do
        local used
        used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        [ "${used:-9999}" -lt 2000 ] && break
        sleep 3
    done
}

log "main" "GPUs:"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/    /'
# one-time global clean before launching lanes (safe: nothing else should be running yet)
pkill -9 -f "vllm serve" 2>/dev/null || true; sleep 4

run_lane 0 8000 "${LANE0[@]}" &
P0=$!
run_lane 1 8001 "${LANE1[@]}" &
P1=$!
wait "$P0" "$P1"

touch "$REPO/JOB_DONE.flag"
log "main" "======== ALL LANES DONE ========"
