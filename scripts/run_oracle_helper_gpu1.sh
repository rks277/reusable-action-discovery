#!/usr/bin/env bash
# Rebalancing helper: once GPU1 frees (lane 1 of run_oracle_2lane_job.sh finishes its
# queue), steal lane 0's REMAINING models from the TAIL end so the two lanes converge
# from opposite ends. Done-flag gated (logs/done_<id>.flag) => race-safe with lane 0:
# each side skips a model whose flag already exists, so worst case is one mid-queue model
# runs twice (dedupe at scoring). Serves on GPU1:8001, same parser rules as the main job.
# Launch (after main job is already running):
#   cd ~/reusable-action-discovery \
#   && setsid nohup bash scripts/run_oracle_helper_gpu1.sh > logs/oracle_helper.out 2>&1 < /dev/null &
set -euo pipefail; trap '' HUP
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$REPO/.venv/bin/python"; VLLM="$REPO/.venv/bin/vllm"
GPU=1; PORT=8001; URL="http://127.0.0.1:$PORT/v1"
mkdir -p "$REPO/logs"
log() { echo "[$(date '+%H:%M:%S')] [helper] $*"; }
parser_for() { [ "$1" = "--qwen" ] && echo "qwen3_xml" || echo "hermes"; }

# Lane 0's models in REVERSE (tail first). 32B/27B are done / lane-0-owned; their flags
# will already exist by the time we reach them, so they're skipped.
ENTRIES=(
  "--qwen3dense|Qwen/Qwen3-0.6B|--max-model-len 32768|32"
  "--qwen3dense|Qwen/Qwen3-1.7B|--max-model-len 32768|32"
  "--qwen25|Qwen/Qwen2.5-3B-Instruct|--max-model-len 32768|32"
  "--qwen3dense|Qwen/Qwen3-4B|--max-model-len 32768|32"
  "--qwen3dense|Qwen/Qwen3-8B|--max-model-len 32768|32"
  "--qwen3dense|Qwen/Qwen3-14B|--max-model-len 32768|24"
)

_drain() {
    local PID="$1"
    kill "$PID" 2>/dev/null || true; sleep 4; kill -9 "$PID" 2>/dev/null || true
    nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | while read -r p; do [ -n "$p" ] && kill -9 "$p" 2>/dev/null || true; done
    for i in $(seq 1 40); do
        local used; used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        [ "${used:-9999}" -lt 2000 ] && break; sleep 3
    done
}

log "waiting for GPU$GPU to free (lane 1 to finish its queue) ..."
for i in $(seq 1 240); do   # up to 20 min
    used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && [ "${used:-9999}" -lt 2000 ]; then
        log "GPU$GPU free (${used}MiB), starting steal loop"; break
    fi
    sleep 5
done

for entry in "${ENTRIES[@]}"; do
    IFS='|' read -r FAM HF FLAGS CONC <<< "$entry"
    ID="${HF//\//_}"; DONE="$REPO/logs/done_${ID}.flag"
    if [ -f "$DONE" ]; then log "SKIP $HF (already done)"; continue; fi
    PARSER="$(parser_for "$FAM")"
    log "======== $HF (conc=$CONC, parser=$PARSER) ========"
    CUDA_VISIBLE_DEVICES="$GPU" setsid nohup "$VLLM" serve "$HF" \
        --port "$PORT" --enable-auto-tool-choice --tool-call-parser "$PARSER" \
        --no-enable-log-requests --enforce-eager --gpu-memory-utilization 0.90 \
        $FLAGS > "$REPO/logs/vllm_helper_${ID}.log" 2>&1 < /dev/null &
    PID=$!
    healthy=0
    for i in $(seq 1 180); do
        sleep 5
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { log "healthy after $((i*5))s"; healthy=1; break; }
    done
    if [ "$healthy" -ne 1 ]; then log "ERROR: $HF not healthy; skipping"; kill -9 "$PID" 2>/dev/null||true; _drain "$PID"; continue; fi
    # re-check flag right before running (lane 0 may have grabbed it while we loaded)
    if [ -f "$DONE" ]; then log "SKIP $HF (lane 0 took it during load)"; _drain "$PID"; continue; fi
    VLLM_BASE_URL="$URL" "$PYTHON" "$REPO/scripts/run_oracle_sweep.py" \
        "$FAM" --model "$HF" --concurrency "$CONC" \
        && touch "$DONE" || log "WARN: sweep failed for $HF"
    _drain "$PID"
done
log "======== HELPER DONE ========"
