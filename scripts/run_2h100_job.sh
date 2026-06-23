#!/bin/bash
# Single two-phase grid_v3 sweep for 2x H100 via vLLM -- hardened to run unattended.
#
#   Phase 1 (parallel): Qwen2.5-7B-Instruct (GPU0) + Qwen2.5-14B-Instruct (GPU1), bf16.
#   Phase 2 (after both finish): Qwen2.5-72B-Instruct on BOTH GPUs (FP8, TP=2).
#
# AFK safety:
#   * ignores SIGHUP (survives SSH disconnect even without nohup)
#   * each model runs in a resume-until-complete loop: if the sweep process OR the
#     vLLM server dies, it restarts the server and re-runs only the missing/errored
#     episodes (run_grid_sweep_v3 --resume drops error rows), up to MAX_RESUME tries
#   * disk preflight, server max-num-seqs caps to avoid KV OOM, cleanup trap
#
# Launch (detached):  setsid nohup bash scripts/run_2h100_job.sh > logs/job.out 2>&1 &
set -uo pipefail
trap '' HUP                                   # survive disconnects
cd "$HOME/reusable-action-discovery" || exit 1
mkdir -p logs

PY="${PY:-.venv/bin/python}"
CTX="${CTX:-32768}"                           # max-model-len (Qwen2.5 native)
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_RESUME="${MAX_RESUME:-12}"                 # resume attempts per model
NEED_GB="${NEED_GB:-200}"                      # free-disk preflight (bf16 72B is ~145GB)
TOTAL_EPS=180                                  # dense grid: 9 cells (N{9,10,11}xT{2,3,4}) x 20 reps
SWEEP_ARGS="--dense"                           # dense grid mode (not the 100-cell scatter)
SERVER_PIDS=()
log () { echo "$(date '+%F %T') $*" | tee -a logs/job.log; }
# Kill ALL vLLM servers by name (PIDs started in background subshells don't reach
# the parent, so name-based is the only reliable teardown), then wait for GPU
# memory to actually free before the next phase loads.
kill_servers () {
  pkill -f vllm.entrypoints 2>/dev/null
  for _ in $(seq 1 30); do
    local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "${used:-99999}" -lt 2000 ] && break    # both GPUs drained
    sleep 2
  done
}
cleanup () { log "cleanup: killing vLLM servers"; kill_servers; }
trap cleanup EXIT INT TERM

command -v curl >/dev/null || { log "FATAL: curl not found"; exit 1; }
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
log "JOB START | free disk ${FREE_GB}GB (need ~${NEED_GB}GB) | ctx=$CTX util=$GPU_UTIL"
if [ "${FREE_GB:-0}" -lt "$NEED_GB" ]; then
  log "FATAL: only ${FREE_GB}GB free; need ~${NEED_GB}GB for weights. Free space or use a prequant FP8 repo."
  echo FAIL > JOB_DONE.flag; exit 1
fi

tag_of () { local n="${1##*/}"; n="${n//./-}"; echo "${n//:/-}"; }
latest_dir () { ls -dt runs/grid_sweep_v3_"$(tag_of "$1")"_* 2>/dev/null | head -1; }
success_count () {  # $1=run dir -> count of non-error episodes
  local f="$1/episodes.jsonl"; [ -f "$f" ] || { echo 0; return; }
  local tot err; tot=$(grep -c . "$f"); err=$(grep -c '"error"' "$f")
  echo $(( tot - err ))
}
healthy () { curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1; }

start_server () {  # $1=cuda $2=port $3=logfile $4=maxseqs ; rest -> model args. echoes PID
  local dev="$1" port="$2" logf="$3" mseq="$4"; shift 4
  CUDA_VISIBLE_DEVICES="$dev" "$PY" -m vllm.entrypoints.openai.api_server \
    --port "$port" --max-model-len "$CTX" --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs "$mseq" --no-enable-log-requests "$@" >> "$logf" 2>&1 &
  echo $!
}

wait_health () {  # $1=port $2=label -> 0 if healthy within ~25min
  for _ in $(seq 1 750); do healthy "$1" && { log "$2 healthy"; return 0; }; sleep 2; done
  log "ERROR: $2 (port $1) never healthy"; return 1
}

# Runs one model to completion: (re)starts its server as needed, resumes the sweep
# until TOTAL_EPS successful episodes exist or MAX_RESUME is exhausted.
# args: model cuda port logfile maxseqs concurrency
run_until_complete () {
  local model="$1" dev="$2" port="$3" logf="$4" mseq="$5" conc="$6"
  local label; label="$(tag_of "$model")"
  local pid="" attempt=0 dir done
  while :; do
    if ! healthy "$port"; then
      log "[$label] starting/restarting server (attempt server)"
      pid=$(start_server "$dev" "$port" "$logf" "$mseq" "${@:7}")
      SERVER_PIDS+=("$pid")
      wait_health "$port" "$label" || { log "[$label] server failed to come up"; return 1; }
    fi
    dir="$(latest_dir "$model")"
    if [ -z "$dir" ]; then
      log "[$label] first pass (fresh run)"
      VLLM_CONCURRENCY="$conc" VLLM_BASE_URL="http://127.0.0.1:$port/v1" \
        "$PY" -m scripts.run_grid_sweep_v3 --model "$model" $SWEEP_ARGS >> "logs/sweep_${label}.log" 2>&1
    else
      done=$(success_count "$dir")
      if [ "$done" -ge "$TOTAL_EPS" ]; then log "[$label] COMPLETE ($done/$TOTAL_EPS) at $dir"; return 0; fi
      attempt=$((attempt+1))
      if [ "$attempt" -gt "$MAX_RESUME" ]; then
        log "[$label] GIVING UP after $MAX_RESUME resumes ($done/$TOTAL_EPS) at $dir"; return 1
      fi
      log "[$label] resume #$attempt ($done/$TOTAL_EPS done) at $dir"
      VLLM_CONCURRENCY="$conc" VLLM_BASE_URL="http://127.0.0.1:$port/v1" \
        "$PY" -m scripts.run_grid_sweep_v3 --model "$model" $SWEEP_ARGS --resume "$dir" >> "logs/sweep_${label}.log" 2>&1
    fi
  done
}

stop_server () { kill "$1" 2>/dev/null; for _ in $(seq 1 30); do kill -0 "$1" 2>/dev/null || break; sleep 1; done; }

# ---------------- Phase 1: 7B (GPU0) + 14B (GPU1) in parallel ----------------
log "=== PHASE 1: 7B + 14B (parallel) ==="
run_until_complete Qwen/Qwen2.5-7B-Instruct  0 8001 logs/vllm_7b.log  256 "${P1_CONC:-64}" \
  --model Qwen/Qwen2.5-7B-Instruct  --dtype bfloat16 &
J7=$!
run_until_complete Qwen/Qwen2.5-14B-Instruct 1 8002 logs/vllm_14b.log 192 "${P1_CONC:-64}" \
  --model Qwen/Qwen2.5-14B-Instruct --dtype bfloat16 &
J14=$!
wait $J7;  R7=$?;  log "phase-1 7B finished rc=$R7"
wait $J14; R14=$?; log "phase-1 14B finished rc=$R14"
log "stopping phase-1 servers (freeing both GPUs for TP=2)"; kill_servers

# ---------------- Phase 2: 72B FP8, TP=2 across both GPUs ---------------------
log "=== PHASE 2: 72B (FP8, TP=2) ==="
run_until_complete Qwen/Qwen2.5-72B-Instruct 0,1 8003 logs/vllm_72b.log "${P2_MAXSEQS:-32}" "${P2_CONC:-16}" \
  --model Qwen/Qwen2.5-72B-Instruct --quantization fp8 --tensor-parallel-size 2
R72=$?; log "phase-2 72B finished rc=$R72"

log "JOB DONE  (7B rc=$R7, 14B rc=$R14, 72B rc=$R72)"
printf 'DONE 7B=%s 14B=%s 72B=%s\n' "$R7" "$R14" "$R72" > JOB_DONE.flag
