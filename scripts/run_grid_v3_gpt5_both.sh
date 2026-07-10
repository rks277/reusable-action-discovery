#!/usr/bin/env bash
# grid_v3 DENSE sweep for the OpenAI reasoning tier, minimal reasoning, run LOCALLY
# via the OpenAI API (no GPU). Runs gpt-5-nano first (cheap) then gpt-5 (full), each
# 180 eps -> runs/grid_v3/dense/grid_sweep_v3_gpt-5-nano_<ts>/ and .../gpt-5_<ts>/.
# OPENAI_REASONING_EFFORT=minimal => lowest reasoning setting (closest to "no thinking"
# for a reasoning model), which also slashes output-token cost.
# The 20-turn no-progress -> fail rule is already built into run_grid_sweep_v3
# (NO_PROGRESS_WINDOW=20). Pair this with scripts/cost_watchdog.py for the spend cap.
# Launch:
#   cd ~/... && setsid nohup bash scripts/run_grid_v3_gpt5_both.sh > logs/grid_gpt5_both.out 2>&1 < /dev/null &
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$REPO/.venv/bin/python"
mkdir -p "$REPO/logs"
export OPENAI_REASONING_EFFORT=minimal
export PYTHONPATH="$REPO"
cd "$REPO"
CAP_FLAG="$REPO/COST_CAP_TRIPPED.flag"
rm -f "$CAP_FLAG"          # clear any stale trip from a prior run
log() { echo "[$(date '+%H:%M:%S')] $*"; }

for MODEL in gpt-5-nano gpt-5; do
    # Respect the EXTERNAL watchdog's verdict: it owns the cost calc + kill decision
    # and writes this flag. We only obey it -- no cost logic lives here.
    if [ -f "$CAP_FLAG" ]; then
        log "COST CAP tripped ($(cat "$CAP_FLAG")); skipping remaining models"; break
    fi
    log "======== $MODEL (reasoning=minimal) ========"
    "$PYTHON" -m scripts.run_grid_sweep_v3 --model "$MODEL" --dense \
        || log "WARN: sweep failed/killed for $MODEL"
done
log "======== GPT-5 BOTH DONE ========"
