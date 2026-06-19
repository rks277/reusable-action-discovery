#!/bin/bash
cd /Users/danielwang/Documents/GitHub/reusable-action-discovery
TS=$(date +%Y%m%d_%H%M%S)
LOG=runs/oss_region_logs/orchestrator_${TS}.log
NVAL="10,20,30,40,50,60,70,80,90"
echo "=== ORCHESTRATOR START $(date) ===" | tee -a "$LOG"
for m in qwen2.5:7b qwen2.5:3b qwen2.5:1.5b; do
  echo "=== [$(date)] $m  TOOLWORLD start ===" | tee -a "$LOG"
  python -u -m scripts.run_haiku_region_sweep \
      --model "$m" --n-points 180 --reps 1 --n-hi 20 --conc 8 >>"$LOG" 2>&1
  echo "=== [$(date)] $m  TOOLWORLD done (exit $?) ===" | tee -a "$LOG"
  echo "=== [$(date)] $m  WOODWORLD start ===" | tee -a "$LOG"
  python -u -m scripts.run_woodworld_region_sweep \
      --model "$m" --variant iso_recipe --budget-mult 1.2 \
      --n-values "$NVAL" --reps 1 --conc 8 >>"$LOG" 2>&1
  echo "=== [$(date)] $m  WOODWORLD done (exit $?) ===" | tee -a "$LOG"
done
echo "=== ORCHESTRATOR ALL DONE $(date) ===" | tee -a "$LOG"
