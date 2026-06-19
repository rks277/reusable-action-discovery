#!/bin/bash
# Waits for the grid_v3 sweep to finish, then finalizes (summary + figures).
cd "$HOME/reusable-action-discovery" || exit 1
echo "watcher started $(date)" > sweep_WATCH.log
while pgrep -f "python -m scripts.run_grid_sweep_v3" >/dev/null; do
  sleep 60
done
echo "sweep ended $(date), finalizing..." >> sweep_WATCH.log
mkdir -p runs/grid_v3_qwen_summary
.venv/bin/python -m scripts._overnight_finalize > runs/grid_v3_qwen_summary/finalize.log 2>&1
echo "ALL DONE $(date)" > sweep_DONE.flag
echo "finalize done $(date)" >> sweep_WATCH.log
