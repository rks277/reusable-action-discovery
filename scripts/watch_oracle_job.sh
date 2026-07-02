#!/usr/bin/env bash
# Monitor the Qwen3.5 oracle job on a remote Lambda box.
# Usage (from LOCAL machine):
#   bash scripts/watch_oracle_job.sh ubuntu@<IP>
# Or on the box itself:
#   bash scripts/watch_oracle_job.sh local
set -euo pipefail

TARGET="${1:-local}"
REPO="~/reusable-action-discovery"

CMD=$(cat <<'REMOTE'
set -e
REPO=~/reusable-action-discovery
cd "$REPO"

echo "=== JOB STATUS ==="
if [ -f JOB_DONE.flag ]; then
    echo "DONE (JOB_DONE.flag present)"
else
    if pgrep -f "run_qwen35_oracle_job" > /dev/null 2>&1; then
        echo "RUNNING (job script alive)"
    else
        echo "NOT RUNNING (job script not found -- may have crashed)"
    fi
    if pgrep -f "vllm serve" > /dev/null 2>&1; then
        VLLM_MODEL=$(pgrep -af "vllm serve" | grep -o 'Qwen/[^ ]*' | head -1)
        echo "vLLM: serving $VLLM_MODEL"
        curl -s http://127.0.0.1:8000/v1/models 2>/dev/null | python3 -c \
          "import sys,json; d=json.load(sys.stdin); [print('  model:',m['id']) for m in d.get('data',[])]" \
          2>/dev/null || echo "  (health check failed)"
    fi
fi

echo ""
echo "=== RECENT LOG (last 12 lines) ==="
tail -12 logs/oracle_job.log 2>/dev/null || echo "(no log yet)"

echo ""
echo "=== RUN PROGRESS ==="
for d in runs/oracle/oracle_pilot_* runs/oracle/oracle_sweep_* 2>/dev/null; do
    [ -d "$d" ] || continue
    ep="$d/episodes.jsonl"
    if [ -f "$ep" ]; then
        n=$(wc -l < "$ep" 2>/dev/null || echo 0)
        ok=$(grep -vc '"error"' "$ep" 2>/dev/null || echo 0)
        # estimate cost (haiku 1/5, sonnet 3/15, opus 5/25 -- Qwen uses vllm unpriced)
        echo "  $(basename "$d"): $ok ok / $n total rows"
    fi
done

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader 2>/dev/null || echo "(nvidia-smi not available)"
REMOTE
)

if [ "$TARGET" = "local" ]; then
    bash -c "$CMD"
else
    ssh "$TARGET" "$CMD"
fi
