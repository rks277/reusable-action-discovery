#!/usr/bin/env bash
# Step 4: publication-grade paired tool-transfer rerun (docs/qwen-tool-transfer-rerun-spec.md §9).
# Both q8_0 tags, A2 (--announce-n), EFR4, 24 paired seeds 2000-2023.
set -uo pipefail
cd ~/reusable-action-discovery
source .venv/bin/activate

for m in qwen-rl-base-q8:latest qwen-rl-urn-final:latest; do
  safe=$(echo "$m" | tr ":/" "__")
  echo "##################### RUN $m #####################"
  env PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.arm_a1_announce \
    --model "$m" --announce-n --empty-fence-retry 4 --seeds $(seq 2000 2023) --conc 4 \
    > ~/tool_rerun_${safe}.log 2>&1
  echo "done $m -> ~/tool_rerun_${safe}.log"
done
echo "===== ALL DONE ====="
