#!/usr/bin/env bash
# Mechanics bridge training (docs/qwen-finetune-transfer-plan.md "Mechanics bridge training --
# root-cause diagnosis and fix"): regenerate the corpus, train the pistar arm with both bugs fixed
# (+ error_recovery), merge -> GGUF -> Ollama, then eval. Designed to run fully unattended via nohup on
# the GPU box: every stage writes to a single-line STATUS file (cheap to check: `ssh $BOX 'cat
# ~/reusable-action-discovery/PIPELINE_STATUS'`) and an append-only timestamped PROGRESS log (`tail -f
# ~/reusable-action-discovery/PIPELINE_PROGRESS.log`), so you can check back after any amount of time --
# including with your laptop asleep, since this runs entirely on the box's own compute once launched
# detached (see launch command at the bottom).
#
# Launch (from the box, already ssh'd in, repo synced, venv active):
#   nohup bash scripts/box/run_mechanics_bridge_training.sh > PIPELINE_PROGRESS.log 2>&1 </dev/null &
#   disown
#   echo "launched pid $!"
#
# Check progress from anywhere:
#   ssh $BOX 'cat ~/reusable-action-discovery/PIPELINE_STATUS'                 # one-line: current stage
#   ssh $BOX 'tail -40 ~/reusable-action-discovery/PIPELINE_PROGRESS.log'      # recent detail
#
# If a stage fails, the script stops (set -e) and PIPELINE_STATUS ends with "FAILED: <stage>" -- the
# detail is in PIPELINE_PROGRESS.log right above the failure.

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STATUS_FILE="PIPELINE_STATUS"
MODEL_TAG="qwen-ft-pistar-mechbridge"
GGUF_NAME="pistar-mechbridge-f16.gguf"

t0=$(date +%s)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

stage() {
  local name="$1"
  echo "RUNNING: $name (started $(date '+%H:%M:%S'), $(( ($(date +%s) - t0) / 60 ))min elapsed)" > "$STATUS_FILE"
  log "===== STAGE START: $name ====="
}

stage_done() {
  local name="$1"
  echo "DONE: $name ($(( ($(date +%s) - t0) / 60 ))min elapsed total)" > "$STATUS_FILE"
  log "===== STAGE DONE: $name ====="
}

on_error() {
  local line="$1"
  echo "FAILED at line $line ($(( ($(date +%s) - t0) / 60 ))min elapsed total) -- see PIPELINE_PROGRESS.log" > "$STATUS_FILE"
  log "===== PIPELINE FAILED at line $line ====="
}
trap 'on_error $LINENO' ERR

log "Mechanics bridge training starting. model_tag=$MODEL_TAG"

# ---------------------------------------------------------------- 1. corpus
stage "1/9 corpus regen + selftest"
python -m scripts.creator.tool_disposition_benchmark.phase3_demos --selftest
python -m scripts.creator.tool_disposition_benchmark.phase3_demos
stage_done "1/9 corpus regen + selftest"

# ---------------------------------------------------------------- 2. dry-run gate
stage "2/9 dry-run gate (tokenizer only, no GPU)"
python -m scripts.creator.tool_disposition_benchmark.train_lora --arm pistar --dry-run
stage_done "2/9 dry-run gate"

# Ollama keeps qwen2.5-coder:14b resident in GPU memory after any inference call (OLLAMA_MAX_LOADED_
# MODELS=1, OLLAMA_NUM_PARALLEL=8 reserves KV-cache for 8 parallel slots -- tens of GB even for a 14B
# model). Training needs the full GPU to itself to load its own 4-bit copy for QLoRA -- stop the Ollama
# service before any GPU-heavy stage so the two never compete for VRAM (diagnosed 2026-07-06: a prior
# smoke-test inference call left Ollama holding ~57GB, leaving <500MB free and OOMing the smoke train).
stage "stopping Ollama before GPU-heavy stages (frees VRAM for training/merge)"
sudo systemctl stop ollama
sleep 2
stage_done "stopping Ollama before GPU-heavy stages"

# ---------------------------------------------------------------- 3. smoke train
stage "3/9 smoke train (20 steps, no save)"
python -m scripts.creator.tool_disposition_benchmark.train_lora --arm pistar --qlora --backend hf --smoke
stage_done "3/9 smoke train"

# ---------------------------------------------------------------- 4. real train
stage "4/9 real train (pistar arm, ~15min expected)"
python -m scripts.creator.tool_disposition_benchmark.train_lora --arm pistar --qlora --backend hf
stage_done "4/9 real train"

# ---------------------------------------------------------------- 5. merge
stage "5/9 merge LoRA -> bf16"
python -m scripts.creator.tool_disposition_benchmark.merge_lora \
  --adapter runs/phase3_ft/pistar --out runs/phase3_merged/pistar
stage_done "5/9 merge LoRA -> bf16"

# ---------------------------------------------------------------- 6. GGUF convert + Ollama create
stage "6/9 GGUF convert + Ollama create"
if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggerganov/llama.cpp
fi
pip install -q gguf sentencepiece protobuf
python llama.cpp/convert_hf_to_gguf.py runs/phase3_merged/pistar --outfile "$HOME/$GGUF_NAME" --outtype f16
sudo systemctl start ollama    # back on now that training/merge (GPU-heavy) are done
sleep 3
ollama pull qwen2.5-coder:14b || true   # no-op if already pulled
ollama show --modelfile qwen2.5-coder:14b > ref.modelfile
{ echo "FROM $HOME/$GGUF_NAME"; grep -vE "^FROM |^# " ref.modelfile; } > ft.modelfile
ollama create "$MODEL_TAG" -f ft.modelfile
stage_done "6/9 GGUF convert + Ollama create"

# ---------------------------------------------------------------- 7. smoke eval (1 seed each)
stage "7/9 smoke eval (1 urn seed + 1 tool seed)"
python -u -m scripts.creator.tool_disposition_benchmark.urn_session \
  --model "${MODEL_TAG}:latest" --announce-n --seeds 2000 --conc 1
python -u -m scripts.creator.tool_disposition_benchmark.arm_a1_announce \
  --model "${MODEL_TAG}:latest" --announce-n --seeds 2000 --conc 1
stage_done "7/9 smoke eval"

# ---------------------------------------------------------------- 8. urn A2 sanity eval (24 seeds)
stage "8/9 urn A2 sanity eval (24 seeds)"
python -u -m scripts.creator.tool_disposition_benchmark.urn_session \
  --model "${MODEL_TAG}:latest" --announce-n --seeds $(seq 2000 2023) --conc 8
stage_done "8/9 urn A2 sanity eval"

# ---------------------------------------------------------------- 9. tool A2 eval (12 seeds) -- the test
stage "9/9 tool A2 eval (12 seeds) -- the actual test"
python -u -m scripts.creator.tool_disposition_benchmark.arm_a1_announce \
  --model "${MODEL_TAG}:latest" --announce-n --conc 4
stage_done "9/9 tool A2 eval"

# ---------------------------------------------------------------- summary
# NOT gated by set -e's trap: a cosmetic summary-parsing bug must never overwrite a successful
# "ALL DONE" status after ~45min of real pipeline work. Failures here just print and move on.
log "===== PIPELINE COMPLETE -- summarizing tool eval ====="
python3 - "$MODEL_TAG" <<'PYEOF' || log "(summary script failed -- harmless, inspect runs/ manually)"
import glob, json, sys
tag = sys.argv[1].replace(":", "_").replace("/", "_")
dirs = sorted(glob.glob(f"runs/arm_a1_announce_{tag}*_n-announced/seed_*/sessions.jsonl"))
if not dirs:
    print("no tool eval sessions found -- check path pattern manually")
else:
    hit_cap = malformed = unknown = 0
    problems_seen = []
    for p in dirs:
        d = json.loads(open(p).readline())
        hit_cap += d.get("hit_cap", False)
        malformed += d.get("n_malformed_tool_calls", 0)
        unknown += d.get("n_unknown_tool_calls", 0)
        problems_seen.append(d.get("problems_seen", 0))
    n = len(dirs)
    print(f"seeds={n}  hit_cap={hit_cap}/{n}  malformed_calls={malformed}  unknown_calls={unknown}")
    print(f"problems_seen per seed: {problems_seen}  (60 = ran the full session)")
PYEOF

echo "ALL DONE ($(( ($(date +%s) - t0) / 60 ))min total). rsync runs/ back from your laptop:
  rsync -az \$BOX:~/reusable-action-discovery/runs/ ./runs/" > "$STATUS_FILE"
log "===== rsync runs/ back from the laptop before releasing the box ====="
