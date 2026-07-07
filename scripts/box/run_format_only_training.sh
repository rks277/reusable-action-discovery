#!/usr/bin/env bash
# "Adapter F" / format-only training (docs/qwen-finetune-transfer-plan.md "Corpus regeneration"
# follow-up, 2026-07-07): train on ZERO urn exposure (anchor + mechanics_bridge + error_recovery only,
# via train_lora.py --skip-urn) to test whether the tool-modality degradation seen in the corpus-fix
# retrain (near-total silence in some seeds, reversion to raw-JSON-without-<tool_call>-tags in others)
# is caused by gradient interference from co-training with urn's dominant, long, text-heavy signal --
# or whether it's something more basic that persists regardless of what's mixed into training.
#
# Distinct MODEL_TAG/output paths from every prior run (qwen-ft-pistar-mechbridge, qwen-ft-pistar-
# corpusfix) so nothing gets overwritten.
#
# Launch (from the box, already ssh'd in, repo synced, venv active):
#   nohup bash scripts/box/run_format_only_training.sh > PIPELINE_PROGRESS_FORMATONLY.log 2>&1 </dev/null &
#   disown
#   echo "launched pid $!"
#
# Check progress from anywhere:
#   ssh $BOX 'cat ~/reusable-action-discovery/PIPELINE_STATUS_FORMATONLY'
#   ssh $BOX 'tail -40 ~/reusable-action-discovery/PIPELINE_PROGRESS_FORMATONLY.log'

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STATUS_FILE="PIPELINE_STATUS_FORMATONLY"
MODEL_TAG="qwen-ft-format-only"
GGUF_NAME="format-only-f16.gguf"
ADAPTER_OUT="runs/phase3_ft/format_only"
MERGED_OUT="runs/phase3_merged/format_only"

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
  echo "FAILED at line $line ($(( ($(date +%s) - t0) / 60 ))min elapsed total) -- see PIPELINE_PROGRESS_FORMATONLY.log" > "$STATUS_FILE"
  log "===== PIPELINE FAILED at line $line ====="
}
trap 'on_error $LINENO' ERR

log "Format-only (Adapter F) training starting. model_tag=$MODEL_TAG"

# ---------------------------------------------------------------- 1. corpus (idempotent, already fixed)
stage "1/8 corpus regen + selftest"
python -m scripts.creator.tool_disposition_benchmark.phase3_demos --selftest
python -m scripts.creator.tool_disposition_benchmark.phase3_demos
stage_done "1/8 corpus regen + selftest"

# ---------------------------------------------------------------- 2. dry-run gate
stage "2/8 dry-run gate (tokenizer only, no GPU)"
python -m scripts.creator.tool_disposition_benchmark.train_lora --arm pistar --skip-urn --dry-run
stage_done "2/8 dry-run gate"

stage "stopping Ollama before GPU-heavy stages (frees VRAM for training/merge)"
sudo systemctl stop ollama
sleep 2
stage_done "stopping Ollama before GPU-heavy stages"

# ---------------------------------------------------------------- 3. smoke train
stage "3/8 smoke train (20 steps, no save)"
python -m scripts.creator.tool_disposition_benchmark.train_lora --arm pistar --skip-urn --qlora \
  --backend hf --smoke
stage_done "3/8 smoke train"

# ---------------------------------------------------------------- 4. real train (no urn -- Adapter F)
stage "4/8 real train (format-only, zero urn exposure, ~10min expected)"
python -m scripts.creator.tool_disposition_benchmark.train_lora --arm pistar --skip-urn --qlora \
  --backend hf --out "$ADAPTER_OUT"
stage_done "4/8 real train"

# ---------------------------------------------------------------- 5. merge
stage "5/8 merge LoRA -> bf16"
python -m scripts.creator.tool_disposition_benchmark.merge_lora \
  --adapter "$ADAPTER_OUT" --out "$MERGED_OUT"
stage_done "5/8 merge LoRA -> bf16"

# ---------------------------------------------------------------- 6. GGUF convert + Ollama create
stage "6/8 GGUF convert + Ollama create"
if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggerganov/llama.cpp
fi
pip install -q gguf sentencepiece protobuf
python llama.cpp/convert_hf_to_gguf.py "$MERGED_OUT" --outfile "$HOME/$GGUF_NAME" --outtype f16
sudo systemctl start ollama
sleep 3
ollama pull qwen2.5-coder:14b || true   # no-op if already pulled
ollama show --modelfile qwen2.5-coder:14b > ref.modelfile
{ echo "FROM $HOME/$GGUF_NAME"; grep -vE "^FROM |^# " ref.modelfile; } > ft.modelfile
ollama create "$MODEL_TAG" -f ft.modelfile
stage_done "6/8 GGUF convert + Ollama create"

# ---------------------------------------------------------------- 7. smoke eval (1 seed)
# No urn eval anywhere in this pipeline -- this adapter was never trained on urn, a urn eval wouldn't
# tell us anything about the question this run is asking (is tool-modality degradation caused by
# urn co-training interference).
stage "7/8 smoke eval (1 tool seed)"
python -u -m scripts.creator.tool_disposition_benchmark.arm_a1_announce \
  --model "${MODEL_TAG}:latest" --announce-n --seeds 2000 --conc 1
stage_done "7/8 smoke eval"

# ---------------------------------------------------------------- 8. tool A2 eval (12 seeds) -- the test
stage "8/8 tool A2 eval (12 seeds) -- the actual test"
python -u -m scripts.creator.tool_disposition_benchmark.arm_a1_announce \
  --model "${MODEL_TAG}:latest" --announce-n --conc 4
stage_done "8/8 tool A2 eval"

# ---------------------------------------------------------------- summary
# Same three checks as the corpus-fix retrain's summary, PLUS a new one: a scan for tool-call-shaped
# JSON appearing in plain `content` with no real tool_calls attached -- the raw-JSON-without-<tool_call>
# -tags reversion pattern the corpus-fix retrain's transcripts showed (seed_2002), which the existing
# malformed/unknown counters do NOT catch (it doesn't match any of the 5 patterns those counters were
# built to detect).
log "===== PIPELINE COMPLETE -- summarizing tool eval ====="
python3 - "$MODEL_TAG" <<'PYEOF' || log "(summary script failed -- harmless, inspect runs/ manually)"
import glob, json, re, sys
from collections import Counter
tag = sys.argv[1].replace(":", "_").replace("/", "_")
dirs = sorted(glob.glob(f"runs/arm_a1_announce_{tag}*_n-announced/seed_*/sessions.jsonl"))
TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"?/?(write_script|run_script|submit_answer|list_scripts|read_script)')
if not dirs:
    print("no tool eval sessions found -- check path pattern manually")
else:
    hit_cap = malformed = unknown = n_correct = n_submitted = n_problems_seen = 0
    hand_rates = []
    problems_seen = []
    all_contents = []
    raw_json_reversions = 0
    for p in dirs:
        d = json.loads(open(p).readline())
        hit_cap += d.get("hit_cap", False)
        malformed += d.get("n_malformed_tool_calls", 0)
        unknown += d.get("n_unknown_tool_calls", 0)
        n_correct += d.get("n_correct", 0)
        n_submitted += d.get("n_submitted", 0)
        problems_seen.append(d.get("problems_seen", 0))
        n_problems_seen += d.get("problems_seen", 0)
        esbh = d.get("eff_solve_by_hand")
        if esbh is not None:
            hand_rates.append(esbh)
        for m in d.get("transcript", []):
            if m.get("role") == "assistant":
                c = m.get("content") or ""
                if c:
                    all_contents.append(c)
                if not m.get("tool_calls") and TOOL_NAME_RE.search(c):
                    raw_json_reversions += 1
    n = len(dirs)
    print(f"=== LEGIBILITY (necessary but NOT sufficient) ===")
    print(f"seeds={n}  hit_cap={hit_cap}/{n}  malformed_calls={malformed}  unknown_calls={unknown}")
    print(f"problems_seen per seed: {problems_seen}  (60 = ran the full session)")
    print(f"\n=== REAL ENGAGEMENT/COMPETENCE ===")
    print(f"n_correct={n_correct}  n_submitted={n_submitted}  n_problems_seen={n_problems_seen}  "
          f"(submission rate {n_submitted/n_problems_seen:.1%} if healthy should be near 100%)")
    print(f"eff_solve_by_hand per seed: {hand_rates}")
    print(f"\n=== REPETITION-COLLAPSE CHECK ===")
    if all_contents:
        cnt = Counter(all_contents)
        top_str, top_n = cnt.most_common(1)[0]
        frac = top_n / len(all_contents)
        print(f"{len(all_contents)} content-bearing assistant turns, {len(cnt)} distinct "
              f"({len(cnt)/len(all_contents):.1%} unique)")
        print(f"most common string: x{top_n} ({frac:.0%} of all content-bearing turns): {top_str[:80]!r}")
        if frac > 0.15:
            print("*** STILL COLLAPSED: a single string dominates >15% of content-bearing turns ***")
        else:
            print("OK: no dominant repeated string")
    else:
        print("0 content-bearing assistant turns across all seeds")
    print(f"\n=== RAW-JSON-WITHOUT-<tool_call>-TAGS REVERSION CHECK (new -- corpus-fix retrain's blind spot) ===")
    print(f"turns with a tool-name-shaped JSON fragment in plain content and NO real tool_calls: "
          f"{raw_json_reversions}")
    if raw_json_reversions > 0:
        print("*** REVERSION DETECTED: model is emitting tool-call-shaped text without the <tool_call> "
              "XML wrapper on at least one turn -- this is the pre-fine-tune base-model failure mode "
              "the training was supposed to have corrected ***")
    else:
        print("OK: no raw-JSON reversion detected")
    print(f"\n=== READ (pre-registered, docs/qwen-finetune-transfer-plan.md \"Corpus regeneration\") ===")
    healthy = (frac <= 0.15 if all_contents else True) and raw_json_reversions == 0 and \
              (n_submitted / n_problems_seen > 0.5 if n_problems_seen else False)
    if healthy:
        print("LEGIBLE + ENGAGED with zero urn exposure -> ENTANGLEMENT CONFIRMED as the cause. "
              "Next: real adapter merge (Adapter P + Adapter F) per docs/adapter-merge-transfer-plan.md.")
    else:
        print("STILL DEGENERATE even with zero urn exposure -> entanglement is NOT the (sole) cause. "
              "Something more basic (LoRA config/recipe) needs investigation before adapter-merge.")
PYEOF

echo "ALL DONE ($(( ($(date +%s) - t0) / 60 ))min total). rsync runs/ back from your laptop:
  rsync -az \$BOX:~/reusable-action-discovery/runs/ ./runs/" > "$STATUS_FILE"
log "===== rsync runs/ back from the laptop before releasing the box ====="
