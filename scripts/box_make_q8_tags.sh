#!/usr/bin/env bash
# Step 1 of the Qwen tool-transfer rerun (docs/qwen-tool-transfer-rerun-spec.md §9):
# produce BOTH q8_0 Ollama tags at num_ctx 8192 through the identical merge->GGUF path.
# Run on the box from ~/reusable-action-discovery with the venv active.
set -euo pipefail
cd ~/reusable-action-discovery
source .venv/bin/activate

BASE_ID="Qwen/Qwen2.5-Coder-14B-Instruct"
ADAPTER="runs/rl_urn_pilot/checkpoint/adapter"

echo "===== [1/5] merge adapter -> bf16 (CPU; torch is CPU-only on this box) ====="
PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.merge_lora \
  --adapter "$ADAPTER" --out runs/rl_urn_pilot/merged --device cpu

echo "===== [2/5] RL-final bf16 -> q8_0 GGUF ====="
python llama.cpp/convert_hf_to_gguf.py runs/rl_urn_pilot/merged \
  --outfile "$HOME/rl-urn-final-q8_0.gguf" --outtype q8_0

echo "===== [3/5] save the SAME base ($BASE_ID) in bf16 + adapter tokenizer/template ====="
python - <<PY
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("$BASE_ID", dtype=torch.bfloat16)
m.save_pretrained("runs/base_bf16", safe_serialization=True)
AutoTokenizer.from_pretrained("$ADAPTER").save_pretrained("runs/base_bf16")
print("saved runs/base_bf16")
PY

echo "===== [4/5] base bf16 -> q8_0 GGUF (identical convert path) ====="
python llama.cpp/convert_hf_to_gguf.py runs/base_bf16 \
  --outfile "$HOME/base-q8_0.gguf" --outtype q8_0

echo "===== [5/5] create both Ollama tags off the stock qwen tool template + num_ctx 8192 ====="
ollama show --modelfile qwen2.5-coder:14b | grep -vE "^FROM |^# " > "$HOME/tmpl.txt"
{ echo "FROM $HOME/rl-urn-final-q8_0.gguf"; echo "PARAMETER num_ctx 8192"; cat "$HOME/tmpl.txt"; } > "$HOME/final.modelfile"
{ echo "FROM $HOME/base-q8_0.gguf";        echo "PARAMETER num_ctx 8192"; cat "$HOME/tmpl.txt"; } > "$HOME/base.modelfile"
ollama create qwen-rl-urn-final -f "$HOME/final.modelfile"
ollama create qwen-rl-base-q8   -f "$HOME/base.modelfile"

echo "===== DONE. tags: ====="
ollama list | grep -E "qwen-rl-urn-final|qwen-rl-base-q8|qwen2.5-coder:14b"
