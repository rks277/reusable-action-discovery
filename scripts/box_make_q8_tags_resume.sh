#!/usr/bin/env bash
# Resume Step 1 after the tokenizer-config patch: reuse the already-merged bf16 weights
# (runs/rl_urn_pilot/merged, 28GB) and finish producing both q8_0 tags.
set -euo pipefail
cd ~/reusable-action-discovery
source .venv/bin/activate

BASE_ID="Qwen/Qwen2.5-Coder-14B-Instruct"
ADAPTER="runs/rl_urn_pilot/checkpoint/adapter"

echo "===== [1/4] copy patched tokenizer into the merged dir (weights already present) ====="
cp "$ADAPTER"/tokenizer.json "$ADAPTER"/tokenizer_config.json runs/rl_urn_pilot/merged/
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('runs/rl_urn_pilot/merged'); print('merged tokenizer OK')"

echo "===== [2/4] RL-final bf16 -> q8_0 GGUF ====="
python llama.cpp/convert_hf_to_gguf.py runs/rl_urn_pilot/merged \
  --outfile "$HOME/rl-urn-final-q8_0.gguf" --outtype q8_0

echo "===== [3/4] save SAME base ($BASE_ID) bf16 + patched tokenizer, then -> q8_0 GGUF ====="
python - <<PY
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("$BASE_ID", dtype=torch.bfloat16)
m.save_pretrained("runs/base_bf16", safe_serialization=True)
AutoTokenizer.from_pretrained("$ADAPTER").save_pretrained("runs/base_bf16")
print("saved runs/base_bf16")
PY
python llama.cpp/convert_hf_to_gguf.py runs/base_bf16 \
  --outfile "$HOME/base-q8_0.gguf" --outtype q8_0

echo "===== [4/4] create both Ollama tags off the stock qwen tool template + num_ctx 8192 ====="
ollama show --modelfile qwen2.5-coder:14b | grep -vE "^FROM |^# " > "$HOME/tmpl.txt"
{ echo "FROM $HOME/rl-urn-final-q8_0.gguf"; echo "PARAMETER num_ctx 8192"; cat "$HOME/tmpl.txt"; } > "$HOME/final.modelfile"
{ echo "FROM $HOME/base-q8_0.gguf";        echo "PARAMETER num_ctx 8192"; cat "$HOME/tmpl.txt"; } > "$HOME/base.modelfile"
ollama create qwen-rl-urn-final -f "$HOME/final.modelfile"
ollama create qwen-rl-base-q8   -f "$HOME/base.modelfile"

echo "===== DONE. tags: ====="
ollama list | grep -E "qwen-rl-urn-final|qwen-rl-base-q8|qwen2.5-coder:14b"
