# GPU box setup runbook — tool-disposition / online-tool-investment

Self-contained procedure for standing up an **ephemeral H100 box** to run the Qwen work for the
online-tool-investment project (`docs/online-tool-investment-plan.md`). Boxes here are transient —
fresh IP each time, **no persistent disk** — so this whole runbook must be re-run every time. All
commands verified 2026-07-04 on a fresh Ubuntu H100 (the Qwen A2 reruns).

Two backends are used at different phases:
- **Ollama** — for the urn/tool *inference* runs (`urn_session.py`, `arm_a1_announce.py`). Simple, one model at a time. Phases 1–2.
- **vLLM** — for serving a *fine-tuned* checkpoint in Phase 3 (LoRA-merged weights; Ollama GGUF isn't fine-tunable). Also has function-calling for the MCP harness.

Architecture: the **sweep/eval code runs LOCALLY** (this repo) and makes HTTP calls to the box.
For local models the box is free (no per-token cost); the Anthropic key stays local for any Claude
comparison. `lomekwi/raw_chat.py` routes any model string containing `:` to Ollama, or to vLLM when
`LOCAL_BACKEND=vllm`.

---

## A. Ollama setup (urn / tool inference runs)

Replace `$BOX` with `ubuntu@<ip>`.

### 1. Install Ollama + concurrency override
```bash
ssh $BOX 'curl -fsSL https://ollama.com/install.sh | sh'
ssh $BOX 'sudo mkdir -p /etc/systemd/system/ollama.service.d
echo "[Service]
Environment=\"OLLAMA_NUM_PARALLEL=8\"
Environment=\"OLLAMA_MAX_LOADED_MODELS=1\"" | sudo tee /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama && sleep 3 && systemctl is-active ollama'
```
`OLLAMA_NUM_PARALLEL=8` = concurrent request slots (matches `--conc 8`); `MAX_LOADED_MODELS=1` keeps
one model resident (we run sizes sequentially).

> **Gotcha (large models):** `NUM_PARALLEL=8 ×` the default ~32k context overflows the H100's VRAM for
> a 32B model's KV cache → partial CPU offload → ~8× slowdown (~8.6 tok/s). Fix: make a
> context-capped variant, e.g. `ollama create qwen2.5-coder:32b-ctx8k` from a Modelfile
> (`FROM qwen2.5-coder:32b\nPARAMETER num_ctx 8192`). 14b and below are fine at defaults.

### 2. Pull models (background — ~37 GB for the full ladder)
```bash
ssh $BOX 'nohup bash -c "for s in 0.5b 1.5b 3b 7b 14b 32b; do ollama pull qwen2.5-coder:\$s; done" \
  > ~/pull_models.log 2>&1 </dev/null & echo launched'
# watch: ssh $BOX 'ollama list | grep qwen2.5-coder'
```
For a tool run you only need `14b`. For the urn ladder you need all six.

### 3. Sync repo + build venv + write .env  (from the LAPTOP, repo root)
```bash
rsync -az --exclude=.git --exclude=runs --exclude=figs --exclude=external --exclude=.venv \
  --exclude=data --exclude=__pycache__ ./ $BOX:~/reusable-action-discovery/
ssh $BOX 'cd ~/reusable-action-discovery && python3 -m venv .venv && source .venv/bin/activate \
  && pip install -q --upgrade pip openai python-dotenv numpy \
  && printf "OLLAMA_BASE_URL=http://localhost:11434/v1\nOLLAMA_API_KEY=ollama\n" > .env'
```
> **Critical:** `.env` must point at **localhost** (`http://localhost:11434/v1`), NOT `ollama.com`
> (that's Ollama Cloud and will silently route elsewhere). Verify: `ssh $BOX 'grep OLLAMA_BASE_URL ~/reusable-action-discovery/.env'`.

If you edited harness files locally after the initial sync, re-rsync just those (the tool-call fixes
in `lomekwi/raw_chat.py` and the A2 arm in `urn_session.py`/`arm_a1_announce.py`/`prompts.py`/
`driver.py`/`session_state.py` must be present on the box).

### 4. Smoke-test one seed before any real run
```bash
ssh $BOX 'cd ~/reusable-action-discovery && source .venv/bin/activate && timeout 180 env PYTHONPATH=. \
  python -u -m scripts.creator.tool_disposition_benchmark.urn_session \
  --model qwen2.5-coder:14b --announce-n --seeds 2000 --conc 1' 2>&1 | grep -A6 "URN FIDELITY"
```
Expect a clean report with `unparsed decisions = 0`. If tool calls come back garbled, confirm the
`raw_chat.py` fixes synced (see §C).

### 5. Launch the real runs (background on the box; poll the log)
```bash
# urn A2 ladder — all 6 sizes, 24 seeds, sequential per size:
ssh $BOX 'cd ~/reusable-action-discovery && source .venv/bin/activate && nohup bash -c "
for s in 0.5b 1.5b 3b 7b 14b 32b; do
  echo \"##### urn A2 qwen2.5-coder:\$s #####\"
  env PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.urn_session \
    --model qwen2.5-coder:\$s --announce-n --seeds \$(seq 2000 2023) --conc 8
done" > ~/urn_ladder_a2.log 2>&1 </dev/null & echo launched'

# tool A2 (14b, 12 seeds):
ssh $BOX 'cd ~/reusable-action-discovery && source .venv/bin/activate && nohup env PYTHONPATH=. \
  python -u -m scripts.creator.tool_disposition_benchmark.arm_a1_announce \
  --model qwen2.5-coder:14b --announce-n --conc 4 > ~/qwen14b_tool_a2.log 2>&1 </dev/null & echo launched'
```
`--announce-n` = the A2 (N-disclosed) arm; drop it for the no-N condition. Poll with
`ssh $BOX 'tail ~/urn_ladder_a2.log'`. Ladder ~a few min; tool run ~5–6 min (local early-stop on).

### 6. SYNC RESULTS BACK before releasing the box  ← don't skip this
```bash
rsync -az $BOX:~/reusable-action-discovery/runs/ ./runs/
```
> **Lesson learned (2026-07-03):** the Qwen A2 raw run dirs were left on the box and lost when it was
> torn down — only the aggregate numbers in the docs survived. Always rsync `runs/` back before
> releasing the box.

---

## B. LoRA training + vLLM serving (Phase 3)

### B0. LoRA training (`train_lora.py`) — verified 2026-07-04 on the H100
Stack (venv): `transformers peft accelerate datasets bitsandbytes` + torch (cu13 build works with the
box's driver). `train_lora.py` is `--backend hf` (Trainer+PEFT) — **Unsloth was skipped: it pins older
transformers and fights the 5.13 install.** Two transformers-5.13 quirks are already handled in the
script (empty-`apply_chat_template([])`; `BatchEncoding` return from `tokenize=True`).
```bash
# gate first (free, tokenizer only): template round-trip + tokenized-length dist -> sets max_seq_len
PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.train_lora --arm pistar --dry-run
# real run BOTH arms (nohup; ~22 min/arm, ~44 min total on one H100):
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for arm in pistar eager; do PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.train_lora \
  --arm $arm --qlora --backend hf; done
```
> **VRAM (the OOM lesson):** plain **bf16 LoRA OOMs** at ~step 6 — a single long tool-bridge session
> (up to ~21k tokens; the smoke test's short sessions miss it) blows past 80 GB on the backward pass
> (14B bf16 base = 28 GB + 21k-token activations + a ~6-13 GB logits tensor over the 152k vocab).
> **Fix = `--qlora` (4-bit base, frees ~20 GB) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.**
> This is the spec's "VRAM-tight fallback" and is now the default for this model on an 80 GB card.
> If it still OOMs, trim `--max-seq-len` to drop the 1-2 longest sessions (keeps most of the bridge).
> Adapters (~550 MB/arm) land in `runs/phase3_ft/{pistar,eager}/`; **rsync them back before releasing.**

### B1. vLLM serving (serve a fine-tuned checkpoint)
After LoRA training, merge the adapter and serve the merged weights:
```bash
ssh $BOX 'source .venv/bin/activate && pip install -q vllm && nohup vllm serve <merged-model-path-or-hf-id> \
  --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.92 --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes > ~/vllm.log 2>&1 </dev/null & echo launched'
```
`hermes` is the Qwen2.5 tool-call template. Then locally, route the harness to it:
```bash
LOCAL_BACKEND=vllm VLLM_BASE_URL=http://localhost:8000/v1 VLLM_API_KEY=EMPTY \
PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.arm_a1_announce \
  --model <served-model-id> --announce-n --conc 4
```
> The `LOCAL_BACKEND=vllm` branch in `raw_chat.py` is **untested in this project** — smoke-test with a
> single seed first. (Older `docs/old/creator-vllm-runbook.md` has more vLLM detail for the MCP harness.)

Tunnel if the box port isn't public: `ssh -N -L 8000:localhost:8000 $BOX` then use `localhost:8000`.

---

## C. Known Qwen-via-Ollama tool-call quirks (already fixed in `lomekwi/raw_chat.py`)

Qwen2.5-Coder emits tool calls in shapes Ollama's parser misses; the fixes are `_repair_triple_quoted_strings`
(Python `"""…"""` inside tool-call JSON → invalid JSON) and shape-C in `_coerce_tool_call_obj`
(script-name placed in the tool-call's `name` slot; disambiguated by the write_script-only `code` key).
If a fresh box shows near-zero builds / garbled tool calls, the cause is almost always that these
fixes didn't sync — re-rsync `lomekwi/raw_chat.py`.

---

## Quick reference

| thing | value |
|---|---|
| Ollama env | `OLLAMA_NUM_PARALLEL=8`, `OLLAMA_MAX_LOADED_MODELS=1` |
| `.env` | `OLLAMA_BASE_URL=http://localhost:11434/v1` (localhost, not cloud) |
| venv deps | `openai python-dotenv numpy` (+ `vllm` for Phase 3) |
| concurrency | urn `--conc 8`; tool `--conc 4` (memory-validated) |
| eval seeds | 2000–2011 (paired w/ Claude); urn ladder uses 2000–2023 |
| A2 flag | `--announce-n` (both harnesses) |
| **always** | `rsync runs/ back` before releasing the box |
