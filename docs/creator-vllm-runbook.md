# CREATOR v5 on Qwen via vLLM (H100 box) — runbook

Architecture: the **sweep runs locally** (this repo) and makes HTTP calls to the **vLLM box**
(serves the Qwen weights) for the model, and to **Anthropic** for the tiny ask-classifier judge
(Haiku, ~8 tokens/ambiguous-episode). So the box only needs to serve vLLM; the Anthropic key
stays on this side. One model per vLLM server → serve 7B, sweep, swap to 14B, sweep.

## 1. On the box — serve the model (one at a time)

```bash
# 7B (bf16; H100 80GB has tons of room for a big KV cache)
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.92 --max-model-len 8192
# optional max-speed: add  --quantization fp8   (Hopper native)
```
Swap to 14B later: same command with `Qwen/Qwen2.5-14B-Instruct`.
(Confirm reachable: `curl http://<IP>:8000/v1/models` from this machine.)

## 2. On this machine — point the harness at the box

```bash
export LOCAL_BACKEND=vllm
export VLLM_BASE_URL=http://<IP>:8000/v1
# VLLM_API_KEY defaults to "EMPTY" (fine unless you set --api-key on the server)
```
Routing: `_provider_for` sends `Qwen/...` to the `vllm` provider when `LOCAL_BACKEND=vllm`
(no `reasoning_effort` arg, which vLLM rejects). Claude judge still routes to Anthropic.

## 3. Timing probe (do this first — replaces the estimates with measured numbers)

```bash
PYTHONPATH=. python -m scripts.creator.run_creator_eval_hard_sweep \
  --models Qwen/Qwen2.5-7B-Instruct --n 20 --limit 5 \
  --tool-policy costly --withhold --concurrency 8
```
Read per-episode wall-clock:
```bash
python -c "import json,glob,os; d=sorted(glob.glob('runs/creator_eval_hard_*/'),key=os.path.getmtime)[-1]; \
es=[json.loads(l) for l in open(d+'episodes.jsonl')]; \
print('median elapsed_s', sorted(e['elapsed_s'] for e in es)[len(es)//2])"
```
Multiply median × (400 / concurrency) for a full-run estimate; tune `--concurrency` up until the
box saturates (watch vLLM throughput / GPU util).

## 4. Full v5 runs (per model)

```bash
# 7B
PYTHONPATH=. python -m scripts.creator.run_creator_eval_hard_sweep \
  --models Qwen/Qwen2.5-7B-Instruct --n 20 --limit 400 \
  --tool-policy costly --withhold --concurrency 48
# then swap the vLLM server to 14B and:
PYTHONPATH=. python -m scripts.creator.run_creator_eval_hard_sweep \
  --models Qwen/Qwen2.5-14B-Instruct --n 20 --limit 400 \
  --tool-policy costly --withhold --concurrency 48
```
The gated-hard N=20 batches are already built + cached (`runs/creator_eval_hard_cache_N20_gated.json`),
so these start model calls immediately.

## 5. Analyze + plot (same tools as the Claude trio)

```bash
PYTHONPATH=. python -m scripts.creator.analyze_creator_eval_hard runs/creator_eval_hard_<ts>/
PYTHONPATH=. python -m scripts.creator.plot_creator_eval_one \
  runs/creator_eval_hard_<ts>/episodes.jsonl "v5 Qwen-7B — costed+gated" figs/creator/fig_creator_v5_qwen7b.png
```
To put Qwen on the model-size axis in the plotters, add entries to `scripts.plot_metric_lines.CLAUDE_B`
(e.g. `"qwen2.5:7b": 7`) or adapt — the plotters key off that size map.

## Notes / caveats
- **Judge stays Haiku** (cross-model consistency with the Claude runs). To run fully offline, add
  `--judge Qwen/Qwen2.5-7B-Instruct` (the served model classifies asks too).
- **Protocol-following risk:** sub-7B Qwen may ignore the `EVALUATE:` convention or the `ANSWER_i:`
  format → looks like low recognition / low solve for *formatting* reasons, not disposition. The
  native `<invoke>` fallback is Claude-specific and won't help Qwen. Check the probe transcripts.
- **Concurrency:** 48 is a starting point; vLLM continuous batching can often take more. Raise until
  GPU util plateaus.

---

## Mistral-Large-2 123B (DENSE) — single 2×H100 box, FP8

Goal: a genuine ~123B **dense** point (all params active) — the largest dense model that fits one
2×H100 (160 GB). FP8 weights ≈ 123 GB; KV+overhead fits the remaining ~37 GB at low concurrency.

**Box:** 1× node with **2×H100-80GB (NVLink)**, ~250 GB disk, CUDA 12.x.

**Prereq — gated weights:** `mistralai/Mistral-Large-Instruct-2407` is **gated** on HF (Mistral
Research License). Need an **HF token that has accepted the license** on the model page. Set
`HF_TOKEN=...` on the box before serving. (Alt: `...-2411` is the updated Large-2; either works.)

### 1. Setup (on box)
```bash
pip install -U "vllm==0.23.*"
export HF_TOKEN=<token-with-mistral-large-2-access>
export VLLM_USE_FLASHINFER_SAMPLER=0          # dodge ninja/nvcc JIT crash (as in Qwen runs)
```

### 2. Serve (FP8, TP=2)
```bash
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-Large-Instruct-2407 \
  --served-model-name mistral-large-2 \
  --quantization fp8 --kv-cache-dtype fp8 \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.93 \
  --port 8000
```
Weights ~123 GB / 160 GB → ~77% just for weights; `kv-cache-dtype fp8` + capped context keeps KV in
budget. If it OOMs on load, drop `--max-model-len` to 12288 or `--gpu-memory-utilization` to 0.90.

### 3. Tunnel (cloud SG blocks 8000)
```bash
ssh -N -L 18000:localhost:8000 ubuntu@<box-ip>
```

### 4. Run v5 (laptop)
```bash
set -a; . ./.env; set +a
LOCAL_BACKEND=vllm VLLM_BASE_URL=http://localhost:18000/v1 VLLM_API_KEY=EMPTY \
PYTHONPATH=. python -m scripts.creator.run_creator_eval_hard_sweep \
  --models mistral-large-2 --n 20 --limit 400 \
  --tool-policy costly --withhold --concurrency 6 \
  --judge claude-haiku-4-5-20251001
```
`mistral-large-2` routes to vLLM (raw_chat `_provider_for` now matches `mistral*`); non-reasoning,
so no `reasoning_effort` is sent. **Concurrency 6** to start (KV is tight on 2×H100) — raise if GPU
util has headroom. Plots: it's already in `plot_creator_eval_metrics.SIZE_MAP`? No — add
`("mistral-large", ("Mistral-Large-2", 123))` (dense → 123B is both total and active).
