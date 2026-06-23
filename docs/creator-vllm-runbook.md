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
