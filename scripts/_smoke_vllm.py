"""One-episode smoke test for a vLLM-served model (uses VLLM_BASE_URL).

Runs a single dense-grid-ish episode and reports turns, empty-content NO-OPs
(should be 0), and the result row. Usage:
  VLLM_BASE_URL=http://127.0.0.1:8001/v1 python -m scripts._smoke_vllm Qwen/Qwen2.5-7B-Instruct
"""
import asyncio, sys, time
from scripts.toolworld_v3 import run
from scripts import sweep_config as cfg

MODEL = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
T = int(sys.argv[3]) if len(sys.argv) > 3 else 3


async def main():
    budget = cfg.budget_for(N)
    t0 = time.time()
    result, trace = await run(MODEL, n=N, n_types=T, relabel_seed=0, drop_seed=0,
                              hint=cfg.HINT, max_turns=2 * budget + 2 * N + 80, budget=budget)
    dt = time.time() - t0
    empties = sum(1 for x in trace if (x.get("agent_text") or "").strip() == "")
    print(f"\n==== {MODEL}  N={N} T={T}  wall={dt:.1f}s ====")
    print(f"turns={len(trace)}  empty_agent_texts={empties}  "
          f"solved={result['solved']}  built={result['built_machine']}  "
          f"actions={result['total_actions']}  noops={result['noop_total']}")
    print("first 6 actions:", [x["action"] for x in trace[:6]])


asyncio.run(main())
