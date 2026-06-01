"""Budget sweep for the two reasoning models (gpt-5, gemini-2.5-pro), deferred
from the headline run until raw_chat.py gave reasoning models token headroom.

Same world/budget/reps as run_budget_sweep.py (T=3, n=8, budget=36, 8 reps,
rep r => relabel_seed=drop_seed=r). These episodes use the SAME per-rep worlds
as the headline Anthropic run, so the two columns are paired-comparable and can
be concatenated with that run's episodes.jsonl for analysis.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from scripts.toolworld_v2 import run

MODELS = [
    ("openai", "gpt-5"),
    ("google", "gemini-2.5-pro"),
]
N, T, REPS = 8, 3, 8
CONCURRENCY = {"openai": 4, "google": 4}


def Hn(n): return sum(1.0 / k for k in range(1, n + 1))
BUDGET = round(1.2 * (N * Hn(N) + N))      # ~36


async def main():
    load_dotenv()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"budget_sweep_newmodels_T{T}_n{N}_b{BUDGET}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    out_path.write_text("")
    print(f"Writing to {out_path}  (T={T}, n={N}, budget={BUDGET}, "
          f"{len(MODELS)} models x {REPS} reps)", flush=True)

    sems = {p: asyncio.Semaphore(CONCURRENCY[p]) for p, _ in MODELS}
    lock = asyncio.Lock()

    async def one(prov, model, rep):
        async with sems[prov]:
            t0 = time.time()
            try:
                result, trace = await run(model, n=N, n_types=T, relabel_seed=rep,
                                          drop_seed=rep, hint=True, max_turns=200,
                                          budget=BUDGET)
                row = {"model": model, "n": N, "n_types": T, "hint": True,
                       "budget": BUDGET, "relabel_seed": rep, "drop_seed": rep,
                       "labels": result["labels"],
                       "actions": [x["action"] for x in trace],
                       "agent_texts": [x["agent_text"] for x in trace],
                       "obs": [x["obs"] for x in trace],
                       "solved": result["solved"],
                       "total_actions": result["total_actions"],
                       "elapsed_s": round(time.time() - t0, 2)}
            except Exception as e:
                row = {"model": model, "n": N, "n_types": T, "hint": True,
                       "budget": BUDGET, "relabel_seed": rep, "drop_seed": rep,
                       "error": f"{type(e).__name__}: {e}",
                       "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                built = (not row.get("error")) and any("fuse into" in o for o in row["obs"])
                print(f"  {model[:28]:28s} rep={rep} solved={row.get('solved')} "
                      f"built={built} actions={row.get('total_actions')} "
                      f"{'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(p, m, r) for p, m in MODELS for r in range(REPS)))
    print(f"\nDone. Episodes at: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
