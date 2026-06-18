"""Cross-model budget sweep for the explicit-pickup world (v3).

Identical to run_budget_sweep.py (same N/T/REPS/BUDGET and paired per-rep
worlds from sweep_config) except each model plays the v3 world: examine drops
the key + part on the GROUND and a free `pickup` action collects them. pickup
costs no budget, so the BUDGETED cost model -- and thus the budget calibration
-- is identical to v2; episodes stay paired-comparable to the v2 sweep.

Each row carries "variant": "v3_pickup" so replay_toolworld / analyze_toolworld
route to the v3 State and exclude pickups from action-cost metrics.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from scripts.toolworld_v3 import run
from scripts import sweep_config as cfg

MODELS = cfg.ANTHROPIC_MODELS
N, T, REPS = cfg.N, cfg.T, cfg.REPS
BUDGET = cfg.BUDGET
CONCURRENCY = cfg.CONCURRENCY


async def main():
    load_dotenv()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"budget_sweep_v3_T{T}_n{N}_b{BUDGET}_{ts}"
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
                                          drop_seed=rep, hint=cfg.HINT,
                                          max_turns=cfg.MAX_TURNS, budget=BUDGET)
                row = {"model": model, "n": N, "n_types": T, "hint": cfg.HINT,
                       "variant": "v3_pickup", "budget": BUDGET,
                       "relabel_seed": rep, "drop_seed": rep,
                       "labels": result["labels"],
                       "actions": [x["action"] for x in trace],
                       "agent_texts": [x["agent_text"] for x in trace],
                       "obs": [x["obs"] for x in trace],
                       "solved": result["solved"],
                       "total_actions": result["total_actions"],
                       "pickups": result["pickups"],
                       "usage": result["usage"],
                       "elapsed_s": round(time.time() - t0, 2)}
            except Exception as e:
                row = {"model": model, "n": N, "n_types": T, "hint": cfg.HINT,
                       "variant": "v3_pickup", "budget": BUDGET,
                       "relabel_seed": rep, "drop_seed": rep,
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
