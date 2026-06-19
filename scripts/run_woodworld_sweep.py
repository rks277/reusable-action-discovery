"""Cross-model woodworld sweep (build-or-fail under a B = round(1.2 N) budget).

Each model plays the obfuscated woodworld at several N straddling N* (~13), under
a strict total-action budget. Outcome per episode: BUILT+SOLVED / BUILT-ran-out /
FAILED-never-built / BRUTE+SOLVED. The discriminating metric is built_axe rate
vs N (calibration) plus solve-rate and action efficiency.

Grid: models x N_VALUES x REPS. rep r => (relabel_seed=r, gather_seed=r), shared
across models so each rep is the same world (paired comparison).
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from scripts.woodworld import run
from scripts import woodworld_config as cfg

MODELS = cfg.OSS_MODELS   # or cfg.ANTHROPIC_MODELS / cfg.NEWMODELS

N_VALUES, REPS = cfg.N_VALUES, cfg.REPS
CONCURRENCY = cfg.CONCURRENCY


async def main():
    load_dotenv()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"woodworld_sweep_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    out_path.write_text("")
    print(f"Writing to {out_path}  (N={N_VALUES}, {len(MODELS)} models x {REPS} reps)",
          flush=True)

    sems = {p: asyncio.Semaphore(CONCURRENCY[p]) for p, _ in MODELS}
    lock = asyncio.Lock()

    async def one(prov, model, n, rep):
        budget = cfg.budget_for(n)
        async with sems[prov]:
            t0 = time.time()
            try:
                result, trace = await run(model, n=n, relabel_seed=rep, gather_seed=rep,
                                          hint=cfg.HINT, max_turns=cfg.MAX_TURNS,
                                          budget=budget)
                row = {"model": model, "n": n, "hint": cfg.HINT, "budget": budget,
                       "relabel_seed": rep, "gather_seed": rep,
                       "labels": result["labels"],
                       "actions": [x["action"] for x in trace],
                       "agent_texts": [x["agent_text"] for x in trace],
                       "obs": [x["obs"] for x in trace],
                       "solved": result["solved"],
                       "total_actions": result["total_actions"],
                       "built_axe": result["built_axe"],
                       "build_turn": result["build_turn"],
                       "t_star": result["t_star"],
                       "use_axe_count": result["use_axe_count"],
                       "wood_by_source": result["wood_by_source"],
                       "final_wood": result["final_wood"],
                       "stopped_reason": result["stopped_reason"],
                       "usage": result["usage"],
                       "elapsed_s": round(time.time() - t0, 2)}
            except Exception as e:
                row = {"model": model, "n": n, "hint": cfg.HINT, "budget": budget,
                       "relabel_seed": rep, "gather_seed": rep,
                       "error": f"{type(e).__name__}: {e}",
                       "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  {model[:28]:28s} N={n:>2} rep={rep} "
                      f"solved={row.get('solved')} built={row.get('built_axe')} "
                      f"actions={row.get('total_actions')}/{budget} "
                      f"{'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(p, m, n, r)
                           for p, m in MODELS for n in N_VALUES for r in range(REPS)))
    print(f"\nDone. Episodes at: {out_path}")
    print(f"Next: python -m scripts.replay_woodworld {out_path}")
    print(f"      python -m scripts.analyze_woodworld {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
