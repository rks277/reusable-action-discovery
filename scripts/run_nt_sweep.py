"""Budgeted sweep over an n x T grid (the design-surface generalization of
run_budget_sweep.py).

Set the three arrays below -- N_VALUES, T_VALUES, REPS -- and this runs every
(model, n, T, rep) cell. The action budget is DERIVED PER CELL from the
coupon-collector expectation via sweep_config.budget_for(n), so each n gets the
budget appropriate to its grind cost (a fixed scalar budget would make large-n
cells trivially easy and small-n cells impossible -- see the design-surface
note: n is the "build margin" lever, T the "discoverability" lever).

Rep r => (relabel_seed=r, drop_seed=r), shared across every cell, so the same
rep index is the same world at a given (n, T) -> paired comparison across models.

Output: one JSONL row per episode (same schema as run_budget_sweep.py, plus the
per-cell budget), so replay_toolworld / analyze_budget load it unchanged. Note
each row carries its own n / n_types / budget -- analysis that pools across the
grid must group by those, not just by model.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from scripts.toolworld_v2 import run
from scripts import sweep_config as cfg

# --- the grid you set ---------------------------------------------------
N_VALUES = [4, 8, 12, 20]      # doors per episode (build-margin axis)
T_VALUES = [3]              # distinct byproduct types (recipe-search axis)
REPS = 10                       # runs per (model, n, T) cell

MODELS = cfg.ANTHROPIC_MODELS  # swap to cfg.NEWMODELS, or concatenate, as needed
CONCURRENCY = cfg.CONCURRENCY


def max_turns_for(budget: int) -> int:
    """Keep the turn cap safely above the budget so the BUDGET binds first, not
    the turn limit (no-op turns don't count toward the budget, hence the slack)."""
    return max(cfg.MAX_TURNS, round(budget * 1.5))


async def main():
    load_dotenv()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"nt_sweep_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    out_path.write_text("")

    cells = [(prov, model, n, T, rep)
             for prov, model in MODELS
             for n in N_VALUES
             for T in T_VALUES
             for rep in range(REPS)]
    print(f"Writing to {out_path}\n"
          f"Grid: {len(MODELS)} models x n in {N_VALUES} x T in {T_VALUES} "
          f"x {REPS} reps = {len(cells)} episodes", flush=True)
    print("Per-n budgets: "
          + ", ".join(f"n={n}->b={cfg.budget_for(n)}" for n in N_VALUES), flush=True)

    sems = {p: asyncio.Semaphore(CONCURRENCY[p]) for p, _ in MODELS}
    lock = asyncio.Lock()

    async def one(prov, model, n, T, rep):
        budget = cfg.budget_for(n)
        async with sems[prov]:
            t0 = time.time()
            try:
                result, trace = await run(model, n=n, n_types=T, relabel_seed=rep,
                                          drop_seed=rep, hint=cfg.HINT,
                                          max_turns=max_turns_for(budget),
                                          budget=budget)
                row = {"model": model, "n": n, "n_types": T, "hint": cfg.HINT,
                       "budget": budget, "relabel_seed": rep, "drop_seed": rep,
                       "labels": result["labels"],
                       "actions": [x["action"] for x in trace],
                       "agent_texts": [x["agent_text"] for x in trace],
                       "obs": [x["obs"] for x in trace],
                       "solved": result["solved"],
                       "total_actions": result["total_actions"],
                       "noop_total": result["noop_total"],
                       "refusals": result["refusals"],
                       "unparsed": result["unparsed"],
                       "usage": result["usage"],
                       "elapsed_s": round(time.time() - t0, 2)}
            except Exception as e:
                row = {"model": model, "n": n, "n_types": T, "hint": cfg.HINT,
                       "budget": budget, "relabel_seed": rep, "drop_seed": rep,
                       "error": f"{type(e).__name__}: {e}",
                       "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                built = (not row.get("error")) and any("fuse into" in o for o in row["obs"])
                print(f"  {model[:24]:24s} n={n:<2} T={T} rep={rep} b={budget} "
                      f"solved={row.get('solved')} built={built} "
                      f"actions={row.get('total_actions')} "
                      f"{'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(*c) for c in cells))
    print(f"\nDone. Episodes at: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
