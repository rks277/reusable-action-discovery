"""Sweep over an explicit array of FIXED budgets at constant n and T.

Unlike run_nt_sweep.py (where the budget is DERIVED per cell from n via
sweep_config.budget_for), here n and T are held fixed and the budget is a plain
scalar swept across BUDGET_VALUES below. This isolates the budget itself as the
independent variable: same world difficulty (n, T), only the action allowance
changes, so the solve-rate / wasted-action curve is read directly against budget
(e.g. where the "build the tool vs. brute-force" trade-off flips).

Rep r => (relabel_seed=r, drop_seed=r), shared across every cell, so the same
rep index is the same world -> paired comparison across models AND across
budgets (the budget never touches world mechanics, so the worlds stay identical
as the budget varies).

Output: one JSONL row per episode (same schema as run_nt_sweep.py), so
replay_toolworld / analyze_budget load it unchanged. Each row carries its own
budget -- analysis that pools across the sweep must group by budget.
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

# --- the sweep you set --------------------------------------------------
BUDGET_VALUES = [1000] #[18, 36, 54, 72]  # fixed action budgets to sweep
N = cfg.N                                 # constant doors per episode
T = cfg.T                                 # constant byproduct types
REPS = cfg.REPS                           # runs per (model, budget) cell

MODELS = cfg.ANTHROPIC_MODELS  # swap to cfg.NEWMODELS, or concatenate, as needed
CONCURRENCY = cfg.CONCURRENCY


def max_turns_for(budget: int) -> int:
    """Keep the turn cap safely above the budget so the BUDGET binds first, not
    the turn limit (no-op turns don't count toward the budget, hence the slack)."""
    return max(cfg.MAX_TURNS, round(budget * 1.5))


async def main():
    load_dotenv()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"budget_arr_sweep_T{T}_n{N}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    out_path.write_text("")

    cells = [(prov, model, budget, rep)
             for prov, model in MODELS
             for budget in BUDGET_VALUES
             for rep in range(REPS)]
    print(f"Writing to {out_path}\n"
          f"Sweep: {len(MODELS)} models x budget in {BUDGET_VALUES} "
          f"x {REPS} reps = {len(cells)} episodes (n={N}, T={T})", flush=True)

    sems = {p: asyncio.Semaphore(CONCURRENCY[p]) for p, _ in MODELS}
    lock = asyncio.Lock()

    async def one(prov, model, budget, rep):
        async with sems[prov]:
            t0 = time.time()
            try:
                result, trace = await run(model, n=N, n_types=T, relabel_seed=rep,
                                          drop_seed=rep, hint=cfg.HINT,
                                          max_turns=max_turns_for(budget),
                                          budget=budget)
                row = {"model": model, "n": N, "n_types": T, "hint": cfg.HINT,
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
                row = {"model": model, "n": N, "n_types": T, "hint": cfg.HINT,
                       "budget": budget, "relabel_seed": rep, "drop_seed": rep,
                       "error": f"{type(e).__name__}: {e}",
                       "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                built = (not row.get("error")) and any("fuse into" in o for o in row["obs"])
                print(f"  {model[:24]:24s} b={budget:<3} rep={rep} "
                      f"solved={row.get('solved')} built={built} "
                      f"actions={row.get('total_actions')} "
                      f"{'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(*c) for c in cells))
    print(f"\nDone. Episodes at: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
