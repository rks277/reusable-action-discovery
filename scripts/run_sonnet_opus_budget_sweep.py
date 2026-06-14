"""Sweep Sonnet and Opus over a budget array at n=12, T=3, then display results.

Holds the world difficulty fixed (n=12 doors, T=3 byproduct types) and sweeps the
strict action budget across BUDGET_VALUES, for both Sonnet and Opus. This isolates
the budget as the independent variable: same worlds, only the action allowance
changes, so the solve-rate / wasted-action curve is read directly against budget
and you can see where each model's "build the tool vs. brute-force" trade-off flips.

Rep r => (relabel_seed=r, drop_seed=r), shared across every cell, so the same rep
index is the same world -> paired comparison across models AND across budgets (the
budget never touches world mechanics, so the worlds stay identical as budget varies).

Output: one JSONL row per episode (same schema as run_budget_arr_sweep.py), so
replay_toolworld / analyze_budget load it unchanged. After the sweep it prints a
per-(model, budget) results table to stdout.

Usage: PYTHONPATH=. python -m scripts.run_sonnet_opus_budget_sweep
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from scripts.toolworld_v2 import run
from scripts import sweep_config as cfg
from scripts.analyze_budget import per_episode, short

# --- the sweep you set --------------------------------------------------
N = 12                                          # constant doors per episode
T = 3                                           # constant byproduct types
BUDGET_VALUES = [20, 40, 80, 160, 320, 640, 1280]  # fixed action budgets to sweep
REPS = cfg.REPS                                 # runs per (model, budget) cell

MODELS = [
    ("anthropic", "claude-sonnet-4-6"),
    ("anthropic", "claude-opus-4-8"),
]
CONCURRENCY = cfg.CONCURRENCY


def max_turns_for(budget: int) -> int:
    """Keep the turn cap safely above the budget so the BUDGET binds first, not
    the turn limit (no-op turns don't count toward the budget, hence the slack)."""
    return max(cfg.MAX_TURNS, round(budget * 1.5))


async def main():
    load_dotenv()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"sonnet_opus_budget_sweep_T{T}_n{N}_{ts}"
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
    rows: list[dict] = []

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
                rows.append(row)
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                built = (not row.get("error")) and any("fuse into" in o for o in row["obs"])
                print(f"  {model[:24]:24s} b={budget:<4} rep={rep} "
                      f"solved={row.get('solved')} built={built} "
                      f"actions={row.get('total_actions')} "
                      f"{'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(*c) for c in cells))
    print(f"\nDone. Episodes at: {out_path}")

    display_results(rows)


def display_results(rows: list[dict]) -> None:
    """Per-(model, budget) summary: solve rate, built rate, wasted-action
    fraction, mean build turn, and the outcome breakdown."""
    ok = [r for r in rows if not r.get("error")]
    nerr = len(rows) - len(ok)
    eps = [(r["model"], r["budget"], per_episode(r)) for r in ok]

    by = defaultdict(list)
    for model, budget, e in eps:
        by[(model, budget)].append(e)

    order = ["claude-sonnet-4-6", "claude-opus-4-8"]
    models = [m for m in order if any(mm == m for mm, _ in by)] \
        + [m for m, _ in by if m not in order]
    # de-dup while preserving order
    models = list(dict.fromkeys(models))

    print(f"\nResults (n={N}, T={T}, {REPS} reps/cell"
          f"{f'; {nerr} errored episodes excluded' if nerr else ''}):\n")
    print(f"{'model':14s} {'budget':>7} {'eps':>3} {'solve':>6} {'built':>6} "
          f"{'waste%':>7} {'build@':>7} | "
          f"built+solved  built+ranout  brute+solved  neverbuilt")
    print("-" * 112)
    for m in models:
        for budget in BUDGET_VALUES:
            g = by.get((m, budget))
            if not g:
                continue
            solve = statistics.mean(e["solved"] for e in g)
            built = statistics.mean(e["built"] for e in g)
            waste = statistics.mean(e["waste_frac"] for e in g)
            bts = [e["build_turn"] for e in g if e["build_turn"] is not None]
            bt = statistics.mean(bts) if bts else float("nan")
            oc = defaultdict(int)
            for e in g:
                oc[e["outcome"]] += 1
            print(f"{short(m):14s} {budget:>7} {len(g):>3} {solve:>6.2f} "
                  f"{built:>6.2f} {waste*100:>6.0f}% {bt:>7.1f} | "
                  f"{oc['built+solved']:>11}  {oc['built+ranout']:>12}  "
                  f"{oc['brute+solved']:>12}  {oc['neverbuilt']:>10}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
