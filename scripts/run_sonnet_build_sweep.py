"""Sonnet-only build-timing sweep: does it build the tool, and when?

Holds the world fixed (n=12, T=3) and sweeps the announced action budget, but
unlike run_sonnet_opus_budget_sweep this runs ONLY Sonnet, 25 reps/cell, and
KILLS each episode the instant the machine is first built (stop_on_build=True in
toolworld_v2.run). The budget is still announced to the agent (so it shapes the
build-vs-brute decision), but we never spend actions past the build -- the only
quantities of interest are whether it builds and at which action.

Rep r => (relabel_seed=r, drop_seed=r), shared across cells (paired worlds).

Output: one JSONL row per episode (same schema as the other budget sweeps; a
non-builder simply runs to its budget/turn cap and records built=False). After
the sweep it prints a per-budget build-rate + build-timing table.

Usage: PYTHONPATH=. python -m scripts.run_sonnet_build_sweep
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

# --- the sweep you set --------------------------------------------------
N = 12                                                       # constant doors
T = 3                                                        # constant types
BUDGET_VALUES = [20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
REPS = 25                                                    # reps per budget

MODELS = [("anthropic", "claude-sonnet-4-6")]                # Sonnet only
CONCURRENCY = cfg.CONCURRENCY


def max_turns_for(budget: int) -> int:
    """Turn cap kept above the budget so the announced BUDGET binds first, not
    the turn limit (no-op turns don't count toward the budget). On build the
    episode stops early anyway, so this only matters for non-builders."""
    return max(cfg.MAX_TURNS, round(budget * 1.5))


async def main():
    load_dotenv()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"sonnet_build_sweep_T{T}_n{N}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    out_path.write_text("")

    cells = [(prov, model, budget, rep)
             for prov, model in MODELS
             for budget in BUDGET_VALUES
             for rep in range(REPS)]
    print(f"Writing to {out_path}\n"
          f"Sweep: Sonnet x budget in {BUDGET_VALUES} x {REPS} reps "
          f"= {len(cells)} episodes (n={N}, T={T}); stop_on_build=True", flush=True)

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
                                          budget=budget, stop_on_build=True)
                row = {"model": model, "n": N, "n_types": T, "hint": cfg.HINT,
                       "budget": budget, "relabel_seed": rep, "drop_seed": rep,
                       "labels": result["labels"],
                       "actions": [x["action"] for x in trace],
                       "agent_texts": [x["agent_text"] for x in trace],
                       "obs": [x["obs"] for x in trace],
                       "solved": result["solved"],
                       "built_machine": result["built_machine"],
                       "build_turn": result["build_turn"],
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
                bt = row.get("build_turn")
                print(f"  sonnet b={budget:<4} rep={rep:<2} "
                      f"built={row.get('built_machine')} "
                      f"build_turn={bt} actions={row.get('total_actions')} "
                      f"{'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(*c) for c in cells))
    print(f"\nDone. Episodes at: {out_path}")
    display_results(rows)


def display_results(rows: list[dict]) -> None:
    """Per-budget: build rate and, among builders, the action at which the
    machine was built (build_turn is 0-based; actions-to-build = build_turn+1)."""
    ok = [r for r in rows if not r.get("error")]
    nerr = len(rows) - len(ok)
    by = defaultdict(list)
    for r in ok:
        by[r["budget"]].append(r)

    print(f"\nSonnet build timing (n={N}, T={T}, {REPS} reps/budget"
          f"{f'; {nerr} errored excluded' if nerr else ''}):\n")
    print(f"{'budget':>7} {'eps':>4} {'built':>6} {'build_rate':>10} | "
          f"{'min':>4} {'median':>7} {'mean':>6} {'max':>4}  (actions to build, builders only)")
    print("-" * 86)
    for b in BUDGET_VALUES:
        g = by.get(b)
        if not g:
            continue
        builders = [r for r in g if r.get("built_machine")]
        nb = len(builders)
        # actions-to-build = build_turn + 1 (1-based action count at the build)
        atb = sorted((r["build_turn"] + 1) for r in builders
                     if r.get("build_turn") is not None)
        if atb:
            cell = (f"{atb[0]:>4} {statistics.median(atb):>7.1f} "
                    f"{statistics.mean(atb):>6.1f} {atb[-1]:>4}")
        else:
            cell = f"{'-':>4} {'-':>7} {'-':>6} {'-':>4}"
        print(f"{b:>7} {len(g):>4} {nb:>6} {nb/len(g):>10.2f} | {cell}")


if __name__ == "__main__":
    asyncio.run(main())
