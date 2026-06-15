"""Opus-only build-timing sweep: does it build the tool, and when?

Opus counterpart of run_sonnet_build_sweep.py -- SAME world (n=12, T=3), SAME
announced budgets, SAME reps, and stop_on_build=True, so the two episodes.jsonl
files are directly comparable (the only thing of interest is the per-budget
proportion of runs that BUILD the machine).

Difference from the Sonnet version: a hallucination/runaway-cost SAFEGUARD is
wired in (no_progress_window). Opus's known failure mode is to confabulate the
environment's responses and then trust that fiction over the real game, which on
high budgets spirals into 40M+ cache-read-token episodes (see HALLUCINATION_
FINDINGS.md). stop_on_build already truncates before the post-build spiral; the
safeguard additionally aborts any PRE-build flounder -- an episode that goes
NO_PROGRESS_WINDOW actions with no new byproduct type, no new key, no new door,
and no build. A genuine builder always trips one of those before the window, so
the safeguard never converts a would-be builder into a false non-builder; it only
caps cost on episodes that were never going to build (recorded built=False, with
stopped_reason="no_progress").

Rep r => (relabel_seed=r, drop_seed=r), shared across cells (paired worlds) and
identical to the Sonnet sweep's worlds.

Usage: PYTHONPATH=. python -m scripts.run_opus_build_sweep
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from scripts.toolworld_v2 import run
from scripts import sweep_config as cfg

# --- the sweep (matched to run_sonnet_build_sweep.py) -------------------
N = 12                                                       # constant doors
T = 3                                                        # constant types
BUDGET_VALUES = [20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
REPS = 25                                                    # default reps/budget
# The three largest budgets are the priciest per-episode (longest non-builders),
# so they run fewer reps to save budget. Everything <=640 keeps the full 25 to
# stay matched with the Sonnet build sweep.
REPS_BIG = 10
BIG_BUDGETS = {1280, 2560, 5120}


def reps_for(budget: int) -> int:
    return REPS_BIG if budget in BIG_BUDGETS else REPS

MODELS = [("anthropic", "claude-opus-4-8")]                 # Opus only
CONCURRENCY = cfg.CONCURRENCY

# --- hallucination / runaway-cost safeguard ----------------------------
# Abort an episode after this many executed actions with NO real state progress
# (no new byproduct type, no new key, no new door, no build). Sized far above any
# legitimate build trajectory (build candidates are <=C(T,2)+T, reachable in a
# few dozen actions) so it can only fire on a flounder, never on a builder.
NO_PROGRESS_WINDOW = 64


def max_turns_for(budget: int) -> int:
    """Turn cap kept above the budget so the announced BUDGET binds first, not
    the turn limit (no-op turns don't count toward the budget). On build the
    episode stops early anyway, so this only matters for non-builders."""
    return max(cfg.MAX_TURNS, round(budget * 1.5))


async def main():
    load_dotenv()
    # Resume into an existing run dir if OPUS_BUILD_RESUME points at one; this
    # lets us retune reps mid-sweep without re-spending on completed episodes.
    resume = os.environ.get("OPUS_BUILD_RESUME")
    if resume:
        out_dir = Path(resume)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("runs") / f"opus_build_sweep_T{T}_n{N}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    out_path.touch()  # append-only; never truncate (preserves any prior episodes)

    # already-completed (budget, rep) cells -> skip them on resume
    done = set()
    rows: list[dict] = []
    for line in out_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append(r)
        if not r.get("error"):
            done.add((r["budget"], r["relabel_seed"]))

    cells = [(prov, model, budget, rep)
             for prov, model in MODELS
             for budget in BUDGET_VALUES
             for rep in range(reps_for(budget))
             if (budget, rep) not in done]
    plan = {b: reps_for(b) for b in BUDGET_VALUES}
    print(f"Writing to {out_path}\n"
          f"Sweep: Opus x budget reps {plan} (n={N}, T={T}); "
          f"stop_on_build=True, no_progress_window={NO_PROGRESS_WINDOW}\n"
          f"resume: {len(done)} cells already done, {len(cells)} remaining",
          flush=True)

    sems = {p: asyncio.Semaphore(CONCURRENCY[p]) for p, _ in MODELS}
    lock = asyncio.Lock()  # `rows` (prior episodes) already loaded above

    async def one(prov, model, budget, rep):
        async with sems[prov]:
            t0 = time.time()
            try:
                result, trace = await run(model, n=N, n_types=T, relabel_seed=rep,
                                          drop_seed=rep, hint=cfg.HINT,
                                          max_turns=max_turns_for(budget),
                                          budget=budget, stop_on_build=True,
                                          no_progress_window=NO_PROGRESS_WINDOW)
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
                       "stopped_reason": result["stopped_reason"],
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
                print(f"  opus b={budget:<4} rep={rep:<2} "
                      f"built={row.get('built_machine')} "
                      f"build_turn={bt} actions={row.get('total_actions')} "
                      f"stop={row.get('stopped_reason')} "
                      f"{'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(*c) for c in cells))
    print(f"\nDone. Episodes at: {out_path}")
    display_results(rows)


def display_results(rows: list[dict]) -> None:
    """Per-budget: build rate and, among builders, the action at which the
    machine was built (build_turn is 0-based; actions-to-build = build_turn+1).
    Also reports how episodes ended (stopped_reason), so any safeguard aborts are
    visible -- they should all be non-builders."""
    ok = [r for r in rows if not r.get("error")]
    nerr = len(rows) - len(ok)
    by = defaultdict(list)
    for r in ok:
        by[r["budget"]].append(r)

    print(f"\nOpus build timing (n={N}, T={T}; {REPS} reps/budget, "
          f"{REPS_BIG} for {sorted(BIG_BUDGETS)}"
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
        atb = sorted((r["build_turn"] + 1) for r in builders
                     if r.get("build_turn") is not None)
        if atb:
            cell = (f"{atb[0]:>4} {statistics.median(atb):>7.1f} "
                    f"{statistics.mean(atb):>6.1f} {atb[-1]:>4}")
        else:
            cell = f"{'-':>4} {'-':>7} {'-':>6} {'-':>4}"
        print(f"{b:>7} {len(g):>4} {nb:>6} {nb/len(g):>10.2f} | {cell}")

    # safeguard audit: every no_progress abort must be a non-builder
    reasons = Counter(r.get("stopped_reason") for r in ok)
    print(f"\nstopped_reason tally: {dict(reasons)}")
    bad = [r for r in ok if r.get("stopped_reason") == "no_progress"
           and r.get("built_machine")]
    if bad:
        print(f"  !! WARNING: {len(bad)} safeguard aborts were builders "
              f"(false negatives) -- raise NO_PROGRESS_WINDOW.")
    else:
        nph = sum(1 for r in ok if r.get("stopped_reason") == "no_progress")
        print(f"  safeguard fired on {nph} episode(s), all non-builders (OK).")


if __name__ == "__main__":
    asyncio.run(main())
