"""Haiku build-rate-vs-(T, N) region sweep: run-to-solve over a sample of world
configurations, so we can map P(built | solved) across the doors x byproduct-types
plane.

Unlike run_haiku_build_sweep.py (constant n=12, T=3, swept budget, stop_on_build),
this sweep:
  - VARIES (n, n_types) per cell across the rectangle T in [2,10], N in [1,40];
  - uses the grind-calibrated budget cfg.budget_for(n) (T-independent) -- "the
    computed budget";
  - RUNS TO SOLVE (stop_on_build=False), so an episode ends on `solved` or
    `out_of_budget`. We later ask, among the SOLVED episodes of each (T, N) cell,
    what proportion BUILT the machine (vs brute-forced).

Sampling: 30 (T, N) points drawn uniformly (fixed SEED) from the 360-point
rectangle -- NO build<grind filter, so the map also covers configs where building
is the irrational choice. Each point gets REPS=10 paired seeds (rep r =>
relabel_seed = drop_seed = r). All raw episode data is saved to episodes.jsonl;
the sampled points + seed go to points.json for reproducibility.

The hallucination/flounder SAFEGUARD (no_progress_window) is wired in exactly as
the build sweeps: abort an episode after NO_PROGRESS_WINDOW executed actions with
no new type/key/door/build. A genuine solver always makes such progress, so the
safeguard only caps doomed brute-force flounders (recorded stopped_reason=
"no_progress"), never converting a would-be builder/solver into a false negative.

Usage: PYTHONPATH=. python -m scripts.run_haiku_region_sweep              # 30 pts x 10 reps, N<=40
       PYTHONPATH=. python -m scripts.run_haiku_region_sweep --n-points 100 --reps 1 --n-hi 20
       PYTHONPATH=. python -m scripts.run_haiku_region_sweep --smoke      # 1 episode
       HAIKU_REGION_RESUME=runs/<dir> PYTHONPATH=. python -m scripts.run_haiku_region_sweep
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from scripts.toolworld_v2 import run
from scripts import sweep_config as cfg

# --- the sweep (defaults reproduce the original 30x10, N<=40 run) ------
MODEL = "claude-haiku-4-5-20251001"
SEED = 12345                 # fixes which (T, N) points are drawn
N_POINTS = 30                # sampled (T, N) cells
REPS = 10                    # paired seeds per cell (rep r => relabel=drop=r)
CONC = 12                    # in-flight episodes (local; cfg.CONCURRENCY untouched)

T_LO, T_HI = 2, 10           # byproduct types: 2..10 (recipe needs 2 distinct)
N_LO, N_HI = 1, 40           # doors: 1..N_HI (overridable via --n-hi)

# Abort after this many executed actions with NO real state progress (no new
# byproduct type, key, door, or build). Sized far above any legitimate solve
# trajectory so it can only fire on a flounder, never on a real solver/builder.
NO_PROGRESS_WINDOW = 64

# Per-MTok pricing (input, output, cache-write, cache-read), keyed by model short
# name. Used only for the optional --max-cost kill switch; cost is summed from each
# episode's logged `usage` (cache fields included for correctness though currently 0).
PRICING = {                       # $/1M tokens: (input, output, cache_write, cache_read)
    "haiku":  (1.0,  5.0,  1.25, 0.10),
    "sonnet": (3.0, 15.0,  3.75, 0.30),
    "opus":   (5.0, 25.0,  6.25, 0.50),
}


def episode_cost(usage: dict, price) -> float:
    """Dollar cost of one episode from its logged token usage."""
    pin, pout, pcw, pcr = price
    return (usage.get("input_tokens", 0) * pin
            + usage.get("output_tokens", 0) * pout
            + usage.get("cache_write_tokens", 0) * pcw
            + usage.get("cache_read_tokens", 0) * pcr) / 1e6


def max_turns_for(budget: int) -> int:
    """Keep the turn cap above the announced budget so the BUDGET binds first."""
    return max(cfg.MAX_TURNS, round(budget * 1.5))


def sample_points(n_points: int, seed: int, n_hi: int):
    """Uniform sample (no build<grind filter) of (t, n) from the rectangle. When
    n_points >= the rectangle size, return the FULL rectangle (full coverage) --
    random.sample would otherwise raise."""
    rectangle = [(t, n) for t in range(T_LO, T_HI + 1) for n in range(N_LO, n_hi + 1)]
    if n_points >= len(rectangle):
        return sorted(rectangle)
    return sorted(random.Random(seed).sample(rectangle, n_points))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--n-points", type=int, default=N_POINTS)
    p.add_argument("--reps", type=int, default=REPS)
    p.add_argument("--n-hi", type=int, default=N_HI)
    p.add_argument("--conc", type=int, default=CONC,
                   help="in-flight episodes (local override; default tuned for the API)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max-cost", type=float, default=None,
                   help="kill switch: once cumulative logged cost (USD) reaches this, "
                        "stop launching new episodes (in-flight ones finish).")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


async def main():
    load_dotenv()
    args = parse_args()
    smoke = args.smoke

    model = args.model
    short = model.split("-")[1] if model.startswith("claude-") else model  # haiku/sonnet/opus

    resume = os.environ.get("HAIKU_REGION_RESUME")
    if resume:
        out_dir = Path(resume)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = "smoke" if smoke else f"p{args.n_points}_r{args.reps}_Nle{args.n_hi}"
        out_dir = Path("runs") / f"{short}_region_sweep_{tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    out_path.touch()  # append-only; never truncate

    points = sample_points(args.n_points, args.seed, args.n_hi)
    reps = range(args.reps)
    if smoke:
        points, reps = points[:1], range(1)   # one cell, one seed
    # record the sampled design for reproducibility
    (out_dir / "points.json").write_text(json.dumps(
        {"seed": args.seed, "n_points": args.n_points, "reps": args.reps,
         "t_range": [T_LO, T_HI], "n_range": [N_LO, args.n_hi],
         "points": [list(p) for p in points],
         "budgets": {str(n): cfg.budget_for(n) for _, n in points}}, indent=2))

    # already-completed (n_types, n, relabel_seed) cells -> skip them on resume
    done = set()
    rows: list[dict] = []
    for line in out_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append(r)
        if not r.get("error"):
            done.add((r["n_types"], r["n"], r["relabel_seed"]))

    cells = [(t, n, rep)
             for (t, n) in points
             for rep in reps
             if (t, n, rep) not in done]
    print(f"Writing to {out_path}\n"
          f"Sweep: {short} ({model}) run-to-solve over {len(points)} (T,N) points x {len(reps)} reps "
          f"= {len(points) * len(reps)} episodes (budget = grind-calibrated budget_for(n); "
          f"stop_on_build=False, no_progress_window={NO_PROGRESS_WINDOW}, conc={args.conc})\n"
          f"resume: {len(done)} cells already done, {len(cells)} remaining",
          flush=True)

    sem = asyncio.Semaphore(args.conc)
    lock = asyncio.Lock()

    price = PRICING.get(short)
    cap = args.max_cost
    state = {"spent": 0.0, "capped": False}   # cumulative logged cost + kill flag

    async def one(t, n, rep):
        budget = cfg.budget_for(n)
        # cost-cap kill switch: once tripped, drain the queue without API calls.
        if cap is not None and state["capped"]:
            return
        async with sem:
            if cap is not None and state["capped"]:   # may have tripped while queued
                return
            t0 = time.time()
            try:
                result, trace = await run(model, n=n, n_types=t, relabel_seed=rep,
                                          drop_seed=rep, hint=cfg.HINT,
                                          max_turns=max_turns_for(budget),
                                          budget=budget, stop_on_build=False,
                                          no_progress_window=NO_PROGRESS_WINDOW)
                row = {"model": model, "n": n, "n_types": t, "hint": cfg.HINT,
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
                row = {"model": model, "n": n, "n_types": t, "hint": cfg.HINT,
                       "budget": budget, "relabel_seed": rep, "drop_seed": rep,
                       "error": f"{type(e).__name__}: {e}",
                       "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                rows.append(row)
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                if price and row.get("usage"):
                    state["spent"] += episode_cost(row["usage"], price)
                cost_note = f"${state['spent']:.2f}" if cap is not None else ""
                print(f"  T={t:<2} N={n:<2} b={budget:<4} rep={rep:<2} "
                      f"solved={row.get('solved')} built={row.get('built_machine')} "
                      f"actions={row.get('total_actions')} "
                      f"stop={row.get('stopped_reason')} {cost_note} "
                      f"{'(err)' if row.get('error') else ''}", flush=True)
                if cap is not None and state["spent"] >= cap and not state["capped"]:
                    state["capped"] = True
                    print(f"  !! COST CAP HIT: logged spend ${state['spent']:.2f} "
                          f">= ${cap:.2f}. No new episodes will launch "
                          f"(in-flight ones finish).", flush=True)

    await asyncio.gather(*(one(*c) for c in cells))
    print(f"\nDone. Episodes at: {out_path}")
    summarize(rows)


def summarize(rows: list[dict]) -> None:
    ok = [r for r in rows if not r.get("error")]
    nerr = len(rows) - len(ok)
    solved = [r for r in ok if r.get("solved")]
    built_solved = [r for r in solved if r.get("built_machine")]
    reasons = Counter(r.get("stopped_reason") for r in ok)
    print(f"\n{len(ok)} episodes ({nerr} errored) | solved {len(solved)} "
          f"| built&solved {len(built_solved)} "
          f"| P(built|solved)={len(built_solved)/len(solved):.2f}"
          if solved else f"\n{len(ok)} episodes, none solved")
    print(f"stopped_reason tally: {dict(reasons)}")
    bad = [r for r in ok if r.get("stopped_reason") == "no_progress"
           and r.get("built_machine")]
    if bad:
        print(f"  !! WARNING: {len(bad)} safeguard aborts were builders -- "
              f"raise NO_PROGRESS_WINDOW.")
    mt = [r for r in ok if r.get("stopped_reason") == "max_turns"]
    if mt:
        print(f"  !! WARNING: {len(mt)} episodes hit max_turns (budget should bind first).")


if __name__ == "__main__":
    asyncio.run(main())
