"""Sweep over the beach v2 world (scripts/run_beach_v2_llm.py) -- the no-movement,
coordinate-addressed build-vs-grind analog of run_beach_sweep.py.

Identical sweep machinery to run_beach_sweep.py (set MODELS / GRID_VALUES /
PAPERS_VALUES / REPS; every cell runs concurrently with per-provider in-flight
caps), reusing its per-cell calibration (durability_for / total_rocks_for) so the
two world levers stay comparable:
  - grid_size    : the search space (n x n cells -- sand to dig, rocks to search)
  - papers_needed: how many scraps must be collected before the map forms

The only difference from v1 is the environment: v2 removes movement, the agent
sees the whole grid, and inspect/dig are addressed by coordinate. The map is
still the reusable TOOL (collect P scraps -> map -> read the treasure coordinate
-> one dig); brute force is digging sand cells until you hit the chest, capped by
shovel durability.

Rep r => seed=r, shared across every cell, so the same rep index is the SAME
world at a given (grid, papers) -> paired comparison across models AND against the
v1 sweep (same seeds, same World.generate).

Output: one JSONL row per episode, schema-identical to run_beach_sweep.py, so
scripts/analyze_beach_v2.py (which reuses the v1 analyzer) loads it unchanged.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Work whether invoked as `python scripts/run_beach_v2_sweep.py` or
# `python -m scripts.run_beach_v2_sweep`: ensure the repo root is importable so
# both `scripts.*` and `lomekwi.*` resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_beach_v2_llm import run                              # noqa: E402

# --- the strip you sweep -----------------------------------------------
# We sweep STRIP geometry: a WIDTH x ROWS beach. WIDTH is the swept axis; ROWS is
# a fixed height (5 = the n x 5 strip; 1 = a 1-D strip). Everything is calibrated
# to the cell count area = WIDTH * ROWS.
MODELS = [
    ("anthropic", "claude-haiku-4-5-20251001"),
    ("anthropic", "claude-sonnet-4-6"),
    ("anthropic", "claude-opus-4-8"),
    # Local Gemma via Ollama -- unhooked for now, bring back later.
    # ("ollama", "gemma4:1b"),
    # ("ollama", "gemma4:4b"),
    # ("ollama", "gemma4:12b"),
    # ("openai", "gpt-5"),
    # ("google", "gemini-2.5-pro"),
]
# swept axes: full (n x m) grid, ONE rep per cell (one episode per (n, m)).
WIDTH_VALUES = list(range(1, 11))    # width n = 1..10
ROWS = None                          # None => SQUARE (n x n, area = n^2);
                                     # int => n x ROWS strip (area = n * ROWS)
PAPERS_VALUES = list(range(1, 11))   # m = scraps to form the map = 1..10
REPS = 1                             # one rep across the grid; rep r => seed=r

# Per-cell calibration, all from area = WIDTH * ROWS. Leave the *_FRAC knobs and
# set the pinned override to None to derive; set an int override to fix a value.
USE_DURABILITY = False     # False -> unbreakable shovel: BUDGET is the ONLY cap.
                           # True  -> shovel breaks after `dur` digs (the old 2-cap world).
DURABILITY = None          # int to pin digs, else round(DURABILITY_FRAC * area)
TOTAL_ROCKS = None         # int to pin rocks, else round(ROCK_DENSITY * area)
BUDGET = None              # int to pin budget, else round(BUDGET_FRAC * area)

DURABILITY_FRAC = 0.2      # digs   = round(area * this)  -- caps brute-force (if USE_DURABILITY)
ROCK_DENSITY = 0.4         # rocks  = round(area * this)  -- search difficulty
# Budget is a toolworld-style TOTAL-action cap, separate from durability. Floored
# at total_rocks + 2 so a thorough builder can inspect every rock + use map + dig
# without starving; 0.5*area sits just above worst-case build but below area.
BUDGET_FRAC = 0.5          # budget = round(area * this), floored at rocks + 2

HINT = True
OBFUSCATE = True           # letter-obfuscate sand/rock/treasure/map/paper per cell
MAX_TURNS = 300            # generous; durability/budget are the real caps

# per-provider in-flight episode cap (bounds rate-limit / token bursts).
# ollama is local (one server) -> keep it small to avoid thrashing.
CONCURRENCY = {"anthropic": 6, "openai": 4, "google": 4, "ollama": 2}


async def main():
    load_dotenv()
    # `--model <substr>` runs only the matching roster entries (keeps the full
    # MODELS list in the script while letting one run target a single model).
    roster = MODELS
    if "--model" in sys.argv:
        want = sys.argv[sys.argv.index("--model") + 1]
        roster = [(p, m) for (p, m) in MODELS if want in m]
        if not roster:
            raise SystemExit(f"--model {want!r} matched no entry in MODELS")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / "beach_v2" / f"beach_v2_sweep_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    out_path.write_text("")

    # Expand cells over the swept axes (WIDTH x PAPERS); resolve per-cell rocks /
    # durability / budget from area = WIDTH * rows, where rows = WIDTH when ROWS is
    # None (SQUARE n x n) else ROWS (n x ROWS strip). Rocks are clamped to the grid
    # (never more than every cell but the rock-free treasure cell); a (width,
    # papers) combo is dropped if even a full grid can't hold the P paper rocks.
    geom = "square (n x n)" if ROWS is None else f"strip (n x {ROWS})"
    cells, skipped = [], []
    for prov, model in roster:
        for width in WIDTH_VALUES:
            rows_eff = width if ROWS is None else ROWS
            for papers in PAPERS_VALUES:
                area = width * rows_eff
                cap = area - 1   # all cells but the rock-free treasure cell
                rocks = (TOTAL_ROCKS if TOTAL_ROCKS is not None
                         else round(ROCK_DENSITY * area))
                rocks = min(rocks, cap)   # overflow -> entire grid is rocks
                if not USE_DURABILITY:
                    dur = None            # unbreakable shovel -> budget is the only cap
                elif DURABILITY is not None:
                    dur = DURABILITY
                else:
                    dur = max(1, round(DURABILITY_FRAC * area))
                if rocks < papers:        # can't place the P paper rocks -> infeasible
                    skipped.append((width, papers, rocks))
                    continue
                budget = (BUDGET if BUDGET is not None
                          else max(round(BUDGET_FRAC * area), rocks + 2))
                for rep in range(REPS):
                    cells.append((prov, model, width, rows_eff, papers, rocks,
                                  dur, budget, rep))

    print(f"Writing to {out_path}\n"
          f"{geom} sweep: {len(roster)} model(s) {[m for _,m in roster]} x width in "
          f"{WIDTH_VALUES} x papers in {PAPERS_VALUES} x {REPS} reps = "
          f"{len(cells)} episodes\n"
          f"  per cell: durability="
          f"{'OFF (budget-only)' if not USE_DURABILITY else (DURABILITY or f'round({DURABILITY_FRAC}*area)')}, "
          f"rocks={TOTAL_ROCKS or f'round({ROCK_DENSITY}*area)'}, "
          f"budget={BUDGET or f'round({BUDGET_FRAC}*area)'}", flush=True)
    if skipped:
        uniq = sorted(set(skipped))
        print(f"  skipped {len(uniq)} infeasible (width, papers) combo(s) "
              f"[rocks don't fit grid]: "
              + ", ".join(f"w{w}/P{p}/rocks{r}" for w, p, r in uniq), flush=True)

    sems = {p: asyncio.Semaphore(CONCURRENCY[p]) for p, _ in roster}
    lock = asyncio.Lock()

    async def one(prov, model, width, rows_eff, papers, rocks, dur, budget, rep):
        async with sems[prov]:
            t0 = time.time()
            cfg = {"model": model, "grid_size": width, "rows": rows_eff, "seed": rep,
                   "papers_needed": papers, "total_rocks": rocks,
                   "shovel_durability": dur, "budget": budget, "hint": HINT}
            try:
                result, trace = await run(
                    model, grid_size=width, rows=rows_eff, seed=rep, papers_needed=papers,
                    total_rocks=rocks, shovel_durability=dur, hint=HINT,
                    max_turns=MAX_TURNS, budget=budget, obfuscate=OBFUSCATE,
                    relabel_seed=width * 131 + papers * 17 + rep)
                row = {
                    **cfg,
                    "obfuscate": result.get("obfuscate"),
                    "labels": result.get("labels"),
                    "treasure": result["treasure"],
                    "won": result["won"],
                    "built_map": result["built_map"],
                    "build_turn": result["build_turn"],
                    "used_map": result["used_map"],
                    "digs": result["digs"], "wasted_digs": result["wasted_digs"],
                    "papers_collected": result["papers_collected"],
                    "durability_left": result["durability_left"],
                    "total_actions": result["total_actions"],
                    "turns": result["turns"],
                    "noop_total": result["noop_total"],
                    "refusals": result["refusals"],
                    "stopped_reason": result["stopped_reason"],
                    "actions": [x["action"] for x in trace],
                    "agent_texts": [x["agent_text"] for x in trace],
                    "messages": [x["message"] for x in trace],
                    "usage": result["usage"],
                    "elapsed_s": round(time.time() - t0, 2),
                }
            except Exception as e:
                row = {**cfg, "error": f"{type(e).__name__}: {e}",
                       "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  {model[:24]:24s} {width}x{rows_eff} P={papers} rocks={rocks} "
                      f"dur={str(dur):<4} bud={budget:<3} rep={rep} won={row.get('won')} "
                      f"built_map={row.get('built_map')} digs={row.get('digs')} "
                      f"actions={row.get('total_actions')} "
                      f"{'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(*c) for c in cells))
    print(f"\nDone. Episodes at: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
