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
from scripts.run_beach_sweep import durability_for, total_rocks_for  # noqa: E402

# --- the grid you set ---------------------------------------------------
MODELS = [
    # Local Gemma via Ollama (the ":" in each id routes to the ollama provider;
    # edit the family tag to match `ollama list`, e.g. gemma3, if needed).
    ("ollama", "gemma4:1b"),
    ("ollama", "gemma4:4b"),
    ("ollama", "gemma4:12b"),
    # Anthropic models unhooked for now -- bring back later.
    # ("anthropic", "claude-opus-4-8"),
    # ("anthropic", "claude-sonnet-4-6"),
    # ("anthropic", "claude-haiku-4-5-20251001"),
    # ("openai", "gpt-5"),
    # ("google", "gemini-2.5-pro"),
]
# swept axes
GRID_VALUES = [4, 5, 6]      # n x n beach (grid_size); the search/grind space
PAPERS_VALUES = [2, 3, 4, 5]  # scraps needed to form the map (papers_needed)
REPS = 10                     # runs per cell; rep r => seed=r

# Optional constant overrides. Leave as None to DERIVE per cell (recommended, so
# the levers stay calibrated to grid/papers); set an int to pin it everywhere.
DURABILITY = None          # int to fix shovel digs, else durability_for(grid)
TOTAL_ROCKS = None         # int to fix rock count, else total_rocks_for(grid)

HINT = True
MAX_TURNS = 200            # generous; durability/grid are the real caps

# per-provider in-flight episode cap (bounds rate-limit / token bursts).
# ollama is local (one server) -> keep it small to avoid thrashing.
CONCURRENCY = {"anthropic": 6, "openai": 4, "google": 4, "ollama": 2}


async def main():
    load_dotenv()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"beach_v2_sweep_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    out_path.write_text("")

    # Expand cells over the swept axes; resolve per-cell durability/total_rocks
    # (constant override or derived) and CLAMP the rock count to the grid: it can
    # never exceed every cell but the one rock-free treasure cell, so a count
    # that would overflow just fills the whole grid with rocks. A (grid, papers)
    # combo is dropped only if even a full grid can't hold the P paper rocks.
    cells, skipped = [], []
    for prov, model in MODELS:
        for grid in GRID_VALUES:
            for papers in PAPERS_VALUES:
                cap = grid * grid - 1   # all cells but the rock-free treasure cell
                rocks = TOTAL_ROCKS if TOTAL_ROCKS is not None else total_rocks_for(grid)
                rocks = min(rocks, cap)   # overflow -> entire grid is rocks
                dur = DURABILITY if DURABILITY is not None else durability_for(grid)
                if rocks < papers:        # can't place the P paper rocks -> infeasible
                    skipped.append((grid, papers, rocks))
                    continue
                for rep in range(REPS):
                    cells.append((prov, model, grid, papers, rocks, dur, rep))

    print(f"Writing to {out_path}\n"
          f"Grid: {len(MODELS)} models x grid in {GRID_VALUES} x papers in "
          f"{PAPERS_VALUES} x {REPS} reps = {len(cells)} episodes "
          f"(durability={DURABILITY or 'f(grid)'}, "
          f"total_rocks={TOTAL_ROCKS or 'f(grid)'})", flush=True)
    if skipped:
        uniq = sorted(set(skipped))
        print(f"  skipped {len(uniq)} infeasible (grid, papers) combo(s) "
              f"[rocks don't fit grid]: "
              + ", ".join(f"grid{g}/P{p}/rocks{r}" for g, p, r in uniq), flush=True)

    sems = {p: asyncio.Semaphore(CONCURRENCY[p]) for p, _ in MODELS}
    lock = asyncio.Lock()

    async def one(prov, model, grid, papers, rocks, dur, rep):
        async with sems[prov]:
            t0 = time.time()
            cfg = {"model": model, "grid_size": grid, "seed": rep,
                   "papers_needed": papers, "total_rocks": rocks,
                   "shovel_durability": dur, "hint": HINT}
            try:
                result, trace = await run(
                    model, grid_size=grid, seed=rep, papers_needed=papers,
                    total_rocks=rocks, shovel_durability=dur, hint=HINT,
                    max_turns=MAX_TURNS)
                row = {
                    **cfg,
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
                print(f"  {model[:24]:24s} grid={grid} P={papers} rocks={rocks} "
                      f"dur={dur:<2} rep={rep} won={row.get('won')} "
                      f"built_map={row.get('built_map')} digs={row.get('digs')} "
                      f"actions={row.get('total_actions')} "
                      f"{'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(*c) for c in cells))
    print(f"\nDone. Episodes at: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
