"""Sweep over the beach treasure-hunt world (scripts/run_beach_llm.py), the
build-vs-grind analog of run_nt_sweep.py.

Set the arrays below -- MODELS, DURABILITY_VALUES, GRID_VALUES, REPS (+ the
fixed PAPERS_NEEDED / TOTAL_ROCKS) -- and this runs every
(model, durability, grid, rep) cell concurrently (per-provider in-flight caps).

WHY shovel_durability is the headline axis: it is the beach's "build margin"
lever, exactly as N is in the toolworld sweep. The map is the reusable TOOL
(collect P scraps -> map -> read the treasure coordinate -> one walk + dig);
brute force is digging cells until you hit the chest, capped by the shovel's
durability. A tiny durability makes blind digging hopeless, so building the map
is forced; a large durability makes grinding a viable (often cheaper) escape
hatch, so building becomes an economic CHOICE. Sweeping durability traces where
the model flips from grind to build.

Rep r => seed=r, shared across every cell, so the same rep index is the SAME
world (treasure / rock / paper / start-cell layout) at a given grid -> paired
comparison across models and durabilities.

Output: one JSONL row per episode carrying the full run() instrumentation
(won / built_map / used_map / digs / ... ) plus the per-cell config and a
compact trace (actions, agent_texts, messages), so downstream analysis loads it
unchanged. Each row carries its own grid / durability, so pooling across the
grid must group by those, not just by model.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Work whether invoked as `python scripts/run_beach_sweep.py` or
# `python -m scripts.run_beach_sweep`: ensure the repo root is importable so
# both `scripts.*` and `lomekwi.*` resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_beach_llm import run  # noqa: E402

# --- the grid you set ---------------------------------------------------
MODELS = [
    ("anthropic", "claude-opus-4-8"),
    ("anthropic", "claude-sonnet-4-6"),
    ("anthropic", "claude-haiku-4-5-20251001"),
    # ("openai", "gpt-5"),
    # ("google", "gemini-2.5-pro"),
]
DURABILITY_VALUES = [5]   # build-vs-grind lever (shovel digs)
GRID_VALUES = [5]                        # n x n beach
REPS = 20                                # runs per cell; rep r => seed=r

# fixed world knobs (held constant across the grid)
PAPERS_NEEDED = 4
TOTAL_ROCKS = 12
HINT = True
MAX_TURNS = 200                          # generous; durability is the real cap

# per-provider in-flight episode cap (bounds rate-limit / token bursts)
CONCURRENCY = {"anthropic": 6, "openai": 4, "google": 4}


async def main():
    load_dotenv()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"beach_sweep_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    out_path.write_text("")

    cells = [(prov, model, dur, grid, rep)
             for prov, model in MODELS
             for dur in DURABILITY_VALUES
             for grid in GRID_VALUES
             for rep in range(REPS)]
    print(f"Writing to {out_path}\n"
          f"Grid: {len(MODELS)} models x durability in {DURABILITY_VALUES} x "
          f"grid in {GRID_VALUES} x {REPS} reps = {len(cells)} episodes",
          flush=True)

    sems = {p: asyncio.Semaphore(CONCURRENCY[p]) for p, _ in MODELS}
    lock = asyncio.Lock()

    async def one(prov, model, dur, grid, rep):
        async with sems[prov]:
            t0 = time.time()
            try:
                result, trace = await run(
                    model, grid_size=grid, seed=rep, papers_needed=PAPERS_NEEDED,
                    total_rocks=TOTAL_ROCKS, shovel_durability=dur, hint=HINT,
                    max_turns=MAX_TURNS)
                row = {
                    "model": model, "grid_size": grid, "seed": rep,
                    "papers_needed": PAPERS_NEEDED, "total_rocks": TOTAL_ROCKS,
                    "shovel_durability": dur, "hint": HINT,
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
                row = {
                    "model": model, "grid_size": grid, "seed": rep,
                    "papers_needed": PAPERS_NEEDED, "total_rocks": TOTAL_ROCKS,
                    "shovel_durability": dur, "hint": HINT,
                    "error": f"{type(e).__name__}: {e}",
                    "elapsed_s": round(time.time() - t0, 2),
                }
            async with lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  {model[:24]:24s} dur={dur:<2} grid={grid} rep={rep} "
                      f"won={row.get('won')} built_map={row.get('built_map')} "
                      f"digs={row.get('digs')} actions={row.get('total_actions')} "
                      f"{'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(*c) for c in cells))
    print(f"\nDone. Episodes at: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
