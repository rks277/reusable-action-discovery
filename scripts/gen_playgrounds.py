"""Generate a POOL of free-exploration 'playground' sessions to be injected later as
prior experience into budgeted game episodes (see the recognition/playground experiment).

Each session is an independent run(playground=True) call (fresh context = your "refresh
between sessions") in a RELABELED world (own relabel_seed, disjoint from the game's reps),
so the agent learns the device's MECHANICS without leaking the game's specific recipe.
No budget, no goal: the agent tinkers for --steps actions; we harvest the transcript.

The pool is SHARED across game models (we inject action->observation text, which is
model-agnostic), so a single pool generated once serves Haiku/Sonnet/Opus games. Each
saved session records whether it built the machine and the build turn, so downstream
injection can split into:
  - saw-build      : sessions that built (full transcript, incl. exploiting the machine)
  - saw-exploration: sessions that did NOT build, OR a built session TRUNCATED before its
                     build_turn (a matched prefix -- same world/early moves, no build shown)

Usage: PYTHONPATH=. python -m scripts.gen_playgrounds --n 8 --n-types 3 --sessions 10 \
           --steps 15 --model claude-haiku-4-5-20251001
Output: playgrounds/T{t}_n{n}/pool.jsonl  (one session per line) + meta in the same dir.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from scripts.toolworld_v2 import run
from scripts import sweep_config as cfg

# relabel seeds for playgrounds start here -- DISJOINT from game reps (0..) so the
# practice worlds are differently labeled and use a different recipe pair.
PG_SEED_BASE = 1000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--n-types", type=int, default=3)
    p.add_argument("--sessions", type=int, default=10)
    p.add_argument("--steps", type=int, default=15, help="free-exploration action cap")
    p.add_argument("--model", default="claude-haiku-4-5-20251001",
                   help="explorer model (pool is model-agnostic once formatted as action->obs)")
    p.add_argument("--conc", type=int, default=6)
    return p.parse_args()


async def main():
    load_dotenv()
    a = parse_args()
    out_dir = Path("playgrounds") / f"T{a.n_types}_n{a.n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pool.jsonl"
    out_path.touch()

    short = a.model.split("-")[1] if a.model.startswith("claude-") else a.model
    print(f"Generating {a.sessions} playground sessions | {short} | n={a.n} T={a.n_types} "
          f"| steps={a.steps} | relabel seeds {PG_SEED_BASE}..{PG_SEED_BASE+a.sessions-1}",
          flush=True)

    sem = asyncio.Semaphore(a.conc)
    lock = asyncio.Lock()
    rows = []

    async def one(i):
        seed = PG_SEED_BASE + i
        async with sem:
            t0 = time.time()
            result, trace = await run(a.model, n=a.n, n_types=a.n_types,
                                      relabel_seed=seed, drop_seed=seed,
                                      hint=cfg.HINT, max_turns=a.steps,
                                      budget=None, stop_on_build=False,
                                      no_progress_window=None, playground=True)
            row = {"session": i, "model": a.model, "n": a.n, "n_types": a.n_types,
                   "relabel_seed": seed, "drop_seed": seed,
                   "labels": result["labels"],
                   "steps": len(trace),
                   "built": result["built_machine"], "build_turn": result["build_turn"],
                   "solved": result["solved"], "opened": result["opened"],
                   "actions": [x["action"] for x in trace],
                   "obs": [x["obs"] for x in trace],
                   "agent_texts": [x["agent_text"] for x in trace],
                   "usage": result["usage"], "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                rows.append(row)
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  session {i:<2} seed={seed} recipe={row['labels']['recipe']} "
                      f"steps={row['steps']} built={row['built']} "
                      f"build_turn={row['build_turn']} opened={row['opened']}", flush=True)

    await asyncio.gather(*(one(i) for i in range(a.sessions)))

    built = [r for r in rows if r["built"]]
    (out_dir / "meta.json").write_text(json.dumps(
        {"model": a.model, "n": a.n, "n_types": a.n_types, "sessions": a.sessions,
         "steps": a.steps, "seed_base": PG_SEED_BASE,
         "n_built": len(built), "n_explore_only": len(rows) - len(built)}, indent=2))
    print(f"\nDone. {len(rows)} sessions -> {out_path}")
    print(f"  saw-build candidates (built):       {len(built)}")
    print(f"  saw-exploration candidates (no build): {len(rows) - len(built)} "
          f"(+ can truncate any built session before its build_turn for matched prefixes)")


if __name__ == "__main__":
    asyncio.run(main())
