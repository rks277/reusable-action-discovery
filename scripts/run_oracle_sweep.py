"""Sweep over the oracle-discovery environment (scripts/run_oracle_llm.py).

Same machinery as run_beach_v2_sweep.py: expand cells over the swept axes, run every
cell concurrently under per-provider in-flight caps, append one JSONL row per ROLLOUT to
runs/oracle/<ts>/episodes.jsonl. Each rollout is all-decoy; scoring into n
oracle-assignments is a downstream pass (scripts/oracle_counterfactual.py), so budget and
surcharge stay free post-hoc knobs.

Swept axes: model x N (number of tools) x difficulty x reps. Rep r => seed=r (shared
across cells, so the same rep is the SAME problem at a given difficulty -> paired).

CLI:  python scripts/run_oracle_sweep.py [--model <substr>]
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_oracle_llm import run                            # noqa: E402
from scripts.oracle_sweep_config import (MODELS, QWEN3_MODELS,      # noqa: E402
                                         CONCURRENCY, budget_for)

# --- swept axes ------------------------------------------------------------------------
# Difficulty spans the trio's manual-reliability cliffs (pilot: haiku~11, opus~17-20,
# sonnet~20) so each model passes through its reliable -> unreliable -> hopeless regimes,
# which is where tool discovery starts to matter.
N_VALUES = [4, 8, 12, 16, 20]          # linear step 4
DIFFICULTY_VALUES = [14, 16, 18, 20, 22]   # linear step 2

# Per-N rep counts so that N × reps ≈ constant (~20) across all N values,
# giving equal numbers of scored oracle-assignment rows per (model, N, difficulty) cell.
#   N=4:  5 reps → 4×5=20    N=8:  3 reps → 8×3=24
#   N=12: 2 reps → 12×2=24   N=16: 1 rep  → 16×1=16
#   N=20: 1 rep  → 20×1=20
REPS_BY_N = {4: 5, 8: 3, 12: 2, 16: 1, 20: 1}
PROBLEM_KIND = "long_division"
OBFUSCATE = False                     # non-semantic config names by default
SEQUENTIAL = True                     # one tool call per turn (enforced in host + prompt)
MAX_TURNS = 300

# budget per cell derived from difficulty (override to pin a constant if desired).
BUDGET = None                         # int to pin; None -> budget_for(difficulty)


async def main():
    load_dotenv()
    base = QWEN3_MODELS if "--qwen" in sys.argv else MODELS
    roster = base
    if "--model" in sys.argv:
        want = sys.argv[sys.argv.index("--model") + 1]
        roster = [(p, m) for (p, m) in base if want in m]
        if not roster:
            raise SystemExit(f"--model {want!r} matched no entry")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / "oracle" / f"oracle_sweep_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    out_path.write_text("")

    cells = []
    for prov, model in roster:
        for n in N_VALUES:
            reps = REPS_BY_N.get(n, 1)
            for difficulty in DIFFICULTY_VALUES:
                budget = BUDGET if BUDGET is not None else budget_for(difficulty, n=n)
                for rep in range(reps):
                    cells.append((prov, model, n, difficulty, budget, rep))

    scored_per_cell = {n: n * REPS_BY_N.get(n, 1) for n in N_VALUES}
    print(f"Writing to {out_path}\n"
          f"oracle sweep: {len(roster)} model(s) {[m for _, m in roster]} x N in "
          f"{N_VALUES} x difficulty in {DIFFICULTY_VALUES} x reps={REPS_BY_N} = "
          f"{len(cells)} rollouts\n"
          f"  scored rows/cell: {scored_per_cell}\n"
          f"  budget={'pinned ' + str(BUDGET) if BUDGET else 'budget_for(difficulty, n)'}, "
          f"obfuscate={OBFUSCATE}", flush=True)

    sems = {p: asyncio.Semaphore(CONCURRENCY.get(p, 4)) for p, _ in roster}
    lock = asyncio.Lock()

    async def one(prov, model, n, difficulty, budget, rep):
        async with sems[prov]:
            t0 = time.time()
            cfg = {"model": model, "n": n, "difficulty": difficulty, "budget": budget,
                   "seed": rep, "problem_kind": PROBLEM_KIND}
            try:
                result, _trace = await run(
                    model, n=n, difficulty=difficulty, seed=rep, budget=budget,
                    problem_kind=PROBLEM_KIND, obfuscate=OBFUSCATE,
                    relabel_seed=n * 131 + difficulty * 17 + rep, max_turns=MAX_TURNS,
                    sequential=SEQUENTIAL)
                row = {**result, "elapsed_s": round(time.time() - t0, 2)}
            except Exception as e:
                row = {**cfg, "error": f"{type(e).__name__}: {e}",
                       "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                sub = row.get("submit")
                print(f"  {model[:22]:22s} n={n:<2} d={difficulty} bud={budget:<5} "
                      f"rep={rep} calls={len(row.get('tool_calls') or [])} "
                      f"submit={'?' if sub is None else ('OK' if sub.get('correct') else 'X')} "
                      f"reason={row.get('stopped_reason')} "
                      f"{'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(*c) for c in cells))
    print(f"\nDone. Rollouts at: {out_path}\n"
          f"Score with: python scripts/oracle_counterfactual.py {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
