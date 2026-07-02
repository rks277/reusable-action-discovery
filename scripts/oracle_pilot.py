"""Difficulty pilot for the oracle environment: run each model with TOOLS OFF (n=0,
only submit_answer) across a difficulty range, and record manual accuracy + the token
cost S at a correct submission. This calibrates (a) the difficulty band where a manual
solve is still reliable (so a manual win is real, not a coin flip) and (b) the manual
token-cost curve S(difficulty) used to set budgets in oracle_sweep_config.budget_for.

Budget is set huge so the live meter never truncates -- we want the TRUE manual cost.

Output: runs/oracle/oracle_pilot_<ts>/episodes.jsonl (one row per episode; schema is the
run() result so the $ watchdog's cost_of works). Prints a per-(model, difficulty) table
of accuracy + median S, and suggested budget_for values.

CLI:  python scripts/oracle_pilot.py [--model <substr>]
Under the cost watchdog:
  python scripts/_beach_sweep_capped.py 15 'runs/oracle/oracle_pilot_*' scripts.oracle_pilot [--model haiku]
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_oracle_llm import run                              # noqa: E402
from scripts.oracle_sweep_config import MODELS, QWEN3_MODELS, CONCURRENCY  # noqa: E402

DIFFICULTY_VALUES = [8, 11, 14, 17, 20]     # dividend digit counts to probe
REPS = 4
BUDGET = 200_000                            # huge -> never truncate; measure true cost
BUDGET_MULT = 0.8                           # suggested budget = MULT * median manual S


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
    out_dir = Path("runs") / "oracle" / f"oracle_pilot_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    out_path.write_text("")

    cells = [(prov, model, d, rep)
             for prov, model in roster
             for d in DIFFICULTY_VALUES
             for rep in range(REPS)]
    print(f"Writing to {out_path}\n"
          f"pilot (TOOLS OFF): {[m for _, m in roster]} x difficulty in "
          f"{DIFFICULTY_VALUES} x {REPS} reps = {len(cells)} episodes", flush=True)

    sems = {p: asyncio.Semaphore(CONCURRENCY.get(p, 4)) for p, _ in roster}
    lock = asyncio.Lock()

    async def one(prov, model, d, rep):
        async with sems[prov]:
            t0 = time.time()
            try:
                result, _ = await run(model, n=0, difficulty=d, seed=rep,
                                      budget=BUDGET, max_turns=40)
                row = {**result, "elapsed_s": round(time.time() - t0, 2)}
            except Exception as e:
                row = {"model": model, "difficulty": d, "seed": rep,
                       "error": f"{type(e).__name__}: {e}",
                       "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                sub = row.get("submit")
                print(f"  {model[:22]:22s} d={d:<2} rep={rep} "
                      f"correct={'?' if sub is None else sub.get('correct')} "
                      f"S={None if sub is None else sub.get('meter_raw')} "
                      f"{'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(*c) for c in cells))
    _report(out_path)


def _report(out_path: Path):
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    # (model, difficulty) -> list of (correct, S)
    cells = defaultdict(list)
    for r in rows:
        sub = r.get("submit")
        correct = bool(sub and sub.get("correct"))
        S = sub.get("meter_raw") if sub else None
        cells[(r["model"], r["difficulty"])].append((correct, S))

    print(f"\n=== pilot results ===\n"
          f"{'model':28s} {'diff':>4s} {'acc':>5s} {'n':>3s} {'medS(correct)':>13s} "
          f"{'suggest_budget':>14s}")
    for (model, d) in sorted(cells):
        vals = cells[(model, d)]
        acc = sum(c for c, _ in vals) / len(vals)
        S_ok = [s for c, s in vals if c and s is not None]
        medS = statistics.median(S_ok) if S_ok else None
        sugg = int(round(BUDGET_MULT * medS)) if medS else None
        print(f"{str(model)[:28]:28s} {d:>4d} {acc:>5.2f} {len(vals):>3d} "
              f"{('' if medS is None else f'{medS:.0f}'):>13s} "
              f"{('' if sugg is None else str(sugg)):>14s}")
    print(f"\nsuggested budget = {BUDGET_MULT} * median manual S among CORRECT solves.\n"
          f"Pick the largest difficulty with acc still high (manual win is real) as the "
          f"sweep's hard end; paste medS values into oracle_sweep_config._manual_cost.")


if __name__ == "__main__":
    asyncio.run(main())
