"""Playground-injection experiment on a single cell (default Opus, N=12, T=6).

Paired design: the SAME game worlds (relabel=drop=rep, rep in 0..reps-1) are played
under each arm, so the only thing that varies is the injected playground context.

Arms (--arms):
  none     : no playground context (baseline)
  explore  : inject a saw-EXPLORATION playground (a session that did NOT build)
  build    : inject a saw-BUILD playground (a session that built AND used the machine)

The injected text is action->observation pairs only (model-agnostic demonstration) from
a chosen stored session in playgrounds/T{pt}_n{pn}/pool.jsonl. The playground world is
relabeled (different recipe), so only the MECHANIC transfers, never the game's recipe.

Budget is the proper grind-calibrated cfg.budget_for(n); runs go to solve-or-out-of-budget.

Usage: PYTHONPATH=. python -m scripts.run_playground_experiment \
         --model claude-opus-4-8 --n 12 --n-types 6 --reps 10 \
         --pool playgrounds/T3_n8/pool.jsonl --explore-session 9 --build-session 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from scripts.toolworld_v2 import run
from scripts import sweep_config as cfg

NO_PROGRESS_WINDOW = 64
CONC = 8


def format_playground(sess: dict) -> str:
    """Render a stored playground session as injected 'prior experience' text:
    action -> observation pairs only (no reasoning), so it is model-agnostic."""
    lines = []
    for a, o in zip(sess["actions"], sess["obs"]):
        verb = " ".join(str(x) for x in a)
        lines.append(f"  > {verb}\n    {o}")
    body = "\n".join(lines)
    return (
        "## A practice session on a SIMILAR device (different parts, same kind of machine)\n"
        "Earlier you got to freely tinker with a comparable device to learn how it works. "
        "The parts had different names, but the device behaves the same way. Here is exactly "
        "what happened during that practice:\n\n"
        f"{body}\n\n"
        "(That was a different device; the labels below are different, but the underlying "
        "mechanics are identical.)")


def load_session(pool_path: Path, sid: int) -> dict:
    for line in pool_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["session"] == sid:
                return r
    raise SystemExit(f"session {sid} not found in {pool_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="claude-opus-4-8")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--n-types", type=int, default=6)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--pool", default="playgrounds/T3_n8/pool.jsonl")
    p.add_argument("--explore-session", type=int, default=9)
    p.add_argument("--build-session", type=int, default=4)
    p.add_argument("--arms", nargs="+", default=["none", "explore", "build"])
    return p.parse_args()


async def main():
    load_dotenv()
    a = parse_args()
    pool = Path(a.pool)
    explore_ctx = format_playground(load_session(pool, a.explore_session))
    build_ctx = format_playground(load_session(pool, a.build_session))
    ARM_CTX = {"none": None, "explore": explore_ctx, "build": build_ctx}

    short = a.model.split("-")[1] if a.model.startswith("claude-") else a.model
    budget = cfg.budget_for(a.n)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"{short}_playground_expt_T{a.n_types}_n{a.n}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    out_path.touch()
    (out_dir / "design.json").write_text(json.dumps(
        {"model": a.model, "n": a.n, "n_types": a.n_types, "reps": a.reps,
         "budget": budget, "pool": str(pool),
         "explore_session": a.explore_session, "build_session": a.build_session,
         "arms": a.arms}, indent=2))

    cells = [(arm, rep) for arm in a.arms for rep in range(a.reps)]
    print(f"{short} playground experiment | n={a.n} T={a.n_types} budget={budget} "
          f"| arms={a.arms} x {a.reps} reps = {len(cells)} episodes "
          f"| explore=s{a.explore_session} build=s{a.build_session}", flush=True)

    sem = asyncio.Semaphore(CONC)
    lock = asyncio.Lock()
    rows = []

    async def one(arm, rep):
        async with sem:
            t0 = time.time()
            try:
                result, trace = await run(
                    a.model, n=a.n, n_types=a.n_types, relabel_seed=rep, drop_seed=rep,
                    hint=cfg.HINT, max_turns=max(cfg.MAX_TURNS, round(budget * 1.5)),
                    budget=budget, stop_on_build=False,
                    no_progress_window=NO_PROGRESS_WINDOW,
                    playground_context=ARM_CTX[arm])
                row = {"arm": arm, "rep": rep, "model": a.model, "n": a.n,
                       "n_types": a.n_types, "budget": budget,
                       "relabel_seed": rep, "drop_seed": rep,
                       "labels": result["labels"],
                       "actions": [x["action"] for x in trace],
                       "obs": [x["obs"] for x in trace],
                       "agent_texts": [x["agent_text"] for x in trace],
                       "solved": result["solved"], "built_machine": result["built_machine"],
                       "build_turn": result["build_turn"],
                       "total_actions": result["total_actions"],
                       "usage": result["usage"], "stopped_reason": result["stopped_reason"],
                       "elapsed_s": round(time.time() - t0, 2)}
            except Exception as e:
                row = {"arm": arm, "rep": rep, "model": a.model, "n": a.n,
                       "n_types": a.n_types, "error": f"{type(e).__name__}: {e}",
                       "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                rows.append(row)
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  arm={arm:<8} rep={rep:<2} solved={row.get('solved')} "
                      f"built={row.get('built_machine')} actions={row.get('total_actions')} "
                      f"stop={row.get('stopped_reason')} {'(err)' if row.get('error') else ''}",
                      flush=True)

    await asyncio.gather(*(one(arm, rep) for arm, rep in cells))
    print(f"\nDone -> {out_path}")
    summarize(rows)
    print(f"RUN_DIR={out_dir}")


def summarize(rows):
    import re
    def collected(r):
        s = set()
        for o in r.get("obs", []):
            s.update(re.findall(r"and a (\w+)\.", o))
            s.update(re.findall(r"hold \d+ (\w+)\(s\)", o))
        return s
    by = defaultdict(list)
    for r in rows:
        if not r.get("error"):
            by[r["arm"]].append(r)
    print("\narm        n  solved  built  held_both  recognition(P built|held)")
    for arm in ["none", "explore", "build"]:
        rs = by.get(arm, [])
        if not rs:
            continue
        s = sum(bool(r["solved"]) for r in rs)
        b = sum(bool(r["built_machine"]) for r in rs)
        held = [r for r in rs if set(r["labels"]["recipe"]) <= collected(r)]
        rec = sum(bool(r["built_machine"]) for r in held) / len(held) if held else float("nan")
        print(f"  {arm:<8} {len(rs):>2}  {s:>5}  {b:>5}  {len(held):>8}  {rec:>10.2f}")


if __name__ == "__main__":
    asyncio.run(main())
