"""Woodworld build/solve-rate region sweep over the (p, N) plane.

Mirrors scripts/run_haiku_region_sweep.py, but for woodworld and over
(gather probability p, target wood N) instead of (byproduct types T, doors N):
  - VARIES (p, n) per cell across the rectangle p in {0.2..1.0}, N in [2, 20];
  - budget is the grind-calibrated cfg budget_for(n, p) = round(1.2 * n / p);
  - HINT OFF (pure discovery);
  - RUNS TO SOLVE (stop_on_build=False) -> episode ends on solved / out_of_budget
    / no_progress. We then map build rate (P built) and solve rate (P solved)
    over the plane with a Gaussian-pooled heatmap (plot_woodworld_region_heatmap.py).

1 rep/cell by default (the KDE pooling smooths single 0/1 draws, the toolworld
convention). p starts at 0.2 so the worst-case budget is 1.2*20/0.2 = 120 actions.

The no_progress safeguard SCALES with 1/p: at low p an honest gatherer hits long
unlucky failure runs, so a fixed window would falsely abort it. window = round(12/p)
gives (1-p)^(12/p) ~ e^-12 false-abort probability while still catching craft/use
flailing. budget always binds before the window where window > budget (small cells).

Usage: PYTHONPATH=. python -m scripts.run_woodworld_region_sweep --reps 1 --max-cost 15
       PYTHONPATH=. python -m scripts.run_woodworld_region_sweep --smoke
       WOODWORLD_REGION_RESUME=runs/<dir> PYTHONPATH=. python -m scripts.run_woodworld_region_sweep
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import scripts.woodworld_config as cfg          # sets DEFAULT_SCHEME='letter'
from scripts.validate_woodworld import budget_for
from scripts.woodworld import run

MODEL = "claude-haiku-4-5-20251001"
N_LO, N_HI = 2, 20                                # target wood
P_VALUES = [round(0.2 + 0.1 * i, 1) for i in range(9)]   # 0.2 .. 1.0
REPS = 1
CONC = 12
NO_PROG_K = 12                                    # window = round(NO_PROG_K / p)

PRICING = {  # $/1M tokens: (input, output, cache_write, cache_read)
    "haiku":  (1.0,  5.0,  1.25, 0.10),
    "sonnet": (3.0, 15.0,  3.75, 0.30),
    "opus":   (5.0, 25.0,  6.25, 0.50),
}


def episode_cost(usage: dict, price) -> float:
    pin, pout, pcw, pcr = price
    return (usage.get("input_tokens", 0) * pin
            + usage.get("output_tokens", 0) * pout
            + usage.get("cache_write_tokens", 0) * pcw
            + usage.get("cache_read_tokens", 0) * pcr) / 1e6


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--hint", action="store_true", help="enable the subtle hint")
    ap.add_argument("--max-cost", type=float, default=None,
                    help="kill switch: stop launching new episodes once cumulative "
                         "logged cost (USD) reaches this (in-flight ones finish).")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


async def main():
    load_dotenv()
    args = parse_args()
    model = args.model
    short = model.split("-")[1] if model.startswith("claude-") else model

    resume = os.environ.get("WOODWORLD_REGION_RESUME")
    if resume:
        out_dir = Path(resume)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        h = "hint" if args.hint else "nohint"
        tag = "smoke" if args.smoke else f"r{args.reps}_{h}_N{N_LO}-{N_HI}"
        out_dir = Path("runs") / f"{short}_woodworld_region_{tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    out_path.touch()

    points = [(p, n) for p in P_VALUES for n in range(N_LO, N_HI + 1)]
    reps = range(args.reps)
    if args.smoke:
        points, reps = [(0.2, 20)], range(1)     # one expensive-ish low-p cell
    (out_dir / "points.json").write_text(json.dumps(
        {"p_values": P_VALUES, "n_range": [N_LO, N_HI], "reps": args.reps,
         "hint": args.hint, "model": model}, indent=2))

    # resume: skip already-completed (p, n, rep)
    done = set()
    rows: list[dict] = []
    for line in out_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append(r)
        if not r.get("error"):
            done.add((r["gather_prob"], r["n"], r["relabel_seed"]))

    cells = [(p, n, rep) for (p, n) in points for rep in reps
             if (p, n, rep) not in done]
    print(f"Writing to {out_path}\n"
          f"Sweep: {short} ({model}) HINT {'ON' if args.hint else 'OFF'} over {len(points)} (p,N) cells x "
          f"{len(reps)} rep = {len(points) * len(reps)} episodes "
          f"(budget=budget_for(n,p), run-to-solve, conc={CONC})\n"
          f"resume: {len(done)} done, {len(cells)} remaining", flush=True)

    sem = asyncio.Semaphore(CONC)
    lock = asyncio.Lock()
    price = PRICING.get(short)
    cap = args.max_cost
    state = {"spent": 0.0, "capped": False}

    async def one(p, n, rep):
        if cap is not None and state["capped"]:
            return
        budget = budget_for(n, p)
        window = round(NO_PROG_K / p)
        max_turns = max(cfg.MAX_TURNS, round(budget * 1.5))
        async with sem:
            if cap is not None and state["capped"]:
                return
            t0 = time.time()
            try:
                result, trace = await run(model, n=n, relabel_seed=rep, gather_seed=rep,
                                          hint=args.hint, gather_prob=p, budget=budget,
                                          max_turns=max_turns, stop_on_build=False,
                                          no_progress_window=window)
                row = {"model": model, "n": n, "gather_prob": p, "hint": args.hint,
                       "budget": budget, "relabel_seed": rep, "gather_seed": rep,
                       "labels": result["labels"],
                       "actions": [x["action"] for x in trace],
                       "agent_texts": [x["agent_text"] for x in trace],
                       "obs": [x["obs"] for x in trace],
                       "solved": result["solved"],
                       "built_axe": result["built_axe"],
                       "build_turn": result["build_turn"],
                       "t_star": result["t_star"],
                       "use_axe_count": result["use_axe_count"],
                       "wood_by_source": result["wood_by_source"],
                       "total_actions": result["total_actions"],
                       "final_wood": result["final_wood"],
                       "action_efficiency": result["action_efficiency"],
                       "noop_total": result["noop_total"],
                       "usage": result["usage"],
                       "stopped_reason": result["stopped_reason"],
                       "elapsed_s": round(time.time() - t0, 2)}
            except Exception as e:
                row = {"model": model, "n": n, "gather_prob": p, "hint": args.hint,
                       "budget": budget, "relabel_seed": rep, "gather_seed": rep,
                       "error": f"{type(e).__name__}: {e}",
                       "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                rows.append(row)
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                if price and row.get("usage"):
                    state["spent"] += episode_cost(row["usage"], price)
                cost_note = f"${state['spent']:.2f}" if cap is not None else ""
                print(f"  p={p} N={n:<2} b={budget:<3} rep={rep} "
                      f"solved={row.get('solved')} built={row.get('built_axe')} "
                      f"act={row.get('total_actions')} stop={row.get('stopped_reason')} "
                      f"{cost_note} {'(err)' if row.get('error') else ''}", flush=True)
                if cap is not None and state["spent"] >= cap and not state["capped"]:
                    state["capped"] = True
                    print(f"  !! COST CAP HIT: ${state['spent']:.2f} >= ${cap:.2f}; "
                          f"no new episodes launch (in-flight finish).", flush=True)

    await asyncio.gather(*(one(*c) for c in cells))
    print(f"\nDone. Episodes at: {out_path}")
    summarize(rows)
    print(f"\nNext: python -m scripts.plot_woodworld_region_heatmap {out_dir}")


def summarize(rows: list[dict]) -> None:
    ok = [r for r in rows if not r.get("error")]
    solved = [r for r in ok if r.get("solved")]
    built = [r for r in ok if r.get("built_axe")]
    reasons = Counter(r.get("stopped_reason") for r in ok)
    print(f"\n{len(ok)} episodes ({len(rows) - len(ok)} errored) | "
          f"solve rate {len(solved)/len(ok):.2f} | build rate {len(built)/len(ok):.2f}"
          if ok else "no episodes")
    print(f"stopped_reason tally: {dict(reasons)}")
    bad = [r for r in ok if r.get("stopped_reason") == "no_progress" and r.get("built_axe")]
    if bad:
        print(f"  !! WARNING: {len(bad)} no_progress aborts were builders -- raise NO_PROG_K.")
    mt = [r for r in ok if r.get("stopped_reason") == "max_turns"]
    if mt:
        print(f"  !! WARNING: {len(mt)} episodes hit max_turns (budget should bind first).")


if __name__ == "__main__":
    asyncio.run(main())
