"""Coupon-collector WoodWorld region sweep over the (T, N) plane.

Sweeps T = tree-resource types (2..8, x-axis) x N = number of distinct wood kinds
(2..20, y-axis), 1 rep/cell = 133 episodes, Haiku, nohint. Budget per cell =
round(1.2 * N * H_N) (1.2x the expected coupon-collector grind). Runs to solve
(stop_on_build=False) so we can map recognition (P built | held sticks) and solve rate.

Run from the REPO ROOT so lomekwi.*/scripts.* resolve:
  PYTHONPATH=. python tool-wood-discrepancy/run_coupon_sweep.py --max-cost 10
  PYTHONPATH=. python tool-wood-discrepancy/run_coupon_sweep.py --smoke
  COUPON_RESUME=tool-wood-discrepancy/runs/<dir> PYTHONPATH=. python tool-wood-discrepancy/run_coupon_sweep.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# sibling imports (this folder has a hyphen -> not an importable package)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coupon_woodworld as cw                      # noqa: E402
import budget_coupon as bc                         # noqa: E402

from dotenv import load_dotenv                     # noqa: E402
import lomekwi.obfuscation as _obf                 # noqa: E402

_obf.DEFAULT_SCHEME = "letter"                      # single-letter wood/axe tokens

MODEL = "claude-haiku-4-5-20251001"
T_LO, T_HI = 2, 8                                   # tree-resource types (x-axis)
N_LO, N_HI = 2, 20                                  # distinct wood kinds (y-axis)
REPS = 1
CONC = 12

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
    ap.add_argument("--t-values", default=None,
                    help="comma-separated T values (default 2..8)")
    ap.add_argument("--n-values", default=None,
                    help="comma-separated N values (default 2..20)")
    ap.add_argument("--budget-mult", type=float, default=1.2,
                    help="budget = round(mult * N * H_N); 1.2 = ToolWorld slack regime")
    ap.add_argument("--recipe", choices=["same", "distinct"], default="same",
                    help="axe recipe: 'same' = stick+stick (2 of one look-alike); "
                         "'distinct' = two distinct resource types (ToolWorld-shaped)")
    ap.add_argument("--max-cost", type=float, default=None,
                    help="kill switch: stop launching new episodes once cumulative "
                         "logged cost (USD) reaches this (in-flight ones finish).")
    ap.add_argument("--conc", type=int, default=CONC)
    ap.add_argument("--smoke", action="store_true",
                    help="one expensive cell (T=8, N=20)")
    return ap.parse_args()


async def main():
    load_dotenv()
    args = parse_args()
    model = args.model
    short = model.split("-")[1] if model.startswith("claude-") else model

    t_values = ([int(x) for x in args.t_values.split(",") if x.strip()]
                if args.t_values else list(range(T_LO, T_HI + 1)))
    n_values = ([int(x) for x in args.n_values.split(",") if x.strip()]
                if args.n_values else list(range(N_LO, N_HI + 1)))
    t_lo, t_hi = min(t_values), max(t_values)
    n_lo, n_hi = min(n_values), max(n_values)

    resume = os.environ.get("COUPON_RESUME")
    if resume:
        out_dir = Path(resume)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        mtag = "" if abs(args.budget_mult - 1.2) < 1e-9 else f"mult{args.budget_mult:g}_"
        rtag = "" if args.recipe == "same" else f"{args.recipe}_"
        tag = "smoke" if args.smoke else f"{rtag}{mtag}r{args.reps}_nohint_T{t_lo}-{t_hi}_N{n_lo}-{n_hi}"
        out_dir = Path(__file__).resolve().parent / "runs" / f"{short}_coupon_{tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    out_path.touch()

    points = [(t, n) for t in t_values for n in n_values]
    reps = range(args.reps)
    if args.smoke:
        points, reps = [(max(t_values), max(n_values))], range(1)
    (out_dir / "points.json").write_text(json.dumps(
        {"t_values": t_values, "n_values": n_values,
         "t_range": [t_lo, t_hi], "n_range": [n_lo, n_hi],
         "reps": args.reps, "model": model, "budget_mult": args.budget_mult,
         "task": "coupon_woodworld"}, indent=2))

    done = set()
    rows: list[dict] = []
    for line in out_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append(r)
        if not r.get("error"):
            done.add((r["n_types"], r["n"], r["relabel_seed"]))

    cells = [(t, n, rep) for (t, n) in points for rep in reps
             if (t, n, rep) not in done]
    print(f"Writing to {out_path}\n"
          f"Sweep: {short} ({model}) coupon-collector over {len(points)} (T,N) cells x "
          f"{len(reps)} rep = {len(points) * len(reps)} episodes "
          f"(budget=round({args.budget_mult}*N*H_N), nohint, run-to-solve, conc={args.conc})\n"
          f"resume: {len(done)} done, {len(cells)} remaining", flush=True)

    sem = asyncio.Semaphore(args.conc)
    lock = asyncio.Lock()
    price = PRICING.get(short)
    cap = args.max_cost
    state = {"spent": 0.0, "capped": False}

    async def one(t, n, rep):
        if cap is not None and state["capped"]:
            return
        budget = bc.budget_for_coupon(n, mult=args.budget_mult)
        max_turns = max(300, round(budget * 1.5))
        async with sem:
            if cap is not None and state["capped"]:
                return
            t0 = time.time()
            try:
                # no_progress_window=None: the coupon tail is legitimately "no new
                # progress" (duplicate gathers waiting for the last kind); budget binds.
                result, trace = await cw.run(
                    model, n_kinds=n, n_types=t, relabel_seed=rep, draw_seed=rep,
                    hint=False, budget=budget, max_turns=max_turns,
                    no_progress_window=None, stop_on_build=False,
                    recipe_mode=args.recipe)
                row = {"model": model, "n_types": t, "n": n, "n_kinds": n,
                       "recipe_mode": args.recipe, "hint": False, "budget": budget,
                       "relabel_seed": rep, "draw_seed": rep,
                       "labels": result["labels"],
                       "actions": [x["action"] for x in trace],
                       "agent_texts": [x["agent_text"] for x in trace],
                       "obs": [x["obs"] for x in trace],
                       "solved": result["solved"],
                       "distinct_held": result["distinct_held"],
                       "built_axe": result["built_axe"],
                       "held_ingredients": result["held_ingredients"],
                       "build_turn": result["build_turn"],
                       "t_star": result["t_star"],
                       "use_axe_count": result["use_axe_count"],
                       "craft_attempts": result["craft_attempts"],
                       "total_actions": result["total_actions"],
                       "noop_total": result["noop_total"],
                       "usage": result["usage"],
                       "stopped_reason": result["stopped_reason"],
                       "elapsed_s": round(time.time() - t0, 2)}
            except Exception as e:
                row = {"model": model, "n_types": t, "n": n, "n_kinds": n,
                       "hint": False, "budget": budget,
                       "relabel_seed": rep, "draw_seed": rep,
                       "error": f"{type(e).__name__}: {e}",
                       "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                rows.append(row)
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                if price and row.get("usage"):
                    state["spent"] += episode_cost(row["usage"], price)
                cost_note = f"${state['spent']:.2f}" if cap is not None else ""
                print(f"  T={t} N={n:<2} b={budget:<3} rep={rep} "
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
    print(f"\nNext: PYTHONPATH=. python tool-wood-discrepancy/plot_coupon_heatmaps.py {out_dir}")


def summarize(rows: list[dict]) -> None:
    ok = [r for r in rows if not r.get("error")]
    if not ok:
        print("no episodes")
        return
    solved = [r for r in ok if r.get("solved")]
    built = [r for r in ok if r.get("built_axe")]
    held = [r for r in ok if r.get("held_ingredients")]
    recog = [r for r in held if r.get("built_axe")]
    reasons = Counter(r.get("stopped_reason") for r in ok)
    print(f"\n{len(ok)} episodes ({len(rows) - len(ok)} errored) | "
          f"solve rate {len(solved)/len(ok):.2f} | build rate {len(built)/len(ok):.2f} | "
          f"recognition P(built|held sticks) {len(recog)/max(1,len(held)):.2f} "
          f"({len(recog)}/{len(held)})")
    print(f"stopped_reason tally: {dict(reasons)}")
    mt = [r for r in ok if r.get("stopped_reason") == "max_turns"]
    if mt:
        print(f"  !! WARNING: {len(mt)} episodes hit max_turns (budget should bind first).")


if __name__ == "__main__":
    asyncio.run(main())
