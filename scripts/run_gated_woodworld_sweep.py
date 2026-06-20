"""Gated WoodWorld region sweep over the (T, N) plane -- multi-layer recipe variant.

Default = the TWO-LAYER recipe (r0+r1 -> part, part+part -> axe; see scripts/gated_woodworld.py)
gated coupon-collector world. T = tree-resource types (2..6, x) x N = distinct wood kinds
(10..30, y), 1 rep/cell = 105 episodes, Haiku, nohint, run-to-solve. Budget per cell =
sweep_config.budget_for(N) = round(1.2*(N*H_N + N)) (== ToolWorld grind-calibrated; building
stays OPTIONAL).

Run from the REPO ROOT so lomekwi.*/scripts.* resolve:
  PYTHONPATH=. python scripts/run_gated_woodworld_sweep.py --max-cost 30
  PYTHONPATH=. python scripts/run_gated_woodworld_sweep.py --smoke
  GATED_WOODWORLD_RESUME=runs/<dir> PYTHONPATH=. python scripts/run_gated_woodworld_sweep.py
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

import scripts.gated_woodworld as gw
import scripts.sweep_config as cfg
from dotenv import load_dotenv

MODEL = "claude-haiku-4-5-20251001"
T_LO, T_HI = 2, 6
N_LO, N_HI = 10, 30
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
    ap.add_argument("--t-values", default=None, help="comma-separated T (default 2..6)")
    ap.add_argument("--n-values", default=None, help="comma-separated N (default 10..30)")
    ap.add_argument("--recipe", choices=["two_layer", "distinct", "same"],
                    default="two_layer", help="axe recipe depth (default two_layer)")
    ap.add_argument("--max-cost", type=float, default=None,
                    help="kill switch: stop launching once cumulative logged cost (USD) "
                         "reaches this (in-flight episodes finish).")
    ap.add_argument("--conc", type=int, default=CONC)
    ap.add_argument("--smoke", action="store_true", help="one expensive cell (T=6, N=30)")
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

    resume = os.environ.get("GATED_WOODWORLD_RESUME")
    if resume:
        out_dir = Path(resume)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = "smoke" if args.smoke else f"r{args.reps}_nohint_T{t_lo}-{t_hi}_N{n_lo}-{n_hi}"
        out_dir = Path("runs") / f"{short}_gatedwood_{args.recipe}_{tag}_{ts}"
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
         "reps": args.reps, "model": model, "recipe": args.recipe,
         "task": "gated_woodworld",
         "budgets": {str(n): cfg.budget_for(n) for n in n_values}}, indent=2))

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
          f"Sweep: {short} ({model}) GATED woodworld (recipe={args.recipe}) over "
          f"{len(points)} (T,N) cells x {len(reps)} rep = {len(points) * len(reps)} "
          f"episodes (budget=sweep_config.budget_for(N), nohint, run-to-solve, "
          f"conc={args.conc})\nresume: {len(done)} done, {len(cells)} remaining", flush=True)

    sem = asyncio.Semaphore(args.conc)
    lock = asyncio.Lock()
    price = PRICING.get(short)
    cap = args.max_cost
    state = {"spent": 0.0, "capped": False}

    async def one(t, n, rep):
        if cap is not None and state["capped"]:
            return
        budget = cfg.budget_for(n)
        max_turns = max(400, round(budget * 1.5))
        async with sem:
            if cap is not None and state["capped"]:
                return
            t0 = time.time()
            try:
                result, trace = await gw.run(
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
                       "built_part": result["built_part"],
                       "built_axe": result["built_axe"],
                       "held_base": result["held_base"],
                       "held_two_parts": result["held_two_parts"],
                       "part_turn": result["part_turn"],
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
                print(f"  T={t} N={n:<2} b={budget:<4} rep={rep} "
                      f"solved={row.get('solved')} part={row.get('built_part')} "
                      f"axe={row.get('built_axe')} act={row.get('total_actions')} "
                      f"stop={row.get('stopped_reason')} {cost_note} "
                      f"{'(err)' if row.get('error') else ''}", flush=True)
                if cap is not None and state["spent"] >= cap and not state["capped"]:
                    state["capped"] = True
                    print(f"  !! COST CAP HIT: ${state['spent']:.2f} >= ${cap:.2f}; "
                          f"no new episodes launch (in-flight finish).", flush=True)

    await asyncio.gather(*(one(*c) for c in cells))
    print(f"\nDone. Episodes at: {out_path}")
    summarize(rows)
    print(f"\nNext: PYTHONPATH=. python scripts/plot_gated_woodworld_heatmap.py {out_dir}")


def summarize(rows: list[dict]) -> None:
    ok = [r for r in rows if not r.get("error")]
    if not ok:
        print("no episodes")
        return
    solved = [r for r in ok if r.get("solved")]
    part = [r for r in ok if r.get("built_part")]
    axe = [r for r in ok if r.get("built_axe")]
    base = [r for r in ok if r.get("held_base")]
    recog = [r for r in base if r.get("built_axe")]              # P(axe | held_base)
    step1 = [r for r in base if r.get("built_part")]             # P(part | held_base)
    step2_den = [r for r in ok if r.get("built_part")]
    step2 = [r for r in step2_den if r.get("built_axe")]         # P(axe | part)
    used = [r for r in axe if (r.get("use_axe_count") or 0) > 0]
    reasons = Counter(r.get("stopped_reason") for r in ok)
    print(f"\n{len(ok)} episodes ({len(rows) - len(ok)} errored) | "
          f"solve {len(solved)/len(ok):.2f} | built_part {len(part)/len(ok):.2f} | "
          f"built_axe {len(axe)/len(ok):.2f}")
    print(f"  RECOGNITION P(built_axe | held_base) = {len(recog)/max(1,len(base)):.2f} "
          f"({len(recog)}/{len(base)})")
    print(f"  step1 P(built_part | held_base) = {len(step1)/max(1,len(base)):.2f} "
          f"({len(step1)}/{len(base)}) | "
          f"step2 P(built_axe | built_part) = {len(step2)/max(1,len(step2_den)):.2f} "
          f"({len(step2)}/{len(step2_den)})")
    print(f"  of axe-builders, used-axe {len(used)}/{max(1,len(axe))}")
    print(f"  stopped_reason tally: {dict(reasons)}")
    mt = [r for r in ok if r.get("stopped_reason") == "max_turns"]
    if mt:
        print(f"  !! WARNING: {len(mt)} episodes hit max_turns (budget should bind first).")


if __name__ == "__main__":
    asyncio.run(main())
