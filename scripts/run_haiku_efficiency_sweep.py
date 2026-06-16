"""Haiku tool-EFFICIENCY region sweep: once the tool is built and handed to the
agent, how cleanly does it exploit it to finish? This is the downstream complement
to the recognition sweep (run_haiku_region_sweep.py), which asked whether the agent
builds at all. Here we hold recognition constant by FORCING the build.

For each recorded run in a completed region sweep (the source square), we:
  1. find the first turn `ab` where the agent held BOTH recipe ingredients (via the
     "hold N TYPE(s)" obs regex, == analyze_recognition_latency.acquire_both_turn);
     runs that never held both are OMITTED;
  2. replay actions[0..ab] through a fresh world (world_from_labels -> scheme-
     independent) as the agent's CONTEXT, using its recorded reasoning text;
  3. inject ONE artificial forced action -- the correct combine of the two recipe
     types -- which builds the machine;
  4. hand control to the LIVE agent and continue until solved or out of budget;
  5. count REDUNDANT continuation actions = live actions that make no progress,
     where progress = opening a door OR operating the machine on a door.

The action budget = cfg.budget_for(n) and caps the ENTIRE run, including the
replayed context and the forced combine (live remaining = budget - (ab + 2)).
Runs whose context already meets the budget (ab + 2 >= budget) are DEGENERATE: no
live budget remains, so they are recorded directly without an API call.

Usage:
  PYTHONPATH=. python -m scripts.run_haiku_efficiency_sweep
  PYTHONPATH=. python -m scripts.run_haiku_efficiency_sweep --max-cost 12
  PYTHONPATH=. python -m scripts.run_haiku_efficiency_sweep --smoke   # 2 cells
  HAIKU_EFF_RESUME=runs/<dir> PYTHONPATH=. python -m scripts.run_haiku_efficiency_sweep
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

from scripts.toolworld_v2 import run, world_from_labels
from scripts.analyze_recognition_latency import acquire_both_turn
from scripts.run_haiku_region_sweep import (PRICING, episode_cost, max_turns_for,
                                            NO_PROGRESS_WINDOW)  # noqa: F401
from scripts import sweep_config as cfg

MODEL = "claude-haiku-4-5-20251001"
SOURCE = "runs/haiku_region_sweep_p100_r1_Nle20_20260615_172029"
CONC = 12


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--source", default=SOURCE,
                   help="completed region sweep dir (or its episodes.jsonl) to fork from")
    p.add_argument("--max-cost", type=float, default=None,
                   help="kill switch: once cumulative logged cost (USD) reaches this, "
                        "stop launching new episodes (in-flight ones finish).")
    p.add_argument("--smoke", action="store_true",
                   help="run only ~2 cells (one ordinary + the degenerate, if present)")
    return p.parse_args()


def load_source(source: str) -> list[dict]:
    path = Path(source)
    if path.is_dir():
        path = path / "episodes.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [r for r in rows if not r.get("error")]


async def main():
    load_dotenv()
    args = parse_args()
    model = args.model
    short = model.split("-")[1] if model.startswith("claude-") else model

    src_rows = load_source(args.source)

    resume = os.environ.get("HAIKU_EFF_RESUME")
    if resume:
        out_dir = Path(resume)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = "smoke" if args.smoke else "Nle20"
        out_dir = Path("runs") / f"{short}_efficiency_sweep_{tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    out_path.touch()
    (out_dir / "meta.json").write_text(json.dumps(
        {"model": model, "source": args.source, "conc": CONC,
         "redundant_def": "live action that neither opens a door nor operates the "
                          "machine on a door", "budget": "cfg.budget_for(n), caps the "
                          "whole run incl. context + forced combine"}, indent=2))

    # build the work list: one (held-both) run per source row
    jobs = []
    skipped_no_both = 0
    for r in src_rows:
        recipe = r["labels"]["recipe"]
        ab = acquire_both_turn(r.get("obs"), set(recipe))
        if ab is None:
            skipped_no_both += 1
            continue
        jobs.append((r, ab))
    if args.smoke:
        degen = [(r, ab) for (r, ab) in jobs
                 if ab + 2 >= cfg.budget_for(r["n"])]
        ordinary = [(r, ab) for (r, ab) in jobs
                    if ab + 2 < cfg.budget_for(r["n"])]
        jobs = ordinary[:1] + degen[:1]

    # resume: skip cells already in episodes.jsonl
    done = set()
    rows: list[dict] = []
    for line in out_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
        done.add((row["n_types"], row["n"], row["relabel_seed"]))
    jobs = [(r, ab) for (r, ab) in jobs
            if (r["n_types"], r["n"], r["relabel_seed"]) not in done]

    print(f"Writing to {out_path}\n"
          f"Efficiency sweep: {short} ({model}) | source {args.source}\n"
          f"  {len(src_rows)} source episodes | {skipped_no_both} omitted (never held both) "
          f"| {len(jobs) + len(done)} held-both runs | {len(done)} done, {len(jobs)} to run\n"
          f"  (forced winning combine; redundant = live action not opening a door / "
          f"using the machine; conc={CONC})", flush=True)

    sem = asyncio.Semaphore(CONC)
    lock = asyncio.Lock()
    price = PRICING.get(short)
    cap = args.max_cost
    state = {"spent": 0.0, "capped": False}

    async def one(r: dict, ab: int):
        n, t, rep = r["n"], r["n_types"], r["relabel_seed"]
        budget = cfg.budget_for(n)
        recipe = r["labels"]["recipe"]
        forced_combine = ["combine", recipe[0], recipe[1]]
        base = {"model": model, "source_run": str(args.source), "n": n, "n_types": t,
                "hint": r.get("hint", cfg.HINT), "budget": budget,
                "relabel_seed": rep, "drop_seed": r.get("drop_seed", rep),
                "labels": r["labels"], "acquire_both_turn": ab,
                "len_prefix": ab + 2, "live_remaining": budget - (ab + 2),
                "forced_combine": forced_combine}

        # degenerate: no live budget remains once context + forced combine are spent
        if ab + 2 >= budget:
            row = {**base, "degenerate": True, "redundant_continuation": 0,
                   "live_actions": 0, "live_redundant_flags": [],
                   "live_opened_delta": 0, "live_solved": False,
                   "continuation_actions": [], "continuation_obs": [],
                   "prefix_obs_verified": None,
                   "stopped_reason": "no_live_budget",
                   "usage": {k: 0 for k in ("input_tokens", "output_tokens",
                             "cache_read_tokens", "cache_write_tokens",
                             "reasoning_tokens", "calls")},
                   "elapsed_s": 0.0, "error": None}
            await record(row)
            return

        if cap is not None and state["capped"]:
            return
        async with sem:
            if cap is not None and state["capped"]:
                return
            t0 = time.time()
            try:
                world = world_from_labels(r["labels"], n)
                # context = actions[0..ab] with the agent's recorded reasoning, then
                # the artificial winning combine.
                prefix = [(list(a), txt) for a, txt in
                          zip(r["actions"][:ab + 1], r["agent_texts"][:ab + 1])]
                prefix.append((forced_combine, "combine " + " ".join(recipe)))
                result, trace = await run(
                    model, n=n, n_types=t, relabel_seed=rep,
                    drop_seed=r.get("drop_seed", rep), hint=r.get("hint", cfg.HINT),
                    max_turns=max_turns_for(budget), budget=budget,
                    stop_on_build=False, no_progress_window=None,
                    world_override=world, forced_prefix=prefix)
                # verify the replayed context matched the recorded transcript byte-for-byte
                rec_obs = r["obs"][:ab + 1]
                got_obs = [x["obs"] for x in trace[:ab + 1]]
                verified = (got_obs == rec_obs)
                cont = trace[ab + 2:]   # live continuation (after context + forced combine)
                row = {**base, "degenerate": False,
                       "redundant_continuation": result["redundant_continuation"],
                       "live_actions": result["live_actions"],
                       "live_redundant_flags": result["live_redundant_flags"],
                       "live_opened_delta": result["live_opened_delta"],
                       "live_solved": result["solved"],
                       "continuation_actions": [x["action"] for x in cont],
                       "continuation_obs": [x["obs"] for x in cont],
                       "prefix_obs_verified": verified,
                       "stopped_reason": result["stopped_reason"],
                       "usage": result["usage"],
                       "elapsed_s": round(time.time() - t0, 2), "error": None}
            except Exception as e:
                row = {**base, "degenerate": False, "error": f"{type(e).__name__}: {e}",
                       "elapsed_s": round(time.time() - t0, 2)}
            await record(row)

    async def record(row: dict):
        async with lock:
            rows.append(row)
            with out_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if price and row.get("usage"):
                state["spent"] += episode_cost(row["usage"], price)
            cost_note = f"${state['spent']:.2f}" if cap is not None else ""
            print(f"  T={row['n_types']:<2} N={row['n']:<2} ab={row.get('acquire_both_turn')} "
                  f"redund={row.get('redundant_continuation')} "
                  f"live={row.get('live_actions')} solved={row.get('live_solved')} "
                  f"verif={row.get('prefix_obs_verified')} "
                  f"stop={row.get('stopped_reason')} {cost_note} "
                  f"{'(err)' if row.get('error') else ''}", flush=True)
            if cap is not None and state["spent"] >= cap and not state["capped"]:
                state["capped"] = True
                print(f"  !! COST CAP HIT: ${state['spent']:.2f} >= ${cap:.2f}. "
                      f"No new episodes will launch (in-flight ones finish).", flush=True)

    await asyncio.gather(*(one(r, ab) for (r, ab) in jobs))
    print(f"\nDone. Episodes at: {out_path}")
    summarize(rows)


def summarize(rows: list[dict]) -> None:
    ok = [r for r in rows if not r.get("error")]
    nerr = len(rows) - len(ok)
    ran = [r for r in ok if not r.get("degenerate")]
    degen = [r for r in ok if r.get("degenerate")]
    solved = [r for r in ran if r.get("live_solved")]
    bad_verif = [r for r in ran if r.get("prefix_obs_verified") is False]
    reasons = Counter(r.get("stopped_reason") for r in ok)
    rc = [r["redundant_continuation"] for r in ran]
    print(f"\n{len(ok)} runs ({nerr} errored) | {len(ran)} live, {len(degen)} degenerate "
          f"| live-solved {len(solved)}/{len(ran)}")
    if rc:
        rc_s = sorted(rc)
        print(f"redundant_continuation: mean {sum(rc)/len(rc):.2f}  "
              f"median {rc_s[len(rc_s)//2]}  max {max(rc)}  (n={len(rc)})")
    print(f"stopped_reason tally: {dict(reasons)}")
    if bad_verif:
        print(f"  !! WARNING: {len(bad_verif)} runs had prefix_obs_verified=False -- "
              f"context reconstruction did NOT match the recorded transcript.")


if __name__ == "__main__":
    asyncio.run(main())
