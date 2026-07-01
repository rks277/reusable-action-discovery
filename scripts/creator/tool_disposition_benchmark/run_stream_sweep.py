"""Multi-seed sweep for the tool-amortization STREAM (the non-composable moderate 5+5 design).

Runs the SAME fixed family set over many seeds (seeds vary arrival order + instance sampling),
concurrently, and aggregates the decision readouts (mean_lateness, bait-rate = fraction of seeds
that build >=1 one-off, decision tallies, regret). Built for robustness: each seed is isolated in
its own try/except with a retry, results are appended to a combined jsonl as they finish (so a
crash/interrupt never loses completed seeds), and failed seeds are listed at the end for re-run.

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.run_stream_sweep \
      --model haiku --n-seeds 20 --concurrency 10 --a0-dir runs/a0_moderate_tuned
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.creator.tool_disposition_benchmark.driver import run_session
from scripts.creator.tool_disposition_benchmark.session_state import SessionState
from scripts.creator.tool_disposition_benchmark.stream_builder import StreamSpec, build_stream
from scripts.creator.tool_disposition_benchmark.skirental_scorer import score_run
from scripts.creator.tool_disposition_benchmark.run_stream_session import slots_to_problems, CLAUDE
from scripts.creator.tool_disposition_benchmark.family_kit import set_profile, profile

# The locked non-composable moderate 5+5 set (calibrated in runs/a0_moderate_tuned).
RECURRING = [("euclid_gcd_chain", 15), ("factorial_mod", 15), ("lcg", 15),
             ("int_div_sum", 15), ("count_inversions", 15)]
ONEOFFS = ["kaprekar_routine", "look_and_say", "luhn_sum", "continued_frac", "mod_pair_sum"]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--recurring", nargs="+", default=None,
                    help="recurring classes as 'family:size' (default = the Haiku 5+5 set)")
    ap.add_argument("--one-offs", nargs="+", default=None,
                    help="one-off families (default = the Haiku 5+5 set)")
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="explicit seeds (else 0..n-1)")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--magnitude", type=int, default=1)
    ap.add_argument("--one-off-magnitude", type=int, default=1)
    ap.add_argument("--arrival", default="random_oneoff_early")
    ap.add_argument("--a0-dir", default="runs/a0_moderate_tuned")
    ap.add_argument("--token-cap", type=int, default=200_000)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--retries", type=int, default=2, help="retry attempts per seed on failure")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    load_dotenv()
    set_profile(args.model)                      # per-model difficulty tuning of the family kit
    model = CLAUDE.get(args.model, args.model)
    recurring = ([(p.rsplit(":", 1)[0], int(p.rsplit(":", 1)[1])) for p in args.recurring]
                 if args.recurring else RECURRING)
    oneoffs = args.one_offs if args.one_offs else ONEOFFS
    seeds = args.seeds if args.seeds is not None else list(range(args.n_seeds))
    print(f"difficulty profile = {profile()}")
    tag = args.tag or f"stream_sweep_{args.model}_n{len(seeds)}"
    base = Path("runs") / tag
    base.mkdir(parents=True, exist_ok=True)
    combined = base / "sweep_results.jsonl"
    print(f"sweep: model={args.model} seeds={seeds} conc={args.concurrency} budget={args.budget} "
          f"arrival={args.arrival} -> {base}", flush=True)

    client = RawChat()
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async def one(seed: int) -> dict:
        async with sem:
            seed_dir = base / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            spec = StreamSpec(recurring=recurring, n_one_offs=len(oneoffs), one_offs=oneoffs,
                              magnitude=args.magnitude, one_off_magnitude=args.one_off_magnitude,
                              arrival=args.arrival, seed=seed)
            slots = build_stream(spec)
            problems = slots_to_problems(slots)
            (seed_dir / "stream.json").write_text(json.dumps(slots, indent=2))
            (seed_dir / "config.json").write_text(json.dumps(
                {"budget": args.budget, "recurring": [list(x) for x in recurring],
                 "one_offs": oneoffs, "seed": seed, "magnitude": args.magnitude}))

            last_err = None
            for attempt in range(args.retries + 1):
                t0 = time.time()
                try:
                    state = SessionState(problems=problems, budget=args.budget)
                    row = await run_session(client, model, state, token_cap=args.token_cap,
                                            max_tokens=args.max_tokens)
                    row["model_key"] = args.model
                    (seed_dir / "sessions.jsonl").write_text(json.dumps(row) + "\n")
                    res = score_run(str(seed_dir), args.a0_dir, args.model, args.magnitude)
                    agg = res["aggregate"]
                    rec = {"seed": seed, "ok": True, "solve": row.get("n_correct"),
                           "N": row.get("N"), "scripts": row.get("n_scripts_written"),
                           "spent_tokens": row.get("spent_tokens"),
                           "mean_lateness": agg["mean_lateness"],
                           "regret": agg["total_regret_budget"],
                           "decision_counts": agg["decision_counts"],
                           "n_oneoffs_built": agg["n_oneoffs_built"],
                           "build_rate_recurring": agg["build_rate_recurring"],
                           "reuse_rate": agg["reuse_rate"], "rebuilds": agg["total_rebuilds"],
                           "wrongly_built": [c["family"] for c in res["classes"]
                                             if c["decision"] == "wrongly-built"],
                           "wrongly_skipped": [c["family"] for c in res["classes"]
                                               if c["decision"] == "wrongly-skipped"],
                           "elapsed_s": round(time.time() - t0, 1)}
                    break
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    traceback.print_exc()
                    print(f"[seed {seed}] attempt {attempt + 1} failed: {last_err}", flush=True)
            else:
                rec = {"seed": seed, "ok": False, "error": last_err}

            async with lock:
                with combined.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                if rec.get("ok"):
                    print(f"[seed {seed}] OK solve={rec['solve']}/{rec['N']} lateness={rec['mean_lateness']:.2f} "
                          f"oneoffs_built={rec['n_oneoffs_built']} regret={rec['regret']:.0f} "
                          f"wrong_built={rec['wrongly_built']} wrong_skip={rec['wrongly_skipped']}", flush=True)
                else:
                    print(f"[seed {seed}] FAIL {rec['error']}", flush=True)
            return rec

    results = await asyncio.gather(*(one(s) for s in seeds))

    ok = [r for r in results if r.get("ok")]
    fail = [r for r in results if not r.get("ok")]

    def ms(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return (float("nan"), float("nan"))
        return (statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0)

    print("\n================ SWEEP SUMMARY ================", flush=True)
    print(f"seeds ok: {len(ok)}/{len(seeds)}   failed: {[r['seed'] for r in fail]}")
    if ok:
        sm, ss = ms([r["solve"] for r in ok])
        lm, ls = ms([r["mean_lateness"] for r in ok])
        rm, rs = ms([r["regret"] for r in ok])
        bait = [1 if r["n_oneoffs_built"] > 0 else 0 for r in ok]
        nob_m, nob_s = ms([r["n_oneoffs_built"] for r in ok])
        wsk_m, wsk_s = ms([len(r["wrongly_skipped"]) for r in ok])
        p = statistics.mean(bait)
        se = (p * (1 - p) / len(bait)) ** 0.5
        print(f"solve/80:          mean={sm:.1f} sd={ss:.1f}")
        print(f"mean_lateness:     mean={lm:.2f} sd={ls:.2f}   (0 = builds on first sight)")
        print(f"regret vs opt:     mean={rm:.0f} sd={rs:.0f}")
        print(f"one-offs built:    mean={nob_m:.2f} sd={nob_s:.2f}   (0 = never took the bait)")
        print(f"recurring starved: mean={wsk_m:.2f} sd={wsk_s:.2f} classes/seed")
        print(f"BAIT RATE (>=1 one-off built): {p:.2f} +/- {se:.2f}  ({sum(bait)}/{len(bait)} seeds)")
    print(f"\nper-seed results -> {combined}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
