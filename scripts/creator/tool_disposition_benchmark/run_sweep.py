"""Concurrent runner for the tool-disposition benchmark. Each model (optionally x seeds) runs
ONE persistent session over the same N-problem dataset; concurrency is across SESSIONS (each
session is internally sequential). Streams one session record per line to sessions.jsonl.

  # local smoke: Haiku over a small dataset
  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.run_sweep \
      --models haiku --n 8 --sig-figs 6 --token-cap 120000 --concurrency 2

  # full sweep across the model ladder
  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.run_sweep \
      --models haiku sonnet opus --n 50 --sig-figs 6 --token-cap 400000 --concurrency 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.creator.tool_disposition_benchmark.dataset import load_or_build
from scripts.creator.tool_disposition_benchmark.driver import run_session
from scripts.creator.tool_disposition_benchmark.session_state import SessionState, write_budget

CLAUDE = {"haiku": "claude-haiku-4-5-20251001",
          "sonnet": "claude-sonnet-4-6", "opus": "claude-opus-4-8"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="claude aliases (haiku/sonnet/opus) or raw model ids")
    ap.add_argument("--n", type=int, default=30, help="number of distinct problems")
    ap.add_argument("--sig-figs", type=int, default=6)
    ap.add_argument("--magnitude", type=float, default=1.0,
                    help="input-size scale (calibrate with calibrate.py; <1 = easier by hand)")
    ap.add_argument("--seed", type=int, default=0, help="dataset seed")
    ap.add_argument("--shuffle", action="store_true",
                    help="random sample of the feasible pool instead of first-N-in-order")
    ap.add_argument("--reps", type=int, default=1, help="sessions per model (varies nothing but "
                    "the sampling of the model; same dataset)")
    ap.add_argument("--token-cap", type=int, default=200_000, help="cumulative tokens per session")
    ap.add_argument("--no-token-cap", action="store_true",
                    help="'no-cap' arm: hide the budget from the model (no budget paragraph, no "
                    "tokens_remaining); --safety-cap is still enforced silently to bound cost")
    ap.add_argument("--safety-cap", type=int, default=600_000,
                    help="silent hard ceiling used when --no-token-cap is set")
    ap.add_argument("--max-tokens", type=int, default=2048, help="per-call output cap")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    load_dotenv()
    models = [CLAUDE.get(m, m) for m in args.models]
    problems = load_or_build(args.n, args.sig_figs, args.seed, args.magnitude, args.shuffle)
    budget = write_budget(len(problems))
    announce_cap = not args.no_token_cap
    cap = args.safety_cap if args.no_token_cap else args.token_cap
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"tool_disposition_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sessions.jsonl"
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    total = len(models) * args.reps
    cap_desc = f"SILENT safety ceiling={cap:,} (no announced cap)" if args.no_token_cap else f"cap={cap:,}"
    print(f"{len(problems)} problems (D={args.sig_figs}, budget={budget}) x {len(models)} models "
          f"x {args.reps} reps -> {out_path} ({total} sessions, {cap_desc})", flush=True)

    client = RawChat()
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    done = 0

    async def one(model: str, rep: int):
        nonlocal done
        async with sem:
            t0 = time.time()
            try:
                state = SessionState(problems=problems, budget=budget)
                row = await run_session(client, model, state, token_cap=cap,
                                        max_tokens=args.max_tokens, announce_cap=announce_cap)
                row["rep"] = rep
            except Exception as e:
                row = {"model": model, "rep": rep, "N": len(problems),
                       "error": f"{type(e).__name__}: {e}", "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                done += 1
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                lbl = model.split("/")[-1]
                print(f"  {done}/{total}  {lbl} rep{rep}: "
                      f"solve={row.get('n_correct')}/{row.get('N')} "
                      f"scripts={row.get('n_scripts_written')} "
                      f"persist={row.get('persistence')} reuse={row.get('reusability')} "
                      f"tok={row.get('spent_tokens')} "
                      f"{'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(m, r) for m in models for r in range(args.reps)))
    print(f"done -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
