"""Run a tool-amortization STREAM through the persistent-session harness, then score it.

Builds a stream (family_kit families with controlled reuse horizons / arrival / distractors via
stream_builder), presents it one problem at a time through the SAME persistent session used by
run_gsmhard_session (write_script / run_script / submit_answer, enforced write budget, token cap),
saves the transcript + hidden labels, and runs the ski-rental scorer.

Exact-integer grading: each problem's sig_figs = the gold's digit count, so correct_to_sigfigs
reduces to exact-integer match. At magnitude 10 all golds are < 2^53, so float grading stays exact.

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.run_stream_session \
      --models haiku --magnitude 10 --arrival spread --a0-dir runs/a0_oracle_gap_20260630_105027
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
from scripts.creator.tool_disposition_benchmark.driver import run_session
from scripts.creator.tool_disposition_benchmark.session_state import SessionState
from scripts.creator.tool_disposition_benchmark.stream_builder import StreamSpec, build_stream
from scripts.creator.tool_disposition_benchmark.skirental_scorer import score_run

CLAUDE = {"haiku": "claude-haiku-4-5-20251001",
          "sonnet": "claude-sonnet-4-6", "opus": "claude-opus-4-8"}

# default recurring classes (all 4 recurring families, size 5); one-offs from the distinct pool
DEFAULT_RECURRING = [("product3", 5), ("weighted_sum", 5), ("lcg", 5), ("modpow", 5)]


def slots_to_problems(slots: list[dict]) -> list[dict]:
    """Stream slots -> session problems (order preserved). sig_figs = gold digit count -> exact int."""
    probs = []
    for s in slots:
        g = int(s["gold"])
        probs.append({"idx": s["slot_index"], "item_idx": s["slot_index"], "keys": s["keys"],
                      "question": s["question"], "inputs": s["inputs"],
                      "gold": float(g), "sig_figs": max(1, len(str(abs(g))))})
    return probs


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["haiku"])
    ap.add_argument("--magnitude", type=int, default=10, help="recurring-class magnitude")
    ap.add_argument("--n-one-offs", type=int, default=4,
                    help="distinct one-off problems (easy pool has 4, hard pool 5)")
    ap.add_argument("--one-off-difficulty", choices=["easy", "hard"], default="hard")
    ap.add_argument("--one-off-magnitude", type=int, default=None,
                    help="override one-off magnitude (else easy->recurring mag, hard->A0 default). "
                         "For Sonnet at recurring m=1000, set this to match the model's calibrated band.")
    ap.add_argument("--arrival", default="oneoff_per_block",
                    help="oneoff_per_block = random but one one-off guaranteed per 4-slot block "
                         "(stress-tests eager-vs-evidence early every seed); random/spread/blocked/"
                         "back/interleaved also available")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--announce", action="store_true", help="reveal class structure to the model")
    ap.add_argument("--budget", type=int, default=5,
                    help="enforced write budget; 5 = #recurring(4)+1 slack; use 4 for the sharp test")
    ap.add_argument("--token-cap", type=int, default=200_000)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--a0-dir", default="runs/a0_oracle_gap_20260630_105027",
                    help="A0 results dir for cost constants (a_hand, h, C)")
    args = ap.parse_args()

    load_dotenv()
    spec = StreamSpec(recurring=DEFAULT_RECURRING, n_one_offs=args.n_one_offs,
                      one_off_difficulty=args.one_off_difficulty, magnitude=args.magnitude,
                      one_off_magnitude=args.one_off_magnitude,
                      arrival=args.arrival, announce=args.announce, seed=args.seed)
    slots = build_stream(spec)
    problems = slots_to_problems(slots)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"stream_disposition_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stream.json").write_text(json.dumps(slots, indent=2))
    (out_dir / "config.json").write_text(json.dumps(
        {**vars(args), "recurring": DEFAULT_RECURRING}, indent=2))
    out_path = out_dir / "sessions.jsonl"
    recur = {s["family"] for s in slots if s["is_recurring"]}
    oneoff = {s["family"] for s in slots if not s["is_recurring"]}
    print(f"stream: {len(problems)} problems | recurring={sorted(recur)} "
          f"one-offs({args.one_off_difficulty})={sorted(oneoff)} | m={args.magnitude} "
          f"arrival={args.arrival} budget={args.budget} -> {out_dir}", flush=True)

    client = RawChat()
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async def one(mkey: str):
        model = CLAUDE.get(mkey, mkey)
        async with sem:
            t0 = time.time()
            try:
                state = SessionState(problems=problems, budget=args.budget)
                row = await run_session(client, model, state, token_cap=args.token_cap,
                                        max_tokens=args.max_tokens)
                row["model_key"] = mkey
            except Exception as e:
                row = {"model": model, "model_key": mkey, "N": len(problems),
                       "error": f"{type(e).__name__}: {e}", "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  {mkey}: solve={row.get('n_correct')}/{row.get('N')} "
                      f"scripts={row.get('n_scripts_written')} used={row.get('n_problems_used_script')} "
                      f"tok={row.get('spent_tokens')} {'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(m) for m in args.models))
    print(f"\nsession transcripts -> {out_path}\n", flush=True)

    # score each model
    for mkey in args.models:
        print(f"================ SCORING {mkey} ================", flush=True)
        try:
            score_run(str(out_dir), args.a0_dir, mkey, args.magnitude)
        except Exception as e:
            print(f"  scoring failed for {mkey}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
