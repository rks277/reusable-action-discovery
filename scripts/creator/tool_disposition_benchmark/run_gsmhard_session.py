"""Run the tool-disposition IN-SESSION benchmark (same harness as run_sweep.py: persistent
session, write_script/run_script/submit_answer tools, 0.1N write budget, token cap, driver loop)
but over the GSM-Hard variation problems instead of the CC.jsonl CREATOR set.

Each of a seed's N problems is presented as ONE concrete instance (the variation file's `example`
row). GSM-Hard answers are exact integers, so each problem's `sig_figs` is set to the gold's full
digit count -> `correct_to_sigfigs` reduces to exact-integer match (faithful to GSM-Hard's exact
answer, while reusing the harness/prompts/grading unchanged).

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.run_gsmhard_session \
      --models haiku --seed 0 --magnitude 1.0 --token-cap 200000 --max-tokens 2048
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
from scripts.creator.tool_disposition_benchmark.gsmhard_variations import _mtag
from scripts.creator.tool_disposition_benchmark.session_state import (
    SessionState, write_budget)

CLAUDE = {"haiku": "claude-haiku-4-5-20251001",
          "sonnet": "claude-sonnet-4-6", "opus": "claude-opus-4-8"}
DATASET_DIR = Path("scripts/creator/tool_disposition_benchmark/datasets")


def load_gsmhard_problems(seed: int, magnitude: float) -> list[dict]:
    """The seed's variation problems as session problems (one `example` instance each), with
    sig_figs == the gold's digit count so grading is exact-integer."""
    path = DATASET_DIR / f"gsmhard_var_seed{seed}_m{_mtag(magnitude)}.json"
    problems = []
    for p in json.loads(path.read_text()):
        ex = p["example"]
        gold = float(ex["gold"])
        d = len(str(abs(int(round(gold)))))           # exact match for an integer answer
        problems.append({
            "idx": p["idx"], "item_idx": p["src_idx"], "keys": p["keys"],
            "question": ex["question"], "inputs": ex["inputs"],
            "gold": gold, "sig_figs": d,
        })
    return problems


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["haiku"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--magnitude", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--token-cap", type=int, default=200_000)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=3)
    args = ap.parse_args()

    load_dotenv()
    models = [CLAUDE.get(m, m) for m in args.models]
    problems = load_gsmhard_problems(args.seed, args.magnitude)
    budget = write_budget(len(problems))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"gsmhard_disposition_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sessions.jsonl"
    (out_dir / "config.json").write_text(json.dumps({**vars(args), "source": "gsmhard"}, indent=2))
    total = len(models) * args.reps
    print(f"{len(problems)} GSM-Hard problems (seed {args.seed}, m{args.magnitude}, budget={budget}) "
          f"x {len(models)} models x {args.reps} reps -> {out_path} (cap={args.token_cap:,})",
          flush=True)

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
                row = await run_session(client, model, state, token_cap=args.token_cap,
                                        max_tokens=args.max_tokens)
                row["rep"] = rep
            except Exception as e:
                row = {"model": model, "rep": rep, "N": len(problems),
                       "error": f"{type(e).__name__}: {e}", "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                done += 1
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                lbl = model.split("/")[-1]
                print(f"  {done}/{total}  {lbl} rep{rep}: solve={row.get('n_correct')}/{row.get('N')} "
                      f"scripts={row.get('n_scripts_written')} used={row.get('n_problems_used_script')} "
                      f"persist={row.get('persistence')} reuse={row.get('reusability')} "
                      f"tok={row.get('spent_tokens')} {'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(m, r) for m in models for r in range(args.reps)))
    print(f"done -> {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
