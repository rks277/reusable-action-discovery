"""Standalone BY-HAND (no tools / no environment / no MCP) eval of models on the CREATOR
disposition-bench problems: just present each question and grade the model's answer at the
dataset's sig figs. Reports per-seed and pooled solve rate.

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.byhand_eval \
      --models haiku --seeds 0 1 --n 20 --sig-figs 6 --magnitude 0.01
"""

from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.creator.creator_exec import _parse_answer
from scripts.creator.tool_disposition_benchmark.dataset import load_or_build
from scripts.creator.tool_disposition_benchmark.grading import correct_to_sigfigs
from scripts.creator.tool_disposition_benchmark.prompts import (byhand_problem_prompt,
                                                                byhand_system_prompt)

CLAUDE = {"haiku": "claude-haiku-4-5-20251001",
          "sonnet": "claude-sonnet-4-6", "opus": "claude-opus-4-8"}


async def solve(client, model, p, max_tokens):
    text = await client.chat(model, byhand_system_prompt(),
                             [{"role": "user", "content": byhand_problem_prompt(p)}],
                             max_tokens=max_tokens)
    return correct_to_sigfigs(_parse_answer(text or ""), p["gold"], p["sig_figs"])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["haiku"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--sig-figs", type=int, default=6)
    ap.add_argument("--magnitude", type=float, default=0.01)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=10)
    args = ap.parse_args()

    load_dotenv()
    client = RawChat()
    sem = asyncio.Semaphore(args.concurrency)

    async def graded(model, p):
        async with sem:
            try:
                return await solve(client, model, p, args.max_tokens)
            except Exception:
                return False

    print(f"by-hand (no tools)  N={args.n}/seed  D={args.sig_figs}  m={args.magnitude}  "
          f"seeds={args.seeds}\n", flush=True)
    for m in args.models:
        model = CLAUDE.get(m, m)
        pooled_ok = pooled_n = 0
        per_seed = []
        for seed in args.seeds:
            probs = load_or_build(args.n, args.sig_figs, seed, args.magnitude, args.shuffle)
            flags = await asyncio.gather(*(graded(model, p) for p in probs))
            ok = sum(flags)
            per_seed.append(f"seed{seed} {ok}/{len(probs)}")
            pooled_ok += ok
            pooled_n += len(probs)
            print(f"  {m:<7} seed{seed}: {ok}/{len(probs)} = {ok/len(probs):.3f}", flush=True)
        print(f"  {m:<7} POOLED: {pooled_ok}/{pooled_n} = {pooled_ok/pooled_n:.3f}  "
              f"({', '.join(per_seed)})\n", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
