"""Calibrate the difficulty so unaided by-hand solving lands near a target (default 50%).
By-hand difficulty is driven mainly by input MAGNITUDE (not sig figs), so this sweeps `--magnitudes`
at a fixed `--sig-figs`: for each magnitude it builds a fresh sample, asks the model to solve BY HAND
(no tools), grades at D, and reports the by-hand solve rate. Picks the magnitude closest to target.

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.calibrate \
      --model haiku --sample 20 --sig-figs 6 --magnitudes 0.02 0.1 0.3 1.0 --target 0.5
"""

from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.creator.creator_exec import _parse_answer
from scripts.creator.tool_disposition_benchmark.dataset import build_dataset
from scripts.creator.tool_disposition_benchmark.grading import correct_to_sigfigs
from scripts.creator.tool_disposition_benchmark.prompts import (byhand_problem_prompt,
                                                                byhand_system_prompt)

CLAUDE = {"haiku": "claude-haiku-4-5-20251001",
          "sonnet": "claude-sonnet-4-6", "opus": "claude-opus-4-8"}


async def _byhand(client, model, problem, d, max_tokens):
    text = await client.chat(model, byhand_system_prompt(),
                             [{"role": "user", "content": byhand_problem_prompt(problem, d)}],
                             max_tokens=max_tokens)
    return correct_to_sigfigs(_parse_answer(text or ""), problem["gold"], d)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sig-figs", type=int, default=6)
    ap.add_argument("--magnitudes", nargs="+", type=float, default=[0.02, 0.1, 0.3, 1.0])
    ap.add_argument("--target", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    load_dotenv()
    model = CLAUDE.get(args.model, args.model)
    client = RawChat()
    sem = asyncio.Semaphore(8)

    print(f"{model.split('/')[-1]}: by-hand @ D={args.sig_figs}, sample {args.sample}\n")
    rates = {}
    for m in sorted(args.magnitudes):
        problems = build_dataset(args.sample, args.sig_figs, args.seed, m)

        async def one(p):
            async with sem:
                return await _byhand(client, model, p, args.sig_figs, args.max_tokens)
        flags = await asyncio.gather(*(one(p) for p in problems))
        rate = sum(flags) / len(flags) if flags else float("nan")
        rates[m] = rate
        print(f"  magnitude={m:<5}: by-hand solve = {sum(flags)}/{len(flags)} = {rate:.2f}")

    best = min(rates, key=lambda m: abs(rates[m] - args.target))
    print(f"\nclosest to target {args.target:.2f}: magnitude={best} (rate {rates[best]:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
