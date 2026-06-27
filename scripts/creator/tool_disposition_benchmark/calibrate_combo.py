"""Calibrate the combinatorial-DP difficulty knob: sweep `size` and measure a model's BY-HAND
solve rate (no tools), to pick the size that lands near ~50% — the regime where tool use is a
genuine choice (the analog of calibrate.py's magnitude search for the CREATOR track).

Grades by EXACT integer equality (golds are exact big ints). Reports overall and per-family
solve rate per size so we can also see which families carry the difficulty.

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.calibrate_combo \
      --model sonnet --sizes 1 2 3 4 5 6 --sample 35
"""

from __future__ import annotations

import argparse
import asyncio
import re
from collections import defaultdict

from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.creator.tool_disposition_benchmark.dataset_combo import build_dataset

CLAUDE = {"haiku": "claude-haiku-4-5-20251001",
          "sonnet": "claude-sonnet-4-6", "opus": "claude-opus-4-8"}

SYS = ("You solve counting problems by hand (no tools). Reason carefully, then end with a single "
       "final line `ANSWER: <integer>` giving the exact count as a plain integer (no commas).")


def parse_int(text: str) -> int | None:
    if not text:
        return None
    m = re.findall(r"ANSWER:\s*(-?[\d,]+)", text)
    raw = m[-1] if m else None
    if raw is None:                      # fall back to the last integer in the text
        nums = re.findall(r"-?\d[\d,]*", text)
        raw = nums[-1] if nums else None
    if raw is None:
        return None
    try:
        return int(raw.replace(",", ""))
    except ValueError:
        return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--sizes", type=float, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--sample", type=int, default=35, help="problems per size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    load_dotenv()
    client = RawChat()
    model = CLAUDE.get(args.model, args.model)
    sem = asyncio.Semaphore(args.concurrency)

    async def solve(p):
        async with sem:
            try:
                text = await client.chat(model, SYS,
                                         [{"role": "user", "content": p["question"]}],
                                         max_tokens=args.max_tokens)
            except Exception:
                return p, None
            return p, parse_int(text or "")

    print(f"model={model}  sample={args.sample}/size  grading=exact-int\n")
    print(f"  {'size':>5}{'by-hand':>9}{'  per-family (solved/n)'}")
    for size in args.sizes:
        problems = build_dataset(args.sample, size, args.seed)
        results = await asyncio.gather(*(solve(p) for p in problems))
        n_ok = sum(1 for p, a in results if a is not None and a == p["gold"])
        fam = defaultdict(lambda: [0, 0])
        for p, a in results:
            fam[p["family"]][0] += int(a is not None and a == p["gold"])
            fam[p["family"]][1] += 1
        per = " ".join(f"{k.replace('fam_',''):>4}:{v[0]}/{v[1]}" for k, v in sorted(fam.items()))
        print(f"  {size:>5g}{f'{n_ok}/{len(results)}':>9} ({n_ok/len(results):.0%})  {per}")


if __name__ == "__main__":
    asyncio.run(main())
