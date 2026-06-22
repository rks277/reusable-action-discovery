"""Sweep the CREATOR C·R·E episode across the Claude trio and write episodes.jsonl.

Mirrors scripts/run_toolworld_sweep.py: async, per-provider Semaphore, a write Lock,
one JSON line per episode. All three models share one RawChat (last_usage is read
synchronously right after each await, so concurrent episodes don't clobber it).

  PYTHONPATH=. python -m scripts.creator.run_creator_sweep            # full set
  PYTHONPATH=. python -m scripts.creator.run_creator_sweep --k 10 --models haiku
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
from scripts.creator.creator_ablation import strip_value
from scripts.creator.creator_runner import run_creator

DATA = Path("external/CC.jsonl")
CLAUDE = {  # short name -> full model id (the id's split("-")[1] gives the short name)
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
}
CONCURRENCY = 6  # matches sweep_config CONCURRENCY["anthropic"]


def load_items(k: int | None) -> list[tuple[int, dict]]:
    """The fixed, strippable subset (same indices for every model). k caps it for pilots."""
    items = []
    for i, line in enumerate(DATA.read_text().splitlines()):
        if not line.strip():
            continue
        d = json.loads(line)
        if strip_value(d["question"], d["solution"]) is not None:
            items.append((i, d))
        if k is not None and len(items) >= k:
            break
    return items


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, help="cap to first K strippable items (pilot)")
    ap.add_argument("--models", nargs="+", default=["haiku", "sonnet", "opus"],
                    choices=list(CLAUDE))
    ap.add_argument("--reps", type=int, default=1)
    args = ap.parse_args()

    load_dotenv()
    items = load_items(args.k)
    models = [CLAUDE[m] for m in args.models]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"creator_sweep_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    print(f"{len(items)} strippable items x {len(models)} models x {args.reps} rep(s) "
          f"-> {out_path}", flush=True)

    client = RawChat()
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    done = 0
    total = len(items) * len(models) * args.reps

    async def one(model: str, idx: int, item: dict, rep: int):
        nonlocal done
        async with sem:
            t0 = time.time()
            try:
                row = await run_creator(client, model, item, idx)
                row["rep"] = rep
            except Exception as e:
                row = {"model": model, "item_idx": idx, "rep": rep,
                       "error": f"{type(e).__name__}: {e}",
                       "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                done += 1
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                if done % 25 == 0 or done == total:
                    print(f"  {done}/{total}  last: {model.split('-')[1]} "
                          f"ask={row.get('asked')} built={row.get('built')} "
                          f"correct={row.get('correct')} {'(err)' if row.get('error') else ''}",
                          flush=True)

    await asyncio.gather(*(one(m, idx, item, rep)
                           for m in models
                           for rep in range(args.reps)
                           for idx, item in items))
    print(f"done -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
