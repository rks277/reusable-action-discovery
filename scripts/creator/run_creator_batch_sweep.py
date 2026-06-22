"""CREATOR v3 sweep — visible variant-batch, announced-budget, across the Claude trio.

Precomputes each item's batch ONCE (parallel + disk-cached, like v2), then runs the
announced-budget episode. Mirrors run_creator_heldout_sweep.py.

  PYTHONPATH=. python -m scripts.creator.run_creator_batch_sweep            # full set
  PYTHONPATH=. python -m scripts.creator.run_creator_batch_sweep --k 8 --models haiku
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.creator.creator_batch import make_batch
from scripts.creator.creator_runner_batch import run_batch

DATA = Path("external/CC.jsonl")
CLAUDE = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
}
CONCURRENCY = 6
N_VARIANTS = 8
BUILD_WORKERS = 8
CACHE = Path(f"runs/creator_batch_cache_N{N_VARIANTS}.json")


def build_usable(k: int | None):
    """(idx, item, batch) for items with >=2 clean inputs and N valid variants.
    Parallel + disk-cached (each make_batch spawns reference subprocesses)."""
    items = [(i, json.loads(l)) for i, l in enumerate(DATA.read_text().splitlines()) if l.strip()]
    cache: dict[str, object] = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    if k is None:
        todo = [(i, d) for i, d in items if str(i) not in cache]
        if todo:
            print(f"building {len(todo)} uncached batches on {BUILD_WORKERS} threads "
                  f"({len(items) - len(todo)} cached)...", flush=True)
            with ThreadPoolExecutor(max_workers=BUILD_WORKERS) as ex:
                res = list(ex.map(lambda it: make_batch(it[1], N_VARIANTS, seed=it[0]), todo))
            for (i, _), r in zip(todo, res):
                cache[str(i)] = r
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(cache))
        usable = [(i, d, cache[str(i)]) for i, d in items if cache.get(str(i)) is not None]
    else:
        usable = []
        for i, d in items:
            if str(i) not in cache:
                cache[str(i)] = make_batch(d, N_VARIANTS, seed=i)
            if cache[str(i)] is not None:
                usable.append((i, d, cache[str(i)]))
            if len(usable) >= k:
                break
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache))
    return usable


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=["haiku", "sonnet", "opus"], choices=list(CLAUDE))
    args = ap.parse_args()

    load_dotenv()
    print("building batches (runs reference solutions)...", flush=True)
    usable = build_usable(args.k)
    models = [CLAUDE[m] for m in args.models]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"creator_batch_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    total = len(usable) * len(models)
    print(f"{len(usable)} usable items x {len(models)} models -> {out_path}", flush=True)

    client = RawChat()
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    done = 0

    async def one(model, idx, item, batch):
        nonlocal done
        async with sem:
            t0 = time.time()
            try:
                row = await run_batch(client, model, item, idx, batch)
            except Exception as e:
                row = {"model": model, "item_idx": idx,
                       "error": f"{type(e).__name__}: {e}", "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                done += 1
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                if done % 25 == 0 or done == total:
                    print(f"  {done}/{total}  last: {model.split('-')[1]} "
                          f"ask={row.get('asked')} built={row.get('built')} "
                          f"correct={row.get('n_correct')}/{row.get('n_variants')} "
                          f"{'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(m, idx, item, batch)
                           for m in models for idx, item, batch in usable))
    print(f"done -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
