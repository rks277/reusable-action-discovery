"""CREATOR v2 sweep — announced held-out generalization across the Claude trio.

Precomputes each item's held-out payload ONCE (so all models are scored on identical
tuples), then runs the announced single-turn episode. Mirrors run_creator_sweep.py.

  PYTHONPATH=. python -m scripts.creator.run_creator_heldout_sweep            # full set
  PYTHONPATH=. python -m scripts.creator.run_creator_heldout_sweep --k 10 --models haiku
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
from scripts.creator.creator_heldout import make_heldout
from scripts.creator.creator_runner_heldout import run_heldout

DATA = Path("external/CC.jsonl")
CLAUDE = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
}
CONCURRENCY = 6
M_HELDOUT = 5
BUILD_WORKERS = 8
# Payloads are deterministic (seed=idx) and model-independent, so cache them once.
# A future crash / the Sonnet+Opus runs then skip the (local, subprocess-heavy) build.
CACHE = Path(f"runs/creator_heldout_payloads_M{M_HELDOUT}.json")


def _to_json(payload):
    """make_heldout's tuple form -> JSON-safe lists (None passes through)."""
    if payload is None:
        return None
    arg_names, shown, sgold, heldout = payload
    return [arg_names, shown, sgold, [[vals, gold] for vals, gold in heldout]]


def build_usable(k: int | None):
    """(idx, item, payload) for items with numeric inputs and M valid held-out tuples.
    Parallel + disk-cached: only uncached items are (re)built, across BUILD_WORKERS
    threads (each make_heldout spawns subprocesses, so threads parallelize well)."""
    items = [(i, json.loads(l)) for i, l in enumerate(DATA.read_text().splitlines()) if l.strip()]
    cache: dict[str, object] = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text())

    def need(i):
        return str(i) not in cache

    if k is None:
        todo = [(i, d) for i, d in items if need(i)]
        if todo:
            print(f"building {len(todo)} uncached payloads on {BUILD_WORKERS} threads "
                  f"({len(items) - len(todo)} cached)...", flush=True)
            with ThreadPoolExecutor(max_workers=BUILD_WORKERS) as ex:
                results = list(ex.map(lambda it: _to_json(make_heldout(it[1], M_HELDOUT, seed=it[0])), todo))
            for (i, _), r in zip(todo, results):
                cache[str(i)] = r
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(cache))
        usable = [(i, d, cache[str(i)]) for i, d in items if cache.get(str(i)) is not None]
    else:  # pilot: build sequentially in order until K usable (small)
        usable = []
        for i, d in items:
            if need(i):
                cache[str(i)] = _to_json(make_heldout(d, M_HELDOUT, seed=i))
            if cache[str(i)] is not None:
                usable.append((i, d, cache[str(i)]))
            if len(usable) >= k:
                break
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache))
    return usable


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, help="cap to first K usable items (pilot)")
    ap.add_argument("--models", nargs="+", default=["haiku", "sonnet", "opus"],
                    choices=list(CLAUDE))
    args = ap.parse_args()

    load_dotenv()
    print("building held-out payloads (runs reference solutions)...", flush=True)
    usable = build_usable(args.k)
    models = [CLAUDE[m] for m in args.models]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"creator_heldout_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    total = len(usable) * len(models)
    print(f"{len(usable)} usable items x {len(models)} models -> {out_path}", flush=True)

    client = RawChat()
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    done = 0

    async def one(model: str, idx: int, item: dict, payload):
        nonlocal done
        async with sem:
            t0 = time.time()
            try:
                row = await run_heldout(client, model, item, idx, payload)
            except Exception as e:
                row = {"model": model, "item_idx": idx,
                       "error": f"{type(e).__name__}: {e}",
                       "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                done += 1
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                if done % 25 == 0 or done == total:
                    print(f"  {done}/{total}  last: {model.split('-')[1]} "
                          f"shown={row.get('shown_correct')} gen={row.get('generalizes')} "
                          f"{'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(m, idx, item, payload)
                           for m in models
                           for idx, item, payload in usable))
    print(f"done -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
