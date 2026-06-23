"""CREATOR v4 sweep — template + N value-rows + free evaluate tool, across N values.

  PYTHONPATH=. python -m scripts.creator.run_creator_eval_sweep \
      --models haiku --limit 200 --ns 5 20 100
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
from scripts.creator.creator_batch import _spans
from scripts.creator.creator_eval_tool import make_template_batch, run_eval_tool
from scripts.creator.creator_heldout import parse_inputs

DATA = Path("external/CC.jsonl")
CLAUDE = {"haiku": "claude-haiku-4-5-20251001",
          "sonnet": "claude-sonnet-4-6", "opus": "claude-opus-4-8"}
CONCURRENCY = 6
BUILD_WORKERS = 8


def _feasible(item: dict) -> bool:
    """N-independent, subprocess-free: >=2 numeric inputs, all cleanly locatable in the
    question text. Lets us pick candidates without running any reference golds."""
    inputs = parse_inputs(item["solution"])
    return len(inputs) >= 2 and _spans(item["question"], inputs) is not None


def build_usable(N: int, limit: int | None):
    """`limit` items that yield a clean template batch at this N. Candidates are chosen by
    the cheap text-only feasibility check (with headroom), so we only run golds for those."""
    items = [(i, json.loads(l)) for i, l in enumerate(DATA.read_text().splitlines()) if l.strip()]
    cand = [(i, d) for i, d in items if _feasible(d)]
    if limit:
        cand = cand[:int(limit * 1.5) + 20]   # headroom for items that fail gold-build at this N
    cache = Path(f"runs/creator_eval_cache_N{N}.json")
    store: dict[str, object] = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [(i, d) for i, d in cand if str(i) not in store]
    if todo:
        print(f"  N={N}: building {len(todo)} candidate batches...", flush=True)
        with ThreadPoolExecutor(max_workers=BUILD_WORKERS) as ex:
            res = list(ex.map(lambda it: make_template_batch(it[1], N, seed=it[0]), todo))
        for (i, _), r in zip(todo, res):
            store[str(i)] = r
        cache.write_text(json.dumps(store))
    usable = [(i, d, store[str(i)]) for i, d in cand if store.get(str(i)) is not None]
    return usable[:limit] if limit else usable


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["haiku"], choices=list(CLAUDE))
    ap.add_argument("--ns", nargs="+", type=int, default=[5, 20, 100])
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    load_dotenv()
    models = [CLAUDE[m] for m in args.models]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"creator_eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"

    print("building template batches...", flush=True)
    jobs = []  # (N, idx, item, batch)
    for N in args.ns:
        usable = build_usable(N, args.limit)
        print(f"  N={N}: {len(usable)} usable items", flush=True)
        for idx, item, batch in usable:
            jobs.append((N, idx, item, batch))
    total = len(jobs) * len(models)
    print(f"{total} episodes -> {out_path}", flush=True)

    client = RawChat()
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    done = 0

    async def one(model, N, idx, item, batch):
        nonlocal done
        async with sem:
            t0 = time.time()
            try:
                row = await run_eval_tool(client, model, item, idx, batch, N)
            except Exception as e:
                row = {"model": model, "item_idx": idx, "N": N,
                       "error": f"{type(e).__name__}: {e}", "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                done += 1
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                if done % 25 == 0 or done == total:
                    print(f"  {done}/{total}  last: N={row.get('N')} "
                          f"ask={row.get('asked')} tool={row.get('used_tool')} "
                          f"calls={row.get('n_eval_calls')} "
                          f"correct={row.get('n_correct')}/{row.get('N')} "
                          f"{'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(m, N, idx, item, batch)
                           for m in models for (N, idx, item, batch) in jobs))
    print(f"done -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
