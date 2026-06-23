"""CREATOR v4-hard sweep — deceptively-hard arithmetic + no Curiosity gate, at N=20.

  PYTHONPATH=. python -m scripts.creator.run_creator_eval_hard_sweep \
      --models haiku sonnet opus --n 20 --limit 400
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
from scripts.creator.creator_eval_hard import make_hard_batch
from scripts.creator.creator_eval_tool import run_eval_tool
from scripts.creator.creator_heldout import parse_inputs

DATA = Path("external/CC.jsonl")
# Short aliases for the Claude trio; any other --models entry is passed through verbatim
# as a raw model id (e.g. a vLLM-served name like "Qwen/Qwen2.5-7B-Instruct").
CLAUDE = {"haiku": "claude-haiku-4-5-20251001",
          "sonnet": "claude-sonnet-4-6", "opus": "claude-opus-4-8"}
BUILD_WORKERS = 8


def _feasible(item: dict, withhold: bool) -> bool:
    inputs = parse_inputs(item["solution"])
    return len(inputs) >= (2 if withhold else 1) and _spans(item["question"], inputs) is not None


def build_usable(N: int, limit: int | None, withhold: bool):
    items = [(i, json.loads(l)) for i, l in enumerate(DATA.read_text().splitlines()) if l.strip()]
    cand = [(i, d) for i, d in items if _feasible(d, withhold)]
    if limit:
        cand = cand[:int(limit * 1.7) + 30]
    tag = "gated" if withhold else "open"
    cache = Path(f"runs/creator_eval_hard_cache_N{N}_{tag}.json")
    store = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [(i, d) for i, d in cand if str(i) not in store]
    if todo:
        print(f"  building {len(todo)} hard batches (N={N}, {tag})...", flush=True)
        with ThreadPoolExecutor(max_workers=BUILD_WORKERS) as ex:
            res = list(ex.map(lambda it: make_hard_batch(it[1], N, seed=it[0], withhold=withhold), todo))
        for (i, _), r in zip(todo, res):
            store[str(i)] = r
        cache.write_text(json.dumps(store))
    usable = [(i, d, store[str(i)]) for i, d in cand if store.get(str(i)) is not None]
    return usable[:limit] if limit else usable


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["haiku", "sonnet", "opus"],
                    help="claude aliases (haiku/sonnet/opus) or raw model ids (e.g. vLLM-served names)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--tool-policy", default="free", choices=["free", "costly", "budget1"])
    ap.add_argument("--withhold", action="store_true", help="re-enable Curiosity gate (blank one shared input)")
    ap.add_argument("--concurrency", type=int, default=10,
                    help="parallel episodes; 10 for Anthropic, 32-64 for a vLLM box")
    ap.add_argument("--judge", default=None,
                    help="ask-classifier model (default: Haiku). Set to a local model to stay fully offline.")
    args = ap.parse_args()

    load_dotenv()
    models = [CLAUDE.get(m, m) for m in args.models]   # alias -> id, else pass through
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"creator_eval_hard_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"

    print("building hard batches...", flush=True)
    usable = build_usable(args.n, args.limit, args.withhold)
    total = len(usable) * len(models)
    print(f"{len(usable)} usable x {len(models)} models -> {out_path} ({total} episodes)", flush=True)

    client = RawChat()
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    done = 0

    async def one(model, idx, item, batch):
        nonlocal done
        async with sem:
            t0 = time.time()
            try:
                row = await run_eval_tool(client, model, item, idx, batch, args.n,
                                          tool_policy=args.tool_policy, judge_model=args.judge)
            except Exception as e:
                row = {"model": model, "item_idx": idx, "N": args.n,
                       "error": f"{type(e).__name__}: {e}", "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                done += 1
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                if done % 25 == 0 or done == total:
                    lbl = model.split("-")[1] if model.startswith("claude-") else model
                    print(f"  {done}/{total}  last: {lbl} "
                          f"tool={row.get('used_tool')} correct={row.get('n_correct')}/{row.get('N')} "
                          f"{'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(m, idx, item, batch)
                           for m in models for idx, item, batch in usable))
    print(f"done -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
