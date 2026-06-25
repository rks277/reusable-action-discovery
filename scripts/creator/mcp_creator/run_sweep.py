"""Concurrent runner for the MCP CREATOR eval (mirrors run_creator_eval_hard_sweep.py).

  # local smoke (no vLLM/MCP): in-process backend + a Claude model
  PYTHONPATH=. python -m scripts.creator.mcp_creator.run_sweep \
      --backend inproc --models haiku --limit 2 --n 20 --token-cap 50000 --concurrency 2

  # real MCP wire + Qwen on the vLLM box (tool-calling flags required, see runbook)
  LOCAL_BACKEND=vllm VLLM_BASE_URL=http://localhost:18000/v1 VLLM_API_KEY=EMPTY \
  PYTHONPATH=. python -m scripts.creator.mcp_creator.run_sweep \
      --backend mcp --models Qwen/Qwen2.5-7B-Instruct --limit 50 --n 20 \
      --token-cap 50000 --concurrency 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from concurrent.futures import ThreadPoolExecutor

from lomekwi.raw_chat import RawChat
from scripts.creator.creator_batch import _spans
from scripts.creator.creator_heldout import parse_inputs
from scripts.creator.mcp_creator.backends import make_backend
from scripts.creator.mcp_creator.driver import run_episode
from scripts.creator.mcp_creator.episode_state import build_episode

DATA = Path("external/CC.jsonl")
CLAUDE = {"haiku": "claude-haiku-4-5-20251001",
          "sonnet": "claude-sonnet-4-6", "opus": "claude-opus-4-8"}


def _feasible(item: dict) -> bool:
    inputs = parse_inputs(item["solution"])
    return len(inputs) >= 1 and _spans(item["question"], inputs) is not None


def usable_items(limit: int, n_test: int, base_seed: int) -> list[tuple[int, dict]]:
    """Items that pass the cheap span check AND actually build into an episode (i.e.
    make_hard_batch yields n_test+1 distinct finite rows; some items overflow/degenerate
    under hard-resampling — e.g. compound interest, M/M/1 queue — and must be dropped, or
    the MCP server would exit on startup -> 'McpError: Connection closed'). The buildable
    set is deterministic in (n_test, base_seed), so it's cached and reused across rungs."""
    items = [(i, json.loads(l)) for i, l in enumerate(DATA.read_text().splitlines()) if l.strip()]
    cand = [(i, d) for i, d in items if _feasible(d)]
    cache = Path(f"runs/mcp_feasible_N{n_test}_seed{base_seed}.json")
    if cache.exists():
        good = set(json.loads(cache.read_text()))
        valid = [(i, d) for i, d in cand if i in good]
    else:
        pool = cand if not limit else cand[:int(limit * 1.7) + 25]   # over-provision for drops

        def buildable(it):
            i, d = it
            try:
                return build_episode(d, i, base_seed + i, n_test) is not None
            except Exception:
                return False
        with ThreadPoolExecutor(max_workers=8) as ex:
            flags = list(ex.map(buildable, pool))
        valid = [it for it, ok in zip(pool, flags) if ok]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([i for i, _ in valid]))
    return valid[:limit] if limit else valid


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="claude aliases (haiku/sonnet/opus) or raw ids (e.g. Qwen/Qwen2.5-7B-Instruct)")
    ap.add_argument("--backend", default="mcp", choices=["mcp", "inproc"])
    ap.add_argument("--n", type=int, default=20, help="number of held-out test problems")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--token-cap", type=int, default=50_000)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0, help="base seed; episode seed = seed + item_idx")
    ap.add_argument("--max-tokens", type=int, default=2048, help="per-call output cap")
    args = ap.parse_args()

    load_dotenv()
    models = [CLAUDE.get(m, m) for m in args.models]
    usable = usable_items(args.limit, args.n, args.seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"mcp_creator_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    total = len(usable) * len(models)
    print(f"{len(usable)} items x {len(models)} models -> {out_path} ({total} episodes, "
          f"backend={args.backend}, cap={args.token_cap})", flush=True)

    client = RawChat()
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    done = 0

    async def one(model: str, idx: int, item: dict):
        nonlocal done
        async with sem:
            t0 = time.time()
            backend = None
            try:
                backend = await make_backend(args.backend, item, idx, args.seed + idx, args.n)
                row = await run_episode(client, model, backend, token_cap=args.token_cap,
                                        max_tokens=args.max_tokens)
            except Exception as e:
                row = {"model": model, "item_idx": idx, "N": args.n,
                       "error": f"{type(e).__name__}: {e}", "elapsed_s": round(time.time() - t0, 2)}
            finally:
                if backend is not None:
                    await backend.aclose()
            async with lock:
                done += 1
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                if done % 25 == 0 or done == total:
                    lbl = model.split("/")[-1]
                    print(f"  {done}/{total}  last: {lbl} "
                          f"solve={row.get('n_correct')}/{row.get('N')} "
                          f"tested={row.get('called_begin_test')} "
                          f"tok={row.get('spent_tokens')} {'(err)' if row.get('error') else ''}",
                          flush=True)

    await asyncio.gather(*(one(m, idx, item) for m in models for idx, item in usable))
    print(f"done -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
