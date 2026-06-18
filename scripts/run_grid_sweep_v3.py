"""Random-grid sweep over (N, T) for the explicit-pickup world (v3), Haiku only.

We sample SAMPLE_N points uniformly WITHOUT replacement from the grid
N in [1, 20] x T in [2, 10] (180 cells) and run one v3 episode per sampled cell.
The degenerate boundaries are excluded: T<2 has no distinct byproduct pair (the
machine is unbuildable, pure-grind worlds) and N=0 has no doors (trivially solved).
Per the request, pickup is FREE, so the BUDGET scales exactly as in v2/v3:
budget = sweep_config.budget_for(N) (a function of N only; T does not change it).

(toolworld_v3 still tolerates N=0 / T<2 worlds so older runs replay, but this
sweep no longer samples them.)

Each row carries variant="v3_pickup" so replay/analyze route correctly. Rep i
uses relabel_seed = drop_seed = i (reproducible; SAMPLE_SEED fixes which cells).

Cost: see the USD estimate printed at startup (grounded in real Haiku usage from
prior budget sweeps). Run with --dry-run to print the sampled cells + estimate
and exit WITHOUT calling the API.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from scripts.toolworld_v3 import run
from scripts import sweep_config as cfg
from lomekwi.raw_chat import _provider_for

# --- sweep knobs -------------------------------------------------------
# Model roster: each model gets its own run dir (grid_sweep_v3_<tag>_<ts>) and is
# run sequentially. `--model <id>` overrides the roster with a single model.
# Local Gemma via Ollama (the ":" in each id routes to the ollama provider; edit
# the family tag to match `ollama list`, e.g. gemma3, if needed). Anthropic
# models are unhooked for now -- kept commented to bring back later.
MODELS = [
    "gemma4:1b",
    "gemma4:4b",
    "gemma4:12b",
    # "claude-haiku-4-5-20251001",
    # "claude-sonnet-4-6",
    # "claude-opus-4-8",
]
N_RANGE = range(1, 21)                 # N = 1..20 inclusive (N=0 is trivially solved)
T_RANGE = range(2, 11)                 # T = 2..10 inclusive (T<2 has no recipe pair)
SAMPLE_N = 100                         # random points over the grid
SAMPLE_SEED = 20260618                 # fixes WHICH cells are sampled


def sampled_points() -> list[tuple[int, int]]:
    grid = [(n, t) for n in N_RANGE for t in T_RANGE]
    return random.Random(SAMPLE_SEED).sample(grid, SAMPLE_N)


def max_turns_for(n: int, budget: int) -> int:
    """Turn cap > budget + pickups. Free pickups add up to one turn per examine,
    so turns can reach ~2*budget; give headroom for opens and stray pickups."""
    return 2 * budget + 2 * n + 80


# --- cost estimate (Haiku 4.5: $1/1M in, $5/1M out, $0.10/1M cache-read,
#     $1.25/1M cache-write) -------------------------------------------------
# Anchored on REAL haiku usage from prior v2 budget sweeps (full price, no cache
# discount assumed -> conservative): n=8 ~ $0.064/ep, n=20 ~ $0.134/ep. Linear
# fit cost_v2(N) ~ 0.018 + 0.0058*N. v3's free pickups add ~one turn per examine,
# and each turn re-sends the growing transcript, so per-episode tokens scale
# super-linearly with turns -- we apply a 2.0x (likely) .. 3.0x (upper) factor.
def estimate_usd(points: list[tuple[int, int]]) -> tuple[float, float]:
    def v2_cost(n: int) -> float:
        return 0.018 + 0.0058 * n
    base = sum(v2_cost(n) for n, _ in points)
    return 2.0 * base, 3.0 * base


def model_tag(model: str) -> str:
    if "sonnet" in model: return "sonnet"
    if "haiku" in model: return "haiku"
    if "opus" in model: return "opus"
    # generic (e.g. ollama "gemma4:12b" -> "gemma4-12b"): strip path/punctuation
    return model.split("/")[-1].replace(".", "-").replace(":", "-")


def print_cost_estimate(model: str, points: list[tuple[int, int]]):
    tag = model_tag(model)
    if tag == "haiku":
        lo, hi = estimate_usd(points)
        print(f"  ESTIMATED COST (Haiku 4.5): ~${lo:.0f}-${hi:.0f} USD", flush=True)
    elif tag == "sonnet":
        print(f"  ESTIMATED COST (Sonnet 4.6): ~$15-35 USD "
              f"(caching-dependent; could brush a $30 cap)", flush=True)
    elif tag == "opus":
        print(f"  ESTIMATED COST (Opus 4.8): ~$20-24 USD "
              f"(~1.7-1.9x the Sonnet run; under the $30 cap with less headroom)",
              flush=True)
    elif _provider_for(model) == "ollama":
        print(f"  COST: local (ollama) -- no API spend", flush=True)


async def run_one_model(model: str, points: list[tuple[int, int]],
                        resume_dir: Path | None):
    """Run the sweep for a single model into its own (or a resumed) run dir."""
    tag = model_tag(model)
    concurrency = cfg.CONCURRENCY[_provider_for(model)]
    done_idx: set[int] = set()
    if resume_dir is not None:
        out_dir = resume_dir
        out_path = out_dir / "episodes.jsonl"
        for l in out_path.read_text().splitlines():
            if l.strip():
                done_idx.add(json.loads(l)["relabel_seed"])
        print(f"\nResuming {out_path}: {len(done_idx)} done, "
              f"{len(points) - len(done_idx)} remaining (model={model})", flush=True)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("runs") / f"grid_sweep_v3_{tag}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "episodes.jsonl"
        out_path.write_text("")
        print(f"\nWriting to {out_path}  (model={model}, concurrency={concurrency})",
              flush=True)
    todo = [(i, n, t) for i, (n, t) in enumerate(points) if i not in done_idx]

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def one(i: int, n: int, t: int):
        async with sem:
            budget = cfg.budget_for(n)
            t0 = time.time()
            try:
                result, trace = await run(
                    model, n=n, n_types=t, relabel_seed=i, drop_seed=i,
                    hint=cfg.HINT, max_turns=max_turns_for(n, budget),
                    budget=budget)
                row = {"model": model, "n": n, "n_types": t, "hint": cfg.HINT,
                       "variant": "v3_pickup", "budget": budget,
                       "relabel_seed": i, "drop_seed": i,
                       "labels": result["labels"],
                       "actions": [x["action"] for x in trace],
                       "agent_texts": [x["agent_text"] for x in trace],
                       "obs": [x["obs"] for x in trace],
                       "solved": result["solved"],
                       "total_actions": result["total_actions"],
                       "pickups": result["pickups"],
                       "usage": result["usage"],
                       "elapsed_s": round(time.time() - t0, 2)}
            except Exception as e:
                row = {"model": model, "n": n, "n_types": t, "hint": cfg.HINT,
                       "variant": "v3_pickup", "budget": budget,
                       "relabel_seed": i, "drop_seed": i,
                       "error": f"{type(e).__name__}: {e}",
                       "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  [{tag} {i:3d}] N={n:2d} T={t:2d} solved={row.get('solved')} "
                      f"actions={row.get('total_actions')} "
                      f"pickups={row.get('pickups')} "
                      f"{'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(i, n, t) for (i, n, t) in todo))
    print(f"\nDone ({model}). Episodes at: {out_path}", flush=True)


async def main():
    dry = "--dry-run" in sys.argv
    # roster: --model overrides the configured MODELS list with a single model
    if "--model" in sys.argv:
        roster = [sys.argv[sys.argv.index("--model") + 1]]
    else:
        roster = list(MODELS)
    resume_dir = None
    if "--resume" in sys.argv:
        resume_dir = Path(sys.argv[sys.argv.index("--resume") + 1])
        if len(roster) != 1:
            raise SystemExit("--resume requires a single model (pass --model too)")

    points = sampled_points()
    ns = [n for n, _ in points]
    print(f"v3 grid sweep: {len(points)} points, models={roster}", flush=True)
    print(f"  N in [{min(ns)},{max(ns)}] (mean {sum(ns)/len(ns):.1f}), "
          f"T in [{min(T_RANGE)},{max(T_RANGE)}]; budget=budget_for(N) "
          f"(pickup is free)", flush=True)
    for model in roster:
        print_cost_estimate(model, points)

    if dry:
        print("\n--dry-run: sampled (N,T) cells:")
        for i, (n, t) in enumerate(points):
            print(f"  [{i:3d}] N={n:2d} T={t:2d} budget={cfg.budget_for(n)}")
        print("\nNo API calls made.")
        return

    load_dotenv()
    # Models run sequentially -- ollama serves one model at a time, and per-model
    # run dirs keep results separable. Each row is deterministic in the rep index.
    for model in roster:
        await run_one_model(model, points, resume_dir)
    print("\nAll models done.")


if __name__ == "__main__":
    asyncio.run(main())
