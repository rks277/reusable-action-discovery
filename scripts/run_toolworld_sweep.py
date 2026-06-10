"""Sweep orchestrator for the byproduct->machine tool world (E3 first pass).

Mirrors scripts/run_l6_three_tasks.py: loop over model x n x hint x reps,
each rep a FRESH (relabel_seed, drop_seed) pair, write one JSONL row per
episode. Per-provider concurrency is capped to avoid rate limits.

Each episode is driven by scripts.run_toolworld_llm.run (one episode/call).
The world is deterministic given (relabel_seed, drop_seed, n, hint), so each
row logs exactly enough to REPLAY the trace post-hoc (see replay_toolworld.py):
the actions, the agent texts, the recorded observations, and the seeds.

Locked first-pass grid (E3 headline at n=8 + n=2 below-threshold contrast):
  3 models x n in {2, 8} x subtle-hint only x 8 reps = 48 episodes.
Rep r uses (relabel_seed=r, drop_seed=r) so the SAME rep index is the same
world across models and n -> paired cross-model comparison.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from scripts.toolworld_v2 import run


# --- Grid (locked first pass) ------------------------------------------

MODELS = [
    ("anthropic", "claude-haiku-4-5-20251001"),
    ("anthropic", "claude-sonnet-4-6"),
    ("anthropic", "claude-opus-4-8"),
]
N_VALUES = [8]          # well above the v2 threshold n*=4 (rational to build).
                        # below-threshold anchor (n=3) added later only if needed.
HINTS = [True]          # subtle-hint only (E1 no-hint arm deferred)
REPS = 8
N_TYPES = 3             # multi-type construction (v2): n*=4, real recipe search

# Per-provider concurrency cap. All three models share the Anthropic key, so
# this bounds total in-flight episodes. Opus is ~80% of cost and the slowest;
# keep it modest to avoid rate limits / token bursts.
CONCURRENCY = {"anthropic": 6, "openai": 6, "google": 4}


def seed_pair(rep: int) -> tuple[int, int]:
    """Fresh (relabel_seed, drop_seed) per rep; shared across cells by rep."""
    return rep, rep


async def main():
    load_dotenv()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"toolworld_sweep_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    out_path.write_text("")
    print(f"Writing to {out_path}", flush=True)

    sems = {prov: asyncio.Semaphore(CONCURRENCY[prov])
            for prov, _ in MODELS}
    lock = asyncio.Lock()

    cells = [(prov, model, n, hint, rep)
             for prov, model in MODELS
             for n in N_VALUES
             for hint in HINTS
             for rep in range(REPS)]
    print(f"Total episodes: {len(cells)} "
          f"({len(MODELS)} models x n in {N_VALUES} x "
          f"{len(HINTS)} hint x {REPS} reps)", flush=True)

    async def run_one(prov, model, n, hint, rep):
        relabel_seed, drop_seed = seed_pair(rep)
        async with sems[prov]:
            t0 = time.time()
            try:
                result, trace = await run(
                    model, n=n, relabel_seed=relabel_seed,
                    drop_seed=drop_seed, hint=hint, n_types=N_TYPES,
                )
                row = {
                    "model": model,
                    "n": n,
                    "n_types": N_TYPES,
                    "hint": hint,
                    "relabel_seed": relabel_seed,
                    "drop_seed": drop_seed,
                    "labels": result["labels"],
                    "actions": [x["action"] for x in trace],
                    "agent_texts": [x["agent_text"] for x in trace],
                    "obs": [x["obs"] for x in trace],
                    "solved": result["solved"],
                    "total_actions": result["total_actions"],
                    "usage": result["usage"],
                    "elapsed_s": round(time.time() - t0, 2),
                }
            except Exception as e:
                row = {
                    "model": model, "n": n, "n_types": N_TYPES, "hint": hint,
                    "relabel_seed": relabel_seed, "drop_seed": drop_seed,
                    "error": f"{type(e).__name__}: {e}",
                    "elapsed_s": round(time.time() - t0, 2),
                }
            async with lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(
                    f"  {model[:28]:28s} n={n} rep={rep} "
                    f"solved={row.get('solved')} "
                    f"actions={row.get('total_actions')} "
                    f"{'(err)' if row.get('error') else ''}",
                    flush=True,
                )

    await asyncio.gather(*(run_one(*c) for c in cells))
    print(f"\nDone. Episodes at: {out_path}")
    print("Next: python -m scripts.replay_toolworld <jsonl> to verify replay, "
          "then python -m scripts.analyze_toolworld <jsonl> for metrics+figures.")


if __name__ == "__main__":
    asyncio.run(main())
