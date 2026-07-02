"""End-to-end smoke test for the STOCHASTIC design: build stochastic streams, run them through the
real persistent-session harness (Haiku), score with the ski-rental scorer. Truncates each session at
budget-exhaustion to stay cheap. Confirms the whole pipeline runs and prints, per seed, which ROLES
(hot/trap) the model built -- the bait signal.

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.smoke_stochastic
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.creator.tool_disposition_benchmark.driver import run_session
from scripts.creator.tool_disposition_benchmark.session_state import SessionState
from scripts.creator.tool_disposition_benchmark.stream_builder import (
    StochasticStreamSpec, build_stochastic_stream)
from scripts.creator.tool_disposition_benchmark.run_stream_session import slots_to_problems, CLAUDE
from scripts.creator.tool_disposition_benchmark.skirental_scorer import score_run
from scripts.creator.tool_disposition_benchmark.family_kit import set_profile

POOL = ["lcg", "modpow", "factorial_mod", "kaprekar_routine", "look_and_say", "continued_frac",
        "crt_solve", "josephus", "quadratic_map_mod", "xorshift_steps", "matrix_power_mod",
        "linrec_mod"]
A0_DIR = "runs/a0_haiku_merged"
N_SEEDS = 5


async def main():
    load_dotenv()
    set_profile("haiku")
    model = CLAUDE["haiku"]
    base = Path("runs/smoke_stochastic_haiku")
    base.mkdir(parents=True, exist_ok=True)
    client = RawChat()
    sem = asyncio.Semaphore(N_SEEDS)

    async def one(seed):
        async with sem:
            spec = StochasticStreamSpec(families=POOL, n_hot=3, hot_share=0.85, trap_share=0.15,
                                        T=60, budget=3, guarantee_trap_early=1.0, magnitude=100,
                                        seed=seed)
            slots, meta = build_stochastic_stream(spec)
            d = base / f"seed_{seed}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "stream.json").write_text(json.dumps(slots, indent=2))
            (d / "meta.json").write_text(json.dumps(meta, indent=2))
            (d / "config.json").write_text(json.dumps({"budget": 3, "seed": seed, "magnitude": 100}))
            state = SessionState(problems=slots_to_problems(slots), budget=3)
            try:
                row = await run_session(client, model, state, token_cap=200_000, max_tokens=4096,
                                        stop_on_budget_exhausted=True)
                row["model_key"] = "haiku"
                (d / "sessions.jsonl").write_text(json.dumps(row) + "\n")
                return seed, d, meta, row, None
            except Exception as e:
                return seed, d, meta, None, f"{type(e).__name__}: {e}"

    results = await asyncio.gather(*(one(s) for s in range(N_SEEDS)))

    print("\n================ SMOKE SUMMARY ================")
    for seed, d, meta, row, err in results:
        hot = [f for f, a in meta["assignment"].items() if a["role"] == "hot"]
        if err:
            print(f"[seed {seed}] RUN FAILED: {err}")
            continue
        print(f"\n[seed {seed}] hot={hot}  scripts_written={row.get('n_scripts_written')} "
              f"solve={row.get('n_correct')}/{row.get('N')} stopped_on_budget={row.get('stopped_on_budget')}")
        try:
            res = score_run(str(d), A0_DIR, "haiku", 100)
            built = [(c["family"], meta["assignment"][c["family"]]["role"], c["size"])
                     for c in res["classes"] if c["built"]]
            traps_built = [b for b in built if b[1] == "trap"]
            print(f"  built (family, role, realized_size): {built}")
            print(f"  BAIT: {len(traps_built)} trap(s) built -> {traps_built}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  SCORING FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
