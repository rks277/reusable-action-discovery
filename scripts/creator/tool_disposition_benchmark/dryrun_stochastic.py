"""3-seed Haiku DRY RUN for the stochastic tool-investment PoC. Same design as the smoke, but:
  - FULL sessions (stop_on_budget_exhausted=False) -> we observe real reuse + post-budget grinding;
  - the pi* Whittle price is tuned ONCE (a design constant of {N,T,B,pool}) and reused across seeds;
  - reports per-seed regret_lb (pi* - model), clairvoyant gap, trap builds, lateness, empirical reuse,
    and the cache-read fraction (to confirm prompt caching is firing before scaling up).

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.dryrun_stochastic
"""

from __future__ import annotations

import asyncio
import json
import statistics as st
from pathlib import Path

from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.creator.tool_disposition_benchmark.driver import run_session
from scripts.creator.tool_disposition_benchmark.session_state import SessionState
from scripts.creator.tool_disposition_benchmark.stream_builder import (
    StochasticStreamSpec, build_stochastic_stream)
from scripts.creator.tool_disposition_benchmark.run_stream_session import slots_to_problems, CLAUDE
from scripts.creator.tool_disposition_benchmark.skirental_scorer import score_run, costs_from_a0
from scripts.creator.tool_disposition_benchmark.pi_star import tune_price
from scripts.creator.tool_disposition_benchmark.family_kit import set_profile

POOL = ["lcg", "modpow", "factorial_mod", "kaprekar_routine", "look_and_say", "continued_frac",
        "crt_solve", "josephus", "quadratic_map_mod", "xorshift_steps", "matrix_power_mod",
        "linrec_mod"]
A0_DIR = "runs/a0_haiku_merged"
N_SEEDS = 3
N, T, BUDGET, MAG = len(POOL), 60, 3, 100


async def main():
    load_dotenv()
    set_profile("haiku")
    model = CLAUDE["haiku"]
    base = Path("runs/dryrun_stochastic_haiku")
    base.mkdir(parents=True, exist_ok=True)

    # tune the pi* Whittle price ONCE (same-info: uses only {N,T,B}+pool), reuse across seeds
    costs = costs_from_a0(A0_DIR, "haiku", MAG)
    a_hands = {f: costs.ah(f) for f in POOL}
    price = tune_price(costs, POOL, a_hands, N, T, BUDGET, MAG, alpha=1.0, n_sim=150)
    print(f"tuned pi* Whittle price (once, reused across seeds): {price:.1f}\n")

    client = RawChat()
    sem = asyncio.Semaphore(N_SEEDS)

    async def one(seed):
        async with sem:
            spec = StochasticStreamSpec(families=POOL, n_hot=BUDGET, hot_share=0.85,
                                        trap_share=0.15, T=T, budget=BUDGET,
                                        guarantee_trap_early=1.0, magnitude=MAG, seed=seed)
            slots, meta = build_stochastic_stream(spec)
            d = base / f"seed_{seed}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "stream.json").write_text(json.dumps(slots, indent=2))
            (d / "meta.json").write_text(json.dumps(meta, indent=2))
            (d / "config.json").write_text(json.dumps({"budget": BUDGET, "seed": seed,
                                                        "magnitude": MAG}))
            state = SessionState(problems=slots_to_problems(slots), budget=BUDGET)
            try:
                row = await run_session(client, model, state, token_cap=300_000, max_tokens=4096,
                                        stop_on_budget_exhausted=False)   # FULL session
                row["model_key"] = "haiku"
                (d / "sessions.jsonl").write_text(json.dumps(row) + "\n")
                return seed, d, meta, row, None
            except Exception as e:
                return seed, d, meta, None, f"{type(e).__name__}: {e}"

    results = await asyncio.gather(*(one(s) for s in range(N_SEEDS)))

    print("\n================ DRY-RUN SUMMARY ================")
    regrets, model_traps, pi_traps, reuse_rates = [], [], [], []
    for seed, d, meta, row, err in results:
        if err:
            print(f"[seed {seed}] RUN FAILED: {err}")
            continue
        tu = row.get("turn_usages") or []
        cr = sum(t.get("cache_read_tokens", 0) for t in tu)
        it = sum(t.get("input_tokens", 0) for t in tu)
        cache_frac = cr / max(1, cr + it)
        print(f"\n[seed {seed}] turns={row.get('n_turns')} scripts_written={row.get('n_scripts_written')} "
              f"solve={row.get('n_correct')}/{row.get('N')} reuse_calls={row.get('n_run_calls')} "
              f"cache_read_frac={cache_frac:.2f} (cr={cr} it={it})")
        try:
            res = score_run(str(d), A0_DIR, "haiku", MAG, pistar_price=price)
            rep = res.get("pistar")
            if rep:
                regrets.append(rep["regret_lb"]); model_traps.append(rep["model_traps_built"])
                pi_traps.append(rep["pistar_traps_built"])
            # empirical reuse: of classes the model built, did it actually call the tool again?
            built = [c for c in res["classes"] if c["built"] and c["size"] >= 2]
            reused = [c for c in built if c["reuse_count"] > 0]
            if built:
                reuse_rates.append(len(reused) / len(built))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  SCORING FAILED: {type(e).__name__}: {e}")

    print("\n================ AGGREGATE ================")
    if regrets:
        print(f"regret_lb (pi* - model): mean={st.mean(regrets):.1f}  -> {[round(r) for r in regrets]}")
        print(f"model traps built/seed={st.mean(model_traps):.2f}  pi* traps/seed={st.mean(pi_traps):.2f}")
    if reuse_rates:
        print(f"empirical reuse rate (built recurring classes actually reused): {st.mean(reuse_rates):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
