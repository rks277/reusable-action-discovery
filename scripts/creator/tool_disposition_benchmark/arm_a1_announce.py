"""A1 (announce) arm on the stochastic design: system prompt DISCLOSES the recurrence structure
(RECURRENCE_NOTE, non-prescriptive) — does Haiku STILL build on first sight? Uniform-hard pool (all
a_hand=0, no under-building confound), g=1, full sessions. Primary metric = build lateness (fidelity).

  PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.arm_a1_announce [--model haiku|sonnet|opus]
"""
from __future__ import annotations
import argparse, asyncio, json, statistics as st
from pathlib import Path
from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.creator.tool_disposition_benchmark.driver import run_session
from scripts.creator.tool_disposition_benchmark.session_state import SessionState
from scripts.creator.tool_disposition_benchmark.stream_builder import StochasticStreamSpec, build_stochastic_stream
from scripts.creator.tool_disposition_benchmark.run_stream_session import slots_to_problems, CLAUDE
from scripts.creator.tool_disposition_benchmark.skirental_scorer import actions_from_session, model_builds_from_actions
from scripts.creator.tool_disposition_benchmark.family_kit import set_profile

UNIFORM = ["lcg", "modpow", "continued_frac", "crt_solve", "josephus", "quadratic_map_mod",
           "xorshift_steps", "matrix_power_mod"]
N, T, B, G = len(UNIFORM), 60, 3, 1.0

_ap = argparse.ArgumentParser()
_ap.add_argument("--model", default="haiku", choices=["haiku", "sonnet", "opus"])
MODEL_KEY = _ap.parse_known_args()[0].model
BASE = Path("runs/arm_a1_announce" if MODEL_KEY == "haiku" else f"runs/arm_a1_announce_{MODEL_KEY}")
SEEDS = list(range(2000, 2012))          # 12 -- paired with the urn (urn_haiku / urn_opus, same seeds)
CAP_USD, EST, CONC = (12.0, 1.0, 12) if MODEL_KEY == "haiku" else (12.0, 2.5, 8)
_PRICES = {"haiku": (1.0, 5.0, 0.10, 1.25), "sonnet": (3.0, 15.0, 0.30, 3.75),
           "opus": (5.0, 25.0, 0.50, 6.25)}       # $/1e6 (in, out, cache_read, cache_write)
IN, OUT, CR, CW = _PRICES[MODEL_KEY]
# MAG=100 is only hand-hard for Haiku; at MAG=1000 crt_solve/modpow collapse to a_hand=0.00 for Opus
# too (calibrated 2026-07-03), but josephus stays hand-solvable (~0.5-0.6) at ANY magnitude for Opus
# (no closed form for general K; Opus tracks the O(N) recurrence reliably regardless of length) --
# so for Opus it is PINNED as a single forced trap at the final slot instead: by T-1 there are 0
# remaining draws, so building never pays off and it cannot influence any earlier build decision.
MAG = 100 if MODEL_KEY == "haiku" else 1000
PINNED_TRAP = None if MODEL_KEY == "haiku" else "josephus"


def cost_of(row):
    tu = row.get("turn_usages") or []
    return (sum(t.get("input_tokens", 0) for t in tu) * IN + sum(t.get("output_tokens", 0) for t in tu) * OUT
            + sum(t.get("cache_read_tokens", 0) for t in tu) * CR + sum(t.get("cache_write_tokens", 0) for t in tu) * CW) / 1e6


async def run_one(client, model, seed):
    d = BASE / f"seed_{seed}"
    if (d / "sessions.jsonl").exists():
        return 0.0, "cached"
    slots, meta = build_stochastic_stream(StochasticStreamSpec(
        families=UNIFORM, n_hot=B, T=T, budget=B, guarantee_trap_early=G, magnitude=MAG, seed=seed,
        pinned_last_trap=PINNED_TRAP))
    d.mkdir(parents=True, exist_ok=True)
    (d / "stream.json").write_text(json.dumps(slots, indent=2))
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    (d / "config.json").write_text(json.dumps({"budget": B, "seed": seed, "magnitude": MAG, "arm": "announce"}))
    state = SessionState(problems=slots_to_problems(slots), budget=B)
    state.announce_recurrence = True          # A1: disclose recurrence structure (non-prescriptive)
    row = await run_session(client, model, state, token_cap=300_000, max_tokens=4096,
                            announce_cap=True, stop_on_budget_exhausted=False)
    row["model_key"] = MODEL_KEY
    (d / "sessions.jsonl").write_text(json.dumps(row) + "\n")
    return cost_of(row), "ran"


async def main():
    load_dotenv(); set_profile(MODEL_KEY)
    model = CLAUDE[MODEL_KEY]; BASE.mkdir(parents=True, exist_ok=True)
    client = RawChat()
    cumulative = 0.0; inflight = 0; idx = 0; paused = False; lock = asyncio.Lock()

    async def worker():
        nonlocal cumulative, inflight, idx, paused
        while True:
            async with lock:
                if paused or idx >= len(SEEDS):
                    return
                d = BASE / f"seed_{SEEDS[idx]}"
                will_run = not (d / "sessions.jsonl").exists()
                if will_run and cumulative + (inflight + 1) * EST > CAP_USD:
                    paused = True; return
                seed = SEEDS[idx]; idx += 1; inflight += 1
            try:
                cost, status = await run_one(client, model, seed)
            except Exception as e:
                cost, status = 0.0, f"ERR:{type(e).__name__}"
            async with lock:
                inflight -= 1; cumulative += cost
                print(f"  [seed {seed}] {status:>6}  ${cost:.3f}  cumulative=${cumulative:.2f}", flush=True)

    print(f"A1 announce arm: {MODEL_KEY}, uniform-hard N={N}, MAG={MAG}, pinned_trap={PINNED_TRAP}, "
          f"g={G}, {len(SEEDS)} seeds (cap=${CAP_USD}) ...", flush=True)
    await asyncio.gather(*(worker() for _ in range(CONC)))
    print(f"\n==== {'PAUSED' if paused else 'COMPLETED'}: A1 spend ${cumulative:.2f} ====", flush=True)

    # ---- fidelity report (the primary metric) ----
    lateness, first_sight, nb, match, nseed = [], 0, 0, 0, 0
    for seed in SEEDS:
        d = BASE / f"seed_{seed}"
        if not (d / "sessions.jsonl").exists():
            continue
        slots = json.loads((d / "stream.json").read_text())
        sess = json.loads((d / "sessions.jsonl").read_text().splitlines()[0])
        acts = actions_from_session(sess, slots)
        mb = {c: b for c, b in model_builds_from_actions(acts).items() if b is not None}
        firstseen, seen = [], set()
        for s in sorted(slots, key=lambda z: z["slot_index"]):
            if s["class_id"] not in seen:
                seen.add(s["class_id"]); firstseen.append(s["class_id"])
        built = set(mb)
        for c, b in mb.items():
            lateness.append(b - 1); nb += 1; first_sight += (b == 1)
        if built and built == set(firstseen[:len(built)]):
            match += 1
        nseed += 1
    print(f"\n==== A1 FIDELITY (n={nseed} seeds, {nb} builds) ====")
    if nb:
        print(f"  builds at FIRST SIGHT (lateness 0): {first_sight}/{nb} = {first_sight/nb:.0%}")
        print(f"  mean lateness = {st.mean(lateness):.3f}  max = {max(lateness)}")
        print(f"  built-set == first-B-distinct-arrivals: {match}/{nseed} = {match/nseed:.0%}")
        print(f"  builds/seed = {nb/nseed:.2f}")
        print(f"\n  => lateness ~0 CONFIRMS open-loop under disclosure; lateness>0 would mean it "
              f"closes the loop when told.")


if __name__ == "__main__":
    asyncio.run(main())
