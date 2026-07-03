"""Haiku PoC (frontier anchor) with a HARD spend guard.

g=1.0 headline seeds first (27 new; the 3 dry-run seeds are folded in at scoring -> 30 total), then a
g=0.0 natural-rate control (10). Full sessions. Scored with the EXACT-DP reference (built once,
reused). Dispatch stops before projected spend would exceed CAP_USD, lets in-flight finish, writes
what completed, and HOLDS. Resumable: seeds with an existing sessions.jsonl are skipped (cost 0).

  PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.poc_haiku
"""
from __future__ import annotations
import asyncio, json, statistics as st
from pathlib import Path
from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.creator.tool_disposition_benchmark.driver import run_session
from scripts.creator.tool_disposition_benchmark.session_state import SessionState
from scripts.creator.tool_disposition_benchmark.stream_builder import StochasticStreamSpec, build_stochastic_stream
from scripts.creator.tool_disposition_benchmark.run_stream_session import slots_to_problems, CLAUDE
from scripts.creator.tool_disposition_benchmark.skirental_scorer import (
    score_run, costs_from_a0, exact_pistar_report, actions_from_session, model_builds_from_actions)
from scripts.creator.tool_disposition_benchmark.exact_dp import ExactDP
from scripts.creator.tool_disposition_benchmark.family_kit import set_profile

POOL = ["lcg", "modpow", "factorial_mod", "kaprekar_routine", "look_and_say", "continued_frac",
        "crt_solve", "josephus", "quadratic_map_mod", "xorshift_steps", "matrix_power_mod", "linrec_mod"]
A0_DIR = "runs/a0_haiku_merged"
N, T, B, CAP, MAG = 12, 60, 3, 3, 100
BASE = Path("runs/poc_haiku")
DRYRUN_G1 = [Path(f"runs/dryrun_stochastic_haiku/seed_{s}") for s in (0, 1, 2)]  # reuse (already paid)

CAP_USD = 27.4          # this run's ceiling ($2.61 already spent today -> total < $30)
EST = 1.5               # conservative per-session estimate for the projected-spend guard
CONC = 12               # sessions are independent; per-session time is fixed (~sequential turns)
IN, OUT, CR, CW = 1.0, 5.0, 0.10, 1.25   # Haiku 4.5 $/1M


def cost_of(row):
    tu = row.get("turn_usages") or []
    return (sum(t.get("input_tokens", 0) for t in tu) * IN
            + sum(t.get("output_tokens", 0) for t in tu) * OUT
            + sum(t.get("cache_read_tokens", 0) for t in tu) * CR
            + sum(t.get("cache_write_tokens", 0) for t in tu) * CW) / 1e6


async def run_one(client, model, seed, g):
    d = BASE / f"g{int(g)}_seed_{seed}"
    if (d / "sessions.jsonl").exists():
        return d, 0.0, "cached"
    spec = StochasticStreamSpec(families=POOL, n_hot=B, hot_share=0.85, trap_share=0.15, T=T,
                                budget=B, guarantee_trap_early=g, magnitude=MAG, seed=seed)
    slots, meta = build_stochastic_stream(spec)
    d.mkdir(parents=True, exist_ok=True)
    (d / "stream.json").write_text(json.dumps(slots, indent=2))
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    (d / "config.json").write_text(json.dumps({"budget": B, "seed": seed, "magnitude": MAG}))
    state = SessionState(problems=slots_to_problems(slots), budget=B)
    row = await run_session(client, model, state, token_cap=300_000, max_tokens=4096,
                            stop_on_budget_exhausted=False)
    row["model_key"] = "haiku"
    (d / "sessions.jsonl").write_text(json.dumps(row) + "\n")
    return d, cost_of(row), "ran"


async def main():
    load_dotenv()
    set_profile("haiku")
    model = CLAUDE["haiku"]
    BASE.mkdir(parents=True, exist_ok=True)
    client = RawChat()

    tasks = [(0.0, s) for s in range(1000, 1010)]   # g=0 natural-rate control (fidelity check)
    cumulative = 0.0
    inflight = 0
    idx = 0
    paused = False
    lock = asyncio.Lock()

    async def worker():
        nonlocal cumulative, inflight, idx, paused
        while True:
            async with lock:
                if paused or idx >= len(tasks):
                    return
                # cached seeds cost 0; only guard against seeds we'd actually run
                d = BASE / f"g{int(tasks[idx][0])}_seed_{tasks[idx][1]}"
                will_run = not (d / "sessions.jsonl").exists()
                if will_run and cumulative + (inflight + 1) * EST > CAP_USD:
                    paused = True
                    return
                g, seed = tasks[idx]; idx += 1; inflight += 1
            try:
                _, cost, status = await run_one(client, model, seed, g)
            except Exception as e:
                cost, status = 0.0, f"ERR:{type(e).__name__}"
            async with lock:
                inflight -= 1
                cumulative += cost
                print(f"  [g{int(g)} seed {seed}] {status:>6}  ${cost:.3f}   cumulative=${cumulative:.2f}",
                      flush=True)

    print(f"launching (cap=${CAP_USD}, conc={CONC}) ...", flush=True)
    await asyncio.gather(*(worker() for _ in range(CONC)))

    status = "PAUSED at guard" if paused else "COMPLETED all planned seeds"
    print(f"\n==== run {status}: this-run spend ${cumulative:.2f} "
          f"(+ $2.61 earlier today = ${cumulative + 2.61:.2f}) ====", flush=True)

    # ---- score everything completed, with the exact DP built ONCE ----
    costs = costs_from_a0(A0_DIR, "haiku", MAG)
    a_repr = st.mean(costs.ah(f) for f in POOL)
    dp = ExactDP(costs.R * a_repr - costs.lam * costs.h, costs.u_build(), costs.u_reuse(),
                 N, T, B, alpha=1.0, cap=CAP)

    def score_arm(dirs, label):
        regs, mtr, ptr, pos, reuse = [], [], [], 0, []
        for d in dirs:
            if not (d / "sessions.jsonl").exists():
                continue
            slots = json.loads((d / "stream.json").read_text())
            sess = json.loads((d / "sessions.jsonl").read_text().splitlines()[0])
            role = {s["class_id"]: s["role"] for s in slots}
            acts = actions_from_session(sess, slots)
            mb = model_builds_from_actions(acts)
            rep = exact_pistar_report(slots, costs, B, N, T, POOL, mb, dp=dp)
            regs.append(rep["regret"]); mtr.append(rep["model_traps_built"]); ptr.append(rep["pistar_traps_built"])
            pos += rep["regret"] > 0
            built = [c for c in {a["class_id"] for a in acts if a["action"] in ("build", "rebuild")}]
            # empirical reuse: recurring built classes that got >=1 reuse
            by = {}
            for a in acts:
                by.setdefault(a["class_id"], []).append(a)
            rec_built = [c for c in built if len(by[c]) >= 2]
            reused = [c for c in rec_built if any(a["action"] == "reuse" for a in by[c])]
            if rec_built:
                reuse.append(len(reused) / len(rec_built))
        if not regs:
            print(f"  {label}: no completed seeds"); return
        se = (st.stdev(regs) / len(regs) ** 0.5) if len(regs) > 1 else 0.0
        print(f"  {label} (n={len(regs)}): regret mean={st.mean(regs):.0f} +/- {se:.0f}  "
              f"pos={pos}/{len(regs)}  model_traps/seed={st.mean(mtr):.2f}  pi*_traps/seed={st.mean(ptr):.2f}  "
              f"reuse={st.mean(reuse) if reuse else float('nan'):.2f}")

    print("\n==== EXACT-DP regret ====", flush=True)
    g1_dirs = DRYRUN_G1 + [BASE / f"g1_seed_{s}" for s in range(3, 30)]
    g0_dirs = [BASE / f"g0_seed_{s}" for s in range(1000, 1010)]
    score_arm(g1_dirs, "g=1.0 (consequential, headline)")
    score_arm(g0_dirs, "g=0.0 (natural-rate control)")
    if paused:
        print("\nHOLDING for confirmation to resume the remaining seeds (re-run this script to continue).")


if __name__ == "__main__":
    asyncio.run(main())
