"""Integrate the exact-DP reference (cap=3) and re-score the cached dry-run seeds.

Step 2 (validation): the forward policy's realized value, averaged over belief-sampled (Polya-urn)
streams, must match the DP root value V((),()). Confirms policy_builds implements the DP correctly.

Step 3 (re-score): build the exact DP once (cap=3), run it forward on each cached dry-run stream to
get pi*'s builds, value model vs pi* vs clairvoyant over full realized sizes, and report the exact
same-info regret -- to compare against the retired Whittle numbers (mean regret_lb ~1263 in the dry run).

  PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.rescore_exact
"""
from __future__ import annotations
import json, random, statistics as st
from pathlib import Path

from scripts.creator.tool_disposition_benchmark.skirental_scorer import (
    Costs, costs_from_a0, actions_from_session, model_builds_from_actions)
from scripts.creator.tool_disposition_benchmark.pi_star import value_of_builds, clairvoyant_builds, _class_meta
from scripts.creator.tool_disposition_benchmark.exact_dp import ExactDP

POOL = ["lcg", "modpow", "factorial_mod", "kaprekar_routine", "look_and_say", "continued_frac",
        "crt_solve", "josephus", "quadratic_map_mod", "xorshift_steps", "matrix_power_mod", "linrec_mod"]
A0_DIR = "runs/a0_haiku_merged"
N, T, B, CAP = 12, 60, 3, 3


def _slots_from_seq(seq):
    counts = {}
    for c in seq:
        counts[c] = counts.get(c, 0) + 1
    seen = {}
    slots = []
    for i, c in enumerate(seq):
        seen[c] = seen.get(c, 0) + 1
        slots.append({"slot_index": i, "class_id": c, "family": POOL[c % len(POOL)],
                      "class_size": counts[c], "class_position": seen[c]})
    return slots


def validate_forward(dp, uh, ub, ur, alpha, a_repr, costs, n_mc=4000, seed=0):
    """E[realized forward value over belief-sampled streams] ?= root_value()."""
    scalar = Costs(a_hand={}, h=costs.h, C=costs.C, r=costs.r, R=costs.R, lam=costs.lam,
                   default_a_hand=a_repr)
    assert abs(scalar.u_hand("x") - uh) < 1e-6 and abs(scalar.u_build() - ub) < 1e-6
    rng = random.Random(seed)
    vals = []
    for _ in range(n_mc):
        counts = [0] * N
        seq = []
        for s in range(T):
            w = [alpha + counts[i] for i in range(N)]
            tot = sum(w); r = rng.random() * tot; acc = 0.0
            for i in range(N):
                acc += w[i]
                if r <= acc:
                    counts[i] += 1; seq.append(i); break
        slots = _slots_from_seq(seq)
        builds = dp.policy_builds(slots)
        vals.append(value_of_builds(slots, builds, scalar))
    mc = st.mean(vals); se = st.stdev(vals) / (n_mc ** 0.5)
    rv = dp.root_value()
    print(f"  forward MC value = {mc:.2f} +/- {se:.2f} (n={n_mc})   root_value = {rv:.2f}   "
          f"|diff| = {abs(mc - rv):.2f}  ({'OK' if abs(mc - rv) < 3 * se + 0.5 else 'MISMATCH'})")


def main():
    costs = costs_from_a0(A0_DIR, "haiku", 100)
    a_repr = st.mean(costs.ah(f) for f in POOL)          # representative a_hand for the DP decisions
    uh = costs.R * a_repr - costs.lam * costs.h
    ub, ur = costs.u_build(), costs.u_reuse()
    print(f"representative a_hand (pool mean) = {a_repr:.3f}  ->  u_hand={uh:.1f} u_build={ub:.1f} u_reuse={ur:.1f}")
    print(f"building exact DP once: N={N} T={T} B={B} cap={CAP} ...", flush=True)
    dp = ExactDP(uh, ub, ur, N, T, B, alpha=1.0, cap=CAP)

    print("\n=== step 2: forward-pass validation (belief-MC value vs root_value) ===", flush=True)
    validate_forward(dp, uh, ub, ur, 1.0, a_repr, costs)

    print("\n=== step 3: re-score cached dry-run seeds with the EXACT reference ===", flush=True)
    base = Path("runs/dryrun_stochastic_haiku")
    regrets, mtraps, ptraps = [], [], []
    for d in sorted(base.glob("seed_*")):
        slots = json.loads((d / "stream.json").read_text())
        meta = json.loads((d / "meta.json").read_text())
        sess = json.loads((d / "sessions.jsonl").read_text().splitlines()[0])
        role = {s["class_id"]: s["role"] for s in slots}
        model_b = model_builds_from_actions(actions_from_session(sess, slots))
        pi_b = dp.policy_builds(slots)
        v_model = value_of_builds(slots, model_b, costs)
        v_star = value_of_builds(slots, pi_b, costs)
        v_clair = value_of_builds(slots, clairvoyant_builds(slots, B), costs)
        reg = v_star - v_model
        mt = sum(1 for c, b in model_b.items() if b is not None and role.get(c) == "trap")
        pt = sum(1 for c, b in pi_b.items() if b is not None and role.get(c) == "trap")
        pl = [b - 1 for b in pi_b.values() if b is not None]
        regrets.append(reg); mtraps.append(mt); ptraps.append(pt)
        print(f"  {d.name}: model={v_model:8.1f}  pi*={v_star:8.1f}  clair={v_clair:8.1f}  "
              f"REGRET={reg:8.1f}  (model traps={mt} pi* traps={pt} pi* lateness={st.mean(pl) if pl else 0:.2f})")

    print(f"\n  EXACT regret (pi* - model): mean={st.mean(regrets):.1f}  -> {[round(r) for r in regrets]}")
    print(f"  model traps/seed={st.mean(mtraps):.2f}  pi* traps/seed={st.mean(ptraps):.2f}")
    print(f"  (retired Whittle pi* dry-run regret mean was ~1263; exact is stronger -> expect larger)")


if __name__ == "__main__":
    main()
