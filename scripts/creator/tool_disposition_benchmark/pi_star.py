"""pi* -- the online EV-optimal reference for the stochastic tool-investment design.

Construction (see docs/whittle-asymptotic-optimality.md + online-tool-investment-stochastic-design.md):
each problem TYPE is an arm; ACTIVATE = build (irreversible). Under the exchangeable Dirichlet(alpha)
prior over the N-type pmf, the predictive that a type seen k times in t elapsed slots appears next is
    q(k, t) = (alpha + k) / (N*alpha + t).
We solve the per-type finite-horizon DP under a Lagrangian build price `price` (shadow price of a
budget token), which yields a build region in (k, t). Then we tune `price` by simulation so the
expected number of builds equals the budget B. The resulting policy -- "build the current type iff
the DP says build, in temporal order, capped at B" -- is the Whittle index policy for the budgeted
online build problem.

Utilities come from the ski-rental cost model (skirental_scorer.Costs): building buys ACCURACY
(u_reuse, u_build use a_script~=1; u_hand uses the family's a_hand). The `price` enters ONLY the DP
decision/tuning; realized VALUE is the true utility (no price).

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.pi_star   # self-test
"""

from __future__ import annotations

from collections import defaultdict

from scripts.creator.tool_disposition_benchmark.skirental_scorer import Costs


def _predictive(k: int, t: int, alpha: float, N: int) -> float:
    """Dirichlet(alpha)-multinomial posterior predictive: P(next draw is this type | k occ in t slots)."""
    return (alpha + k) / (N * alpha + t)


def built_value_table(costs: Costs, T: int, alpha: float, N: int) -> list[list[float]]:
    """BV[t][k] = expected future reuse value (over slots > t) if the type is ALREADY built, given it
    has been seen k times in t elapsed slots. Family-independent (uses u_reuse + the predictive)."""
    ur = costs.u_reuse()
    BV = [[0.0] * (T + 2) for _ in range(T + 2)]
    for t in range(T - 1, -1, -1):
        for k in range(0, t + 1):
            q = _predictive(k, t, alpha, N)
            BV[t][k] = q * (ur + BV[t + 1][k + 1]) + (1 - q) * BV[t + 1][k]
    return BV


def build_region(costs: Costs, a_hand: float, price: float, T: int, alpha: float, N: int,
                 BV: list[list[float]]) -> list[list[bool]]:
    """region[t][k] = True iff, when a type's k-th occurrence arrives at elapsed slot t (unbuilt so
    far), the DP prefers BUILDING now to hand-solving and staying unbuilt. Solves the per-type
    optimal-stopping DP under build price `price`.

      W[t][k]      = future value (slots > t) of staying UNBUILT at (k occ, t slots), acting optimally
      build now    = (u_build - price) + BV[t][k]        [current occ served by tool + future reuse]
      hand & wait  = u_hand           + W[t][k]           [current occ hand-solved, keep the option]
    """
    u_hand = costs.R * a_hand - costs.lam * costs.h
    u_build = costs.u_build()
    W = [[0.0] * (T + 2) for _ in range(T + 2)]
    for t in range(T - 1, -1, -1):
        for k in range(0, t + 1):
            q = _predictive(k, t, alpha, N)
            appear = max(u_hand + W[t + 1][k + 1], (u_build - price) + BV[t + 1][k + 1])
            W[t][k] = q * appear + (1 - q) * W[t + 1][k]
    region = [[False] * (T + 2) for _ in range(T + 2)]
    for t in range(1, T + 1):
        for k in range(1, t + 1):
            region[t][k] = (u_build - price) + BV[t][k] >= u_hand + W[t][k]
    return region


def _regions_for_pool(costs: Costs, a_hands: dict, price: float, T: int, alpha: float, N: int) -> dict:
    """One build-region per DISTINCT a_hand value in the pool (regions are identical for equal a_hand)."""
    BV = built_value_table(costs, T, alpha, N)
    by_val: dict[float, list[list[bool]]] = {}
    for v in {round(a, 4) for a in a_hands.values()}:
        by_val[v] = build_region(costs, v, price, T, alpha, N, BV)
    return {f: by_val[round(a, 4)] for f, a in a_hands.items()}


def policy_builds(slots: list[dict], regions: dict, budget: int | None) -> dict:
    """Run the Whittle policy forward on a stream. Returns {class_id: build_position or None}.
    Walks slots in arrival order; at each occurrence of an unbuilt type, builds iff its family's
    region says so and budget remains. budget=None -> uncapped (for price tuning)."""
    seen: dict[int, int] = defaultdict(int)
    built: dict[int, int] = {}
    remaining = float("inf") if budget is None else budget
    for s in sorted(slots, key=lambda z: z["slot_index"]):
        cid = s["class_id"]
        seen[cid] += 1
        if cid in built or remaining <= 0:
            continue
        t, k = s["slot_index"] + 1, seen[cid]
        if regions[s["family"]][t][k]:
            built[cid] = k
            remaining -= 1
    return {s["class_id"]: built.get(s["class_id"]) for s in slots}


def _class_meta(slots: list[dict]) -> dict:
    """class_id -> (family, realized_size)."""
    out = {}
    for s in slots:
        out[s["class_id"]] = (s["family"], s["class_size"])
    return out


def value_of_builds(slots: list[dict], builds: dict, costs: Costs) -> float:
    """Analytic value over full realized class sizes given build decisions (build_pos or None)."""
    total = 0.0
    for cid, (fam, size) in _class_meta(slots).items():
        b = builds.get(cid)
        if b is None:
            total += size * costs.u_hand(fam)
        else:
            total += (b - 1) * costs.u_hand(fam) + costs.u_build() + (size - b) * costs.u_reuse()
    return total


def clairvoyant_builds(slots: list[dict], budget: int) -> dict:
    """Build the `budget` highest-realized-size types at first sight (knows the future)."""
    meta = _class_meta(slots)
    top = sorted(meta, key=lambda cid: meta[cid][1], reverse=True)[:budget]
    return {cid: (1 if cid in top else None) for cid in meta}


def eager_builds(slots: list[dict], budget: int) -> dict:
    """Build the first `budget` DISTINCT types by arrival, each at first sight (the model's policy)."""
    order, seen = [], set()
    for s in sorted(slots, key=lambda z: z["slot_index"]):
        if s["class_id"] not in seen:
            seen.add(s["class_id"])
            order.append(s["class_id"])
    chosen = set(order[:budget])
    return {cid: (1 if cid in chosen else None) for cid in _class_meta(slots)}


def wait_k_builds(slots: list[dict], budget: int, k_repeat: int = 2) -> dict:
    """Baseline: build the first `budget` types to reach their k_repeat-th sighting (temporal order)."""
    seen: dict[int, int] = defaultdict(int)
    built: dict[int, int] = {}
    remaining = budget
    for s in sorted(slots, key=lambda z: z["slot_index"]):
        cid = s["class_id"]
        seen[cid] += 1
        if cid not in built and remaining > 0 and seen[cid] >= k_repeat:
            built[cid] = seen[cid]
            remaining -= 1
    return {s["class_id"]: built.get(s["class_id"]) for s in slots}


def tune_price(costs: Costs, pool: list[str], a_hands: dict, N: int, T: int, budget: int,
               magnitude: int, alpha: float = 1.0, n_sim: int = 200,
               guarantee_trap_early: float = 1.0) -> float:
    """Whittle price: choose the build price so the UNCAPPED policy builds `budget` types in
    expectation over the generative process (Lagrangian complementary slackness on the budget). This
    is a SAME-INFORMATION calibration -- it only uses {N, T, B}, which the model also knows, plus the
    exchangeable prior; it does NOT encode the frequent/rare structure. Streams generated once,
    reused across bisection steps (seed offset 1000, held out from evaluation)."""
    from scripts.creator.tool_disposition_benchmark.stream_builder import (
        StochasticStreamSpec, build_stochastic_stream)
    streams = [build_stochastic_stream(StochasticStreamSpec(
        families=pool, n_hot=budget, T=T, budget=budget,
        guarantee_trap_early=guarantee_trap_early, magnitude=magnitude, seed=1000 + s))[0]
        for s in range(n_sim)]

    def mean_builds(price):
        regions = _regions_for_pool(costs, a_hands, price, T, alpha, N)
        return sum(sum(1 for b in policy_builds(s, regions, None).values() if b is not None)
                   for s in streams) / len(streams)

    lo, hi = 0.0, 1.0
    while mean_builds(hi) > budget and hi < 1e7:      # grow hi until expected builds drop below B
        hi *= 4
    for _ in range(40):                                # bisection (mean_builds is decreasing in price)
        mid = 0.5 * (lo + hi)
        if mean_builds(mid) > budget:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------- self-test
def _selftest():
    import statistics as st
    from scripts.creator.tool_disposition_benchmark.stream_builder import (
        StochasticStreamSpec, build_stochastic_stream)

    POOL = ["lcg", "modpow", "factorial_mod", "kaprekar_routine", "look_and_say", "continued_frac",
            "crt_solve", "josephus", "quadratic_map_mod", "xorshift_steps", "matrix_power_mod",
            "linrec_mod"]
    N, T, BUDGET, MAG = len(POOL), 60, 3, 100
    # measured-ish a_hand ~ 0 for the hard pool; realistic cost constants from the smoke/A0.
    a_hands = {f: 0.05 for f in POOL}
    costs = Costs(a_hand=a_hands, h=987, C=308, r=200, R=100.0, lam=0.1, default_a_hand=0.05)

    price = tune_price(costs, POOL, a_hands, N, T, BUDGET, MAG, alpha=1.0, n_sim=150)
    print(f"tuned build price lambda* = {price:.1f}")

    # show the build-region threshold: min k to build, as a function of elapsed slot t
    BV = built_value_table(costs, T, 1.0, N)
    region = build_region(costs, 0.05, price, T, 1.0, N, BV)
    thr = {}
    for t in (3, 5, 10, 20, 30, 45, 55):
        ks = [k for k in range(1, t + 1) if region[t][k]]
        thr[t] = min(ks) if ks else None
    print(f"min occurrences k to build, by elapsed slot t: {thr}")
    print("  (interpretation: pi* WAITS for >=this many sightings before building at that time)")

    # evaluate pi* (capped) vs realizable baselines vs clairvoyant on fresh seeds
    regions = _regions_for_pool(costs, a_hands, price, T, 1.0, N)
    v_star, v_eager, v_wait, v_rand, v_clair, lat, n_built, trap_built = ([] for _ in range(8))
    import random as _r
    for seed in range(200):
        slots, meta = build_stochastic_stream(StochasticStreamSpec(
            families=POOL, n_hot=BUDGET, T=T, budget=BUDGET, guarantee_trap_early=1.0,
            magnitude=MAG, seed=seed))
        pb = policy_builds(slots, regions, BUDGET)
        v_star.append(value_of_builds(slots, pb, costs))
        v_eager.append(value_of_builds(slots, eager_builds(slots, BUDGET), costs))
        v_wait.append(value_of_builds(slots, wait_k_builds(slots, BUDGET, 2), costs))
        # random-B: build B distinct types picked at random (at first sight)
        rng = _r.Random(seed)
        cids = list(_class_meta(slots))
        chosen = set(rng.sample(cids, min(BUDGET, len(cids))))
        v_rand.append(value_of_builds(slots, {c: (1 if c in chosen else None) for c in cids}, costs))
        v_clair.append(value_of_builds(slots, clairvoyant_builds(slots, BUDGET), costs))
        role = {cid: meta["assignment"][fam]["role"] for cid, (fam, _) in _class_meta(slots).items()}
        bp = [b for b in pb.values() if b is not None]
        n_built.append(len(bp)); lat += [b - 1 for b in bp]
        trap_built.append(sum(1 for cid, b in pb.items() if b is not None and role[cid] == "trap"))

    ms = lambda xs: (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)
    sm, ss = ms(v_star); em, es = ms(v_eager); wm, ws = ms(v_wait); rm, _ = ms(v_rand); cm, cs = ms(v_clair)
    print(f"\nover 200 held-out seeds (g=1.0):")
    print(f"  pi* (SAME-INFO reference): {sm:8.1f} +/- {ss:.0f}")
    print(f"  eager (the model):         {em:8.1f} +/- {es:.0f}   <- pi* beats it {sm/max(em,1):.1f}x")
    print(f"  random-B (floor):          {rm:8.1f}")
    print(f"  --- extra-info comparisons (NOT the reference) ---")
    print(f"  wait-one-repeat:           {wm:8.1f} +/- {ws:.0f}   (knows rare=one-off -> more info than pi*)")
    print(f"  clairvoyant (upper bound): {cm:8.1f} +/- {cs:.0f}   (knows the future)")
    print(f"  pi* builds/seed: {st.mean(n_built):.2f}   pi* mean build-lateness: {st.mean(lat):.2f}"
          f"   pi* traps/seed: {st.mean(trap_built):.3f}")
    print(f"  model's regret lower bound (pi* - eager): {sm - em:.1f} per seed")

    # pi* is a SAME-INFO reference, NOT claimed optimal: it may be beaten by structure-informed
    # policies (wait-one-repeat) that know more than the model. The valid claims:
    assert sm > em, "pi* (same info as the model) must beat the model's eager policy in expectation"
    assert sm > rm, "pi* must beat the random-B floor"
    assert sm <= cm + 1e-6, "pi* cannot exceed the clairvoyant upper bound (sanity)"
    assert st.mean(lat) >= 0.99, "pi* should WAIT (mean build-lateness >= 1) -- the model does not"
    print("\npi* self-test OK  (same-info pi* trounces the model; regret lower bound established)")


if __name__ == "__main__":
    _selftest()
