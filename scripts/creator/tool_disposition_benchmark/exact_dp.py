"""Exact finite-horizon same-information optimum for the online tool-investment benchmark.

The true Bayes-optimal policy for the budgeted build problem, computed by DP over the learner's
belief state -- NO relaxation, NO N->infty assumption. Supersedes the Whittle construction in
pi_star.py (see docs/same-info-optimal-dp.md).

Belief: symmetric Dirichlet(alpha) over the N-type pmf; predictive p_i = (alpha+c_i)/(N*alpha+s).
State (canonical, exploiting type-exchangeability): the MULTISET of counts of built types + the
MULTISET of counts of unbuilt-seen types. From these, s = sum(all counts) and budget = B - #built are
derived; the number of unseen types is U = N - #built - #unbuilt_seen.

Bellman: V(built, unbuilt) = sum over the arriving type of p * Q, where for a built type Q = u_reuse +
V(...), and for an unbuilt/unseen type Q = max(hand, build) (build only if budget>0). Terminal s==T:
0. Base case budget==0: CLOSED FORM -- with no builds left there are no decisions, and by the Polya-urn
exchangeability property every future draw of a type currently at count c has marginal probability
(alpha+c)/(N*alpha+s), so the remaining value is (T-s)/(N*alpha+s) * [u_reuse*built_mass +
u_hand*(rest)]. This terminates the recursion after at most B build decisions (huge pruning).

Utilities are SCALAR (uniform a_hand): the same-info reference treats types exchangeably, so it uses a
single representative a_hand (default: the pool mean). Realized VALUE of a policy's builds is scored
separately with per-family utilities (skirental_scorer.value_of_builds).

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.exact_dp   # self-test + validation
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict

from scripts.creator.tool_disposition_benchmark.skirental_scorer import Costs


def _replace(t: tuple, old: int, new: int) -> tuple:
    """Return multiset `t` with one occurrence of `old` replaced by `new` (kept sorted)."""
    lst = list(t)
    lst.remove(old)
    lst.append(new)
    return tuple(sorted(lst))


def _add(t: tuple, x: int) -> tuple:
    return tuple(sorted(t + (x,)))


def _remove(t: tuple, x: int) -> tuple:
    lst = list(t)
    lst.remove(x)
    return tuple(sorted(lst))


class ExactDP:
    """Exact belief-state DP. Utilities are scalar (uniform a_hand)."""

    def __init__(self, u_hand: float, u_build: float, u_reuse: float,
                 N: int, T: int, B: int, alpha: float = 1.0, cap: int | None = None):
        self.uh, self.ub, self.ur = u_hand, u_build, u_reuse
        self.N, self.T, self.B, self.alpha = N, T, B, alpha
        # cap: if set, an unbuilt type at count `cap` that arrives again is FORCED to build (its count
        # never exceeds cap while unbuilt). This bounds the unbuilt partition to parts <= cap ->
        # controls the state blow-up. V(cap) is a FEASIBLE same-info policy (build-any-type-recurring-
        # cap-times), hence a valid LOWER BOUND on the true optimum, monotone non-decreasing in cap;
        # cap=None is the exact optimum. See docs/same-info-optimal-dp.md.
        self.cap = cap
        self._cache: dict[tuple, float] = {}
        sys.setrecursionlimit(1_000_000)

    # ---- utilities ------------------------------------------------------
    def _base_budget0(self, built: tuple, unbuilt: tuple, s: int) -> float:
        """Closed-form value from a state with NO builds left (budget==0). No decisions remain; by the
        Polya-urn martingale property each of the (T-s) future draws lands on a built type with total
        probability built_mass/(N*alpha+s), earning u_reuse, else earns u_hand."""
        Na_s = self.N * self.alpha + s
        built_mass = len(built) * self.alpha + sum(built)     # sum_{built}(alpha + c_j)
        rem = self.T - s
        return rem / Na_s * (self.ur * built_mass + self.uh * (Na_s - built_mass))

    def V(self, built: tuple, unbuilt: tuple) -> float:
        key = (built, unbuilt)
        v = self._cache.get(key)
        if v is not None:
            return v
        s = sum(built) + sum(unbuilt)
        if s >= self.T:
            self._cache[key] = 0.0
            return 0.0
        budget = self.B - len(built)
        if budget <= 0:
            v = self._base_budget0(built, unbuilt, s)
            self._cache[key] = v
            return v

        Na_s = self.N * self.alpha + s
        U = self.N - len(built) - len(unbuilt)          # unseen types (count 0)
        total = 0.0

        # arriving type is a BUILT type of count cval -> reuse
        for cval, mult in Counter(built).items():
            p = (self.alpha + cval) / Na_s
            total += mult * p * (self.ur + self.V(_replace(built, cval, cval + 1), unbuilt))

        # arriving type is an UNBUILT-SEEN type of count cval -> max(hand, build)
        for cval, mult in Counter(unbuilt).items():
            p = (self.alpha + cval) / Na_s
            v_build = self.ub + self.V(_add(built, cval + 1), _remove(unbuilt, cval))  # budget>0 here
            if self.cap is not None and cval >= self.cap:      # count would exceed cap -> force build
                total += mult * p * v_build
            else:
                v_hand = self.uh + self.V(built, _replace(unbuilt, cval, cval + 1))
                total += mult * p * max(v_hand, v_build)

        # arriving type is UNSEEN (count 0 -> 1) -> max(hand, build); U identical copies
        if U > 0:
            p0 = self.alpha / Na_s
            v_hand = self.uh + self.V(built, _add(unbuilt, 1))
            v_build = self.ub + self.V(_add(built, 1), unbuilt)
            total += U * p0 * max(v_hand, v_build)

        self._cache[key] = total
        return total

    def root_value(self) -> float:
        return self.V((), ())

    # ---- run the optimal policy forward on a realized stream ------------
    def policy_builds(self, slots: list[dict]) -> dict:
        """Play the exact-optimal policy forward on a realized stream. Returns {class_id: build_position
        (1-based occurrence at which it built) or None}. Decisions use the belief DP; the arriving
        type's identity is known, so at each slot we compare the hand vs build successor states."""
        counts: dict[int, int] = defaultdict(int)      # pre-arrival counts per class
        built: dict[int, int] = {}                     # class_id -> build position
        for s in sorted(slots, key=lambda z: z["slot_index"]):
            cid = s["class_id"]
            pre = counts[cid]
            if cid in built:                            # tool exists -> reuse, no decision
                counts[cid] = pre + 1
                continue
            budget = self.B - len(built)
            if budget <= 0:                             # no builds left -> hand
                counts[cid] = pre + 1
                continue
            if self.cap is not None and pre >= self.cap:   # count would exceed cap -> forced build
                built[cid] = pre + 1
                counts[cid] = pre + 1
                continue
            # canonical pre-arrival multisets
            built_ms = tuple(sorted(counts[c] for c in built))
            unbuilt_ms = tuple(sorted(counts[c] for c in counts if c not in built and counts[c] > 0))
            newc = pre + 1
            # HAND successor: this type stays unbuilt, count -> newc
            if pre > 0:
                unbuilt_hand = _replace(unbuilt_ms, pre, newc)
            else:
                unbuilt_hand = _add(unbuilt_ms, 1)      # was unseen
            v_hand = self.uh + self.V(built_ms, unbuilt_hand)
            # BUILD successor: this type becomes built at count newc, removed from unbuilt
            unbuilt_build = _remove(unbuilt_ms, pre) if pre > 0 else unbuilt_ms
            v_build = self.ub + self.V(_add(built_ms, newc), unbuilt_build)
            if v_build > v_hand:
                built[cid] = newc
            counts[cid] = newc
        return {s["class_id"]: built.get(s["class_id"]) for s in slots}


# --------------------------------------------------------------------- brute-force validator
def _bruteforce_root(u_hand, u_build, u_reuse, N, T, B, alpha):
    """Exact vector DP over full count-vectors (no canonicalization). For validating the canonical DP
    on small instances only."""
    from functools import lru_cache
    sys.setrecursionlimit(1_000_000)

    @lru_cache(maxsize=None)
    def V(s, counts, budget, built):
        if s == T:
            return 0.0
        denom = N * alpha + s
        tot = 0.0
        for i in range(N):
            p = (alpha + counts[i]) / denom
            nc = list(counts); nc[i] += 1; nc = tuple(nc)
            if (built >> i) & 1:
                best = u_reuse + V(s + 1, nc, budget, built)
            else:
                best = u_hand + V(s + 1, nc, budget, built)
                if budget > 0:
                    best = max(best, u_build + V(s + 1, nc, budget - 1, built | (1 << i)))
            tot += p * best
        return tot

    return V(0, tuple([0] * N), B, 0)


# --------------------------------------------------------------------- self-test
def _selftest():
    import time
    uh, ub, ur = -98.7, 49.2, 80.0

    # 1) LOSSLESSNESS: canonical root value == brute-force vector DP, several small instances.
    print("=== validation: canonical exact DP  vs  brute-force vector DP (root value) ===")
    for (N, T, B) in [(3, 8, 1), (4, 8, 1), (3, 10, 2), (5, 9, 2), (4, 11, 3), (6, 10, 2)]:
        for alpha in (0.5, 1.0, 2.0):
            dp = ExactDP(uh, ub, ur, N, T, B, alpha)
            got = dp.root_value()
            want = _bruteforce_root(uh, ub, ur, N, T, B, alpha)
            ok = abs(got - want) < 1e-6
            assert ok, f"MISMATCH N={N} T={T} B={B} a={alpha}: canonical={got} brute={want}"
        print(f"  N={N} T={T} B={B}: canonical == brute-force for alpha in (0.5,1,2)  ({got:.3f})")
    print("  lossless: canonical DP reproduces the brute-force optimum exactly.\n")

    # 2) POLICY STRUCTURE: reserve on first sight (s>=1), build on recurrence.
    print("=== optimal-policy structure (N=8,T=14,B=1) ===")
    dp = ExactDP(uh, ub, ur, 8, 14, 1, 1.0)
    for s in (0, 1, 2, 4, 6):
        unbuilt = tuple([1] * s)                         # s distinct singletons, budget free
        # new type arrives (count 0->1): build vs reserve
        v_build = ub + dp.V((1,), unbuilt)
        v_hand = uh + dp.V((), _add(unbuilt, 1))
        tag = "BUILD" if v_build > v_hand else "reserve"
        print(f"  new type at slot s={s}: {tag:>7} (margin {v_build - v_hand:+7.1f})")
    print("  => builds only the opening slot; reserves every later first-sighting.\n")

    # 3) SCALE: root at the real benchmark size.
    print("=== scale probe: N=12, T=60, B=3, alpha=1 ===")
    dp = ExactDP(uh, ub, ur, 12, 60, 3, 1.0)
    t0 = time.time()
    rv = dp.root_value()
    print(f"  root value = {rv:.2f}   states cached = {len(dp._cache):,}   "
          f"time = {time.time() - t0:.1f}s")
    print("\nexact-DP self-test OK")


if __name__ == "__main__":
    _selftest()
