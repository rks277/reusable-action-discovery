"""Coupon-collector economics for the tool-wood-discrepancy experiment.

Grinding (gather only) to collect all N distinct kinds is the classic coupon-collector
problem: E[gathers] = N * H_N, H_N = sum_{k=1..N} 1/k. The budget is 1.2x that.
"""

from __future__ import annotations


def H(n: int) -> float:
    """Nth harmonic number."""
    return sum(1.0 / k for k in range(1, n + 1))


def expected_grind(n_kinds: int) -> float:
    """Expected gathers to collect all N distinct coupons (uniform draws)."""
    return n_kinds * H(n_kinds)


def budget_for_coupon(n_kinds: int, mult: float = 1.2) -> int:
    """Strict total-action cap = round(mult * E[grind]) = round(mult * N * H_N)."""
    return round(mult * expected_grind(n_kinds))


def expected_grind_gated(n_kinds: int) -> float:
    """GATED world grind: coupon-collector over the N KEYS (examine -> random key) plus the
    N opens themselves. Identical to ToolWorld's _grind_cost = N*H_N + N."""
    return n_kinds * H(n_kinds) + n_kinds


def budget_for_gated(n_kinds: int, mult: float = 1.2) -> int:
    """Strict total-action cap for the gated world = round(mult * (N*H_N + N)). Matches
    ToolWorld's budget philosophy exactly, so building stays OPTIONAL (never forced)."""
    return round(mult * expected_grind_gated(n_kinds))


def expected_build(n_kinds: int, n_types: int) -> float:
    """Rough expected actions on the build path (for an OPTIONAL boundary overlay only).
    Phase 1: get 2 sticks -- each gather yields a stick w.p. 1/T, so E[gathers] = 2T,
    during which ~N*(1-(1-1/N)^(2T)) distinct kinds are collected for free.
    Phase 2: recipe search over the look-alike family ~ (C(T,2)+1)/2 attempts + 1 craft.
    Phase 3: each remaining unheld kind via one axe-use -> ~N*(1-1/N)^(2T) uses.
    Approximate; not used unless --boundary is passed to the plotter."""
    if n_kinds <= 1:
        return 1.0
    search = (n_types * (n_types - 1) / 2 + 1) / 2 + 1
    remaining = n_kinds * ((1 - 1.0 / n_kinds) ** (2 * n_types))
    return 2 * n_types + search + remaining
