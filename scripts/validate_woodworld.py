"""Cost model + oracle for woodworld (mirrors scripts/validate_toolworld_v2.py).

Derives, from the EDITABLE recipe table in scripts/woodworld.py, the analytic
brute vs. build costs, the deterministic oracle minimum (the action-efficiency
denominator), and the crossover N* above which building the axe is the rational
play. Also reports P(brute solves) exactly, because gather is stochastic and the
budget is integer-rounded -- so pure gathering is NOT a clean failure at small N
(it solves with real probability there), and built_axe only carries calibration
signal once N straddles N*.

Budget is grind-calibrated: B = round(BUDGET_MULT * E[grind]) = round(BUDGET_MULT
* N / p). At BUDGET_MULT=1.0 a grinder expects exactly N wood (no surplus), so
pure grinding is a coin-flip and building is the path that reliably solves -- the
"forced" regime where solve rate itself is a build signal. (At 1.2 grinding had a
20% surplus and was a viable escape hatch, making built_axe a disposition signal.)
"""

from __future__ import annotations

from math import ceil, comb

from scripts.woodworld import (GATHER_PROB, GOAL_ITEM, RECIPES, TOOL_ITEM,
                               USE_YIELDS)

BUDGET_MULT = 1.0
USE_YIELD = USE_YIELDS[TOOL_ITEM][1]


def budget_for(n: int, p: float = GATHER_PROB) -> int:
    """B = round(BUDGET_MULT * E[grind]) = round(BUDGET_MULT * N / p): the strict
    total-action cap, grind-calibrated. At BUDGET_MULT=1.0 a grinder expects
    exactly N wood (E[grind successes] = B*p = N) -- NO surplus -- so pure
    grinding is a coin-flip (~P(Binom(B,p) >= N) ~ 0.5-0.6) and building the axe
    is now the path that reliably solves. (At 1.2 grinding had a 20% surplus and
    building was merely cheaper, not necessary.)"""
    return round(BUDGET_MULT * n / p)


def _recipes_map():
    """latent output item -> (recipe, yield count) for the first recipe making it."""
    m = {}
    for r in RECIPES:
        for out, c in r["out"].items():
            m.setdefault(out, (r, c))
    return m


def _cost_to_make(item: str, count: int, rmap) -> tuple[int, int]:
    """(wood needed, craft actions) to obtain `count` of `item`, assuming wood is
    the gatherable base. Yields are batched (one craft makes a whole batch)."""
    if item not in rmap:                      # base item (wood): gather it
        return (count if item == GOAL_ITEM else 0, 0)
    recipe, y = rmap[item]
    batches = ceil(count / y)
    wood, crafts = 0, batches
    for inp, c in recipe["in"].items():
        w, cr = _cost_to_make(inp, batches * c, rmap)
        wood += w
        crafts += cr
    return wood, crafts


def axe_cost() -> tuple[int, int]:
    """(wood, crafts) to build one tool from scratch, derived from the current
    recipe table (e.g. (3, 2) for 2 wood->4 sticks + 1 stick + 1 wood->axe)."""
    return _cost_to_make(TOOL_ITEM, 1, _recipes_map())


def brute_expected(n: int, p: float = GATHER_PROB) -> float:
    """E[actions] to reach N wood by gathering alone = N / p."""
    return n / p


def build_expected(n: int, p: float = GATHER_PROB) -> float:
    """E[actions] for the tool path: gather the build wood (~wood/p), the crafts,
    then N wood at USE_YIELD per use (post-build inventory wood is 0)."""
    wood, crafts = axe_cost()
    return wood / p + crafts + n / USE_YIELD


def oracle_min(n: int) -> int:
    """Deterministic best-case action count (every gather succeeds): the smaller
    of pure-gather (N) and build (wood + crafts + ceil(N/USE_YIELD))."""
    wood, crafts = axe_cost()
    return min(n, wood + crafts + ceil(n / USE_YIELD))


def p_brute(n: int, p: float = GATHER_PROB) -> float:
    """Exact P(pure gathering solves within budget) = P(Binom(B, p) >= N)."""
    b = budget_for(n, p)
    if n > b:
        return 0.0
    return sum(comb(b, k) * p**k * (1 - p)**(b - k) for k in range(n, b + 1))


def n_star(p: float = GATHER_PROB) -> int:
    """Smallest N where building both fits the budget and beats brute in expectation."""
    n = 1
    while not (build_expected(n, p) <= budget_for(n, p)
               and build_expected(n, p) < brute_expected(n, p)):
        n += 1
        if n > 10000:
            return -1
    return n


def main():
    wood, crafts = axe_cost()
    print(f"axe cost: {wood} wood + {crafts} crafts | gather p={GATHER_PROB} | "
          f"use yield={USE_YIELD} | budget mult={BUDGET_MULT}")
    print(f"N*  = {n_star()}  (smallest N where build <= budget and build < brute)\n")
    print(f"{'N':>4} {'budget':>7} {'brute_E':>8} {'build_E':>8} {'oracle':>7} "
          f"{'P(brute)':>9} {'build_wins':>11}")
    for n in [4, 8, 12, 13, 14, 16, 20, 30, 50]:
        wins = build_expected(n) <= budget_for(n) and build_expected(n) < brute_expected(n)
        print(f"{n:>4} {budget_for(n):>7} {brute_expected(n):>8.2f} "
              f"{build_expected(n):>8.2f} {oracle_min(n):>7} {p_brute(n):>9.3f} "
              f"{('yes' if wins else 'no'):>11}")


if __name__ == "__main__":
    main()
