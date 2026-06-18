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


def budget_for(n: int, p: float = GATHER_PROB, mult: float = BUDGET_MULT) -> int:
    """B = round(mult * E[grind]) = round(mult * N / p): the strict total-action cap,
    grind-calibrated. At mult=1.0 a grinder expects exactly N wood (E[grind
    successes] = B*p = N) -- NO surplus -- so pure grinding is a coin-flip
    (~P(Binom(B,p) >= N) ~ 0.5-0.6) and building the axe is the path that reliably
    solves. At mult=1.2 grinding has a 20% surplus and is a viable escape hatch
    (building merely cheaper, not necessary) -- the ToolWorld slack regime. `mult`
    defaults to the module BUDGET_MULT; the region sweep can override it per-run."""
    return round(mult * n / p)


def _recipes_map(recipes=RECIPES):
    """latent output item -> (recipe, yield count) for the first recipe making it."""
    m = {}
    for r in recipes:
        for out, c in r["out"].items():
            m.setdefault(out, (r, c))
    return m


def _cost_to_make(item: str, count: int, rmap, goal_item=GOAL_ITEM) -> tuple[int, int]:
    """(wood needed, craft actions) to obtain `count` of `item`, assuming wood is
    the gatherable base. Yields are batched (one craft makes a whole batch)."""
    if item not in rmap:                      # base item (wood): gather it
        return (count if item == goal_item else 0, 0)
    recipe, y = rmap[item]
    batches = ceil(count / y)
    wood, crafts = 0, batches
    for inp, c in recipe["in"].items():
        w, cr = _cost_to_make(inp, batches * c, rmap, goal_item)
        wood += w
        crafts += cr
    return wood, crafts


def axe_cost(recipes=RECIPES, tool_item=TOOL_ITEM, goal_item=GOAL_ITEM) -> tuple[int, int]:
    """(wood, crafts) to build one tool from scratch, derived from a recipe table.
    Defaults to the BASE recipe ((3, 2) for 2 wood->4 sticks + 1 stick+1 wood->axe);
    pass a variant's recipes to get its cost (e.g. iso 2 wood->axe gives (2, 1))."""
    return _cost_to_make(tool_item, 1, _recipes_map(recipes), goal_item)


def brute_expected(n: int, p: float = GATHER_PROB) -> float:
    """E[actions] to reach N wood by gathering alone = N / p."""
    return n / p


def build_expected(n: int, p: float = GATHER_PROB, axe_cost_override=None,
                   use_yield: int = USE_YIELD) -> float:
    """E[actions] for the tool path: gather the build wood (~wood/p), the crafts,
    then N wood at use_yield per use (post-build inventory wood is 0). Pass
    axe_cost_override=(wood, crafts) to evaluate a variant recipe's boundary."""
    wood, crafts = axe_cost_override if axe_cost_override is not None else axe_cost()
    return wood / p + crafts + n / use_yield


def oracle_min(n: int, axe_cost_override=None, use_yield: int = USE_YIELD) -> int:
    """Deterministic best-case action count (every gather succeeds): the smaller
    of pure-gather (N) and build (wood + crafts + ceil(N/use_yield))."""
    wood, crafts = axe_cost_override if axe_cost_override is not None else axe_cost()
    return min(n, wood + crafts + ceil(n / use_yield))


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
