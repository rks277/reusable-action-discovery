"""No-LLM cost validation for the MULTI-TYPE construction variant (v2).

v1 (validate_toolworld.py): one generic byproduct; combine(bp,bp) -> machine.
Construction was trivially discoverable (combine your two duplicates), so the
LLM built the tool ~always and built_tool carried no signal.

v2 hardens CONSTRUCTION (01_world_mechanics.md "construction-hardness knob"):
  - T distinct byproduct TYPES. examine(door) drops ONE byproduct of a
    UNIFORMLY RANDOM type (keys drop alongside as before -> brute UNCHANGED).
  - the machine needs a SPECIFIC unordered pair of distinct types {A,B}.
    combine(x,y) builds iff {type(x),type(y)} == {A,B}; else nothing.
  -> "which two components combine?" is now a genuine discovery (C(T,2)+T
     candidate combines, one correct), not "combine my duplicates".

Brute baseline is identical to v1 (it ignores byproducts):  brute ~ n*H_n + n.
The machine route gains a setup cost: gather >=1 of each recipe type
(~3T/2 examines) + 1 correct combine + 2 actions/door. We Monte-Carlo the
informed-optimal machine cost (knows the recipe) to locate the new threshold,
and separately report a naive-discovery cost (must search for the pair) to
gauge the realistic overhead an LLM faces.
"""

from __future__ import annotations

import itertools
import random
import statistics


def Hn(n: int) -> float:
    return sum(1.0 / k for k in range(1, n + 1))


def simulate_brute(n: int, rng: random.Random) -> int:
    opened = set()
    examines = uses = 0
    while len(opened) < n:
        examines += 1
        j = rng.randrange(n)
        if j not in opened:
            uses += 1
            opened.add(j)
    return examines + uses


def simulate_machine_informed(n: int, T: int, rng: random.Random) -> int:
    """Knows the recipe {0,1}. Examine until holding both types, build, spend
    free drop-keys, then machine-produce + open the rest (2 actions/door)."""
    recipe = {0, 1}
    have_types = set()
    held_keys = set()
    examines = uses = machine_calls = 0
    while not recipe <= have_types:
        examines += 1
        have_types.add(rng.randrange(T))   # random byproduct type
        held_keys.add(rng.randrange(n))     # random door's key, alongside
    combine = 1
    opened = set()
    for j in held_keys:
        if j not in opened:
            uses += 1; opened.add(j)
    for j in range(n):
        if j not in opened:
            machine_calls += 1; uses += 1; opened.add(j)
    return examines + combine + machine_calls + uses


def simulate_machine_discovery(n: int, T: int, rng: random.Random) -> int:
    """Realistic: must DISCOVER the recipe by trying combines. Gathers a couple
    byproducts, then tries candidate type-pairs (random order) until one works,
    paying 1 action per failed combine. Lower bound on real LLM overhead."""
    recipe = frozenset({0, 1})
    have_counts = {}
    held_keys = set()
    examines = uses = machine_calls = combines = 0
    # gather until we hold at least two distinct types (need 2 to even try)
    while len([t for t, c in have_counts.items() if c > 0]) < 2:
        examines += 1
        t = rng.randrange(T)
        have_counts[t] = have_counts.get(t, 0) + 1
        held_keys.add(rng.randrange(n))
    # search candidate distinct-type pairs we can form, until the recipe builds
    built = False
    while not built:
        held = [t for t, c in have_counts.items() if c > 0]
        candidates = [frozenset(p) for p in itertools.combinations(held, 2)]
        rng.shuffle(candidates)
        tried = set()
        for pair in candidates:
            combines += 1
            tried.add(pair)
            if pair == recipe:
                built = True
                break
        if not built:
            # exhausted current holdings; examine more to get new types
            examines += 1
            t = rng.randrange(T)
            have_counts[t] = have_counts.get(t, 0) + 1
            held_keys.add(rng.randrange(n))
    opened = set()
    for j in held_keys:
        if j not in opened:
            uses += 1; opened.add(j)
    for j in range(n):
        if j not in opened:
            machine_calls += 1; uses += 1; opened.add(j)
    return examines + combines + machine_calls + uses


def main():
    T_VALUES = [2, 3, 4]
    NS = [1, 2, 3, 4, 5, 6, 8, 12, 16]
    TR = 5000
    print("Multi-type construction cost validation (brute unchanged ~ n*H_n + n)\n")
    for T in T_VALUES:
        print(f"=== T = {T} byproduct types (recipe = one specific distinct pair) ===")
        print(f"{'n':>3} | {'brute':>7} | {'mach_inf':>8} | {'mach_disc':>9} | wins(inf)?")
        print("-" * 56)
        nstar = None
        for n in NS:
            brute = statistics.mean(simulate_brute(n, random.Random(s)) for s in range(TR))
            inf = statistics.mean(simulate_machine_informed(n, T, random.Random(7000 + s)) for s in range(TR))
            disc = statistics.mean(simulate_machine_discovery(n, T, random.Random(9000 + s)) for s in range(TR))
            wins = "YES" if inf < brute else "no"
            if wins == "YES" and nstar is None:
                nstar = n
            print(f"{n:>3} | {brute:>7.1f} | {inf:>8.1f} | {disc:>9.1f} | {wins}")
        print(f"  -> informed-optimal threshold n* = {nstar} for T={T}\n")


if __name__ == "__main__":
    main()
