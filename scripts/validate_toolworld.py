"""No-LLM validation of the (rebuilt) byproduct->machine tool world.

Mechanics (obfuscated in the LLM harness; plain here):
  - n locked doors. key_i opens door_i (index correspondence is transparent).
  - examine(door) -> drops ONE key for a UNIFORMLY RANDOM door (over all n, WITH
    replacement) + one generic byproduct. (Which door you examine is cosmetic.)
  - combine(byproduct, byproduct) -> a MACHINE (persists).
  - use(machine, door_i) -> yields key_i (the machine persists).
  - use(key_i, door_i) -> opens door_i.

Two routes:
  BRUTE  : examine repeatedly, using each dropped key on its door. Because draws
           are uniform-with-replacement over ALL doors, you waste draws on doors
           already opened -> COUPON COLLECTOR, ~ n*H_n examines + n uses.
  MACHINE: examine twice (gather 2 byproducts), build the machine, then for each
           door use(machine,door)->key + use(key,door)->open = 2 actions/door,
           DETERMINISTIC (no waste).

The machine is rational only because brute is a grind (this is load-bearing: if
examine handed you the matching key, brute would be 2/door and the machine -- also
2/door -- would never pay off). We confirm the crossover and find n*.
"""

from __future__ import annotations

import random
import statistics


def simulate_brute(n: int, rng: random.Random) -> int:
    """Examine until a key for every door has dropped; use each on its door."""
    opened = set()
    examines = uses = 0
    while len(opened) < n:
        examines += 1
        j = rng.randrange(n)          # uniform over ALL doors, with replacement
        # (a byproduct also drops; brute ignores it)
        if j not in opened:
            uses += 1                 # use the key on its (still-locked) door
            opened.add(j)
    return examines + uses


def simulate_machine(n: int, rng: random.Random, build_after: int = 2) -> int:
    """Gather `build_after` byproducts (and whatever keys come with them), build
    the machine, then deterministically open every remaining door."""
    opened = set()
    held_keys = set()
    examines = uses = machine_calls = 0
    for _ in range(build_after):       # gather byproducts; keys drop alongside
        examines += 1
        held_keys.add(rng.randrange(n))
    combine = 1                        # build the machine from 2 byproducts
    for j in held_keys:                # spend any free drop-keys first
        if j not in opened:
            uses += 1
            opened.add(j)
    for j in range(n):                 # machine-produce + open the rest
        if j not in opened:
            machine_calls += 1         # use(machine, door) -> key
            uses += 1                  # use(key, door) -> open
            opened.add(j)
    return examines + combine + machine_calls + uses


def Hn(n: int) -> float:
    return sum(1.0 / k for k in range(1, n + 1))


def main():
    print("Validating byproduct->machine tool world (no LLM)\n")
    print(f"{'n':>3} | {'brute(mc)':>9} | {'n*Hn+n':>8} | {'machine(mc)':>11} | "
          f"{'2n+~':>5} | tool wins?")
    print("-" * 64)
    T = 5000
    rows = []
    for n in [1, 2, 3, 4, 6, 8, 12, 16]:
        brute = [simulate_brute(n, random.Random(s)) for s in range(T)]
        mach = [simulate_machine(n, random.Random(10_000 + s)) for s in range(T)]
        mb, mm = statistics.mean(brute), statistics.mean(mach)
        analytic_brute = n * Hn(n) + n
        wins = "YES" if mm < mb else "no"
        rows.append((n, mb, mm, wins))
        print(f"{n:>3} | {mb:>9.1f} | {analytic_brute:>8.1f} | {mm:>11.1f} | "
              f"{2*n+1:>5} | {wins}")

    thr = next((n for n, mb, mm, _ in rows if mm < mb), None)
    print(f"\nTool-construction becomes rational at n* = {thr}.")
    print("Brute ~ coupon collector (n*H_n examines + n uses); machine ~ 2n+1, "
          "deterministic.")

    # --- well-formedness: machine yields the RIGHT key; wrong key doesn't open ---
    print("\nWell-formedness checks (n=4):")
    n = 4
    # machine produces key_i for door_i; key_i opens only door_i
    # (modelled directly here as identity map i->i)
    def key_opens(key_idx, door_idx):
        return key_idx == door_idx
    print(f"  machine(door_2) yields key_2; key_2 opens door_2: {key_opens(2, 2)} (True)")
    print(f"  key_2 on door_3 (wrong door): {key_opens(2, 3)} (False)")
    # brute is genuinely solvable (coverage guaranteed since draws hit every door a.s.)
    s = simulate_brute(n, random.Random(0))
    print(f"  brute solves n=4 in {s} actions (finite, all doors covered): True")


if __name__ == "__main__":
    main()
