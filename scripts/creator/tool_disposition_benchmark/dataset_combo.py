"""Synthetic combinatorial-DP problem set for the tool-disposition benchmark.

Replaces the CREATOR/MATH tracks with problems whose difficulty is COMPUTATION, not insight:
every family is a counting problem solved by a short 2-D dynamic program, with an exact big-integer
gold. A general memoised solver makes any instance trivial; by hand, small instances are feasible
and large ones are not — so a single continuous `size` knob (the magnitude analog of the CREATOR
track) dials the by-hand baseline, and the rational play under the scarce write budget is to build
a few general counting primitives and reuse them.

DESIGN FINDING (2026-06-25 calibration): difficulty REQUIRES a 2-D DP state the solver must track,
with NO recallable closed form. Families that collapse to a closed form or a single 1-D accumulator
(domino tilings / stairs{1,2} / no-run-of-2 strings are ALL Fibonacci; pure lattice = one binomial;
fixed-coin change = quasi-polynomial) stay ~100% by-hand for a strong model at ANY size — a model
recalls or reliably iterates them, so they are never tool-necessary. This roster is therefore all
genuine 2-D DPs: grid paths WITH OBSTACLES, Delannoy (king moves), multi-dice sums, BOUNDED coin
change, partitions into exactly k parts, and common-subsequence counts.

Each family exposes gen(rng, size) -> (question_text, gold_int, params_dict). Golds use Python big
ints (exact at any size); grade by EXACT integer equality, not sig figs.

  python -m scripts.creator.tool_disposition_benchmark.dataset_combo --n 20 --size 5 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DATASET_DIR = Path("scripts/creator/tool_disposition_benchmark/datasets")


# --------------------------------------------------------------------- reference solvers (exact)
def _lattice_obstacle(a: int, b: int, blocked: frozenset) -> int:
    """Monotone paths (0,0)->(a,b), steps right/up, avoiding blocked lattice points. 2-D DP."""
    dp = [[0] * (b + 1) for _ in range(a + 1)]
    for i in range(a + 1):
        for j in range(b + 1):
            if (i, j) in blocked:
                continue
            if i == 0 and j == 0:
                dp[i][j] = 1
            else:
                dp[i][j] = (dp[i - 1][j] if i else 0) + (dp[i][j - 1] if j else 0)
    return dp[a][b]


def _delannoy(a: int, b: int) -> int:
    """Paths (0,0)->(a,b) with steps right, up, AND diagonal — 2-D DP, no single-binomial form."""
    row = [1] * (b + 1)
    for _ in range(a):
        new = [1] * (b + 1)
        for j in range(1, b + 1):
            new[j] = new[j - 1] + row[j] + row[j - 1]
        row = new
    return row[b]


def _dice_sum_ways(d: int, s: int, target: int) -> int:
    """Number of ORDERED outcomes of d s-sided dice summing to target (2-D DP: dice x sum)."""
    dp = [1]
    for _ in range(d):
        new = [0] * (len(dp) + s)
        for i, v in enumerate(dp):
            if v:
                for f in range(1, s + 1):
                    new[i + f] += v
        dp = new
    return dp[target] if target < len(dp) else 0


def _bounded_coins(amount: int, coins: tuple[int, ...], limit: int) -> int:
    """Ways to make `amount` from coins, each usable at most `limit` times (2-D bounded knapsack)."""
    dp = [0] * (amount + 1)
    dp[0] = 1
    for c in coins:
        for v in range(amount, -1, -1):       # iterate down; add 1..limit copies of c
            tot = 0
            for cnt in range(1, limit + 1):
                if v - cnt * c < 0:
                    break
                tot += dp[v - cnt * c]
            dp[v] += tot
    return dp[amount]


def _partitions_k(n: int, k: int) -> int:
    """Number of partitions of n into EXACTLY k positive parts (2-D DP: n x k)."""
    if k <= 0 or n < k:
        return 0
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, k + 1):
            if i - j >= 0:
                dp[i][j] = dp[i - 1][j - 1] + dp[i - j][j]
    return dp[n][k]


def _bounded_comp(n: int, k: int, m: int) -> int:
    """Ordered k-tuples of positive integers, each at most m, summing to n (2-D DP: parts x sum)."""
    if k <= 0 or n < k or n > k * m:
        return 0
    dp = [0] * (n + 1)
    dp[0] = 1
    for _ in range(k):                         # add one part (value 1..m) at a time
        new = [0] * (n + 1)
        for s in range(n + 1):
            if dp[s]:
                for p in range(1, m + 1):
                    if s + p <= n:
                        new[s + p] += dp[s]
        dp = new
    return dp[n]


# ------------------------------------------------------------------------------------- families
# Each maps the continuous `size` to integer params (per-family base/slope tuned by calibration so
# the families break around a common size); rng varies params so instances are distinct/unmemorised.
def _rint(size: float, base: float, slope: float, rng, jitter: int = 1) -> int:
    return max(1, round(base + slope * size) + rng.randint(-jitter, jitter))


def fam_lattice_obs(rng, size):
    a = _rint(size, 4, 0.9, rng)
    b = _rint(size, 4, 0.9, rng)
    n_block = max(1, round(0.12 * a * b))
    cells = [(i, j) for i in range(a + 1) for j in range(b + 1) if (i, j) not in {(0, 0), (a, b)}]
    rng.shuffle(cells)
    blocked = frozenset(cells[:n_block])
    bl = ", ".join(f"({i},{j})" for i, j in sorted(blocked))
    q = (f"On a grid you start at (0, 0) and want to reach ({a}, {b}), each step moving one unit "
         f"right or one unit up. These lattice points are BLOCKED and cannot be visited: {bl}. "
         f"How many distinct paths avoid every blocked point?")
    return q, _lattice_obstacle(a, b, blocked), {"a": a, "b": b, "blocked": sorted(blocked)}


def fam_delannoy(rng, size):
    a = _rint(size, 4, 0.55, rng)
    b = _rint(size, 4, 0.55, rng)
    q = (f"You start at (0, 0) and want to reach ({a}, {b}). Each step is one unit right, one unit "
         f"up, OR one unit diagonally up-right (+1, +1). How many distinct paths are there?")
    return q, _delannoy(a, b), {"a": a, "b": b}


def fam_dice(rng, size):
    d = _rint(size, 4, 0.9, rng)
    s = rng.choice([6, 6, 8])
    lo, hi = d, d * s
    target = rng.randint(lo + (hi - lo) // 4, hi - (hi - lo) // 4)
    q = (f"You roll {d} fair {s}-sided dice (faces 1..{s}), distinguishable. In how many ordered "
         f"outcomes do the {d} dice sum to exactly {target}?")
    return q, _dice_sum_ways(d, s, target), {"d": d, "s": s, "target": target}


_COIN_SETS = [(1, 2, 5), (1, 3, 4), (2, 3, 5), (1, 2, 5, 10)]


def fam_bounded_coins(rng, size):
    coins = rng.choice(_COIN_SETS)
    limit = rng.choice([2, 3, 4])
    amount = _rint(size, 10, 2.2, rng, jitter=2)
    q = (f"Using coins of denominations {list(coins)}, each denomination available at most {limit} "
         f"times, in how many different ways can you make a total of {amount}? (Order does not "
         f"matter.)")
    return q, _bounded_coins(amount, coins, limit), {"amount": amount, "coins": coins, "limit": limit}


def fam_partitions_k(rng, size):
    n = _rint(size, 12, 2.6, rng, jitter=2)
    k = _rint(size, 3, 0.35, rng)
    k = min(k, n)
    q = (f"In how many ways can the integer {n} be written as a sum of exactly {k} positive "
         f"integers, where order does not matter (e.g. 3+1 and 1+3 count once)?")
    return q, _partitions_k(n, k), {"n": n, "k": k}


def fam_bounded_comp(rng, size):
    k = _rint(size, 3, 0.5, rng)
    m = rng.choice([4, 5, 6])
    n = _rint(size, 10, 2.0, rng, jitter=2)
    n = min(max(n, k), k * m)                   # keep it satisfiable
    q = (f"How many ordered sequences of exactly {k} positive integers, each at most {m}, sum to "
         f"exactly {n}? (Order matters: e.g. (1,2) and (2,1) count separately.)")
    return q, _bounded_comp(n, k, m), {"n": n, "k": k, "m": m}


FAMILIES = [fam_lattice_obs, fam_delannoy, fam_dice, fam_bounded_coins,
            fam_partitions_k, fam_bounded_comp]


def build_dataset(n: int, size: float, seed: int) -> list[dict]:
    """n distinct instances, families cycled for an even mix, params rng-varied per instance."""
    rng = random.Random(7919 * seed + 31)
    problems, seen = [], set()
    guard = 0
    while len(problems) < n and guard < n * 80:
        guard += 1
        fam = FAMILIES[len(problems) % len(FAMILIES)] if guard <= n else rng.choice(FAMILIES)
        q, gold, params = fam(rng, size)
        key = (fam.__name__, json.dumps(params, sort_keys=True))
        if key in seen or gold <= 0:
            continue
        seen.add(key)
        problems.append({"idx": len(problems), "family": fam.__name__, "question": q,
                         "gold": int(gold), "params": params, "size": size, "kind": "combo"})
    return problems


def dataset_path(n: int, size: float, seed: int) -> Path:
    st = f"{size:g}".replace(".", "p")
    return DATASET_DIR / f"combo_N{n}_seed{seed}_S{st}.json"


def load_or_build(n: int, size: float, seed: int) -> list[dict]:
    p = dataset_path(n, size, seed)
    if p.exists():
        return json.loads(p.read_text())
    problems = build_dataset(n, size, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(problems, indent=2))
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--size", type=float, default=5.0, help="difficulty knob (calibrate to ~50% by hand)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    problems = build_dataset(args.n, args.size, args.seed)
    p = dataset_path(args.n, args.size, args.seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(problems, indent=2))
    print(f"built {len(problems)}/{args.n} @ size {args.size} -> {p}")
    for pr in problems[:6]:
        print(f"\n[{pr['idx']}] {pr['family']} gold={pr['gold']}\n  {pr['question'][:140]}")


if __name__ == "__main__":
    main()
