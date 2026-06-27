"""Sample seeded N-question subsets of GSM-Hard, excluding negative/nonsensical answers.

GSM-Hard (reasoning-machines/pal, gsmhardv2.jsonl) is built by swapping one number in a
GSM8K question for a random <=7-digit integer and re-executing the reference program to get
the target. With no constraint check, ~150/1319 items go negative (e.g. morning feed > total
feed). We drop those and any non-finite/non-positive target up front.

Beyond negatives, some surviving items are still nonsensical: a fractional answer to an
inherently discrete-count question (e.g. "students per class" = 531842.5). Those can't be
caught by a target-only rule (money/rate answers are legitimately fractional), so they were
found by manual inspection and listed in SKIP below, keyed by position in each seed's shuffled
stream. A skipped item is replaced by the next clean item in that same deterministic stream.

Output: one .jsonl per seed under datasets/, native GSM-Hard line format (input/code/target).
Run with `--inspect` to dump the candidate stream (skips applied) for manual review.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

SRC = Path("external/gsmhardv2.jsonl")
OUT_DIR = Path("scripts/creator/tool_disposition_benchmark/datasets")
N = 20
SEEDS = [0, 1, 2, 3]

# Positions (0-based) in each seed's shuffled clean stream to skip as nonsensical.
# Populated by manual inspection (see module docstring). Replacements come from later in
# the same stream, so changing this only shifts the tail.
# Found by manual review: fractional answers to discrete-count questions (balls, cars,
# pretzels, girls, ducks, balloons), fractional ages, and degenerate near-zero physical
# quantities. Fractional money / length / weight answers were KEPT (legitimately continuous).
SKIP: dict[int, set[int]] = {
    0: set(),
    1: {4, 6, 15, 16},   # 23634691.5 balls; 7.9e-06 balloons; 2110819.5 age; 6859259.2 cars
    2: {3, 5},           # 12213729.5 pretzels; 5.9e-06 in mosaic
    3: {12, 16},         # 840349.5 girls; 5489.55 ducks
}


def is_clean(d: dict) -> bool:
    t = d.get("target")
    return isinstance(t, (int, float)) and math.isfinite(t) and t > 0


def stream(clean: list[dict], seed: int) -> list[dict]:
    """Deterministic shuffled order of the clean pool for a seed."""
    order = clean[:]
    random.Random(seed).shuffle(order)
    return order


def select(clean: list[dict], seed: int) -> list[dict]:
    """First N items of the seed's stream, skipping SKIP positions."""
    skip = SKIP.get(seed, set())
    return [d for i, d in enumerate(stream(clean, seed)) if i not in skip][:N]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", type=int, metavar="SEED",
                    help="dump the post-skip candidate stream for a seed and exit")
    ap.add_argument("--depth", type=int, default=28, help="how many candidates to dump")
    args = ap.parse_args()

    items = [json.loads(l) for l in SRC.open()]
    clean = [d for d in items if is_clean(d)]

    if args.inspect is not None:
        skip = SKIP.get(args.inspect, set())
        kept = [(i, d) for i, d in enumerate(stream(clean, args.inspect)) if i not in skip]
        for rank, (pos, d) in enumerate(kept[: args.depth]):
            tag = "<<IN-20" if rank < N else "spare"
            print(f"[{rank:2d}] pos={pos:3d} {tag}  target={d['target']}")
            print("    " + " ".join(d["input"].split())[:240])
        return

    print(f"{len(items)} total -> {len(clean)} clean (positive & finite)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        sample = select(clean, seed)
        out = OUT_DIR / f"gsmhard_N{N}_seed{seed}.jsonl"
        with out.open("w") as f:
            for d in sample:
                f.write(json.dumps(d) + "\n")
        assert len(sample) == N and all(d["target"] > 0 for d in sample)
        print(f"seed {seed}: wrote {len(sample)} (skipped {sorted(SKIP.get(seed, set()))}) -> {out}")


if __name__ == "__main__":
    main()
