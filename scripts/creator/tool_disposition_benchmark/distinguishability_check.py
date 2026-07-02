"""Distinguishability / non-composability check for the stochastic-design family pool.

Two usable families must satisfy BOTH:
  (1) NON-COMPOSABLE: no single reasonable script solves both (distinct gen_cluster) -- else they are
      the same tool-identity and not independent build-decisions;
  (2) DISTINGUISHABLE: a reader can tell them apart, so a repeat of the SAME family is recognizable
      as a repeat and NOT confused with a different family.

This script checks (1) exactly (via cluster_of) and FLAGS (2) via lexical similarity on rendered
questions (numbers stripped) -- high overlap = shared scaffold, to inspect the operative clause by
eye. It prints one sample per family for the semantic judgment. Not a model call; a triage tool.

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.distinguishability_check
"""

from __future__ import annotations

import random
import re
from itertools import combinations

from scripts.creator.tool_disposition_benchmark.family_kit import ALL_FAMILIES, cluster_of

# candidate pool: groups A/B/C/E (building buys ACCURACY -> low a_hand), lcg_small merged into lcg.
POOL = [
    # A. modular / recurrence iteration
    "lcg", "modpow", "fib_mod", "factorial_mod", "weighted_checksum",
    # B. iterate-a-map K times
    "collatz_steps", "digitsq_iter", "digit_reverse_add", "kaprekar_routine", "look_and_say",
    # C. number-theoretic / digit
    "euclid_gcd_chain", "base_convert_digitsum", "luhn_sum", "continued_frac",
    # E. division / modulo
    "int_div_sum", "mod_pair_sum",
]

_STOP = set("a an the of to in on for is are be with and or by each every all their its it into "
            "you your what how many number numbers value values compute report result give the "
            "then repeat times start starting apply final resulting report set list sequence "
            "over from at as do get computed take report reports so on that this these which where "
            "when if else than more less most left right first second third fourth i j n k x s "
            "position positions total sum add".split())


def content_words(text: str) -> set[str]:
    text = re.sub(r"\{[^}]*\}", " ", text)          # drop {PLACEHOLDERS}
    toks = re.findall(r"[a-zA-Z]+", text.lower())
    return {t for t in toks if t not in _STOP and len(t) > 2}


def family_signature(name: str, rng: random.Random) -> set[str]:
    """Union of content words across a family's cover templates (operation-flavored words)."""
    fam = ALL_FAMILIES[name]
    sig: set[str] = set()
    for cov in fam.covers:
        sig |= content_words(cov)
    return sig


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def main():
    rng = random.Random(0)
    print(f"candidate pool: {len(POOL)} families\n")

    # (1) NON-COMPOSABILITY: distinct generalization clusters
    clusters = {}
    for f in POOL:
        clusters.setdefault(cluster_of(f), []).append(f)
    print("=== non-composability (generalization clusters) ===")
    collisions = {c: fs for c, fs in clusters.items() if len(fs) > 1}
    for c, fs in clusters.items():
        flag = "  <-- SHARED TOOL" if len(fs) > 1 else ""
        print(f"  cluster {c:<22} {fs}{flag}")
    print(f"\ndistinct clusters in pool: {len(clusters)}  "
          f"(collisions: {collisions or 'none'})\n")

    # (2) SURFACE SIMILARITY: pairwise Jaccard on cover-word signatures
    sigs = {f: family_signature(f, rng) for f in POOL}
    pairs = sorted(((jaccard(sigs[a], sigs[b]), a, b) for a, b in combinations(POOL, 2)),
                   reverse=True)
    print("=== most surface-similar cross-family pairs (Jaccard on cover words) ===")
    for score, a, b in pairs[:10]:
        print(f"  {score:.2f}  {a} <-> {b}   shared={sorted(sigs[a] & sigs[b])[:8]}")
    print()

    # samples for the semantic judgment (one rendered question per family)
    print("=== one sample per family (read the OPERATIVE clause) ===")
    for f in POOL:
        q = ALL_FAMILIES[f].make_member(rng, 100)["question"]
        print(f"  [{f}] {q[:110]}")


if __name__ == "__main__":
    main()
