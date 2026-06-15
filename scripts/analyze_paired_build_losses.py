"""Experiment (A): paired-seed trajectory diff -- why does Opus build less?

The build sweeps share worlds by seed: world (and therefore the winning RECIPE)
depends only on relabel_seed, so for a fixed (budget, seed) cell the three models
face an IDENTICAL world. This script takes every (budget, seed) where a SMALLER
model built the machine but the LARGER model (default Opus) did NOT, and -- with
the environment held perfectly constant -- asks what the larger model did instead.

Because labels.recipe records the exact winning type pair and holdings are
MONOTONIC (combine never consumes byproducts; see toolworld_v2.State.combine),
each loss decomposes rigorously, from the recorded actions+obs alone, into:

  never_gathered      : never held >=1 of BOTH recipe types  -> gathering failure
  gathered_not_tried  : held both, but never issued the winning combine
                        -> discovery / commitment gap (the interesting case)
  tried_not_holding   : issued the winning pair but not while holding both
                        -> sequencing / execution slip

(Every loss is a non-builder by construction, so these three are exhaustive.)

Usage:
  PYTHONPATH=. python -m scripts.analyze_paired_build_losses \
      runs/haiku_build_sweep_T3_n12_20260614_190439/episodes.jsonl \
      runs/sonnet_build_sweep_T3_n12_20260613_003611/episodes.jsonl \
      runs/opus_build_sweep_T3_n12_20260613_212256/episodes.jsonl

By default the LAST path given is the "large" model under study; the rest are the
"small" comparators. Override with --large=Opus / --small=Haiku,Sonnet.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HOLD_RE = re.compile(r"hold (\d+) (\S+?)\(s\)")


def label_for(model: str) -> str:
    m = (model or "").lower()
    return ("Sonnet" if "sonnet" in m else "Opus" if "opus" in m
            else "Haiku" if "haiku" in m else "Fable" if "fable" in m
            else (model or "model"))


def load(path: Path) -> list[dict]:
    if path.is_dir():
        path = path / "episodes.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [r for r in rows if not r.get("error")]


def held_types(obs_list: list[str]) -> set[str]:
    """Set of byproduct types the agent acquired (held >=1 of) at any point.
    Holdings are monotonic, so 'seen in any hold line' == 'held from then on'."""
    seen = set()
    for o in obs_list:
        for _, ty in HOLD_RE.findall(o or ""):
            seen.add(ty)
    return seen


def combine_pairs(actions: list) -> list[frozenset]:
    """Unordered type pairs (or singletons) the agent tried to combine."""
    out = []
    for a in actions:
        if a and a[0] == "combine" and len(a) >= 3:
            out.append(frozenset((a[1], a[2])))
    return out


def first_combine_index(actions: list) -> int | None:
    for i, a in enumerate(actions):
        if a and a[0] == "combine":
            return i
    return None


def classify_loss(opus: dict) -> dict:
    """Decompose one Opus non-builder against its own recorded recipe."""
    recipe = frozenset(opus["labels"]["recipe"])
    rA, rB = sorted(recipe)
    held = held_types(opus.get("obs") or [])
    acquired_both = rA in held and rB in held
    pairs = combine_pairs(opus.get("actions") or [])
    tried_winning = recipe in pairs
    distinct_pairs = len(set(pairs))
    if not acquired_both:
        cat = "never_gathered"
    elif not tried_winning:
        cat = "gathered_not_tried"
    else:
        cat = "tried_not_holding"
    return {
        "category": cat,
        "recipe": (rA, rB),
        "acquired_both": acquired_both,
        "acquired": tuple(sorted(held & recipe)),
        "tried_winning": tried_winning,
        "n_combines": len(pairs),
        "distinct_pairs": distinct_pairs,
        "first_combine_at": first_combine_index(opus.get("actions") or []),
        "total_actions": opus.get("total_actions"),
        "stopped_reason": opus.get("stopped_reason"),
    }


def main() -> None:
    args = [a for a in sys.argv[1:]]
    large_lbl, small_lbls = None, None
    rest = []
    for a in args:
        if a.startswith("--large="):
            large_lbl = a.split("=", 1)[1]
        elif a.startswith("--small="):
            small_lbls = [x.strip() for x in a.split("=", 1)[1].split(",")]
        else:
            rest.append(a)
    if len(rest) < 2:
        sys.exit("need >=2 episodes.jsonl paths")

    runs = {}  # label -> {(budget, seed): row}
    order = []
    for p in rest:
        rows = load(Path(p))
        lbl = label_for(rows[0].get("model"))
        order.append(lbl)
        idx = {}
        for r in rows:
            idx[(r["budget"], r["relabel_seed"])] = r
        runs[lbl] = idx

    large_lbl = large_lbl or order[-1]
    small_lbls = small_lbls or [l for l in order if l != large_lbl]
    big = runs[large_lbl]

    # paired losses: cells where `large` did NOT build but some `small` DID
    losses = []          # one entry per losing (budget, seed) cell for `large`
    beaten_by = defaultdict(set)   # (budget,seed) -> {small labels that built}
    for (budget, seed), lr in big.items():
        if lr.get("built_machine"):
            continue
        winners = [sl for sl in small_lbls
                   if (budget, seed) in runs[sl]
                   and runs[sl][(budget, seed)].get("built_machine")]
        if winners:
            beaten_by[(budget, seed)] = set(winners)
            losses.append((budget, seed, lr, winners))

    n_big = len(big)
    n_big_nonbuild = sum(1 for r in big.values() if not r.get("built_machine"))
    print(f"== Experiment A: paired-seed build losses ({large_lbl} vs "
          f"{'/'.join(small_lbls)}) ==\n")
    print(f"{large_lbl} episodes: {n_big} | non-builders: {n_big_nonbuild} | "
          f"paired losses (a smaller model built the same world): {len(losses)}")
    # how often each smaller model specifically beat the large one head-to-head
    for sl in small_lbls:
        shared = [(b, s) for (b, s) in big
                  if (b, s) in runs[sl]]
        head = sum(1 for (b, s) in shared
                   if not big[(b, s)].get("built_machine")
                   and runs[sl][(b, s)].get("built_machine"))
        rev = sum(1 for (b, s) in shared
                  if big[(b, s)].get("built_machine")
                  and not runs[sl][(b, s)].get("built_machine"))
        print(f"  head-to-head on {len(shared)} shared cells: "
              f"{sl} built & {large_lbl} didn't = {head}  |  "
              f"{large_lbl} built & {sl} didn't = {rev}")

    if not losses:
        print("\nNo paired losses found.")
        return

    # decompose every losing episode
    cls = [classify_loss(lr) for (_, _, lr, _) in losses]
    cats = Counter(c["category"] for c in cls)
    print(f"\n-- Why {large_lbl} lost these {len(losses)} identical worlds --")
    pretty = {"never_gathered": "never gathered both ingredients (gathering)",
              "gathered_not_tried": "gathered both, never tried winning combine (discovery/commitment)",
              "tried_not_holding": "tried winning pair but not while holding both (sequencing)"}
    for cat in ("never_gathered", "gathered_not_tried", "tried_not_holding"):
        k = cats.get(cat, 0)
        print(f"  {k:>3} / {len(losses)} ({k/len(losses):4.0%})  {pretty[cat]}")

    print(f"\n-- exploration economy among losses --")
    np_ = [c["n_combines"] for c in cls]
    dp = [c["distinct_pairs"] for c in cls]
    ta = [c["total_actions"] for c in cls if c["total_actions"] is not None]
    print(f"  combine attempts:        median {statistics.median(np_):.0f}  "
          f"max {max(np_)}")
    print(f"  distinct pairs tried:    median {statistics.median(dp):.0f}  "
          f"max {max(dp)}   (recipe space ~ C(T,2)+T)")
    print(f"  total actions in episode: median {statistics.median(ta):.0f}  "
          f"max {max(ta)}")
    print(f"  stopped_reason: {dict(Counter(c['stopped_reason'] for c in cls))}")

    # exemplars: a few losing cells in detail, spread across budgets
    print(f"\n-- exemplars ({large_lbl}'s fate on worlds a smaller model won) --")
    losses_sorted = sorted(zip(losses, cls), key=lambda z: (z[0][0], z[0][1]))
    shown = 0
    for (budget, seed, lr, winners), c in losses_sorted:
        if shown >= 8:
            break
        shown += 1
        wbits = []
        for w in winners:
            wr = runs[w][(budget, seed)]
            wbits.append(f"{w} built @ action {wr['build_turn'] + 1}")
        rA, rB = c["recipe"]
        print(f"\n  b={budget} seed={seed}  recipe = combine({rA},{rB})")
        print(f"    winners: {', '.join(wbits)}")
        acq = ("both" if c["acquired_both"]
               else f"only {c['acquired'] or '(neither)'}")
        print(f"    {large_lbl}: {c['category']} | acquired {acq} | "
              f"tried_winning={c['tried_winning']} | "
              f"{c['n_combines']} combines ({c['distinct_pairs']} distinct) | "
              f"{c['total_actions']} actions | stop={c['stopped_reason']}")

    print(f"\nFull doc: docs/why-opus-builds-less-experiments.md")


if __name__ == "__main__":
    main()
