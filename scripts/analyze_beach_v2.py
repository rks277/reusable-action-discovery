"""Analysis for run_beach_v2_sweep.py (the no-movement, coordinate-addressed
build-vs-grind surface).

v2 episode rows are schema-identical to v1, so the entire v1 analyzer applies
unchanged: this module just calls scripts.analyze_beach_sweep.main (Wilson-CI
win/build/use rates grouped by (model, durability, grid), the build->use->win
funnel, the win/build heatmaps over the papers x grid surface, token/cost
tables). Figures land alongside the episodes file.

v2-only addition: because there's no movement, total_actions is exactly
inspects + digs, so mean INSPECTS (total_actions - digs) is a clean readout of
the tool-search cost -- how many rock-searches the agent spent hunting scraps.
We print that as an extra table after the shared analysis.

Usage: python -m scripts.analyze_beach_v2 runs/beach_v2_sweep_*/episodes.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from scripts.analyze_beach_sweep import _mean, main as base_main, short


def print_inspects(rows, models, durs):
    """Mean inspects (= total_actions - digs) by (model, durability) -- the
    tool-search cost, isolated now that no movement is mixed into the action
    count."""
    by = defaultdict(list)
    for r in rows:
        ta, dg = r.get("total_actions"), r.get("digs")
        if ta is None or dg is None:
            continue
        by[(r["model"], r["shovel_durability"])].append(ta - dg)

    print("\nMean inspects (rock-searches) by (model, durability)  "
          "[inspects = total_actions - digs]:\n")
    hdr = f"{'model':14s} {'dur':>4} {'eps':>4} {'inspects':>9} {'digs':>6}"
    print(hdr)
    print("-" * len(hdr))
    for m in models:
        for d in durs:
            g = [r for r in rows
                 if r["model"] == m and r["shovel_durability"] == d
                 and not r.get("error")]
            if not g:
                continue
            insp = _mean([r["total_actions"] - r["digs"] for r in g])
            digs = _mean([r["digs"] for r in g])
            print(f"{short(m):14s} {d:>4} {len(g):>4} {insp:>9.1f} {digs:>6.1f}")


def main():
    if len(sys.argv) < 2:
        print("usage: python -m scripts.analyze_beach_v2 "
              "runs/beach_v2_sweep_*/episodes.jsonl")
        sys.exit(1)
    # v2-only tool-search-cost table FIRST, so it prints even if the shared
    # figure step trips its (pre-existing) empty-surface edge case on degenerate
    # data -- e.g. a region where no episode built the map.
    path = Path(sys.argv[1])
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    models = sorted({r["model"] for r in rows})
    durs = sorted({r["shovel_durability"] for r in rows})
    print_inspects(rows, models, durs)

    # Then the full shared v1 analysis (tables + all figures) on the v2 episodes.
    base_main()


if __name__ == "__main__":
    main()
