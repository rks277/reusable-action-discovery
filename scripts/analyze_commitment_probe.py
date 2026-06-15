"""Analyze experiment (J) commitment-probe output: recognition rate per condition.

Reuses per_episode() from analyze_recognition_latency.py (held_both, built,
latency) and groups by the row's `condition` field. Reports, per condition:
  recognition rate  P(built | held both)   <- the headline
  build rate        P(built)
  held-both rate    P(held both)
  distinct combine pairs tried (median), nudge-fire count, episode count.

Reading: commit >> baseline ~ neutral  => closable commitment failure;
         commit ~ baseline             => deeper deficit;
         commit ~ neutral > baseline   => mere-interruption effect.

Usage: PYTHONPATH=. python -m scripts.analyze_commitment_probe \
           runs/commitment_probe_T3_n12_<ts>/episodes.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from scripts.analyze_recognition_latency import per_episode
from scripts.analyze_paired_build_losses import combine_pairs

ORDER = ["baseline", "commit", "neutral"]


def main() -> None:
    path = Path(sys.argv[1])
    if path.is_dir():
        path = path / "episodes.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]

    by = defaultdict(list)
    for r in rows:
        by[r.get("condition", "?")].append(r)

    conds = [c for c in ORDER if c in by] + [c for c in by if c not in ORDER]
    print(f"== Experiment J: commitment probe ({len(rows)} episodes) ==\n")
    print(f"{'condition':>9} {'eps':>4} {'held_both':>9} {'recognition':>12} "
          f"{'build':>6} {'distinct_pairs':>14} {'nudged':>7}")
    print("-" * 72)
    summ = {}
    for c in conds:
        g = by[c]
        pe = [per_episode(r) for r in g]
        hb = [e for e in pe if e["held_both"]]
        built = sum(e["built"] for e in pe)
        built_hb = sum(e["built"] for e in hb)
        recog = built_hb / len(hb) if hb else float("nan")
        dp = [len(set(combine_pairs(r.get("actions") or []))) for r in g]
        nud = sum(1 for r in g if r.get("nudged"))
        summ[c] = dict(n=len(g), hb=len(hb), recog=recog, build=built / len(g))
        print(f"{c:>9} {len(g):>4} {len(hb):>9} {recog:>12.2f} "
              f"{built/len(g):>6.2f} {statistics.median(dp):>14.0f} {nud:>7}")

    if "commit" in summ and "baseline" in summ:
        d = summ["commit"]["recog"] - summ["baseline"]["recog"]
        print(f"\nrecognition lift (commit - baseline): {d:+.2f}")
        if "neutral" in summ:
            dn = summ["commit"]["recog"] - summ["neutral"]["recog"]
            print(f"recognition lift (commit - neutral):  {dn:+.2f}  "
                  f"(controls for mere interruption)")

    # per-budget recognition rate per condition
    print(f"\nrecognition rate P(built|held both) by budget:")
    budgets = sorted({r["budget"] for r in rows})
    print(f"{'budget':>7} " + " ".join(f"{c:>10}" for c in conds))
    for b in budgets:
        cells = []
        for c in conds:
            g = [per_episode(r) for r in by[c] if r["budget"] == b]
            hb = [e for e in g if e["held_both"]]
            bu = sum(e["built"] for e in hb)
            cells.append(f"{bu/len(hb):.2f}({bu}/{len(hb)})" if hb else "   -   ")
        print(f"{b:>7} " + " ".join(f"{x:>10}" for x in cells))


if __name__ == "__main__":
    main()
