"""Analyze a v4-hard (or v4) eval-tool run: per-model Recognition (P used tool), genuine
tool-DECLINE rate (real hand-grind, artifact-stripped), crater rate among declines, and Solve.
The inversion = decline rate rising with capability + decliners cratering.

  PYTHONPATH=. python -m scripts.creator.analyze_creator_eval_hard runs/creator_eval_hard_<ts>/
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from scripts.creator.creator_eval_tool import _parse_answers

NATIVE = re.compile(r'<invoke name=|<function_calls>|antml:invoke', re.I)
FENCE = re.compile(r'```')
ASKAGAIN = re.compile(r'\?|provide|need|require|missing|specify|clarify', re.I)
ORDER = ["haiku", "sonnet", "opus"]


def _short(m):
    return next((s for s in ORDER if s in m), m)


def classify(e):
    """'tool' | 'decline' (real hand-grind) | 'artifact' (clar-loop / native-miss)."""
    if e.get("used_tool"):
        return "tool"
    full = "\n\n".join(e.get("model_texts", []))
    if NATIVE.search(full):
        return "artifact"
    if FENCE.search(full):
        return "tool"           # wrote code we executed via fallback
    answered = bool(_parse_answers(full))
    if ASKAGAIN.search(e.get("model_texts", [""])[-1]) and not answered:
        return "artifact"       # clarification loop
    return "decline" if answered else "artifact"


def main():
    run = Path(sys.argv[1])
    eps = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()]
    me = [e for e in eps if not e.get("error")]
    print(f"{run}  ({len(eps)} episodes)\n")
    hdr = f"{'model':7} {'n':>4} {'Recog(tool)':>11} {'decline':>8} {'crater|dec':>10} {'Solve':>6}"
    print(hdr)
    rows = {}
    for m in ORDER:
        g = [e for e in me if _short(e["model"]) == m]
        if not g:
            continue
        cls = [classify(e) for e in g]
        tool = [e for e, c in zip(g, cls) if c == "tool"]
        dec = [e for e, c in zip(g, cls) if c == "decline"]
        crater = [e for e in dec if e["frac_correct"] == 0]
        solve = [e for e in g if e.get("all_correct")]
        rows[m] = (len(g), len(tool) / len(g), len(dec) / len(g),
                   (len(crater) / len(dec) if dec else float("nan")), len(solve) / len(g))
        n, R, D, C, S = rows[m]
        print(f"{m:7} {n:>4} {R:>11.2f} {D:>8.3f} {C:>10.2f} {S:>6.2f}")
    if {"haiku", "opus"} <= rows.keys():
        print(f"\nINVERSION (decline rate): haiku {rows['haiku'][2]:.3f} -> "
              f"opus {rows['opus'][2]:.3f}  (delta {rows['opus'][2]-rows['haiku'][2]:+.3f})")


if __name__ == "__main__":
    main()
