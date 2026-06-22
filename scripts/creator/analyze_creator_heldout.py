"""CREATOR v2 metrics — held-out generalization, per model.

  Recognition  = P(generalizes)                         tool advances the decoupled reward
  Grind G      = P(shown-correct & not generalizes)     solved shown but doesn't generalize
  Solve(shown) = P(shown-correct)                        ~= Recognition + G
  Diagnostics  = P(has_solve), P(generalizes | has_solve)

  PYTHONPATH=. python -m scripts.creator.analyze_creator_heldout runs/creator_heldout_<ts>/episodes.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def _ratio(num: int, den: int):
    return (num / den if den else float("nan"), den)


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    gen = [r for r in rows if r.get("generalizes")]
    shown = [r for r in rows if r.get("shown_correct")]
    grind = [r for r in shown if not r.get("generalizes")]
    has = [r for r in rows if r.get("has_solve")]
    return {
        "recognition": _ratio(len(gen), n),                 # P(generalizes)
        "grind": _ratio(len(grind), n),                     # P(shown & not gen)
        "solve rate": _ratio(len(shown), n),                # P(shown-correct)
        "has_solve": _ratio(len(has), n),                   # diagnostic
        "gen|solve": _ratio(sum(bool(r.get("generalizes")) for r in has), len(has)),  # diagnostic
    }


SERIES = ["recognition", "grind", "solve rate", "has_solve", "gen|solve"]


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: analyze_creator_heldout.py runs/creator_heldout_<ts>/episodes.jsonl")
    path = Path(sys.argv[1])
    by = defaultdict(list)
    n_skip = n_err = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error"):
            n_err += 1
        elif r.get("skipped"):
            n_skip += 1
        else:
            by[r["model"]].append(r)
    if n_skip or n_err:
        print(f"(excluded {n_skip} skipped, {n_err} errored)")

    order = {"haiku": 0, "sonnet": 1, "opus": 2}
    out = path.parent / "creator_heldout_metrics.jsonl"
    with out.open("w") as f:
        for model in sorted(by, key=lambda m: order.get(m.split("-")[1] if m.startswith("claude-") else m, 9)):
            rows = by[model]
            m = metrics(rows)
            f.write(json.dumps({"model": model, "n_episodes": len(rows), **m}) + "\n")
            rec = m["recognition"][0]; gr = m["grind"][0]; sv = m["solve rate"][0]
            print(f"{model:28} n={len(rows):4}  " +
                  "  ".join(f"{k}={m[k][0]:.2f}(n={m[k][1]})" for k in SERIES) +
                  f"   [check R+G={rec+gr:.2f} vs solve={sv:.2f}]")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
