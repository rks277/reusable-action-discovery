"""CREATOR v3 metrics — C·R·E chain in the visible-batch / announced-budget setting.

  Curiosity   = P(asks for the withheld shared value)
  Recognition = P(built a reusable tool | asked)        build = creator_batch.reused
  Efficiency  = P(all N correct | asked & built)
  Solve       = P(all N correct)                        = C·R·E + G
  (also: mean per-variant accuracy; grind = solved-all & not via ask->build)

  PYTHONPATH=. python -m scripts.creator.analyze_creator_batch runs/creator_batch_<ts>/episodes.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def _ratio(num, den):
    return (num / den if den else float("nan"), den)


def metrics(rows):
    n = len(rows)
    asked = [r for r in rows if r.get("asked")]
    ab = [r for r in asked if r.get("built")]
    allc = [r for r in rows if r.get("all_correct")]
    grind = [r for r in allc if not (r.get("asked") and r.get("built"))]
    mean_acc = sum(r.get("frac_correct", 0.0) for r in rows) / n if n else float("nan")
    return {
        "curiosity": _ratio(len(asked), n),
        "recognition": _ratio(sum(bool(r.get("built")) for r in asked), len(asked)),
        "efficiency": _ratio(sum(bool(r.get("all_correct")) for r in ab), len(ab)),
        "solve rate": _ratio(len(allc), n),
        "grind": _ratio(len(grind), n),
        "mean_acc": (round(mean_acc, 3), n),
    }


SERIES = ["recognition", "curiosity", "efficiency", "solve rate", "grind", "mean_acc"]


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: analyze_creator_batch.py runs/creator_batch_<ts>/episodes.jsonl")
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
    out = path.parent / "creator_batch_metrics.jsonl"
    with out.open("w") as f:
        for model in sorted(by, key=lambda m: order.get(m.split("-")[1] if m.startswith("claude-") else m, 9)):
            rows = by[model]
            m = metrics(rows)
            f.write(json.dumps({"model": model, "n_episodes": len(rows), **m}) + "\n")
            print(f"{model:28} n={len(rows):4}  " +
                  "  ".join(f"{k}={m[k][0]:.2f}(n={m[k][1]})" for k in SERIES))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
