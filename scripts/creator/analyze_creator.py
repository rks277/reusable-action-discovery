"""Compute the four C·R·E metrics per model from a CREATOR sweep, via the
conditioning chain that mirrors ToolWorld (ask -> build -> correct).

  Curiosity   = P(asked)                 over all (non-skipped) episodes
  Recognition = P(built | asked)         denom = askers
  Efficiency  = P(correct | asked&built) denom = asker-builders
  Solve rate  = P(correct)               over all episodes

Each metric is a (value, denominator) pair so plot_metric_lines_creator can draw the
same binomial-SE error bars as the ToolWorld figure.

  PYTHONPATH=. python -m scripts.creator.analyze_creator runs/creator_sweep_<ts>/episodes.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def _ratio(num: int, den: int):
    return (num / den if den else float("nan"), den)


def metrics_creator(rows: list[dict]) -> dict:
    """rows = all non-skipped, non-error episodes for one model."""
    asked = [r for r in rows if r.get("asked")]
    asked_built = [r for r in asked if r.get("built")]
    # Grind = correct WITHOUT going through the ask->build tool path (the residual G in
    # the identity Solve = C*R*E + G). Denominator is all episodes, like solve rate.
    grind = sum(1 for r in rows
                if r.get("correct") and not (r.get("asked") and r.get("built")))
    return {
        "curiosity": _ratio(len(asked), len(rows)),
        "recognition": _ratio(sum(bool(r.get("built")) for r in asked), len(asked)),
        "efficiency": _ratio(sum(bool(r.get("correct")) for r in asked_built), len(asked_built)),
        "solve rate": _ratio(sum(bool(r.get("correct")) for r in rows), len(rows)),
        "grind": _ratio(grind, len(rows)),
    }


def load(path: Path) -> dict[str, list[dict]]:
    by_model: dict[str, list[dict]] = defaultdict(list)
    n_skip = n_err = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error"):
            n_err += 1
            continue
        if r.get("skipped"):
            n_skip += 1
            continue
        by_model[r["model"]].append(r)
    if n_skip or n_err:
        print(f"(excluded {n_skip} skipped, {n_err} errored episodes)")
    return by_model


SERIES = ["recognition", "curiosity", "efficiency", "solve rate", "grind"]


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: analyze_creator.py runs/creator_sweep_<ts>/episodes.jsonl")
    path = Path(sys.argv[1])
    by_model = load(path)
    out = path.parent / "creator_metrics.jsonl"
    with out.open("w") as f:
        # Order rows by the short-name capability ladder when present.
        order = {"haiku": 0, "sonnet": 1, "opus": 2}
        for model in sorted(by_model, key=lambda m: order.get(m.split("-")[1] if m.startswith("claude-") else m, 99)):
            rows = by_model[model]
            m = metrics_creator(rows)
            rec = {"model": model, "n_episodes": len(rows), **m}
            f.write(json.dumps(rec) + "\n")
            print(f"{model:28} n={len(rows):4}  " +
                  "  ".join(f"{k}={m[k][0]:.2f}(n={m[k][1]})" for k in SERIES))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
