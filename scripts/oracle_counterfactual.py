"""Post-processor: turn ONE all-decoy rollout into n scored episodes.

For each candidate tool i, the counterfactual "tool i is the oracle" wins iff the model
either (a) submitted a correct answer manually, or (b) called tool i -- whichever came
first in token spend -- within budget:

    T_i    = meter at FIRST call to tool i              (inf if never called)
    S      = meter at correct manual submission         (inf if never submitted-correct)
    cost_i = min(S, T_i)
    win_i  = cost_i + surcharge_applied <= budget

Because the rollout stores a TOKEN-ONLY meter (meter_raw) per call, both `budget` and the
flat per-call `surcharge` are free post-hoc knobs -- re-tuning them never needs a re-run.
The surcharge applied to T_i is `surcharge * (number of calls up to and including i)`,
i.e. surcharge * (order_of_first_call_to_i + 1); the manual path S carries no surcharge.

Modeling assumption (no validation arm): receiving the oracle's correct answer => the
model submits it and wins.

CLI:  python scripts/oracle_counterfactual.py runs/oracle/<run>/episodes.jsonl
            [--budget B] [--surcharge S]
Writes scored.jsonl (n rows per rollout) next to it and prints win-rate per cell.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

INF = float("inf")
_PASSTHROUGH = ("model", "provider", "n", "difficulty", "seed", "relabel_seed",
                "problem_kind", "stopped_reason")


def _first_calls(row: dict):
    """canonical index i -> (meter_raw, order) at FIRST call to tool i (skip unknowns)."""
    name_to_idx = {name: int(idx) for idx, name in (row.get("labels") or {}).items()}
    first = {}
    for c in row.get("tool_calls", []):
        i = name_to_idx.get(c["tool"])
        if i is None or i in first:
            continue
        first[i] = (c["meter_raw"], c["order"])
    return first


def score_episode(row: dict, budget: int | None = None,
                  surcharge: int | None = None) -> list[dict]:
    """One rollout row -> n scored rows (one per hypothetical oracle i). budget/surcharge
    default to the values recorded on the row."""
    n = row["n"]
    budget = row["budget"] if budget is None else budget
    surcharge = row.get("surcharge", 0) if surcharge is None else surcharge

    sub = row.get("submit")
    S = sub["meter_raw"] if (sub and sub.get("correct")) else INF
    firsts = _first_calls(row)

    out = []
    for i in range(n):
        meter_i, order_i = firsts.get(i, (INF, None))
        # surcharge for reaching tool i: one flat charge per call up to & incl. tool i
        T_i = meter_i + surcharge * (order_i + 1) if meter_i is not INF else INF
        cost_i = min(S, T_i)
        out.append({
            **{k: row.get(k) for k in _PASSTHROUGH},
            "budget": budget, "surcharge": surcharge, "oracle_i": i,
            "T_i": (None if T_i is INF else T_i),
            "S": (None if S is INF else S),
            "cost_i": (None if cost_i is INF else cost_i),
            "win_i": cost_i <= budget,             # False when cost_i is INF
            "win_via": ("manual" if cost_i is not INF and S <= T_i
                        else "tool" if cost_i is not INF else "none"),
        })
    return out


def _cell_key(s: dict):
    return (s["model"], s["n"], s["difficulty"], s["budget"], s["surcharge"])


def main():
    argv = sys.argv[1:]

    def opt(name, cast):
        return cast(argv[argv.index(name) + 1]) if name in argv else None
    budget = opt("--budget", int)
    surcharge = opt("--surcharge", int)
    pos = [a for a in argv if not a.startswith("-")]
    if not pos:
        raise SystemExit("usage: python scripts/oracle_counterfactual.py "
                         "runs/oracle/<run>/episodes.jsonl [--budget B] [--surcharge S]")
    path = Path(pos[0])
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]

    scored, agg = [], defaultdict(lambda: [0, 0])
    for r in rows:
        for s in score_episode(r, budget=budget, surcharge=surcharge):
            scored.append(s)
            a = agg[_cell_key(s)]
            a[0] += int(s["win_i"]); a[1] += 1

    out_path = path.with_name("scored.jsonl")
    out_path.write_text("\n".join(json.dumps(s) for s in scored) + "\n")
    print(f"Scored {len(rows)} rollouts -> {len(scored)} (oracle, rep) rows: {out_path}\n")
    print(f"{'model':28s} {'n':>3s} {'diff':>4s} {'budget':>7s} {'surch':>5s} "
          f"{'win_rate':>9s} {'rows':>5s}")
    for k in sorted(agg):
        wins, tot = agg[k]
        model, n, diff, bud, surch = k
        print(f"{str(model)[:28]:28s} {n:>3d} {diff:>4d} {bud:>7d} {surch:>5d} "
              f"{wins/tot:>9.3f} {tot:>5d}")


if __name__ == "__main__":
    main()
