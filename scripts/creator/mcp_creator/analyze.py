"""Per-model summary for an MCP CREATOR run: solve rate + the protocol-adherence funnel
(wrote script -> ran on example -> began test -> submitted -> solved), tool usage in the
test, token spend vs cap, and tool-calling reliability (malformed/refused calls).

  PYTHONPATH=. python -m scripts.creator.mcp_creator.analyze runs/mcp_creator_<ts>/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(run_dir: str):
    p = Path(run_dir)
    f = p / "episodes.jsonl" if p.is_dir() else p
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def _pct(xs) -> str:
    return f"{(100.0 * sum(1 for x in xs if x) / len(xs)):.0f}%" if xs else "-"


def _mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    rows = _load(sys.argv[1])
    ok = [r for r in rows if not r.get("error")]
    err = [r for r in rows if r.get("error")]
    by_model: dict[str, list] = {}
    for r in ok:
        by_model.setdefault(r["model"], []).append(r)

    print(f"{len(rows)} episodes ({len(err)} errored)\n")
    hdr = (f"{'model':<34}{'n':>4}{'solve':>7}{'frac':>7}{'wrote':>7}{'ranEx':>7}"
           f"{'test':>6}{'subm':>6}{'toolTest':>9}{'tok':>8}{'hitCap':>7}{'malf':>6}")
    print(hdr); print("-" * len(hdr))
    for model, rs in sorted(by_model.items()):
        explore_runs = [(r.get("n_run_calls", 0) - r.get("n_run_calls_in_test", 0)) > 0 for r in rs]
        line = (
            f"{model.split('/')[-1]:<34}{len(rs):>4}"
            f"{_pct([r['all_correct'] for r in rs]):>7}"
            f"{_mean([r['frac_correct'] for r in rs]):>7.2f}"
            f"{_pct([r['wrote_script'] for r in rs]):>7}"
            f"{_pct(explore_runs):>7}"
            f"{_pct([r['called_begin_test'] for r in rs]):>6}"
            f"{_pct([r['submitted'] for r in rs]):>6}"
            f"{_pct([r['used_run_in_test'] for r in rs]):>9}"
            f"{_mean([r['spent_tokens'] for r in rs]):>8.0f}"
            f"{_pct([r['hit_cap'] for r in rs]):>7}"
            f"{_mean([r['n_malformed_tool_calls'] for r in rs]):>6.1f}"
        )
        print(line)
    print("\nfunnel = wrote_script -> ranEx (ran on example) -> test (begin_test) -> subm "
          "(submit_answers) -> solve (all 20 correct); toolTest = % that ran a script during "
          "the test; tok = mean tokens spent; malf = mean malformed tool calls.")


if __name__ == "__main__":
    main()
