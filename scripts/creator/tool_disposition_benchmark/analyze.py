"""Per-model summary for a tool-disposition run: the four headline metrics plus disposition
descriptors.

  solve     = overall solve rate (exact match at the stated significant figures)
  effScript = Efficiency = P(correct | a script was used on that problem)
  effHand   = P(correct | solved by hand, no script)
  reuse     = mean # distinct problems each written script was applied to
  recog     = fraction of script-used problems NOT solvable by hand (tool necessity; recognition.py)
  scripts   = mean scripts written vs the 0.1N budget
  runs/prob = mean run_script calls per problem (chaining depth)

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.analyze runs/tool_disposition_<ts>/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(run_dir: str):
    p = Path(run_dir)
    f = p / "sessions.jsonl" if p.is_dir() else p
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


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

    budget = next((r.get("budget") for r in ok), None)
    print(f"{len(rows)} sessions ({len(err)} errored); write budget = {budget}\n")
    hdr = (f"{'model':<26}{'sess':>5}{'solve':>7}{'effScript':>10}{'effHand':>9}"
           f"{'reuse':>7}{'recog':>7}{'scripts':>9}{'runs/prob':>10}{'tok':>9}")
    print(hdr); print("-" * len(hdr))
    for model, rs in sorted(by_model.items()):
        n_problems = _mean([r["N"] for r in rs])
        recog = [r["recognition_rate"] for r in rs if r.get("recognition_rate") is not None]
        line = (
            f"{model.split('/')[-1]:<26}{len(rs):>5}"
            f"{_mean([r['solve_rate'] for r in rs]):>7.2f}"
            f"{_mean([r['eff_solve_given_script'] for r in rs]):>10.2f}"
            f"{_mean([r['eff_solve_by_hand'] for r in rs]):>9.2f}"
            f"{_mean([r['mean_reuse'] for r in rs]):>7.2f}"
            f"{(_mean(recog) if recog else float('nan')):>7.2f}"
            f"{_mean([r['n_scripts_written'] for r in rs]):>9.2f}"
            f"{_mean([r['n_run_calls'] / max(1, r['N']) for r in rs]):>10.2f}"
            f"{_mean([r['spent_tokens'] for r in rs]):>9.0f}"
        )
        print(line)
    print("\nsolve=exact match @ sig figs; effScript=Efficiency=P(correct|used a script); effHand="
          "P(correct|by hand); reuse=mean distinct problems per written script; recog=fraction of "
          "script-used problems NOT solvable by hand (tool necessity; run recognition.py first to "
          "populate); scripts=mean written vs budget; runs/prob=mean run_script calls per problem.")


if __name__ == "__main__":
    main()
