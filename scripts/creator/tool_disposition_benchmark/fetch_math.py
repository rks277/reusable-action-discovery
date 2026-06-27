"""One-time fetch of the MATH dataset (test split) for the tool-disposition MATH track.

Pulls the two numeric-gold-rich subjects from the LIVE mirror EleutherAI/hendrycks_math (the
original hendrycks/competition_math was served a takedown), extracts each problem's gold answer
from the last \\boxed{} of its solution, and writes one local JSONL per subject into external/ --
matching the external/CC.jsonl pattern so the benchmark has NO Hugging Face dependency at run time.

  pip install datasets            # only needed to run THIS script, once
  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.fetch_math

Reports per-level and numeric-gold counts so you can see how many usable problems exist per level
band (this determines whether N x seeds is feasible per band).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from scripts.creator.tool_disposition_benchmark.grading_math import extract_gold, num_value

SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]
SPLIT = "test"  # published MATH numbers are on test; needed for grader validation
EXTERNAL = Path("external/MATH")


def fetch_subject(subj: str) -> list[dict]:
    ds = load_dataset("EleutherAI/hendrycks_math", subj, split=SPLIT)
    rows = []
    for ex in ds:
        gold = extract_gold(ex["solution"])
        if gold is None:
            continue
        rows.append({"problem": ex["problem"], "level": ex["level"], "type": ex["type"],
                     "solution": ex["solution"], "gold": gold,
                     "numeric": num_value(gold) is not None})
    return rows


def _level_num(lvl) -> str:
    """Normalise 'Level 3' / 3 -> '3' for stable sorting/reporting."""
    s = str(lvl).strip()
    return s.split()[-1] if s.lower().startswith("level") else s


def main():
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    summary = []
    for subj in SUBJECTS:
        rows = fetch_subject(subj)
        out = EXTERNAL / f"math_{subj}.jsonl"
        out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        n_num = sum(r["numeric"] for r in rows)
        summary.append((subj, len(rows), n_num))
        print(f"\n{subj}: {len(rows)} problems with a parseable gold ({n_num} numeric) -> {out}")
        by_level = Counter(_level_num(r["level"]) for r in rows)
        by_level_num = Counter(_level_num(r["level"]) for r in rows if r["numeric"])
        print(f"  {'level':<6}{'all':>6}{'numeric':>9}")
        for lvl in sorted(by_level, key=lambda x: (len(x), x)):
            print(f"  {lvl:<6}{by_level[lvl]:>6}{by_level_num[lvl]:>9}")
        print(f"  {'TOTAL':<6}{len(rows):>6}{n_num:>9}")

    print(f"\n{'=== numeric-gold fraction by subject (conservative) ===':}")
    print(f"  {'subject':<26}{'all':>6}{'numeric':>9}{'frac':>7}")
    for subj, n, nn in sorted(summary, key=lambda x: -(x[2] / x[1] if x[1] else 0)):
        print(f"  {subj:<26}{n:>6}{nn:>9}{(nn / n if n else 0):>7.0%}")


if __name__ == "__main__":
    main()
