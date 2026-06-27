"""Validate the MATH grader (grading_math.is_equiv / correct_math) before trusting any disposition
number — a grading bug masquerades as a finding (cf. the Haiku-judge sig-fig errors).

Two independent checks:

1. FAITHFULNESS UNIT TESTS (instant, no API): a battery of known equivalent / non-equivalent
   answer pairs the canonical Hendrycks normaliser must get right. Aborts if any fail.

2. MODEL RUN + DISAGREEMENT AUDIT: solve a sample of C&P + Number-Theory problems with a real
   model (answer in \\boxed{}), grade each response two ways -- pure string is_equiv (matches the
   published MATH numbers) and correct_math (is_equiv OR numeric-equal) -- and report:
     - accuracy under each grader (should land in the believable frontier range),
     - every DISAGREEMENT between the two graders (each is a candidate grader bug / false negative
       of pure is_equiv), printed for inspection.
   Agreement of two independent graders on the same outputs validates the grader without needing
   an exact external per-subject number.

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.validate_math_grader \
      --model sonnet --sample 120
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.creator.tool_disposition_benchmark.grading_math import (correct_math, is_equiv,
                                                                     last_boxed_only_string,
                                                                     num_value, remove_boxed)

CLAUDE = {"haiku": "claude-haiku-4-5-20251001",
          "sonnet": "claude-sonnet-4-6", "opus": "claude-opus-4-8"}
SUBJECTS = ["counting_and_probability", "number_theory"]

# (a, b, expected_is_equiv) — faithfulness of the canonical normaliser
UNIT_PAIRS = [
    ("\\frac{1}{2}", "\\frac{1}{2}", True),
    ("\\frac12", "\\frac{1}{2}", True),
    ("1/2", "\\frac{1}{2}", True),
    ("0.5", "\\frac{1}{2}", True),
    ("\\dfrac{3}{8}", "\\frac{3}{8}", True),
    ("\\sqrt3", "\\sqrt{3}", True),
    ("\\left(3\\right)", "(3)", True),
    ("50\\%", "50", True),
    ("\\$5", "5", True),
    ("18\\text{ ways.}", "18", True),
    ("x = 7", "7", True),
    ("\\frac{1}{2}", "\\frac{1}{3}", False),
    ("\\frac{10}{11}", "\\frac{1}{11}", False),   # brace-strip bug would call these equal
    ("12", "21", False),
    ("\\sqrt{2}", "\\sqrt{3}", False),
]
# (a, b, expected_correct_math) — numeric fallback beyond pure string equality
NUMERIC_PAIRS = [
    ("0.375", "\\frac{3}{8}", True),
    ("3/8", "0.375", True),
    ("0.25", "\\frac{1}{4}", True),
    ("0.375", "\\frac{3}{7}", False),
    # regressions from the model-run disagreement audit
    ("90900909", "90{,}900{,}909", True),     # LaTeX {,} thousands separator
    ("-\\dfrac{1}{2}", "-\\$0.50", True),      # negative \dfrac
    ("98770", "98,\\!770", True),              # \! thousands separator
]


def run_unit_tests() -> None:
    fails = []
    for a, b, exp in UNIT_PAIRS:
        got = is_equiv(a, b)
        if got != exp:
            fails.append(f"is_equiv({a!r}, {b!r}) = {got}, expected {exp}")
    for a, b, exp in NUMERIC_PAIRS:
        got = correct_math(a, b)
        if got != exp:
            fails.append(f"correct_math({a!r}, {b!r}) = {got}, expected {exp}")
    if fails:
        print("FAITHFULNESS UNIT TESTS FAILED:")
        for f in fails:
            print("  ✗", f)
        raise SystemExit(1)
    print(f"faithfulness unit tests: {len(UNIT_PAIRS) + len(NUMERIC_PAIRS)} passed ✓")


SYS = ("Solve the math problem. Show brief working, then give your final answer on the last line "
       "as \\boxed{ANSWER} in simplest exact form (integer, or reduced fraction like \\frac{a}{b}).")


async def solve(client: RawChat, model: str, problem: str, max_tokens: int) -> str | None:
    text = await client.chat(model, SYS, [{"role": "user", "content": problem}],
                             max_tokens=max_tokens)
    return remove_boxed(last_boxed_only_string(text or ""))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--sample", type=int, default=120, help="problems pooled across C&P + NT")
    ap.add_argument("--numeric-only", action="store_true", help="restrict to numeric-gold problems")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_unit_tests()

    pool = []
    for subj in SUBJECTS:
        for r in (json.loads(l) for l in
                  Path(f"external/MATH/math_{subj}.jsonl").read_text().splitlines() if l.strip()):
            if args.numeric_only and not r["numeric"]:
                continue
            pool.append({**r, "subject": subj})
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    sample = pool[:args.sample]
    print(f"sampled {len(sample)} problems from {SUBJECTS} "
          f"({'numeric-only' if args.numeric_only else 'all parseable'})")

    load_dotenv()
    client = RawChat()
    model = CLAUDE.get(args.model, args.model)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(p):
        async with sem:
            try:
                ans = await solve(client, model, p["problem"], args.max_tokens)
            except Exception as e:
                ans = None
                p["err"] = f"{type(e).__name__}: {e}"
            return {**p, "model_answer": ans,
                    "strict": is_equiv(ans, p["gold"]),
                    "lenient": correct_math(ans, p["gold"])}

    results = await asyncio.gather(*(one(p) for p in sample))

    n = len(results)
    n_extracted = sum(1 for r in results if r["model_answer"] is not None)
    strict = sum(r["strict"] for r in results)
    lenient = sum(r["lenient"] for r in results)
    print(f"\nmodel={model}  n={n}  answer-extracted={n_extracted}/{n}")
    print(f"  accuracy (pure is_equiv, matches published): {strict}/{n} = {strict / n:.3f}")
    print(f"  accuracy (is_equiv + numeric fallback):       {lenient}/{n} = {lenient / n:.3f}")

    # disagreement audit: lenient-correct but strict-wrong = numeric fallback rescued a real answer
    # that pure string-match missed (i.e. a false negative of the published grader). Inspect each.
    disagree = [r for r in results if r["lenient"] and not r["strict"]]
    print(f"\nDISAGREEMENTS (numeric fallback rescued, pure is_equiv missed): {len(disagree)}")
    for r in disagree:
        print(f"  gold={r['gold']!r:<22} model={r['model_answer']!r:<22} "
              f"num: {num_value(r['model_answer'])} vs {num_value(r['gold'])}  [{r['subject'][:4]}]")

    # spot-list a few graded-WRONG (both graders) so we can eyeball for missed equivalences
    wrong = [r for r in results if not r["lenient"] and r["model_answer"] is not None]
    print(f"\nGRADED WRONG by both (sample up to 15 — eyeball for missed equivalences): {len(wrong)}")
    for r in wrong[:15]:
        print(f"  gold={r['gold']!r:<22} model={r['model_answer']!r:<22} [{r['subject'][:4]}]")

    out = Path("runs") / f"math_grader_validation_{args.model}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"model": model, "n": n, "strict": strict, "lenient": lenient,
                               "results": results}, indent=2))
    print(f"\nfull results -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
