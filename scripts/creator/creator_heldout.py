"""CREATOR v2 core — held-out generalization scoring (decouple grind from reward).

The model is asked for `solve(<inputs>)` and told it will be re-run on OTHER values of
those inputs. We score `solve` on held-out value-tuples whose gold answers come from the
reference `tool`+`solution`. Hardcoding the shown instance fails held-out → grind; a
correctly generalized tool passes → recognition. See docs/CREATOR-fork-plan.md "## v2".

- parse_inputs(solution)            -> [(name, value), ...]  (the init block)
- make_heldout(item, M, seed)       -> (arg_names, shown_values, shown_gold, heldout)
- has_solve(code)                   -> bool
- score(code, arg_names, shown_values, shown_gold, heldout) -> dict
"""

from __future__ import annotations

import ast
import random
import re

from scripts.creator.creator_ablation import _ASSIGN
from scripts.creator.creator_exec import (_parse_answer, _run, correct_within_tol,
                                          extract_code)


def parse_inputs(solution: str) -> list[tuple[str, float]]:
    """Numeric `name = value` assignments from the reference solution's init block.
    These are the inputs `solve` must take and that held-out tuples resample."""
    out = []
    for name, raw in _ASSIGN.findall(extract_code(solution)):
        v = float(raw.replace("_", ""))
        out.append((name, int(v) if v.is_integer() else v))
    return out


def has_solve(code: str) -> bool:
    """True iff the code defines a function named `solve`."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return False
    return any(isinstance(n, ast.FunctionDef) and n.name == "solve"
               for n in ast.walk(tree))


def resample(value, rng: random.Random):
    """A non-degenerate new value of the same int/float flavor as `value`."""
    if isinstance(value, int):
        lo = max(1, value // 3) if value else 1
        hi = max(lo + 1, value * 2 if value else 10)
        return rng.randint(lo, hi)
    base = abs(value) or 1.0
    return round(rng.uniform(0.5 * base, 2.0 * base), 4)


def _fmt(v) -> str:
    return repr(v)


def gold_for(item: dict, values: dict[str, float]) -> float | None:
    """Run the reference tool+solution with the init values overridden by `values`;
    return its printed numeric answer (None if the reference errors / prints nothing)."""
    sol = extract_code(item["solution"])
    for name, newv in values.items():
        sol = re.sub(rf"(?m)^(\s*{re.escape(name)}\s*=\s*)[-+]?\d[\d_]*\.?\d*",
                     rf"\g<1>{_fmt(newv)}", sol, count=1)
    stdout, err = _run(extract_code(item["tool"]) + "\n\n" + sol)
    return None if err else _parse_answer(stdout)


def make_heldout(item: dict, M: int = 5, seed: int = 0):
    """Build the shown tuple + M held-out tuples (with reference golds) for `item`.
    Returns (arg_names, shown_values, shown_gold, [(values, gold), ...]) or None if the
    item has no numeric inputs or fewer than M valid held-out tuples can be made."""
    inputs = parse_inputs(item["solution"])
    if not inputs:
        return None
    arg_names = [n for n, _ in inputs]
    shown_values = [v for _, v in inputs]
    # Confirm the reference reproduces the dataset answer with the original inputs.
    shown_gold = gold_for(item, {})
    if shown_gold is None or not correct_within_tol(shown_gold, item["answer"]):
        return None

    rng = random.Random(seed * 100003 + 7)
    heldout, seen = [], set()
    for _ in range(M * 4):
        if len(heldout) >= M:
            break
        vals = {n: resample(v, rng) for n, v in inputs}
        key = tuple(vals[n] for n in arg_names)
        if key in seen or key == tuple(shown_values):
            continue
        g = gold_for(item, vals)
        if g is None:
            continue
        seen.add(key)
        heldout.append(([vals[n] for n in arg_names], g))
    if len(heldout) < M:
        return None
    return arg_names, shown_values, shown_gold, heldout


def _call_solve(code: str, values: list) -> float | None:
    """Append a call to the model's `solve` on `values` and return its ANSWER (the last
    printed ANSWER line, so our appended call wins over the model's own shown-call)."""
    harness = code + '\nprint("ANSWER:", solve(' + ", ".join(_fmt(v) for v in values) + "))"
    stdout, err = _run(harness)
    return None if err else _parse_answer(stdout)


def score(code: str, arg_names: list[str], shown_values: list, shown_gold: float,
          heldout: list) -> dict:
    """Classify the model's submission.

    - shown_correct: the model's OWN output on the shown instance is correct (run the
      code as submitted) — captures grind (inline or solve, correct on the shown values).
    - generalizes: the model defined `solve` and it is correct on the shown values AND
      every held-out tuple — the reward-advancing tool.
    Grind = shown_correct & not generalizes."""
    own_stdout, own_err = _run(code)
    shown_correct = (own_err is None) and correct_within_tol(_parse_answer(own_stdout), shown_gold)

    solve = has_solve(code)
    passed = 0
    generalizes = False
    if solve:
        tool_shown = correct_within_tol(_call_solve(code, shown_values), shown_gold)
        for values, gold in heldout:
            if correct_within_tol(_call_solve(code, values), gold):
                passed += 1
        generalizes = bool(tool_shown and passed == len(heldout))
    return {
        "shown_correct": bool(shown_correct),
        "has_solve": solve,
        "heldout_pass": passed,
        "heldout_total": len(heldout),
        "generalizes": generalizes,
    }
