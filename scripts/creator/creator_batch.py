"""CREATOR v3 core — visible variant-batch (answers-only, announced-budget).

For each base question, build N isomorphic variants rendered back into natural language
(resample the non-withheld inputs; hold ONE input constant across all N and blank it as
the shared Curiosity gate). The model sees N word-problems with one shared quantity
missing; it asks once, then solves all N. See docs/CREATOR-fork-plan.md "## v3" / plan.

- make_batch(item, N, seed) -> {arg_names, withheld_name, withheld_value, variants:[(q,gold)]} | None
- reused(code) -> bool   (a function defined and called >=2x = built-and-reused)
"""

from __future__ import annotations

import ast
import random
import re

from scripts.creator.creator_ablation import (PLACEHOLDER, _standalone_count,
                                               _token_forms)
from scripts.creator.creator_heldout import _fmt, gold_for, parse_inputs, resample


def _spans(question: str, inputs: list[tuple[str, float]]):
    """Match span of each input's value in the ORIGINAL question — exactly one
    standalone occurrence per input, and no two spans overlapping. Returns a
    start-sorted [(start, end, name)] or None if any value is missing/ambiguous/clashes.
    Spans are computed once on the original text so rendering is collision-free even
    when a resampled value coincides with another input's original number."""
    spans = []
    for name, val in inputs:
        hit = None
        for form in _token_forms(_fmt(val)):
            c, pat = _standalone_count(question, form)
            if c == 1:
                m = re.search(pat, question)
                hit = (m.start(), m.end(), name)
                break
        if hit is None:
            return None
        spans.append(hit)
    spans.sort()
    for a, b in zip(spans, spans[1:]):
        if a[1] > b[0]:          # two inputs resolve to overlapping/identical spans
            return None
    return spans


def _render(question: str, spans, repl_by_name: dict[str, str]) -> str:
    """Splice all replacements into the original question in one position-based pass."""
    out, last = [], 0
    for s, e, name in spans:
        out.append(question[last:s])
        out.append(repl_by_name[name])
        last = e
    out.append(question[last:])
    return "".join(out)


def make_batch(item: dict, N: int = 8, seed: int = 0):
    """N variant questions sharing structure, with one input held constant + blanked.
    Returns the batch dict or None (no clean inputs / too few valid variants)."""
    inputs = parse_inputs(item["solution"])
    if len(inputs) < 2:  # need one to withhold and >=1 to vary
        return None
    q = item["question"]
    spans = _spans(q, inputs)
    if spans is None:
        return None

    withheld_name, withheld_val = inputs[-1]   # hold the last init var constant + blank it
    varied = inputs[:-1]
    rng = random.Random(seed * 100003 + 7)
    variants, seen = [], set()
    for _ in range(N * 5):
        if len(variants) >= N:
            break
        vals = {n: resample(v, rng) for n, v in varied}
        vals[withheld_name] = withheld_val
        key = tuple(vals[n] for n, _ in inputs)
        if key in seen:
            continue
        gold = gold_for(item, vals)
        if gold is None:
            continue
        repl = {name: (PLACEHOLDER if name == withheld_name else _fmt(vals[name]))
                for name, _ in inputs}
        qtext = _render(q, spans, repl)
        seen.add(key)
        variants.append((qtext, gold))
    if len(variants) < N:
        return None
    return {
        "arg_names": [n for n, _ in inputs],
        "withheld_name": withheld_name,
        "withheld_value": withheld_val,
        "variants": variants,
    }


_PY_FENCE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_BARE_FENCE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
_REAL_STMT = (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For, ast.While,
              ast.FunctionDef, ast.Import, ast.ImportFrom, ast.If)


def wrote_code(text: str) -> bool:
    """True iff the reply contains a fenced block that parses as real Python — i.e.
    the model chose to externalize its work as code at all (the project's
    code-use disposition), regardless of whether it abstracted a reusable tool.
    A fenced block with only a bare literal / number dump does not count."""
    blocks = _PY_FENCE.findall(text or "") or _BARE_FENCE.findall(text or "")
    for b in blocks:
        try:
            tree = ast.parse(b)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, _REAL_STMT):
                return True
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
                return True
    return False


_LOOP_NODES = (ast.For, ast.While, ast.AsyncFor,
               ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def reused(code: str) -> bool:
    """True iff a defined function is applied across the batch: either called inside a
    loop/comprehension (the idiomatic build — define once, apply to each variant) OR
    called >=2 times statically (unrolled). A lone one-off call counts as not-reused."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return False
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if not defined:
        return False

    def calls_to_defined(node):
        return [c for c in ast.walk(node)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                and c.func.id in defined]

    for n in ast.walk(tree):                       # a defined func called within a loop
        if isinstance(n, _LOOP_NODES) and calls_to_defined(n):
            return True
    counts: dict[str, int] = {}                    # or >=2 static calls
    for c in calls_to_defined(tree):
        counts[c.func.id] = counts.get(c.func.id, 0) + 1
    return any(v >= 2 for v in counts.values())
