"""CREATOR v4-hard — AMPLIFY the Opus tool-decline inversion.

Two levers over v4 (creator_eval_tool):
  (a) DECEPTIVELY-HARD arithmetic: same simple formula facade, but inputs resampled to large /
      messy / decimal values so the problem *looks* one-pass-easy (a capable model declines the
      free tool) yet *is* error-prone in-head (declines crater). Opus declines on simple-looking
      formulas (F=ma, P=VI, fuel=rate*dist, ...); big ugly numbers punish that without changing
      the perceived difficulty.
  (b) NO Curiosity gate: every input is shown in the rows (nothing withheld), removing the asking
      confound (Sonnet under-asks, Haiku over-asks) so Recognition = pure build-vs-grind.

Recognition here = P(used_tool) over ALL episodes; tool-DECLINE = real hand-grind (no tool, emits
answers); the inversion = decline rate rising with capability + decliners cratering.

  make_hard_batch(item, N, seed) -> {template, arg_names, varied_names, withheld_name=None,
                                     rows:[{all vars}], golds:[...]} | None
Reuses creator_eval_tool.run_eval_tool (gate auto-disabled when withheld_name is None).
"""

from __future__ import annotations

import random

from scripts.creator.creator_batch import _render, _spans
from scripts.creator.creator_heldout import _fmt, gold_for, parse_inputs


def hard_resample(value, rng: random.Random):
    """A same-flavour but arithmetically punishing value: 3-4 significant digits, messy.
    Ints stay ints (keeps range()/count formulas valid) but grow to 3-4 digits; floats
    become multi-decimal. Magnitude is deliberately large so mental arithmetic is unreliable."""
    if isinstance(value, int):
        return rng.randint(137, 9973)                 # 3-4 digit int; products -> 6-8 digits
    base = abs(value) or 1.0
    return round(rng.uniform(50 * base, 500 * base) + rng.random(), 3)


def make_hard_batch(item: dict, N: int, seed: int, withhold: bool = False):
    """N rows of hard-resampled inputs rendered into the template.
    withhold=False: all inputs shown (no Curiosity gate).
    withhold=True : hold the LAST input constant (a fixed hard value) and blank it from the
    rows -> the model must ask for it (Curiosity gate), recovering the full C·R·E·Solve chain."""
    inputs = parse_inputs(item["solution"])
    if len(inputs) < (2 if withhold else 1):
        return None
    q = item["question"]
    spans = _spans(q, inputs)
    if spans is None:
        return None
    template = _render(q, spans, {name: name.upper() for name, _ in inputs})

    rng = random.Random(seed * 100003 + 17)
    if withhold:
        withheld_name, withheld_orig = inputs[-1]
        withheld_val = hard_resample(withheld_orig, rng)   # one fixed hard constant
        varied = inputs[:-1]
    else:
        withheld_name, withheld_val, varied = None, None, inputs

    rows, golds, seen = [], [], set()
    for _ in range(N * 6):
        if len(rows) >= N:
            break
        vals = {n: hard_resample(v, rng) for n, v in varied}
        if withhold:
            vals[withheld_name] = withheld_val
        key = tuple(vals[n] for n, _ in inputs)
        if key in seen:
            continue
        gold = gold_for(item, vals)
        if gold is None or gold != gold or abs(gold) == float("inf"):   # skip nan/inf blowups
            continue
        seen.add(key)
        rows.append({n: vals[n] for n, _ in varied})   # only varied vars shown; withheld blanked
        golds.append(gold)
    if len(rows) < N:
        return None
    return {
        "template": template,
        "arg_names": [n for n, _ in inputs],
        "varied_names": [n for n, _ in varied],
        "withheld_name": withheld_name,
        "withheld_value": withheld_val,
        "rows": rows,
        "golds": golds,
    }
