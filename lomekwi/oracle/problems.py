"""Pluggable hard-problem generator + answer checker for the oracle environment.

Default problem: LONG DIVISION. The difficulty knob is the dividend digit count `d`
(divisor ~ ceil(d/2) digits), chosen so solving manually is EXPENSIVE-BUT-POSSIBLE in
tokens -- each extra dividend digit is another subtract/bring-down step the model must
verbalize, so the manual-solve token cost grows steeply while correctness stays
achievable. That is the tool-vs-grind regime the experiment targets.

A Problem bundles the prompt text, the canonical true answer, and a `check(text)` that
normalizes a free-form model answer and compares it to the truth. New problem kinds just
need to return a Problem with a matching checker.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Problem:
    kind: str
    text: str                       # the problem statement shown to the model
    true_answer: str                # canonical human-readable answer
    check: Callable[[str], bool]    # normalize + compare a model answer string


# --- long division ---------------------------------------------------------------------

def _ld_check(quotient: int, remainder: int) -> Callable[[str], bool]:
    """Accept any answer that pins the integer quotient AND remainder. Tolerates
    'q remainder r', 'q r r', 'q R r', 'quotient q remainder r', and -- when r == 0 --
    a bare quotient. Pulls the first two integers as (q, r) for the general case."""
    def check(text: str) -> bool:
        if text is None:
            return False
        t = text.strip().lower().replace(",", "")
        ints = [int(x) for x in re.findall(r"-?\d+", t)]
        if not ints:
            return False
        if remainder == 0:
            # bare quotient is acceptable; also accept 'q r 0' / 'q remainder 0'
            if len(ints) == 1:
                return ints[0] == quotient
            return ints[0] == quotient and ints[1] == 0
        # need both quotient and remainder
        if len(ints) < 2:
            return False
        return ints[0] == quotient and ints[1] == remainder
    return check


def long_division(difficulty: int, seed: int) -> Problem:
    rng = random.Random(seed)
    d = max(2, int(difficulty))
    dv_digits = d
    ds_digits = max(1, d // 2)
    dividend = rng.randint(10 ** (dv_digits - 1), 10 ** dv_digits - 1)
    divisor = rng.randint(10 ** (ds_digits - 1), 10 ** ds_digits - 1)
    q, r = divmod(dividend, divisor)
    text = (f"Compute the integer quotient and remainder of the long division "
            f"{dividend} / {divisor}. Give the quotient and the remainder.")
    true = f"{q} remainder {r}"
    return Problem(kind="long_division", text=text, true_answer=true,
                   check=_ld_check(q, r))


# --- registry --------------------------------------------------------------------------

_GENERATORS = {
    "long_division": long_division,
}


def make_problem(kind: str = "long_division", difficulty: int = 6,
                 seed: int = 0) -> Problem:
    """Build a Problem of the given kind/difficulty/seed. `seed` makes the same cell
    reproducible and paired across models (mirrors the beach/toolworld seeding)."""
    if kind not in _GENERATORS:
        raise ValueError(f"unknown problem kind {kind!r}; have {list(_GENERATORS)}")
    return _GENERATORS[kind](difficulty=difficulty, seed=seed)


if __name__ == "__main__":
    for d in (3, 5, 7):
        p = make_problem("long_division", difficulty=d, seed=1)
        print(f"d={d}: {p.text}\n   true={p.true_answer}  "
              f"check(true)={p.check(p.true_answer)}  check('0')={p.check('0')}")
