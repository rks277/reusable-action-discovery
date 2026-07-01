"""Precision grading for the tool-disposition benchmark.

Unlike the rest of CREATOR (1% relative tolerance, see creator_exec.correct_within_tol),
this benchmark grades by EXACT MATCH at a stated number of significant figures `d`: an
answer is correct iff it rounds to the same `d`-sig-fig value as the gold. Precision is the
difficulty knob — each problem statement tells the model the required `d`.
"""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation


def as_exact_int(v):
    """Exact Python int from an int / integral float / numeric string (commas & underscores
    stripped), via Decimal so huge values NEVER route through float. None if not an integer.

    This is the grading path for integer-exact families whose golds exceed 2**53 (e.g. products,
    Horner polynomials, continued fractions) — float rounding to d sig-figs would mark a correct
    answer wrong because float only holds ~15-16 significant digits."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v) if v.is_integer() else None
    if isinstance(v, str):
        s = v.strip().replace(",", "").replace("_", "")
        try:
            d = Decimal(s)
        except InvalidOperation:
            return None
        return int(d) if d == d.to_integral_value() else None
    return None


def correct_exact_int(ans, gold) -> bool:
    """True iff `ans` equals the integer `gold` EXACTLY (arbitrary precision, no float)."""
    a, g = as_exact_int(ans), as_exact_int(gold)
    return a is not None and g is not None and a == g


def round_sig(x: float, d: int) -> float:
    """Round x to d significant figures. 0 and non-finite values pass through."""
    x = float(x)
    if x == 0.0 or not math.isfinite(x):
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + (d - 1))


def correct_to_sigfigs(ans, gold, d: int) -> bool:
    """True iff `ans` rounds to the same d-significant-figure value as `gold`.

    A tiny relative slack (1e-9) absorbs float-representation noise so that two values that
    genuinely agree to d sig figs compare equal; it is far smaller than one unit in the d-th
    figure, so it never lets a wrong answer through."""
    if ans is None:
        return False
    try:
        a, g = float(ans), float(gold)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(a) and math.isfinite(g)):
        return False
    ra, rg = round_sig(a, d), round_sig(g, d)
    return math.isclose(ra, rg, rel_tol=1e-9, abs_tol=1e-12)


def sigfigs_meaningful(gold: float, d: int) -> bool:
    """True iff all d significant figures actually matter for this gold — i.e. truncating to
    (d-1) figs would be graded WRONG at d. Filters out golds like 50.0 where 'to 6 sig figs'
    is trivial, keeping only problems where the precision requirement bites."""
    if d <= 1:
        return True
    return not correct_to_sigfigs(round_sig(gold, d - 1), gold, d)
