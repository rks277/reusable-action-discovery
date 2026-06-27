"""Grading utilities for the MATH track of the tool-disposition benchmark.

MATH answers are symbolic strings (`\\frac{3}{5}`, `2\\sqrt{3}`, `(-\\infty, 3]`, `42`), not floats
like the CREATOR track. This module houses everything answer-string related so the gold extractor
and the (forthcoming) `is_equiv` equivalence grader live in one place:

  - last_boxed_only_string / remove_boxed -- pull the gold answer out of a MATH `solution` (the
    last \\boxed{...}); the canonical Hendrycks extraction.
  - num_value -- best-effort parse of an answer string to a float (ints, decimals, simple a/b and
    \\frac{a}{b} fractions, percentages). Returns None when the answer is not numeric. Used both
    for the `numeric_only` dataset filter and for benefit-attribution (comparing a script's float
    return to a fractional gold like \\frac{3}{8}).

The full `is_equiv` string-normalising grader is added in the next step (validated against the
published MATH accuracy before any disposition number is trusted).
"""

from __future__ import annotations

import math
import re


def last_boxed_only_string(s: str) -> str | None:
    """Return the LAST `\\boxed{...}` (or `\\fbox{...}`) substring of `s`, braces included, or
    None if there is none. Brace-balanced so nested `{}` inside the answer are handled."""
    idx = s.rfind("\\boxed")
    if idx < 0:
        idx = s.rfind("\\fbox")
        if idx < 0:
            return None
    i = s.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[idx:j + 1]
    return None


def remove_boxed(s: str | None) -> str | None:
    """Strip the `\\boxed{...}` / `\\fbox{...}` wrapper, returning just the inner answer."""
    if s is None:
        return None
    for pre in ("\\boxed{", "\\fbox{"):
        if s.startswith(pre) and s.endswith("}"):
            return s[len(pre):-1].strip()
    # tolerate `\boxed ...` without braces
    m = re.match(r"\\(?:boxed|fbox)\s+(.+)", s)
    return m.group(1).strip() if m else s.strip()


def extract_gold(solution: str) -> str | None:
    """The gold answer for a MATH problem = inner of the last \\boxed{} in its solution."""
    return remove_boxed(last_boxed_only_string(solution))


_FRAC = re.compile(r"^(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}$")
_SIMPLE_FRAC = re.compile(r"^(-?\d+)\s*/\s*(-?\d+)$")


def num_value(x) -> float | None:
    """Best-effort float value of an answer (string or number); None if not numeric.

    Handles plain ints/decimals, `a/b`, `\\frac{a}{b}` / `\\dfrac{a}{b}`, a trailing `%`, `\\$`,
    `\\!`, `\\,`, and surrounding `$...$`. Anything with a radical, variable, interval, or matrix
    returns None (genuinely non-numeric -> not script-addressable)."""
    if isinstance(x, (int, float)):
        return float(x)
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    # strip common LaTeX cosmetics and delimiters
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "").replace(" ", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.strip("$").replace("\\$", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("{,}", "").replace(",", "")  # LaTeX `{,}` and plain thousands separators
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    m = _FRAC.match(s)
    if m:
        sign = -1 if m.group(1) == "-" else 1
        num, den = int(m.group(2)), int(m.group(3))
        return sign * num / den if den else None
    m = _SIMPLE_FRAC.match(s)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        return num / den if den else None
    # last resort: a bare integer left wrapped in braces (residual {…} separators)
    try:
        return float(s.replace("{", "").replace("}", ""))
    except ValueError:
        return None


# --------------------------------------------------------------------------- is_equiv
# Faithful port of the canonical Hendrycks MATH equivalence grader
# (github.com/hendrycks/math, math_equivalence.py). This is the normaliser the PUBLISHED MATH
# accuracy numbers are computed with, so it is the right baseline to validate against. We add a
# numeric fallback in `correct_math` below (string-equiv OR numerically-equal) so e.g. 0.375 and
# \frac{3}{8} grade equal -- which also lets benefit-attribution compare a script's float return
# to a fractional gold.


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    new_str += "{" + a + "}{" + b + "}" + substr[2:]
                else:
                    new_str += "{" + a + "}" + b + substr[2:]
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        ai, bi = int(a), int(b)
        if string != f"{ai}/{bi}":
            return string
        return "\\frac{" + str(ai) + "}{" + str(bi) + "}"
    except ValueError:
        return string


def _remove_right_units(string: str) -> str:
    # "\text{ " only ever describes trailing units in the MATH val set
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace(r"\%", "").replace("%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def is_equiv(str1, str2) -> bool:
    """Canonical MATH string-equivalence (normalise both, compare). False if either is None."""
    if str1 is None or str2 is None:
        return False
    try:
        return _strip_string(str(str1)) == _strip_string(str(str2))
    except Exception:
        return str(str1) == str(str2)


def correct_math(answer, gold) -> bool:
    """The grader the benchmark uses: canonical string-equivalence OR numeric equality (so
    0.375 == \\frac{3}{8}). `answer` may be a model string or a script's numeric return."""
    if answer is None:
        return False
    if is_equiv(answer, gold):
        return True
    a, g = num_value(answer), num_value(gold)
    if a is not None and g is not None:
        return math.isclose(a, g, rel_tol=1e-9, abs_tol=1e-12)
    return False
