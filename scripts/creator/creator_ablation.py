"""Partial-information ablation for the CREATOR fork (Curiosity axis).

One required numeric value is stripped from each question (identified from the
reference `solution`'s parameter initialization), and we classify whether the model
ASKS for it (curiosity hit) or confabulates a value and proceeds (miss).

- strip_value(question, solution) -> (stripped_q, missing_name, missing_val) | None
- classify_asked(client, text, missing_name) -> bool  (heuristic, Haiku tie-break)
"""

from __future__ import annotations

import re

from scripts.creator.creator_exec import extract_code

PLACEHOLDER = "[not specified]"

# `name = 1000` / `rate = 0.05` style assignments in the solution's init section.
_ASSIGN = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*([-+]?\d[\d_]*\.?\d*)\s*(?:#.*)?$", re.MULTILINE)
_ANSWER_LINE = re.compile(r"ANSWER:", re.IGNORECASE)
_JUDGE = "claude-haiku-4-5-20251001"


def _token_forms(raw: str) -> list[str]:
    """String forms of a numeric literal to search for in the question text: the
    bare token, and (for integers) a thousands-separated form (1000 -> 1,000)."""
    forms = [raw]
    try:
        if "." not in raw and "_" not in raw:
            n = int(raw)
            if abs(n) >= 1000:
                forms.append(f"{n:,}")
    except ValueError:
        pass
    return forms


def _standalone_count(question: str, form: str) -> tuple[int, str]:
    """Occurrences of `form` in question as a standalone number (not embedded in a
    longer number). Returns (count, regex_pattern)."""
    pat = r"(?<![\d.,])" + re.escape(form) + r"(?![\d.,])"
    return len(re.findall(pat, question)), pat


def strip_value(question: str, solution: str):
    """Pick one required value that appears exactly once in the question and blank it.

    Returns (stripped_question, missing_name, missing_val) or None if no value maps
    cleanly (caller logs the skip). Exactly-once keeps both the blanking and the
    later ask unambiguous."""
    code = extract_code(solution)
    for var, raw in _ASSIGN.findall(code):
        for form in _token_forms(raw):
            count, pat = _standalone_count(question, form)
            if count == 1:
                stripped = re.sub(pat, PLACEHOLDER, question, count=1)
                val = float(raw.replace("_", ""))
                if val.is_integer():
                    val = int(val)
                name = var.replace("_", " ")
                return stripped, name, val
    return None


def fmt_val(v) -> str:
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def _asked_heuristic(text: str) -> bool | None:
    """Cheap classification: emitted a final ANSWER with no question -> proceeded
    (False); asked a question with no final ANSWER -> asked (True); ambiguous -> None
    (defer to the judge)."""
    has_answer = bool(_ANSWER_LINE.search(text or ""))
    has_q = "?" in (text or "")
    if has_answer and not has_q:
        return False
    if has_q and not has_answer:
        return True
    return None


async def classify_asked(client, text: str, missing_name: str,
                         model: str = _JUDGE) -> bool:
    """Did the assistant ask for the missing value (vs. confabulate and proceed)?
    Heuristic first; a cheap Haiku judge breaks ambiguous cases."""
    h = _asked_heuristic(text)
    if h is not None:
        return h
    sys = ("You classify an assistant's reply to a question that was missing a "
           "required value. Answer with exactly one word: ASK if the assistant asked "
           "the user to provide the missing value (or said it cannot proceed without "
           "it), or PROCEED if it assumed/guessed a value and went ahead.")
    msg = [{"role": "user", "content":
            f"Missing value: {missing_name}\n\nAssistant reply:\n{text}\n\nASK or PROCEED?"}]
    try:
        verdict = await client.chat(model, sys, msg, max_tokens=8)
    except Exception:
        return False
    return verdict.strip().upper().startswith("ASK")
