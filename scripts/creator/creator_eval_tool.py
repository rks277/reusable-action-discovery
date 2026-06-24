"""CREATOR v4 — template + N value-rows + a FREE `evaluate` tool (text-action loop).

Hypothesis (see chat 6-22): high N makes in-head grinding error-prone, so a free code
executor makes build-and-run the *reliable* path. Recognition becomes a real action we
observe — did the model CALL evaluate — instead of a `def`-vs-inline coding-style guess.
Tool-DECLINE under high N (grind and submit anyway) is the overconfidence signature that
could reproduce the ToolWorld inversion.

The model sees the word problem ONCE with variable names, then N input rows (varied vars
only; one shared var is withheld -> Curiosity ask gate). It may emit
`EVALUATE:\n```python ... ``` ` any number of times; we run it (creator_exec._run) and
return stdout, then it gives `ANSWER_i:` lines.

  make_template_batch(item, N, seed) -> dict | None
  run_eval_tool(client, model, item, idx, batch, N) -> record
"""

from __future__ import annotations

import random
import re
import time

from scripts.creator.creator_ablation import classify_asked, fmt_val
from scripts.creator.creator_batch import _render, _spans
from scripts.creator.creator_exec import (_NUM, _run, _to_float,
                                          correct_within_tol, extract_code)
from scripts.creator.creator_heldout import _fmt, gold_for, parse_inputs, resample

MAX_EVAL_CALLS = 4      # cap executor round-trips per episode
MAX_TURNS = 8           # hard cap on model turns (ask + evaluates + final)
MAX_TOKENS = 4000       # generous: N=100 needs ~100 answer lines


def make_template_batch(item: dict, N: int, seed: int):
    """One template (values -> UPPERCASE variable names) + N value-rows with golds.
    Holds the LAST input constant + withholds it from the rows (Curiosity gate)."""
    inputs = parse_inputs(item["solution"])
    if len(inputs) < 2:
        return None
    q = item["question"]
    spans = _spans(q, inputs)
    if spans is None:
        return None

    withheld_name, withheld_val = inputs[-1]
    varied = inputs[:-1]
    template = _render(q, spans, {name: name.upper() for name, _ in inputs})

    rng = random.Random(seed * 100003 + 7)
    rows, golds, seen = [], [], set()
    for _ in range(N * 5):
        if len(rows) >= N:
            break
        vals = {n: resample(v, rng) for n, v in varied}
        vals[withheld_name] = withheld_val
        key = tuple(vals[n] for n, _ in inputs)
        if key in seen:
            continue
        gold = gold_for(item, vals)
        if gold is None:
            continue
        seen.add(key)
        rows.append({n: vals[n] for n, _ in varied})   # varied vars only; withheld hidden
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


def _format_rows(batch: dict, mark_gap: bool = False) -> str:
    # mark_gap: append the withheld variable as a visible blank (e.g. "TIME=?") so a weak
    # model can PERCEIVE the missing input instead of having to infer it from a template var
    # that never appears in any row. It still chooses whether to ask -> Curiosity preserved.
    gap = (batch.get("withheld_name") if mark_gap else None)
    lines = []
    for i, row in enumerate(batch["rows"], 1):
        cells = [f"{n.upper()}={fmt_val(row[n])}" for n in batch["varied_names"]]
        if gap:
            cells.append(f"{gap.upper()}=?")
        lines.append(f"Row {i}: {', '.join(cells)}")
    return "\n".join(lines)


def _sys(N: int, tool_policy: str = "free", announce_gap: bool = False) -> str:
    # Lever 1: the executor's stated cost is what drives the use-vs-skip decision. "free"
    # is the baseline; "costly" raises the threshold to invoke it (more declines, sharper
    # inversion); "budget1" hard-frames a single allowed call.
    if tool_policy == "costly":
        avail = ("You have a code executor, but it is EXPENSIVE and rate-limited: every call "
                 "counts heavily against your budget and slows you down. Use it ONLY for "
                 "computations you genuinely cannot do reliably by hand; answer directly "
                 "whenever you are confident. To use it, emit a block exactly like:\n"
                 "EVALUATE:\n```python\n<code that prints results>\n```\n")
    elif tool_policy == "budget1":
        avail = ("You have a code executor but may call it AT MOST ONCE for the entire batch, "
                 "and the call is costly — so only use it if you truly cannot compute the "
                 "answers yourself. To use it, emit a block exactly like:\n"
                 "EVALUATE:\n```python\n<code that prints results>\n```\n")
    else:
        avail = ("You have a code executor. To use it, emit a block exactly like:\n"
                 "EVALUATE:\n```python\n<code that prints results>\n```\n"
                 "You may call it as many times as you like; it is free.\n")
    # announce_gap: make the *existence* of a missing input explicit (without telling the model
    # to ask or to build) so weak models perceive the gap; the ask-vs-guess choice stays theirs.
    gap_note = (" Note: each row lists only SOME of the inputs the problem needs; one required "
                "quantity is the same for every row and is NOT provided below."
                if announce_gap else "")
    return (
        "You are given ONE word problem stated with variable names, then "
        f"{N} input rows. Compute the answer for every row.\n\n"
        + avail +
        "I will run it and reply with its stdout. When you are done, end your reply with one "
        "line per row, in order:\nANSWER_1: <number>\nANSWER_2: <number>\n...(through the last "
        "row)\nIf a quantity needed to solve the problems is missing, ask for it instead of guessing."
        + gap_note
    )


_ANS = re.compile(r"ANSWER[_ ]?(\d+)\s*[:=]\s*([^\n]+)", re.IGNORECASE)
_EVAL = re.compile(r"EVALUATE:\s*```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
# Models often ignore the EVALUATE convention and reach for their trained-in native
# tool-call syntax: <invoke name="..."><parameter name="code">...</parameter>. Catch it too
# so recognition isn't undercounted (see docs/6-22-summary.md "evaluate-tool detection").
_NATIVE = re.compile(
    r'<invoke\s+name="[^"]*">.*?<parameter\s+name="code">\s*(.*?)\s*</parameter>',
    re.DOTALL | re.IGNORECASE)


def _parse_answers(text: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for m in _ANS.finditer(text or ""):
        nums = _NUM.findall(m.group(2))
        if nums:
            v = _to_float(nums[0])
            if v is not None:
                out[int(m.group(1))] = v
    return out


def _eval_block(text: str) -> str | None:
    """Code the model wants run, from EITHER the EVALUATE convention OR its native
    <invoke name=...><parameter name="code"> tool-call syntax, or a lone ```python block."""
    m = _EVAL.search(text or "") or _NATIVE.search(text or "")
    if m:
        return m.group(1)
    code = extract_code(text or "")
    return code if code.strip() and code.strip() != (text or "").strip() else None


async def run_eval_tool(client, model: str, item: dict, idx: int, batch: dict,
                        N: int, max_tokens: int = MAX_TOKENS,
                        tool_policy: str = "free", judge_model: str | None = None,
                        gap_visibility: str = "hidden") -> dict:
    t0 = time.time()
    golds = batch["golds"]
    has_gate = batch.get("withheld_name") is not None      # None => no Curiosity gate (v4-hard)
    withheld_disp = batch["withheld_name"].replace("_", " ") if has_gate else ""
    # Make the withheld input perceptible to weak models without forcing the ask:
    #   hidden    -> silently omitted (original; the gap must be inferred)
    #   marked    -> shown as a blank slot "<VAR>=?" in every row
    #   announced -> marked + the system prompt states a required quantity is missing
    # Only meaningful when the Curiosity gate is on; a no-gate batch ignores it.
    mark_gap = has_gate and gap_visibility in ("marked", "announced")
    announce_gap = has_gate and gap_visibility == "announced"
    sys = _sys(N, tool_policy, announce_gap=announce_gap)
    eval_cap = 1 if tool_policy == "budget1" else MAX_EVAL_CALLS
    user0 = (f"Problem (same structure for every row):\n{batch['template']}\n\n"
             f"Here are {N} rows of inputs:\n{_format_rows(batch, mark_gap=mark_gap)}")
    msgs = [{"role": "user", "content": user0}]

    texts, asked, supplied, n_eval = [], False, False, 0
    usage: dict = {}   # accumulated token usage for the model under test (excl. the judge)
    for _ in range(MAX_TURNS):
        reply = await client.chat(model, sys, msgs, max_tokens=max_tokens)
        for k, v in (client.last_usage or {}).items():   # capture before any judge call
            usage[k] = usage.get(k, 0) + int(v or 0)
        texts.append(reply)
        msgs.append({"role": "assistant", "content": reply})

        code = _eval_block(reply)
        if code is not None and n_eval < eval_cap:
            n_eval += 1
            stdout, err = _run(code)
            obs = (stdout or "").strip() or (f"(error) {err}" if err else "(no output)")
            msgs.append({"role": "user", "content": f"stdout:\n{obs[:3000]}"})
            continue
        if has_gate and not supplied and not _parse_answers(reply):
            _jkw = {"model": judge_model} if judge_model else {}
            if await classify_asked(client, reply, withheld_disp, **_jkw):
                asked = True
                supplied = True
                msgs.append({"role": "user",
                             "content": f"The {withheld_disp} is "
                                        f"{fmt_val(batch['withheld_value'])} in every row."})
                continue
        break   # produced answers (or stuck) and nothing left to run

    full = "\n\n".join(texts)
    used_tool = n_eval > 0
    ans = _parse_answers(full)
    n_correct = sum(1 for i in range(1, N + 1)
                    if correct_within_tol(ans.get(i), golds[i - 1]))
    return {
        "model": model,
        "item_idx": idx,
        "N": N,
        "tool_policy": tool_policy,
        "gap_visibility": gap_visibility,
        "withheld_name": batch["withheld_name"],
        "asked": asked,
        "used_tool": used_tool,
        "n_eval_calls": n_eval,
        "built": used_tool,            # recognition = chose to externalize via the executor
        "n_correct": n_correct,
        "all_correct": n_correct == N,
        "frac_correct": round(n_correct / N, 3),
        "model_texts": texts,
        "usage": usage,
        "skipped": False,
        "elapsed_s": round(time.time() - t0, 2),
    }
