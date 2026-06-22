"""CREATOR v2 episode — announced held-out generalization (single turn).

The model is told up front that its `solve` will be re-run on other input values (the
visible obstacle), then we score it on held-out tuples. See docs/CREATOR-fork-plan.md "## v2".

run_heldout(client, model, item, idx, payload) -> record dict
The held-out `payload` (from creator_heldout.make_heldout) is precomputed once per item
in the sweep so every model is scored on identical tuples.
"""

from __future__ import annotations

import time

from scripts.creator.creator_exec import extract_code
from scripts.creator.creator_heldout import score

SYS = (
    "You solve a word problem by writing a reusable Python function. Define "
    "`solve(...)` that COMPUTES the answer from its arguments — do NOT hard-code the "
    "answer. Your `solve` function will be re-run on OTHER values of these inputs, so it "
    "must work for any valid inputs, not only the ones given. Then call `solve` on the "
    "given values and end your reply with a line in exactly this format:\n"
    "ANSWER: <number>\n"
    "Put your code in a ```python code block."
)


async def run_heldout(client, model: str, item: dict, idx: int, payload,
                      max_tokens: int = 1500) -> dict:
    t0 = time.time()
    arg_names, shown_values, shown_gold, heldout = payload
    sig = "solve(" + ", ".join(arg_names) + ")"
    user = (f"{item['question']}\n\nImplement `{sig}` — the arguments are, in exactly "
            f"this order: {', '.join(arg_names)}.")

    text = await client.chat(model, SYS, [{"role": "user", "content": user}],
                             max_tokens=max_tokens)
    usage = dict(client.last_usage or {})
    code = extract_code(text)
    s = score(code, arg_names, shown_values, shown_gold, heldout)

    return {
        "model": model,
        "item_idx": idx,
        "question": item["question"],
        "arg_names": arg_names,
        "shown_gold": shown_gold,
        "heldout_total": s["heldout_total"],
        "heldout_pass": s["heldout_pass"],
        "shown_correct": s["shown_correct"],
        "has_solve": s["has_solve"],
        "generalizes": s["generalizes"],
        "code": code,
        "model_text": text,
        "usage": usage,
        "skipped": False,
        "elapsed_s": round(time.time() - t0, 2),
    }
