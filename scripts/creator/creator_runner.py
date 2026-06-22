"""Interactive CREATOR episode (one value-stripped question) for the C·R·E fork.

Turn 1: present the stripped question. If the model asks for the missing value, a
scripted user supplies it and turn 2 runs. All three metrics fall out of this one
episode via the conditioning chain ask -> build -> correct (see docs/CREATOR-fork-plan.md).

run_creator(client, model, item, item_idx) -> record dict
"""

from __future__ import annotations

import time

from scripts.creator.creator_ablation import classify_asked, fmt_val, strip_value
from scripts.creator.creator_exec import built, execute_and_check, extract_code

SYS = (
    "You solve a word problem and report a single numeric answer. You may answer "
    "directly, OR first define a reusable Python function and then call it to solve "
    "the problem -- your choice. If a value you need is missing from the problem, ask "
    "for it instead of guessing. Once you have the answer, end your reply with a line "
    "in exactly this format:\n"
    "ANSWER: <number>\n"
    "If you write code, put it in a ```python code block."
)


def _sum_usage(*usages: dict | None) -> dict:
    keys = ("input_tokens", "output_tokens", "cache_read_tokens",
            "cache_write_tokens", "reasoning_tokens")
    out = {k: 0 for k in keys}
    for u in usages:
        for k in keys:
            out[k] += int((u or {}).get(k, 0) or 0)
    return out


async def run_creator(client, model: str, item: dict, item_idx: int,
                      max_tokens: int = 1500) -> dict:
    t0 = time.time()
    base = {"model": model, "item_idx": item_idx, "question": item["question"],
            "gold_answer": item["answer"]}

    stripped = strip_value(item["question"], item["solution"])
    if stripped is None:
        return {**base, "skipped": True, "reason": "no_strippable_value",
                "elapsed_s": round(time.time() - t0, 2)}
    stripped_q, missing_name, missing_val = stripped

    # Turn 1 -- read usage synchronously right after the await (race-free under asyncio).
    msgs = [{"role": "user", "content": stripped_q}]
    text1 = await client.chat(model, SYS, msgs, max_tokens=max_tokens)
    u1 = dict(client.last_usage or {})

    asked = await classify_asked(client, text1, missing_name)

    texts = [text1]
    u2 = None
    if asked:
        msgs += [{"role": "assistant", "content": text1},
                 {"role": "user", "content": f"The {missing_name} is {fmt_val(missing_val)}."}]
        text2 = await client.chat(model, SYS, msgs, max_tokens=max_tokens)
        u2 = dict(client.last_usage or {})
        texts.append(text2)

    code = extract_code("\n\n".join(texts))
    built_flag = built(code)
    exec_answer, correct, exec_err = execute_and_check(code, item["answer"])

    return {
        **base,
        "stripped_question": stripped_q,
        "missing_name": missing_name,
        "missing_val": missing_val,
        "asked": asked,
        "built": built_flag,
        "exec_answer": exec_answer,
        "correct": bool(correct),
        "exec_error": exec_err,
        "code": code,
        "turns": len(texts),
        "model_texts": texts,
        "usage": _sum_usage(u1, u2),
        "skipped": False,
        "elapsed_s": round(time.time() - t0, 2),
    }
