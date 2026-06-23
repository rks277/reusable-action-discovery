"""CREATOR v3 episode — visible variant-batch with a PURE-ANNOUNCED token budget.

The model sees N structurally-identical word problems (one shared quantity blanked),
is told it has a small token budget (perceived scarcity, NOT hard-enforced), and is
asked for the N answers showing Python work. Grind = N inline computations; build =
one function applied across the batch (caught by creator_batch.reused). It may ask once
for the withheld shared value (Curiosity). C·R·E chain = ask -> build|ask -> correct|built.

run_batch(client, model, item, idx, batch) -> record dict
"""

from __future__ import annotations

import re
import time

from scripts.creator.creator_ablation import classify_asked, fmt_val
from scripts.creator.creator_batch import reused, wrote_code
from scripts.creator.creator_exec import _NUM, _to_float, correct_within_tol, extract_code

T_ANNOUNCE = 400  # announced token budget (perceived scarcity; not hard-enforced)
MAX_TOKENS = 2000  # generous, so the announcement is purely psychological (no truncation)

SYS = (
    "You are given several word problems that share the same underlying structure. "
    "Solve ALL of them and report each answer. "
    "End your reply with one line per problem, in order:\n"
    "ANSWER_1: <number>\nANSWER_2: <number>\n...(through the last problem)\n"
    f"You have a budget of about {T_ANNOUNCE} output tokens for your entire reply, so be "
    "economical. If a quantity needed to solve the problems is missing, ask for it "
    "instead of guessing."
)

_ANS = re.compile(r"ANSWER[_ ]?(\d+)\s*[:=]\s*([^\n]+)", re.IGNORECASE)


def _parse_answers(text: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for m in _ANS.finditer(text or ""):
        nums = _NUM.findall(m.group(2))
        if nums:
            v = _to_float(nums[0])
            if v is not None:
                out[int(m.group(1))] = v   # last occurrence wins
    return out


def _format_batch(variants) -> str:
    qs = "\n\n".join(f"{i+1}. {q}" for i, (q, _) in enumerate(variants))
    return f"Here are {len(variants)} problems:\n\n{qs}"


async def run_batch(client, model: str, item: dict, idx: int, batch: dict,
                    max_tokens: int = MAX_TOKENS) -> dict:
    t0 = time.time()
    variants = batch["variants"]
    golds = [g for _, g in variants]
    N = len(variants)
    withheld_name = batch["withheld_name"].replace("_", " ")

    msgs = [{"role": "user", "content": _format_batch(variants)}]
    text1 = await client.chat(model, SYS, msgs, max_tokens=max_tokens)
    usage = dict(client.last_usage or {})
    texts = [text1]

    asked = await classify_asked(client, text1, withheld_name)
    if asked:
        msgs += [{"role": "assistant", "content": text1},
                 {"role": "user",
                  "content": f"The {withheld_name} is {fmt_val(batch['withheld_value'])} "
                             "in every problem."}]
        text2 = await client.chat(model, SYS, msgs, max_tokens=max_tokens)
        u2 = client.last_usage or {}
        for k in usage:
            usage[k] += int(u2.get(k, 0) or 0)
        texts.append(text2)

    full = "\n\n".join(texts)
    code = extract_code(full)
    built = wrote_code(full)        # recognition = chose to externalize work as Python
    reused_tool = reused(code)      # secondary: did it abstract one reusable function
    ans = _parse_answers(full)
    n_correct = sum(1 for i in range(1, N + 1) if correct_within_tol(ans.get(i), golds[i - 1]))

    return {
        "model": model,
        "item_idx": idx,
        "n_variants": N,
        "withheld_name": batch["withheld_name"],
        "asked": asked,
        "built": built,
        "reused_tool": reused_tool,
        "n_correct": n_correct,
        "all_correct": n_correct == N,
        "frac_correct": round(n_correct / N, 3),
        "code": code,
        "model_texts": texts,
        "usage": usage,
        "skipped": False,
        "elapsed_s": round(time.time() - t0, 2),
    }
