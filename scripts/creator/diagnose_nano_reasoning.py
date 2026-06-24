"""Diagnostic: read gpt-5-nano's (hidden) reasoning via the OpenAI Responses API reasoning
SUMMARY, on real v5 batch problems, and contrast with gpt-5.4-nano (same vendor, R 0.76 vs 0.25).

Un-gated (all values supplied) to isolate RECOGNITION = build-vs-grind. Single turn: we only
need the DECISION (emit EVALUATE vs dump ANSWER lines) + the reasoning summary, not the executor
round-trip. max_output_tokens high (8000) so truncation can't confound.

  PYTHONPATH=. python -m scripts.creator.diagnose_nano_reasoning --n-items 25
"""
from __future__ import annotations
import argparse, asyncio, json, re
from pathlib import Path

from lomekwi.raw_chat import RawChat
from scripts.creator.creator_eval_hard import make_hard_batch
from scripts.creator.creator_eval_tool import _sys, _format_rows, _eval_block, _parse_answers
from scripts.creator.creator_exec import correct_within_tol

MODELS = ["gpt-5-nano", "gpt-5.4-nano"]


async def one(client, model, batch, idx):
    sys = _sys(20, "costly")
    user = (f"Problem (same structure for every row):\n{batch['template']}\n\n"
            f"Here are 20 rows of inputs:\n{_format_rows(batch)}")
    try:
        r = await client.responses.create(
            model=model,
            input=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            reasoning={"effort": "low", "summary": "auto"},
            max_output_tokens=8000)
    except Exception as e:
        return {"model": model, "idx": idx, "error": f"{type(e).__name__}: {str(e)[:160]}"}
    summ, txt = [], []
    for item in r.output:
        if getattr(item, "type", None) == "reasoning":
            for s in (getattr(item, "summary", None) or []):
                summ.append(getattr(s, "text", str(s)))
        if getattr(item, "type", None) == "message":
            for ct in (getattr(item, "content", None) or []):
                txt.append(getattr(ct, "text", ""))
    full = "\n".join(txt)
    used_tool = _eval_block(full) is not None
    ans = _parse_answers(full)
    nc = sum(1 for i in range(1, 21) if correct_within_tol(ans.get(i), batch["golds"][i - 1]))
    return {"model": model, "idx": idx, "used_tool": used_tool, "n_answers": len(ans),
            "n_correct": nc, "reasoning_summary": "\n".join(summ),
            "final_head": full[:200]}


async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-items", type=int, default=25)
    args = ap.parse_args()
    # load CREATOR items from the gated cache (already-usable batches) — reuse withheld items
    cache = json.load(open("runs/creator_eval_hard_cache_N20_gated.json"))
    def walk(o):
        if isinstance(o, dict):
            if "template" in o and "rows" in o and "golds" in o: yield o
            for v in o.values(): yield from walk(v)
        elif isinstance(o, list):
            for v in o: yield from walk(v)
    # supply ALL values (un-gate): fold the withheld var back into every row
    batches = []
    for b in walk(cache):
        if len(batches) >= args.n_items: break
        wn, wv = b.get("withheld_name"), b.get("withheld_value")
        rows = [dict(r, **({wn: wv} if wn else {})) for r in b["rows"]]
        # un-gate: show the withheld var in every row (add it to varied_names so _format_rows prints it)
        vn = b["varied_names"] + ([wn] if wn else [])
        nb = dict(b, rows=rows, varied_names=vn, withheld_name=None)
        batches.append(nb)

    client = RawChat()._openai()
    sem = asyncio.Semaphore(8)
    async def guarded(m, b, i):
        async with sem: return await one(client, m, b, i)
    tasks = [guarded(m, b, i) for m in MODELS for i, b in enumerate(batches)]
    res = await asyncio.gather(*tasks)
    out = Path("runs/diagnose_nano_reasoning.jsonl")
    out.write_text("\n".join(json.dumps(r) for r in res))
    # quick tallies
    for m in MODELS:
        rs = [r for r in res if r["model"] == m and not r.get("error")]
        if rs:
            R = sum(r["used_tool"] for r in rs) / len(rs)
            print(f"{m}: n={len(rs)} R(EVALUATE)={R:.2f} "
                  f"mean_correct={sum(r['n_correct'] for r in rs)/len(rs):.1f}/20")
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
