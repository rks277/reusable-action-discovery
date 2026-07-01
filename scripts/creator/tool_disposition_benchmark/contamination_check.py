"""Contamination spot-check: can a model RECALL AIME answers with no work (pure memory)?

Asks the model for the final integer ONLY, with a tiny output cap so it cannot compute. Elevated
accuracy on OLD problems vs FRESH (2026) — especially on HARD problems (idx 11-15) that can't be
computed in a few tokens — indicates the answer was memorized (training contamination).

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.contamination_check --model haiku
"""
from __future__ import annotations
import argparse, asyncio, json, re
from pathlib import Path
from dotenv import load_dotenv
from lomekwi.raw_chat import RawChat

CLAUDE = {"haiku": "claude-haiku-4-5-20251001", "sonnet": "claude-sonnet-4-6", "opus": "claude-opus-4-8"}
DATA = Path("external/aime/aime_2015_2026.jsonl")
SYS = ("You are taking a closed-book recall test. For each AIME problem output ONLY the final "
       "integer answer (0-999). Do NOT show work or reasoning. Just the number.")


def parse_int(t: str | None):
    ns = re.findall(r"\d{1,3}", t or "")
    return int(ns[-1]) if ns else None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    load_dotenv(".env")
    model = CLAUDE.get(args.model, args.model)
    probs = [json.loads(l) for l in DATA.open()]
    client = RawChat()
    sem = asyncio.Semaphore(args.concurrency)
    res = []

    async def one(p):
        async with sem:
            msgs = [{"role": "user", "content":
                     f"AIME problem. Answer with the integer only.\n\n{p['problem']}\n\nInteger answer:"}]
            try:
                turn = await client.chat_tools(model, SYS, msgs, [], max_tokens=args.max_tokens)
                ans = parse_int(turn.content)
            except Exception:
                ans = None
            res.append({**{k: p[k] for k in ("year", "paper", "problem_idx", "answer")},
                        "pred": ans, "correct": ans == p["answer"]})

    await asyncio.gather(*(one(p) for p in probs))

    def acc(rows):
        return (sum(r["correct"] for r in rows) / len(rows)) if rows else float("nan")
    print(f"=== {model} pure-recall (no work, max_tokens={args.max_tokens}), n={len(res)} ===")
    print("by year:")
    for y in sorted(set(r["year"] for r in res)):
        rows = [r for r in res if r["year"] == y]
        print(f"  {y}: recall_acc={acc(rows):.2f}  (n={len(rows)})")

    def band(i):
        return "easy(1-5)" if i <= 5 else "mid(6-10)" if i <= 10 else "HARD(11-15)"
    print("by difficulty band -- OLD (2015-25) vs FRESH (2026):")
    for b in ["easy(1-5)", "mid(6-10)", "HARD(11-15)"]:
        old = [r for r in res if r["year"] < 2026 and band(r["problem_idx"]) == b]
        fr = [r for r in res if r["year"] == 2026 and band(r["problem_idx"]) == b]
        print(f"  {b:13} old={acc(old):.2f} (n={len(old)})   fresh2026={acc(fr):.2f} (n={len(fr)})")
    out = DATA.parent / f"recall_{args.model}.jsonl"
    with out.open("w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    asyncio.run(main())
