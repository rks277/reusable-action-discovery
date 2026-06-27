"""IN-SESSION by-hand eval: present all N problems of a seed SEQUENTIALLY in ONE shared
conversation (the model accumulates context across problems, exactly like the real disposition
session) but with NO tools and NO budget. Contrasts with byhand_eval.py, where each problem is an
isolated single-turn call. The gap between the two isolates the long-context effect of running the
problems back-to-back.

Prompts mirror prompts.system_prompt / problem_prompt as closely as possible, with the tool +
budget machinery removed and submit_answer replaced by an `ANSWER: <number>` line.

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.byhand_session \
      --models haiku --seeds 0 1 --n 20 --sig-figs 6 --magnitude 0.01
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.creator.creator_exec import _parse_answer
from scripts.creator.tool_disposition_benchmark.dataset import load_or_build
from scripts.creator.tool_disposition_benchmark.grading import correct_to_sigfigs

CLAUDE = {"haiku": "claude-haiku-4-5-20251001",
          "sonnet": "claude-sonnet-4-6", "opus": "claude-opus-4-8"}


def _system(n: int) -> str:
    return (f"You will answer {n} numeric word problems, ONE AT A TIME. Each problem states a "
            "required number of significant figures; your answer is graded correct only if it "
            "matches the exact answer rounded to that many significant figures. The numbers are "
            "large and messy, so the arithmetic is demanding.\n\nYou have NO tools — work each "
            "answer out by hand. End every answer with a single line `ANSWER: <number>` rounded to "
            "the requested significant figures. After you answer, the next problem is presented; "
            "you cannot go back to a problem you have already answered.")


def _problem_turn(p: dict, pos: int, total: int) -> str:
    keys = ", ".join(p["keys"])
    return (f"PROBLEM {pos} of {total}:\n{p['question']}\n\nThe named values in this problem are: "
            f"{p['inputs']!r} (keys: [{keys}]).\nGive your final answer to {p['sig_figs']} "
            f"significant figures, ending with `ANSWER: <number>`.")


async def run_seed(client, model, problems, max_tokens) -> tuple[list[bool], list[dict]]:
    """One shared conversation over all problems in the seed (sequential, context accumulates).
    Returns (correct-flags, transcript messages)."""
    system = _system(len(problems))
    messages: list[dict] = []
    flags: list[bool] = []
    for i, p in enumerate(problems):
        messages.append({"role": "user", "content": _problem_turn(p, i + 1, len(problems))})
        try:
            text = await client.chat(model, system, messages, max_tokens=max_tokens)
        except Exception:
            text = ""
        messages.append({"role": "assistant", "content": text or ""})
        flags.append(correct_to_sigfigs(_parse_answer(text or ""), p["gold"], p["sig_figs"]))
    return flags, messages


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["haiku"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--sig-figs", type=int, default=6)
    ap.add_argument("--magnitude", type=float, default=0.01)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--dump", type=str, default=None, help="dir to write per-seed transcripts")
    args = ap.parse_args()

    load_dotenv()
    client = RawChat()
    dump_dir = Path(args.dump) if args.dump else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)
    print(f"IN-SESSION by-hand (one shared conversation/seed, no tools)  N={args.n}/seed  "
          f"D={args.sig_figs}  m={args.magnitude}  seeds={args.seeds}\n", flush=True)
    for m in args.models:
        model = CLAUDE.get(m, m)
        # seeds run concurrently; each seed is internally sequential (shared context)
        seed_results = await asyncio.gather(*(
            run_seed(client, model, load_or_build(args.n, args.sig_figs, s, args.magnitude,
                                                  args.shuffle), args.max_tokens)
            for s in args.seeds))
        pooled_ok = pooled_n = 0
        for s, (flags, messages) in zip(args.seeds, seed_results):
            if dump_dir:
                (dump_dir / f"{m}_seed{s}.json").write_text(json.dumps(messages, indent=2))
            ok = sum(flags)
            print(f"  {m:<7} seed{s}: {ok}/{len(flags)} = {ok/len(flags):.3f}", flush=True)
            pooled_ok += ok
            pooled_n += len(flags)
        print(f"  {m:<7} POOLED: {pooled_ok}/{pooled_n} = {pooled_ok/pooled_n:.3f}\n", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
