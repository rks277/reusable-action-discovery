"""Recognition pass: for each session, re-run the SAME model on the problems it used a script
on, this time WITH NO TOOLS (pure reasoning). Recognition = the fraction of those problems the
model CANNOT solve by hand — i.e. how necessary its tool use was. High recognition = it reached
for a script on genuinely hard problems (good disposition); low recognition = gratuitous tool use
on problems it could have done unaided.

Rewrites sessions.jsonl in place, adding `recognition_rate` and `recognition_detail` to each
session record (regenerable; safe to re-run).

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.recognition runs/tool_disposition_<ts>/
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.creator.creator_exec import _parse_answer
from scripts.creator.tool_disposition_benchmark.dataset import load_or_build
from scripts.creator.tool_disposition_benchmark.grading import correct_to_sigfigs
from scripts.creator.tool_disposition_benchmark.prompts import (byhand_problem_prompt,
                                                                byhand_system_prompt)


async def _solve_by_hand(client: RawChat, model: str, problem: dict, max_tokens: int) -> bool:
    text = await client.chat(model, byhand_system_prompt(),
                             [{"role": "user", "content": byhand_problem_prompt(problem)}],
                             max_tokens=max_tokens)
    ans = _parse_answer(text or "")
    return correct_to_sigfigs(ans, problem["gold"], problem["sig_figs"])


async def main():
    run_dir = Path(sys.argv[1])
    sess_path = run_dir / "sessions.jsonl" if run_dir.is_dir() else run_dir
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    cfg = json.loads((sess_path.parent / "config.json").read_text())
    problems = load_or_build(cfg["n"], cfg["sig_figs"], cfg["seed"], cfg.get("magnitude", 1.0),
                             cfg.get("shuffle", False))
    by_idx = {p["idx"]: p for p in problems}

    load_dotenv()
    client = RawChat()
    sem = asyncio.Semaphore(8)
    rows = [json.loads(l) for l in sess_path.read_text().splitlines() if l.strip()]

    async def grade(model, p):
        async with sem:
            return await _solve_by_hand(client, model, p, max_tokens)

    for r in rows:
        if r.get("error"):
            continue
        used = [by_idx[rec["idx"]] for rec in r.get("records", [])
                if rec.get("used_script") and rec["idx"] in by_idx]
        if not used:
            r["recognition_rate"] = None
            r["recognition_detail"] = []
            continue
        flags = await asyncio.gather(*(grade(r["model"], p) for p in used))
        # recognition = fraction of script-used problems NOT solvable by hand
        n_unsolved = sum(1 for f in flags if not f)
        r["recognition_rate"] = round(n_unsolved / len(flags), 4)
        r["byhand_solve_rate"] = round(sum(flags) / len(flags), 4)
        r["recognition_detail"] = [{"idx": p["idx"], "byhand_correct": bool(f)}
                                   for p, f in zip(used, flags)]
        print(f"  {r['model'].split('/')[-1]} rep{r.get('rep')}: recognition "
              f"{n_unsolved}/{len(flags)} = {r['recognition_rate']} "
              f"(by-hand {sum(flags)}/{len(flags)})", flush=True)

    sess_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"updated -> {sess_path}")


if __name__ == "__main__":
    asyncio.run(main())
