"""Exploratory: run a model IN-SESSION over the AIME 2026 problems, same tool-disposition harness
as run_gsmhard_session (persistent session, write_script/run_script/submit_answer tools, token cap,
driver loop, exact grading) but with an AIME-appropriate prompt.

Differences from the GSM-Hard / CREATOR runs, on purpose:
  - Framing: AIME answers are integers 0-999, graded by exact match (no 'large messy numbers',
    no significant-figures requirement, no named input values).
  - Write budget is GENEROUS (= N, not the scarce 0.1N): AIME problems don't share a reusable
    tool, so the question is whether the model CHOOSES to compute (brute force), not whether a
    tight budget blocks it. persistence/reusability metrics are therefore not meaningful here.

We override driver.system_prompt / driver.problem_prompt locally (no shared file is modified) so
the proven driver loop is reused verbatim.

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.run_aime_session \
      --models haiku --paper I --token-cap 200000 --max-tokens 2048
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.creator.tool_disposition_benchmark import driver
from scripts.creator.tool_disposition_benchmark.driver import run_session
from scripts.creator.tool_disposition_benchmark.session_state import SessionState, TOOL_SCHEMAS

CLAUDE = {"haiku": "claude-haiku-4-5-20251001",
          "sonnet": "claude-sonnet-4-6", "opus": "claude-opus-4-8"}
DATA = Path("external/aime_2026.jsonl")


def aime_system_prompt(n: int, budget: int, token_cap: int | None = None) -> str:
    cap = (f"\n\nTOKEN BUDGET: you have a HARD cap of {token_cap:,} total tokens (input + output, "
           "summed over all turns) for the whole session. When it runs out the session ends and "
           "whatever you have submitted is scored as-is. Every tool result shows tokens_remaining."
           ) if token_cap is not None else ""
    return (f"You will solve {n} AIME competition problems, ONE AT A TIME. Every answer is an "
            "integer between 0 and 999; it is graded correct only by exact match.\n\nHOW IT WORKS:\n"
            "- You have tools available (listed separately). Using them is entirely optional — how "
            "you reach each answer is up to you. You may reason it out, or write and run code, or "
            "any mix; nothing here favors one approach.\n"
            "- Scripts you write with write_script persist across the session and can be rerun with "
            "run_script; define them as `def solve(inputs):` (you may ignore the inputs argument).\n"
            f"- You may write up to {budget} scripts this session.\n"
            "- Answer the current problem with submit_answer (pass the integer). This advances to "
            f"the next problem; you cannot go back.{cap}")


def aime_byhand_system_prompt(n: int, budget: int, token_cap: int | None = None) -> str:
    """No-code arm: only submit_answer is exposed; the model must reason every answer out by hand."""
    cap = (f"\n\nTOKEN BUDGET: you have a HARD cap of {token_cap:,} total tokens (input + output, "
           "summed over all turns) for the whole session. When it runs out the session ends and "
           "whatever you have submitted is scored as-is."
           ) if token_cap is not None else ""
    return (f"You will solve {n} AIME competition problems, ONE AT A TIME. Every answer is an "
            "integer between 0 and 999; it is graded correct only by exact match.\n\nHOW IT WORKS:\n"
            "- You CANNOT write or run code. There are no scripting tools. Work every answer out "
            "yourself, by reasoning.\n"
            "- The only tool is submit_answer: when you have the integer, call submit_answer with it "
            f"to record it and advance to the next problem; you cannot go back.{cap}")


def aime_problem_prompt(problem: dict, position: int, total: int) -> str:
    return (f"PROBLEM {position} of {total}:\n{problem['question']}\n\n"
            "Give your final integer answer (0-999), then call submit_answer with that integer.")


def load_aime(paper: str) -> list[dict]:
    """AIME 2026 problems as session problems. paper 'I' = first 15, 'II' = last 15, 'both' = 30.
    keys/inputs are empty (not parameterized); sig_figs = answer digit count -> exact-integer match."""
    rows = [json.loads(l) for l in DATA.open()]
    if paper == "I":
        rows = rows[:15]
    elif paper == "II":
        rows = rows[15:]
    out = []
    for i, r in enumerate(rows):
        a = int(r["answer"])
        out.append({"idx": i, "item_idx": r["problem_idx"], "keys": [], "inputs": {},
                    "question": r["problem"], "gold": float(a), "sig_figs": max(1, len(str(abs(a))))})
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["haiku"])
    ap.add_argument("--paper", choices=["I", "II", "both"], default="I")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--budget", type=int, default=None,
                    help="ENFORCED hard cap on scripts written for the whole session (default = N)")
    ap.add_argument("--announce-budget", type=int, default=None,
                    help="budget NUMBER shown in the prompt, decoupled from --budget enforcement "
                         "(default = same as --budget). Lets you tell the model 'up to K scripts' "
                         "while actually enforcing a different cap — to isolate the announcement/"
                         "anchor effect from binding.")
    ap.add_argument("--token-cap", type=int, default=200_000)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--no-scripts", action="store_true",
                    help="no-code arm: expose only submit_answer; the model must solve by hand")
    ap.add_argument("--per-problem", action="store_true",
                    help="isolation mode: run EACH problem as its own 1-problem session (no shared "
                         "token pool, no allocation across problems). Used to measure clean per-problem "
                         "value-of-budget curves. One output row per (problem, rep).")
    ap.add_argument("--no-announce-cap", action="store_true",
                    help="hide the token cap from the model (silent safety ceiling, not an announced "
                         "budget). Use with --per-problem so the token cap is inert instrumentation.")
    args = ap.parse_args()

    load_dotenv()
    driver.problem_prompt = aime_problem_prompt        # local overrides; shared file untouched
    if args.no_scripts:
        driver.system_prompt = aime_byhand_system_prompt
        submit_only = [t for t in TOOL_SCHEMAS() if t["function"]["name"] == "submit_answer"]
        driver.TOOL_SCHEMAS = lambda **_kwargs: submit_only  # strip write/run/list/read tools
    models = [CLAUDE.get(m, m) for m in args.models]
    problems = load_aime(args.paper)
    budget = args.budget if args.budget is not None else len(problems)
    announce = args.announce_budget if args.announce_budget is not None else budget
    if not args.no_scripts:
        # The driver calls system_prompt(n, state.budget, cap) with the ENFORCED budget; ignore it
        # and announce `announce` instead, so the prompt's "up to K scripts" can differ from the
        # enforced cap (anchor-effect isolation). When announce == budget this is the old behavior.
        driver.system_prompt = lambda n, _enforced, cap, _a=announce: aime_system_prompt(n, _a, cap)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")   # microseconds: avoid same-second dir collisions
    out_dir = Path("runs") / f"aime_disposition_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sessions.jsonl"
    (out_dir / "config.json").write_text(json.dumps({**vars(args), "source": "aime_2026"}, indent=2))
    announce_cap = not args.no_announce_cap
    # In per-problem mode each problem is its own 1-problem session; otherwise one N-problem session.
    pidxs = list(range(len(problems))) if args.per_problem else [None]
    total = len(models) * args.reps * len(pidxs)
    print(f"AIME {args.paper}: {len(problems)} problems x {len(models)} models x {args.reps} reps "
          f"{'(PER-PROBLEM)' if args.per_problem else ''} -> {out_path} "
          f"(cap={args.token_cap:,}{'/silent' if not announce_cap else ''}, budget={budget})", flush=True)

    client = RawChat()
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    done = 0

    async def one(model: str, rep: int, pidx: int | None):
        nonlocal done
        async with sem:
            t0 = time.time()
            subset = problems if pidx is None else [problems[pidx]]
            try:
                state = SessionState(problems=subset, budget=budget, announce_budget=announce)
                row = await run_session(client, model, state, token_cap=args.token_cap,
                                        max_tokens=args.max_tokens, announce_cap=announce_cap)
                row["rep"] = rep
                if pidx is not None:
                    row["problem_idx"] = pidx
                    row["item_idx"] = problems[pidx].get("item_idx")
                    row["budget_k"] = 0 if args.no_scripts else budget
            except Exception as e:
                row = {"model": model, "rep": rep, "problem_idx": pidx, "N": len(subset),
                       "error": f"{type(e).__name__}: {e}", "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                done += 1
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                lbl = model.split("/")[-1]
                tag = f"p{pidx} " if pidx is not None else ""
                print(f"  {done}/{total}  {lbl} {tag}rep{rep}: solve={row.get('n_correct')}/{row.get('N')} "
                      f"scripts={row.get('n_scripts_written')} "
                      f"tok={row.get('spent_tokens')} {'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(m, r, p) for m in models for r in range(args.reps) for p in pidxs))
    print(f"done -> {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
