"""Single-episode runner for the oracle-discovery environment.

Builds a hard problem + the n (non-semantic) tool labels, frames the task, and runs ONE
all-decoy rollout via lomekwi.oracle.mcp_host. Returns (result, trace) in the same shape
contract as scripts/run_beach_v2_llm.run, so the sweep merges identity fields and the
counterfactual post-processor scores it into n oracle-assignments.

CLI:  python scripts/run_oracle_llm.py <model> [--n N] [--difficulty D] [--budget B]
                                       [--seed S] [--obfuscate]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Make `lomekwi.*` and `scripts.*` importable when run as a plain file.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lomekwi.oracle import mcp_host                       # noqa: E402
from lomekwi.oracle.config import DEFAULT_N, SURCHARGE_TOKENS, tool_specs_name  # noqa: E402
from lomekwi.oracle.problems import make_problem          # noqa: E402
from lomekwi.obfuscation import assign                    # noqa: E402
from lomekwi.raw_chat import _provider_for                # noqa: E402


SYS_FREE = ("You are a problem-solving agent with access to tools. You operate under a "
            "strict token budget, so act efficiently. Each turn, either call a tool or "
            "call submit_answer. Think briefly, then act.")

SYS_SEQ  = ("You are a problem-solving agent with access to tools. You operate under a "
            "strict token budget, so act efficiently. Call AT MOST ONE tool per turn -- "
            "wait to see the result before deciding what to do next. Each turn, either "
            "call one tool or call submit_answer. Think briefly, then act.")

# Qwen3 emits <think>...</think> blocks by default; appending /no_think suppresses it
# so tool-calling responses are clean and comparable to Anthropic runs.
SYS_SEQ_NOTHINK  = SYS_SEQ  + " /no_think"
SYS_FREE_NOTHINK = SYS_FREE + " /no_think"


def _sys(sequential: bool, model: str) -> str:
    """Pick the right system prompt: suppress Qwen3/3.5 thinking mode when needed."""
    qwen = "qwen3" in model.lower()   # matches qwen3 and qwen3.5
    if sequential:
        return SYS_SEQ_NOTHINK if qwen else SYS_SEQ
    return SYS_FREE_NOTHINK if qwen else SYS_FREE


def build_labels(n: int, obfuscate: bool, relabel_seed: int) -> dict:
    """str(i) -> surface name for the n candidate tools. Default: config names; with
    --obfuscate: fresh per-episode non-semantic alnum labels (uniform-random relabeling)."""
    if obfuscate:
        m = assign([f"tool{i}" for i in range(n)], seed=relabel_seed, scheme="alnum")
        return {str(i): m[f"tool{i}"] for i in range(n)}
    return {str(i): tool_specs_name(i) for i in range(n)}


def build_intro(problem, n: int, budget: int) -> str:
    head = (f"Solve the following problem within a STRICT BUDGET of {budget} tokens "
            f"(every token you read and generate counts toward it).\n\n"
            f"PROBLEM: {problem.text}\n\n")
    if n <= 0:
        # tools-off (pilot): no candidate tools, only submit_answer.
        return head + ("Work the problem out and, when you have the answer, call "
                       "submit_answer with it. You get ONE submission: correct = win, "
                       "wrong = lose.")
    return head + (
        f"You have {n} tools available (plus submit_answer). Exactly one of the {n} "
        f"tools can solve this problem and return the answer directly -- but you do not "
        f"know which one, and the rest return nothing useful. You may also work the "
        f"problem out yourself. When you have the answer, call submit_answer with it. "
        f"You get ONE submission: correct = win, wrong = lose."
    )


async def run(model: str, n: int = DEFAULT_N, difficulty: int = 6, seed: int = 0,
              budget: int = 4000, problem_kind: str = "long_division",
              obfuscate: bool = False, relabel_seed: int | None = None,
              max_turns: int = 300, sequential: bool = True):
    load_dotenv()
    rseed = relabel_seed if relabel_seed is not None else seed
    problem = make_problem(problem_kind, difficulty=difficulty, seed=seed)
    labels = build_labels(n, obfuscate, rseed)
    intro = build_intro(problem, n, budget)
    sys_prompt = _sys(sequential, model)

    result, trace = await mcp_host.run_episode(
        model, sys_prompt, intro, problem, n=n, labels=labels, budget=budget,
        max_turns=max_turns, sequential=sequential)

    result.update({
        "model": model, "provider": _provider_for(model),
        "n": n, "difficulty": difficulty, "problem_kind": problem_kind,
        "seed": seed, "relabel_seed": rseed, "obfuscate": obfuscate,
        "budget": budget, "surcharge": SURCHARGE_TOKENS,
        "sequential": sequential,
        "problem": problem.text, "true_answer": problem.true_answer,
    })
    return result, trace


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--difficulty", type=int, default=6)
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obfuscate", action="store_true")
    args = ap.parse_args()

    result, trace = await run(args.model, n=args.n, difficulty=args.difficulty,
                              seed=args.seed, budget=args.budget,
                              obfuscate=args.obfuscate)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("runs") / "oracle" / f"oracle_{args.model.replace('/', '_')}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({"result": result, "trace": trace},
                                                 indent=2))
    print(json.dumps({k: result[k] for k in
                      ("model", "n", "difficulty", "budget", "stopped_reason",
                       "meter_total", "tool_calls", "submit", "true_answer")},
                     indent=2, default=str))
    print(f"\nSaved: {out/'result.json'}")


if __name__ == "__main__":
    asyncio.run(main())
