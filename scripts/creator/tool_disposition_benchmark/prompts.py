"""Prompts for the tool-disposition benchmark.

METHOD-NEUTRAL BY DESIGN (like mcp_creator/prompts.py): the experiment measures whether a
model spontaneously builds/reuses tools vs. solves by hand. The system prompt states the task,
the mechanics (persistent scripts, the global write budget, the precision requirement, the token
cap) and presents code vs. by-hand as an equally valid free choice — it does NOT recommend code.
The per-problem prompt presents one concrete problem and its required significant figures.
"""

from __future__ import annotations


def system_prompt(n: int, budget: int, token_cap: int | None = None) -> str:
    """token_cap=None: omit the token-budget paragraph entirely (the 'no-cap' arm — the model is
    told nothing about a token budget; a silent safety ceiling is still enforced by the driver)."""
    budget_note = (
        f"\n\nTOKEN BUDGET: you have a HARD cap of {token_cap:,} total tokens (input + output, "
        "summed across all turns) for the whole session. When it runs out, the session ends and "
        "whatever you have submitted is scored as-is. Every tool result shows tokens_remaining."
    ) if token_cap is not None else ""
    return f"""You will answer {n} numeric word problems, ONE AT A TIME. Each problem states a \
required number of significant figures; your answer is graded correct only if it matches the \
exact answer rounded to that many significant figures. Your score is the fraction of the {n} \
problems you answer correctly.

HOW IT WORKS:
- You have tools available (listed separately). Using them is entirely optional — how you reach \
your answers is up to you. You may reason answers out yourself, or write and run code, or any mix; \
nothing here favors one approach.
- Scripts you write with write_script PERSIST across every problem in this session and can be \
rerun on new inputs with run_script (use list_scripts / read_script to recall what you saved). \
You may write at most {budget} scripts for the WHOLE session — once that budget is used up, \
write_script is disabled, but run_script keeps working on whatever you already saved.
- Answer the current problem with submit_answer. This advances to the next problem; you cannot \
go back to a problem you have already submitted.{budget_note}"""


# Optional "awareness" arm: disclose the RECURRENCE STRUCTURE (not the strategy). States that some
# types repeat and some are one-offs, unlabeled -- deliberately does NOT tell the model to wait, so
# we cannot be accused of leading it. Appended to the system prompt when announce_recurrence is set.
RECURRENCE_NOTE = (
    "\n\nAbout the sequence: some of these problems are drawn from recurring TYPES that reappear "
    "several times over the session, while others appear only once. The problems are not labeled "
    "with their type, and you are not told in advance how many times any type will recur.")


def problem_prompt(problem: dict, position: int, total: int) -> str:
    keys = ", ".join(problem["keys"])
    return f"""PROBLEM {position} of {total}:
{problem['question']}

The named values in this problem are: {problem['inputs']!r} (keys: [{keys}]).
Give your final answer to {problem['sig_figs']} significant figures, then call submit_answer \
with that number."""


def byhand_system_prompt(n: int = 1) -> str:
    """For the recognition / calibration pass: solve unaided (no tools), emit one final answer."""
    return ("You answer numeric word problems by hand. For each problem, work out the answer and "
            "end with a single line `ANSWER: <number>` rounded to the requested number of "
            "significant figures. You have no tools.")


def byhand_problem_prompt(problem: dict, sig_figs: int | None = None) -> str:
    d = sig_figs if sig_figs is not None else problem["sig_figs"]
    keys = ", ".join(problem["keys"])
    return (f"{problem['question']}\n\nThe named values are: {problem['inputs']!r} (keys: [{keys}]). "
            f"Give your answer to {d} significant figures as `ANSWER: <number>`.")
