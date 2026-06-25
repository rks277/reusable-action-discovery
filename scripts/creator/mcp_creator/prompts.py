"""System prompt for the MCP CREATOR eval.

METHOD-NEUTRAL BY DESIGN: the experiment measures whether a model spontaneously chooses
to build/use a tool vs. solve by hand (recognition). So this prompt must NOT recommend or
favor writing code — it states the task, the available tools, the phase mechanics, and the
token cap, and presents using-code vs. solving-directly as an equally valid free choice.
The I/O contract still lives in the tool schemas (documentation of how a tool works *if*
chosen), not here.

Takes a `briefing` dict ({template, keys, n_test, example_inputs, example_reference}) so it
works for both the in-process and real-MCP backends."""

from __future__ import annotations


def system_prompt(briefing: dict, token_cap: int) -> str:
    keys = ", ".join(briefing["keys"])
    n = briefing["n_test"]
    return f"""You answer numeric word problems.

You will first see ONE example problem with its correct answer. When you call begin_test you \
will receive {n} more problems of the same kind (same wording, different numbers), and you must \
return a correct numeric answer for every one of them.

EXAMPLE PROBLEM:
{briefing["template"]}
For this example the inputs are {briefing["example_inputs"]!r} and the correct answer is \
{briefing["example_reference"]!r}. (In general a problem's inputs are given as a dict with the \
keys [{keys}].)

HOW IT WORKS:
- You have tools available (listed separately). Using them is entirely optional — how you reach \
your answers is up to you. You may reason the answers out yourself, or write and run code, or \
any mix; nothing here favors one approach.
- Call begin_test when you want the {n} test problems. IMPORTANT: begin_test PERMANENTLY \
disables write_script for the rest of the episode (run_script stays available). So if you intend \
to write any script, you must do it BEFORE calling begin_test — you cannot write scripts once the \
test has begun. (You may also call begin_test right away if you prefer.)
- Submit with submit_answers: a list of {n} numbers in problem order. This is FINAL — submitting \
ENDS the episode immediately (no resubmits, no further actions), so submit only once every answer \
is ready.

TOKEN BUDGET: you have a HARD cap of {token_cap:,} total tokens (input + output, summed across \
all turns) for this whole episode. When it runs out, the episode ends and whatever you have \
submitted is scored as-is. Every tool result shows tokens_remaining."""
