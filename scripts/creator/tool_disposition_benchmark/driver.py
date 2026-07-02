"""Agent loop for ONE tool-disposition session: drives a tool-calling model over N distinct
problems presented one at a time, against a SessionState (in-process). Enforces a single HARD
cumulative token cap for the whole session and never crashes on a malformed/odd tool call.

The model solves the current problem (writing/running/reusing scripts as it sees fit) and calls
submit_answer to advance. We present the next problem as a new user turn whenever the session's
current-problem pointer moves. A problem on which the model produces two consecutive turns with
no tool call is force-advanced (left unsubmitted) so the session keeps going.
"""

from __future__ import annotations

import json
import time

from lomekwi.raw_chat import RawChat
from scripts.creator.tool_disposition_benchmark.prompts import problem_prompt, system_prompt
from scripts.creator.tool_disposition_benchmark.session_state import SessionState, TOOL_SCHEMAS

MIN_CALL_BUDGET = 256
DEFAULT_MAX_TOKENS = 2048


def _est_tokens(system: str, messages: list, turn) -> int:
    n = len(system) + len(json.dumps(messages))
    n += len(turn.content or "") + sum(len(tc["arguments"]) + 24 for tc in turn.tool_calls)
    return max(1, n // 4)


async def run_session(client: RawChat, model: str, state: SessionState, *,
                      token_cap: int = 200_000, max_tokens: int = DEFAULT_MAX_TOKENS,
                      max_turns: int | None = None, announce_cap: bool = True,
                      stop_on_budget_exhausted: bool = False) -> dict:
    """token_cap is always enforced as a hard ceiling. announce_cap=False ('no-cap' arm) hides it
    from the model: the system prompt omits the budget paragraph and tool results omit
    tokens_remaining — token_cap then acts only as a silent safety ceiling on cost."""
    t0 = time.time()
    if max_turns is None:
        max_turns = max(60, 15 * state.n)
    tools = TOOL_SCHEMAS()
    known_tools = {t["function"]["name"] for t in tools}
    system = system_prompt(state.n, state.budget, token_cap if announce_cap else None)
    if getattr(state, "announce_recurrence", False):   # awareness arm (appended so system_prompt's
        from scripts.creator.tool_disposition_benchmark.prompts import RECURRENCE_NOTE  # signature
        system += RECURRENCE_NOTE                       # stays 3-arg for the AIME monkeypatch)

    messages: list[dict] = [{"role": "user",
                             "content": problem_prompt(state.current(), 1, state.n)}]
    presented = 0                            # highest problem index already presented

    spent = n_turns = n_tool_calls = n_malformed = n_unknown = consecutive_no_tool = 0
    usage_estimated = False
    last_finish = None
    turn_usages: list[dict] = []          # per-turn exact usage + which tools it called (for cost model)

    stopped_on_budget = False
    while not state.done and n_turns < max_turns:
        # early-stop pilot: once the write budget is spent, no further BUILD decisions are possible,
        # so all decision signal (bait, lateness, which classes built) is final -- stop to save cost.
        if stop_on_budget_exhausted and state.writes_remaining <= 0:
            stopped_on_budget = True
            break
        remaining = token_cap - spent
        if remaining < MIN_CALL_BUDGET:
            break
        call_max = max(64, min(max_tokens, remaining))
        n_turns += 1
        turn = await client.chat_tools(model, system, messages, tools, max_tokens=call_max)
        last_finish = turn.finish_reason
        u = client.last_usage
        if u and (u.get("input_tokens") or u.get("output_tokens")):
            spent += int(u.get("input_tokens", 0)) + int(u.get("output_tokens", 0))
            turn_usages.append({
                "tools": [tc["name"] for tc in turn.tool_calls],
                "input_tokens": int(u.get("input_tokens", 0)),
                "output_tokens": int(u.get("output_tokens", 0)),
                "cache_read_tokens": int(u.get("cache_read_tokens", 0) or 0),
                "cache_write_tokens": int(u.get("cache_write_tokens", 0) or 0),
                "problem": state.cur})
        else:
            spent += _est_tokens(system, messages, turn)
            usage_estimated = True

        asst = {"role": "assistant", "content": turn.content or ""}
        if turn.tool_calls:
            asst["tool_calls"] = [{"id": tc["id"], "type": "function",
                                   "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                                  for tc in turn.tool_calls]
        messages.append(asst)

        if not turn.tool_calls:
            consecutive_no_tool += 1
            if consecutive_no_tool >= 2:
                # force-advance the current problem (left unsubmitted) so the session continues
                if not state.done:
                    state.cur += 1
                consecutive_no_tool = 0
                if state.done:
                    break
                messages.append({"role": "user",
                                 "content": problem_prompt(state.current(), state.cur + 1, state.n)})
                presented = state.cur
                continue
            messages.append({"role": "user", "content":
                             "Use the tools to proceed (e.g. submit_answer for the current "
                             "problem): write_script / run_script / list_scripts / read_script / "
                             "submit_answer."})
            continue
        consecutive_no_tool = 0

        for tc in turn.tool_calls:
            n_tool_calls += 1
            if tc["args"] is None:
                n_malformed += 1
                result = {"ok": False, "error": "malformed JSON in tool arguments — resend valid "
                          "JSON matching the tool schema."}
            elif tc["name"] not in known_tools:
                n_unknown += 1
                result = {"ok": False, "error": f"no such tool '{tc['name']}'. Available tools: "
                          f"{sorted(known_tools)}."}
            else:
                result = state.call(tc["name"], tc["args"])
            if announce_cap:
                result = {**result, "tokens_remaining": max(0, token_cap - spent)}
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": json.dumps(result)})

        # present the next problem whenever the pointer has advanced
        if not state.done and state.cur > presented:
            messages.append({"role": "user",
                             "content": problem_prompt(state.current(), state.cur + 1, state.n)})
            presented = state.cur

    score = state.score()
    return {
        "model": model,
        **score,
        "spent_tokens": spent,
        "token_cap": token_cap,
        "hit_cap": spent >= token_cap,
        "stopped_on_budget": stopped_on_budget,
        "problems_seen": state.cur,
        "n_turns": n_turns,
        "n_tool_calls": n_tool_calls,
        "n_malformed_tool_calls": n_malformed,
        "n_unknown_tool_calls": n_unknown,
        "usage_estimated": usage_estimated,
        "turn_usages": turn_usages,
        "last_finish_reason": last_finish,
        "scripts": dict(state.scripts),
        "transcript": messages,
        "elapsed_s": round(time.time() - t0, 2),
    }
