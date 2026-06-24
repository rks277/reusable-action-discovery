"""Agent loop for one MCP CREATOR episode: drives a tool-calling model against a backend
(in-proc or real MCP), enforces a HARD cumulative token cap (the only place it can be
enforced — vLLM caps per-request, MCP has no token accounting), and never crashes on a
malformed/odd tool call (counts it instead). Returns a flat record dict."""

from __future__ import annotations

import json
import time

from lomekwi.raw_chat import RawChat
from scripts.creator.mcp_creator.prompts import system_prompt

MIN_CALL_BUDGET = 256     # stop if fewer tokens remain than a call could meaningfully use
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_TURNS = 60


def _est_tokens(system: str, messages: list, turn) -> int:
    """Crude fallback when the server omits usage on tool-call turns (~4 chars/token)."""
    n = len(system) + len(json.dumps(messages))
    n += len(turn.content or "") + sum(len(tc["arguments"]) + 24 for tc in turn.tool_calls)
    return max(1, n // 4)


async def run_episode(client: RawChat, model: str, backend, *, token_cap: int = 50_000,
                      max_tokens: int = DEFAULT_MAX_TOKENS, max_turns: int = DEFAULT_MAX_TURNS) -> dict:
    t0 = time.time()
    briefing = await backend.briefing()
    tools = backend.model_tools()
    system = system_prompt(briefing, token_cap)
    messages: list[dict] = [{"role": "user", "content": "Begin."}]

    known_tools = {t["function"]["name"] for t in tools}
    spent = 0
    n_turns = n_tool_calls = n_malformed = n_unknown = consecutive_no_tool = 0
    usage_estimated = False
    last_finish = None

    while n_turns < max_turns:
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
        else:
            spent += _est_tokens(system, messages, turn)
            usage_estimated = True

        # record assistant turn (OpenAI format)
        asst = {"role": "assistant", "content": turn.content or ""}
        if turn.tool_calls:
            asst["tool_calls"] = [{"id": tc["id"], "type": "function",
                                   "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                                  for tc in turn.tool_calls]
        messages.append(asst)

        if not turn.tool_calls:
            consecutive_no_tool += 1
            if consecutive_no_tool >= 2:
                break
            messages.append({"role": "user", "content":
                             "Use the tools to proceed: write_script / run_script / begin_test / "
                             "submit_answers."})
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
                result = await backend.call(tc["name"], tc["args"])
            result = {**result, "tokens_remaining": max(0, token_cap - spent)}
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": json.dumps(result)})

    score = await backend.results()
    record = {
        "model": model,
        **score,
        "spent_tokens": spent,
        "token_cap": token_cap,
        "hit_cap": spent >= token_cap,
        "n_turns": n_turns,
        "n_tool_calls": n_tool_calls,
        "n_malformed_tool_calls": n_malformed,
        "n_unknown_tool_calls": n_unknown,
        "usage_estimated": usage_estimated,
        "last_finish_reason": last_finish,
        "transcript": messages,
        "elapsed_s": round(time.time() - t0, 2),
    }
    return record
