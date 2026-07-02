"""MCP host + manual tool-use loop for the oracle environment.

Spawns the decoy MCP server over stdio, lists its tools, exposes them to the model via
the provider's native tool schema, and runs a HAND-WRITTEN agentic loop (not the SDK
tool-runner -- we need to stamp a per-call token meter, which the runner would hide).

The load-bearing artifact is ``tool_calls``: an ordered list of every candidate-tool call
with the cumulative TOKEN-ONLY meter (``meter_raw``) at dispatch. Together with the
``submit`` event this is everything ``scripts/oracle_counterfactual.py`` needs to score
the rollout into n oracle-assignments at any budget/surcharge.

Two dialects:
  - anthropic: Messages ``tools`` / ``tool_use`` / ``tool_result``.
  - openai-compatible (Qwen via vLLM/Ollama, or OpenAI): ``tools`` / ``tool_calls`` /
    role:"tool".
Google is intentionally unsupported here.
"""

from __future__ import annotations

import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from lomekwi.oracle.config import SUBMIT_TOOL
from lomekwi.raw_chat import RawChat, _provider_for

import re as _re
MAX_NOOP = 4            # consecutive turns with no tool call -> give up (beach-style)

_THINK_RE = _re.compile(r"<think>.*?</think>", _re.DOTALL)

def _strip_think(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks from visible text."""
    return _THINK_RE.sub("", text).strip()


# --- token meter -----------------------------------------------------------------------

def per_call_cost(usage: dict | None, provider: str) -> int:
    """Token-only cost of one model API call. Folds Anthropic cache read/write back into
    input so the meter measures total prompt work and is cache-invariant + comparable to
    OpenAI's prompt_tokens (which already includes its cached subset)."""
    u = usage or {}
    out = u.get("output_tokens", 0)
    if provider == "anthropic":
        inp = (u.get("input_tokens", 0) + u.get("cache_read_tokens", 0)
               + u.get("cache_write_tokens", 0))
    else:
        inp = u.get("input_tokens", 0)
    return int(out + inp)


# --- MCP tool-schema translation -------------------------------------------------------

def to_anthropic_tools(mcp_tools) -> list[dict]:
    return [{"name": t.name, "description": t.description or "",
             "input_schema": t.inputSchema or {"type": "object", "properties": {}}}
            for t in mcp_tools]


def to_openai_tools(mcp_tools) -> list[dict]:
    return [{"type": "function", "function": {
                "name": t.name, "description": t.description or "",
                "parameters": t.inputSchema or {"type": "object", "properties": {}}}}
            for t in mcp_tools]


# --- the episode -----------------------------------------------------------------------

async def run_episode(model: str, system: str, intro: str, problem, n: int,
                      labels: dict, budget: int, max_turns: int = 300,
                      max_tokens: int = 2048, sequential: bool = True):
    """Run ONE all-decoy rollout. `labels` maps str(i)->surface tool name for the n
    candidates (already chosen by the caller; passed to the server so names match).
    `budget` is the live token-meter safety stop; real win/lose is decided post-hoc by
    oracle_counterfactual at any budget. Returns (result, trace).

    sequential=True (default): enforce one-tool-at-a-time. If the model emits multiple
    tool_use blocks in one response, only the FIRST is processed; the rest are silently
    dropped. A prompt sentence also instructs the model to call one tool per turn."""
    prov = _provider_for(model)
    if prov == "google":
        raise ValueError("google is not supported by the oracle MCP host")
    client = RawChat()

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "lomekwi.oracle.server", str(n), "--labels-json", json.dumps(labels)],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools

            registered = {t.name for t in mcp_tools}

            async def call(name: str) -> str:
                if name not in registered:
                    # model hallucinated a tool name; tell it the valid options
                    return (f"unknown tool '{name}'. "
                            f"Available: {', '.join(sorted(registered - {'submit_answer'}))}")
                try:
                    res = await session.call_tool(name, {})
                    parts = [getattr(b, "text", "") for b in (res.content or [])]
                    return "".join(p for p in parts if p) or ""
                except Exception as e:
                    return f"tool error: {e}"

            if prov == "anthropic":
                return await _loop_anthropic(client, model, system, intro, problem,
                                             mcp_tools, call, budget, max_turns,
                                             max_tokens, labels, sequential)
            return await _loop_openai(client, model, system, intro, problem, mcp_tools,
                                      call, budget, max_turns, max_tokens, labels, prov,
                                      sequential)


def _blank_result(labels):
    return {
        "tool_calls": [], "submit": None, "meter_total": 0,
        "usage": {**{k: 0 for k in RawChat.USAGE_FIELDS}, "calls": 0},
        "turns": 0, "noop_total": 0, "refusals": 0, "stopped_reason": None,
        "agent_texts": [], "labels": labels,
    }


async def _loop_anthropic(client, model, system, intro, problem, mcp_tools, call,
                          budget, max_turns, max_tokens, labels, sequential):
    tools = to_anthropic_tools(mcp_tools)
    msgs = [{"role": "user", "content": intro}]
    r = _blank_result(labels)
    r["sequential"] = sequential
    meter = 0
    noop = 0

    for t in range(max_turns):
        resp = await client.chat_tools_anthropic(model, system, msgs, tools, max_tokens)
        meter += per_call_cost(client.last_usage, "anthropic")
        _accrue_usage(r, client.last_usage)

        texts, tool_uses, assistant_content = [], [], []
        for b in (resp.content or []):
            bt = getattr(b, "type", None)
            if bt == "text":
                texts.append(b.text)
                assistant_content.append({"type": "text", "text": b.text})
            elif bt == "tool_use":
                tool_uses.append(b)
                assistant_content.append({"type": "tool_use", "id": b.id,
                                          "name": b.name, "input": b.input or {}})
        r["agent_texts"].append("\n".join(texts))

        # sequential: keep only the first tool_use; drop the rest before echoing back
        if sequential and len(tool_uses) > 1:
            first = tool_uses[0]
            # rebuild assistant content with only the first tool_use block
            assistant_content = [
                c for c in assistant_content
                if c.get("type") != "tool_use" or c.get("id") == first.id
            ]
            tool_uses = [first]
        msgs.append({"role": "assistant", "content": assistant_content})

        if not tool_uses:
            noop += 1
            r["noop_total"] += 1
            if getattr(resp, "stop_reason", None) == "refusal":
                r["refusals"] += 1
            if noop >= MAX_NOOP:
                r["stopped_reason"] = "noop"
                break
            msgs.append({"role": "user", "content": "Respond by calling a tool: try one "
                         "of the available tools, or call submit_answer with your final "
                         "answer."})
            continue
        noop = 0

        tool_results, done = [], False
        for b in tool_uses:
            if b.name == SUBMIT_TOOL:
                ans = str((b.input or {}).get("answer", ""))
                correct = bool(problem.check(ans))
                r["submit"] = {"meter_raw": meter, "correct": correct, "answer": ans}
                r["stopped_reason"] = "submitted_correct" if correct else "submitted_wrong"
                done = True
                break
            r["tool_calls"].append({"order": len(r["tool_calls"]), "tool": b.name,
                                    "meter_raw": meter, "turn": t})
            out = await call(b.name)
            tool_results.append({"type": "tool_result", "tool_use_id": b.id,
                                 "content": out})
        if done:
            break
        msgs.append({"role": "user", "content": tool_results})

        if meter >= budget:
            r["stopped_reason"] = "out_of_budget"
            break

    r["turns"] = len(r["agent_texts"])
    r["meter_total"] = meter
    r["stopped_reason"] = r["stopped_reason"] or "max_turns"
    return r, _trace(r)


async def _loop_openai(client, model, system, intro, problem, mcp_tools, call, budget,
                       max_turns, max_tokens, labels, prov, sequential):
    tools = to_openai_tools(mcp_tools)
    msgs = [{"role": "user", "content": intro}]
    r = _blank_result(labels)
    r["sequential"] = sequential
    meter = 0
    noop = 0

    for t in range(max_turns):
        resp = await client.chat_tools_openai(model, system, msgs, tools, max_tokens,
                                              provider=prov)
        meter += per_call_cost(client.last_usage, prov)
        _accrue_usage(r, client.last_usage)

        msg = resp.choices[0].message
        tcs = getattr(msg, "tool_calls", None) or []
        r["agent_texts"].append(_strip_think(msg.content or ""))

        # sequential: keep only the first tool_call
        if sequential and len(tcs) > 1:
            tcs = tcs[:1]

        assistant = {"role": "assistant", "content": msg.content or ""}
        if tcs:
            assistant["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments or "{}"}}
                for tc in tcs]
        msgs.append(assistant)

        if not tcs:
            noop += 1
            r["noop_total"] += 1
            if noop >= MAX_NOOP:
                r["stopped_reason"] = "noop"
                break
            msgs.append({"role": "user", "content": "Respond by calling a tool: try one "
                         "of the available tools, or call submit_answer with your final "
                         "answer."})
            continue
        noop = 0

        done = False
        for tc in tcs:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == SUBMIT_TOOL:
                ans = str(args.get("answer", ""))
                correct = bool(problem.check(ans))
                r["submit"] = {"meter_raw": meter, "correct": correct, "answer": ans}
                r["stopped_reason"] = "submitted_correct" if correct else "submitted_wrong"
                done = True
                break
            r["tool_calls"].append({"order": len(r["tool_calls"]), "tool": name,
                                    "meter_raw": meter, "turn": t})
            out = await call(name)
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})
        if done:
            break

        if meter >= budget:
            r["stopped_reason"] = "out_of_budget"
            break

    r["turns"] = len(r["agent_texts"])
    r["meter_total"] = meter
    r["stopped_reason"] = r["stopped_reason"] or "max_turns"
    return r, _trace(r)


def _accrue_usage(r: dict, usage: dict | None) -> None:
    u = usage or {}
    for k in RawChat.USAGE_FIELDS:
        r["usage"][k] += int(u.get(k, 0) or 0)
    r["usage"]["calls"] += 1


def _trace(r: dict) -> list[dict]:
    """Human-readable per-turn summary (parity with beach's trace), for replay/debug."""
    acts = [f"call:{c['tool']}" for c in r["tool_calls"]]
    if r["submit"] is not None:
        acts.append("submit")
    return [{"actions": acts, "agent_texts": r["agent_texts"]}]
