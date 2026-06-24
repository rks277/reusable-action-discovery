"""Minimal multi-provider raw-chat shim for the grammar-world runner.

Unlike lomekwi/providers/* (built for tool-use trials), the grammar world drives
a plain text conversation. This routes (model, system, messages) to the right
backend by model-name prefix and returns the assistant text.

messages: list of {"role": "user"|"assistant", "content": str}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatTurn:
    """One assistant turn from a tool-calling chat. `tool_calls` items are normalized to
    {"id", "name", "arguments" (raw str), "args" (parsed dict or None)}; args is None when
    the model emitted malformed JSON (the caller counts these)."""
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str | None = None


def _is_reasoning(model: str) -> bool:
    """OpenAI o-series / GPT-5 and Gemini 2.5 do hidden reasoning that consumes
    the token budget before any visible text. They return empty at low caps."""
    m = model.lower()
    return (m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3")
            or m.startswith("gemini-2.5") or m.startswith("gemini-3"))


def _provider_for(model: str) -> str:
    m = model.lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    # Ollama tags always carry a ":" (e.g. "gemma4:e2b", "qwen2.5:7b"); check this
    # BEFORE the gemma/gemini branch so local gemma weights route to ollama, not the
    # Google API (whose Gemma names use dashes, no colon).
    if (":" in m or m.startswith("qwen") or m.startswith("llama") or m.startswith("glm")
            or m.startswith("mistral") or m.startswith("mixtral")):
        # A local open-weights model. Route to vLLM (OpenAI-compatible, batched) when
        # LOCAL_BACKEND=vllm, else Ollama. vLLM rejects ollama's reasoning_effort arg,
        # so it gets its own branch below.
        return "vllm" if os.environ.get("LOCAL_BACKEND", "").lower() == "vllm" else "ollama"
    if m.startswith("gemini") or m.startswith("gemma"):
        return "google"
    raise ValueError(f"cannot route model {model!r}")


class RawChat:
    """Lazily constructs one client per provider; reusable across calls."""

    # normalized per-call token usage; fields are raw provider counts (ints).
    # NOTE on semantics (avoids a double-counting trap): Anthropic's input_tokens
    # is the UNCACHED remainder (cache read/write are separate), whereas OpenAI's
    # prompt_tokens INCLUDES its cached_tokens subset. So a cross-provider "total
    # input" is input+cache_read+cache_write for Anthropic but just input for
    # OpenAI/Google. We log the raw fields and leave that reconciliation to analysis.
    USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens",
                    "cache_write_tokens", "reasoning_tokens")

    def __init__(self):
        self._clients: dict[str, Any] = {}
        self.last_usage: dict | None = None  # set per chat() call; None on failure
        self.last_debug: dict | None = None  # set when a call returns empty text

    def _anthropic(self):
        if "anthropic" not in self._clients:
            from anthropic import AsyncAnthropic
            self._clients["anthropic"] = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        return self._clients["anthropic"]

    def _openai(self):
        if "openai" not in self._clients:
            from openai import AsyncOpenAI
            self._clients["openai"] = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return self._clients["openai"]

    def _ollama(self):
        if "ollama" not in self._clients:
            from openai import AsyncOpenAI
            self._clients["ollama"] = AsyncOpenAI(
                api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
                base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            )
        return self._clients["ollama"]

    def _vllm(self):
        if "vllm" not in self._clients:
            from openai import AsyncOpenAI
            self._clients["vllm"] = AsyncOpenAI(
                api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
                base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            )
        return self._clients["vllm"]

    def _google(self):
        if "google" not in self._clients:
            from google import genai
            self._clients["google"] = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            )
        return self._clients["google"]

    @staticmethod
    def _norm_usage(**fields) -> dict:
        """Build a normalized usage dict, coercing None/missing to int 0."""
        return {k: int(fields.get(k) or 0) for k in RawChat.USAGE_FIELDS}

    def _set_usage(self, **fields) -> None:
        """Record per-call usage; never let extraction break the chat call."""
        try:
            self.last_usage = self._norm_usage(**fields)
        except Exception:
            self.last_usage = None

    async def chat(self, model: str, system: str, messages: list[dict], max_tokens: int = 1200) -> str:
        prov = _provider_for(model)
        self.last_usage = None  # reset; stays None if the call/extraction fails
        self.last_debug = None
        # Reasoning models burn the budget on hidden reasoning -> empty visible
        # text at low caps. Give them headroom; keep reasoning "low" so they stay
        # comparable to the no-extended-thinking Anthropic runs.
        reasoning = _is_reasoning(model)
        if reasoning:
            max_tokens = max(max_tokens, 4000)

        if prov == "anthropic":
            # Prompt caching: the transcript only ever grows by appending, so a
            # breakpoint on the last message caches the whole prefix (system +
            # all prior turns). Next turn that breakpoint is an interior prefix
            # -> served as a cache read (~10% of input price); we only pay full
            # price for the new turn. Collapses the quadratic input cost.
            # COPY first: mutating the caller's list would leave a breakpoint on
            # every past message and blow past Anthropic's 4-breakpoint cap.
            cached = [dict(m) for m in messages]
            if cached:
                last = cached[-1]
                blocks = ([{"type": "text", "text": last["content"]}]
                          if isinstance(last["content"], str) else list(last["content"]))
                blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
                last["content"] = blocks
            resp = await self._anthropic().messages.create(
                model=model, max_tokens=max_tokens, system=system, messages=cached,
            )
            u = getattr(resp, "usage", None)
            self._set_usage(
                input_tokens=getattr(u, "input_tokens", 0),
                output_tokens=getattr(u, "output_tokens", 0),
                cache_read_tokens=getattr(u, "cache_read_input_tokens", 0),
                cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0),
            )
            text = "".join(b.text for b in resp.content
                           if getattr(b, "type", None) == "text")
            if not text:  # empty visible text -> capture why
                self.last_debug = {
                    "stop_reason": getattr(resp, "stop_reason", None),
                    "stop_details": str(getattr(resp, "stop_details", None) or ""),
                    "block_types": [getattr(b, "type", None) for b in (resp.content or [])],
                    "output_tokens": getattr(u, "output_tokens", None),
                }
            return text

        if prov in ("openai", "ollama", "vllm"):
            client = {"openai": self._openai, "ollama": self._ollama,
                      "vllm": self._vllm}[prov]()
            oai_msgs = [{"role": "system", "content": system}] + messages
            kwargs = dict(model=model, messages=oai_msgs)
            if prov == "ollama":
                # Local thinking models (e.g. gemma4) emit a long hidden reasoning
                # trace that overruns the token cap -> truncated mid-thought ->
                # empty `content` (counted as a noop) and ~25-min episodes. Turn it
                # off for parity with the no-extended-thinking Anthropic baseline;
                # harmless for non-thinking local models (qwen2.5). vLLM does NOT
                # accept this arg (400, not TypeError), so its branch omits it.
                kwargs["reasoning_effort"] = "none"
            elif reasoning:
                # default "low"; override per-run via OPENAI_REASONING_EFFORT (e.g. "minimal").
                kwargs["reasoning_effort"] = os.environ.get("OPENAI_REASONING_EFFORT", "low")
            # GPT-5 family uses max_completion_tokens; be tolerant.
            try:
                resp = await client.chat.completions.create(max_completion_tokens=max_tokens, **kwargs)
            except TypeError:
                kwargs.pop("reasoning_effort", None)
                resp = await client.chat.completions.create(max_tokens=max_tokens, **kwargs)
            u = getattr(resp, "usage", None)
            ptd = getattr(u, "prompt_tokens_details", None)
            ctd = getattr(u, "completion_tokens_details", None)
            # OpenAI prompt_tokens INCLUDES cached_tokens; cache_write n/a (auto-cache).
            self._set_usage(
                input_tokens=getattr(u, "prompt_tokens", 0),
                output_tokens=getattr(u, "completion_tokens", 0),
                cache_read_tokens=getattr(ptd, "cached_tokens", 0),
                reasoning_tokens=getattr(ctd, "reasoning_tokens", 0),
            )
            return resp.choices[0].message.content or ""

        if prov == "google":
            import asyncio
            from google.genai import types
            # Convert to a single contents list; google uses "model" for assistant.
            contents = []
            for m in messages:
                role = "model" if m["role"] == "assistant" else "user"
                contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
            cfg_kw = dict(system_instruction=system, max_output_tokens=max_tokens)
            if reasoning:
                # Gemini 3.x ignores small positive budgets (treats <min as advisory) but
                # honors 0 = OFF (true no-thinking, matching the Anthropic baseline). Pro-tier
                # models reject 0; fall back to a small positive budget for them.
                budget = int(os.environ.get("GEMINI_THINKING_BUDGET", "512"))
                try:
                    cfg_kw["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)
                except Exception:
                    pass
            cfg = types.GenerateContentConfig(**cfg_kw)
            resp = await asyncio.to_thread(
                self._google().models.generate_content,
                model=model, contents=contents, config=cfg,
            )
            um = getattr(resp, "usage_metadata", None)
            self._set_usage(
                input_tokens=getattr(um, "prompt_token_count", 0),
                output_tokens=getattr(um, "candidates_token_count", 0),
                cache_read_tokens=getattr(um, "cached_content_token_count", 0),
                reasoning_tokens=getattr(um, "thoughts_token_count", 0),
            )
            cand = (resp.candidates or [None])[0]
            parts = getattr(getattr(cand, "content", None), "parts", []) or []
            return "".join(getattr(p, "text", "") or "" for p in parts)

        raise ValueError(prov)

    async def chat_tools(self, model: str, system: str, messages: list[dict],
                         tools: list[dict], max_tokens: int = 1200,
                         tool_choice: str = "auto") -> ChatTurn:
        """Tool-calling chat. `messages` are OpenAI-format (incl. assistant tool_calls and
        role="tool" results); translated to Anthropic blocks when needed. Sets last_usage
        identically to chat(). Returns a ChatTurn (.content, .tool_calls, .finish_reason)."""
        prov = _provider_for(model)
        self.last_usage = None
        self.last_debug = None

        if prov in ("openai", "ollama", "vllm"):
            client = {"openai": self._openai, "ollama": self._ollama,
                      "vllm": self._vllm}[prov]()
            oai_msgs = [{"role": "system", "content": system}] + messages
            kwargs = dict(model=model, messages=oai_msgs, tools=tools, tool_choice=tool_choice)
            try:
                resp = await client.chat.completions.create(max_completion_tokens=max_tokens, **kwargs)
            except TypeError:
                resp = await client.chat.completions.create(max_tokens=max_tokens, **kwargs)
            u = getattr(resp, "usage", None)
            ptd = getattr(u, "prompt_tokens_details", None)
            ctd = getattr(u, "completion_tokens_details", None)
            self._set_usage(
                input_tokens=getattr(u, "prompt_tokens", 0),
                output_tokens=getattr(u, "completion_tokens", 0),
                cache_read_tokens=getattr(ptd, "cached_tokens", 0),
                reasoning_tokens=getattr(ctd, "reasoning_tokens", 0),
            )
            choice = resp.choices[0]
            msg = choice.message
            return ChatTurn(content=msg.content or "",
                            tool_calls=_norm_oai_tool_calls(getattr(msg, "tool_calls", None)),
                            finish_reason=getattr(choice, "finish_reason", None))

        if prov == "anthropic":
            resp = await self._anthropic().messages.create(
                model=model, max_tokens=max_tokens, system=system,
                messages=_oai_msgs_to_anthropic(messages),
                tools=_oai_tools_to_anthropic(tools),
            )
            u = getattr(resp, "usage", None)
            self._set_usage(
                input_tokens=getattr(u, "input_tokens", 0),
                output_tokens=getattr(u, "output_tokens", 0),
                cache_read_tokens=getattr(u, "cache_read_input_tokens", 0),
                cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0),
            )
            content, tcs = "", []
            for b in (resp.content or []):
                if getattr(b, "type", None) == "text":
                    content += b.text
                elif getattr(b, "type", None) == "tool_use":
                    tcs.append({"id": b.id, "name": b.name,
                                "arguments": json.dumps(b.input), "args": dict(b.input)})
            return ChatTurn(content=content, tool_calls=tcs,
                            finish_reason=getattr(resp, "stop_reason", None))

        raise ValueError(f"chat_tools: unsupported provider {prov!r}")


def _norm_oai_tool_calls(tool_calls) -> list[dict]:
    out = []
    for tc in (tool_calls or []):
        raw = tc.function.arguments or ""
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            parsed = None  # malformed JSON args -> caller counts as malformed
        out.append({"id": tc.id, "name": tc.function.name, "arguments": raw, "args": parsed})
    return out


def _oai_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        f = t["function"]
        out.append({"name": f["name"], "description": f.get("description", ""),
                    "input_schema": f.get("parameters", {"type": "object", "properties": {}})})
    return out


def _oai_msgs_to_anthropic(messages: list[dict]) -> list[dict]:
    """OpenAI-format history -> Anthropic content-block messages. Consecutive role="tool"
    results are merged into one user turn (Anthropic requires all tool_results for a given
    assistant turn in a single following user message)."""
    out, i, n = [], 0, len(messages)
    while i < n:
        m = messages[i]
        role = m["role"]
        if role == "tool":
            blocks = []
            while i < n and messages[i]["role"] == "tool":
                t = messages[i]
                blocks.append({"type": "tool_result", "tool_use_id": t["tool_call_id"],
                               "content": str(t["content"])})
                i += 1
            out.append({"role": "user", "content": blocks})
            continue
        if role == "user":
            out.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            blocks = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in (m.get("tool_calls") or []):
                args = tc["function"]["arguments"]
                try:
                    inp = json.loads(args) if args else {}
                except json.JSONDecodeError:
                    inp = {}
                blocks.append({"type": "tool_use", "id": tc["id"],
                               "name": tc["function"]["name"], "input": inp})
            out.append({"role": "assistant", "content": blocks or ""})
        i += 1
    return out
