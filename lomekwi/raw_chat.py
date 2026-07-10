"""Minimal multi-provider raw-chat shim for the grammar-world runner.

Unlike lomekwi/providers/* (built for tool-use trials), the grammar world drives
a plain text conversation. This routes (model, system, messages) to the right
backend by model-name prefix and returns the assistant text.

messages: list of {"role": "user"|"assistant", "content": str}
"""

from __future__ import annotations

import json
import os
import re
import uuid
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


def provider_for(model: str) -> str:
    """Public provider lookup for harness configuration and accounting."""
    return _provider_for(model)


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
        self.last_response_model: str | None = None

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

    async def chat(self, model: str, system: str, messages: list[dict], max_tokens: int = 1200,
                   temperature: float | None = None,
                   reasoning_effort: str | None = None) -> str:
        prov = _provider_for(model)
        self.last_usage = None  # reset; stays None if the call/extraction fails
        self.last_debug = None
        self.last_response_model = None
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
            self.last_response_model = getattr(resp, "model", None)
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
            if temperature is not None:      # e.g. 0 = greedy, for reading a learned policy cleanly
                kwargs["temperature"] = temperature
            if prov == "ollama":
                # Local thinking models (e.g. gemma4) emit a long hidden reasoning
                # trace that overruns the token cap -> truncated mid-thought ->
                # empty `content` (counted as a noop) and ~25-min episodes. Turn it
                # off for parity with the no-extended-thinking Anthropic baseline;
                # harmless for non-thinking local models (qwen2.5). vLLM does NOT
                # accept this arg (400, not TypeError), so its branch omits it.
                kwargs["reasoning_effort"] = "none"
            elif reasoning:
                # An explicit per-run value takes precedence; the environment remains the
                # backwards-compatible default for existing callers.
                kwargs["reasoning_effort"] = (
                    reasoning_effort or os.environ.get("OPENAI_REASONING_EFFORT", "low")
                )
            # GPT-5 family uses max_completion_tokens; be tolerant.
            try:
                resp = await client.chat.completions.create(max_completion_tokens=max_tokens, **kwargs)
            except TypeError:
                kwargs.pop("reasoning_effort", None)
                resp = await client.chat.completions.create(max_tokens=max_tokens, **kwargs)
            u = getattr(resp, "usage", None)
            self.last_response_model = getattr(resp, "model", None)
            ptd = getattr(u, "prompt_tokens_details", None)
            ctd = getattr(u, "completion_tokens_details", None)
            # OpenAI prompt_tokens includes cached/read and cache-write subsets. Older SDKs omit
            # cache-write detail; normalize that case to zero and price the remainder as uncached.
            self._set_usage(
                input_tokens=getattr(u, "prompt_tokens", 0),
                output_tokens=getattr(u, "completion_tokens", 0),
                cache_read_tokens=getattr(ptd, "cached_tokens", 0),
                cache_write_tokens=(getattr(ptd, "cache_write_tokens", 0)
                                    or getattr(ptd, "cache_creation_tokens", 0)),
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
                         tool_choice: str = "auto",
                         temperature: float | None = None,
                         reasoning_effort: str | None = None) -> ChatTurn:
        """Tool-calling chat. `messages` are OpenAI-format (incl. assistant tool_calls and
        role="tool" results); translated to Anthropic blocks when needed. Sets last_usage
        identically to chat(). Returns a ChatTurn (.content, .tool_calls, .finish_reason).
        temperature=0 forces greedy decoding (diagnostic use — e.g. isolating whether a
        fine-tuned checkpoint's early derailment is a sampling-variance artifact or a genuine
        learned-distribution defect); left unset by default so callers keep serving defaults."""
        prov = _provider_for(model)
        self.last_usage = None
        self.last_debug = None
        self.last_response_model = None

        if prov in ("openai", "ollama", "vllm"):
            client = {"openai": self._openai, "ollama": self._ollama,
                      "vllm": self._vllm}[prov]()
            oai_msgs = [{"role": "system", "content": system}] + messages
            kwargs = dict(model=model, messages=oai_msgs, tools=tools, tool_choice=tool_choice)
            if temperature is not None:
                kwargs["temperature"] = temperature
            if prov == "openai" and _is_reasoning(model):
                # Keep tool and text arms symmetric: GPT-5 reasoning consumes the completion
                # budget before emitting a tool call, just as it does before visible text.
                max_tokens = max(max_tokens, 4000)
                kwargs["reasoning_effort"] = (
                    reasoning_effort or os.environ.get("OPENAI_REASONING_EFFORT", "low")
                )
            try:
                resp = await client.chat.completions.create(max_completion_tokens=max_tokens, **kwargs)
            except TypeError:
                kwargs.pop("reasoning_effort", None)
                resp = await client.chat.completions.create(max_tokens=max_tokens, **kwargs)
            u = getattr(resp, "usage", None)
            self.last_response_model = getattr(resp, "model", None)
            ptd = getattr(u, "prompt_tokens_details", None)
            ctd = getattr(u, "completion_tokens_details", None)
            self._set_usage(
                input_tokens=getattr(u, "prompt_tokens", 0),
                output_tokens=getattr(u, "completion_tokens", 0),
                cache_read_tokens=getattr(ptd, "cached_tokens", 0),
                cache_write_tokens=(getattr(ptd, "cache_write_tokens", 0)
                                    or getattr(ptd, "cache_creation_tokens", 0)),
                reasoning_tokens=getattr(ctd, "reasoning_tokens", 0),
            )
            choice = resp.choices[0]
            msg = choice.message
            content = msg.content or ""
            tcs = _norm_oai_tool_calls(getattr(msg, "tool_calls", None))
            if not tcs and prov in ("ollama", "vllm") and tools:
                known = {t["function"]["name"] for t in tools}
                tcs, content = _extract_untagged_tool_call(content, known)
            return ChatTurn(content=content, tool_calls=tcs,
                            finish_reason=getattr(choice, "finish_reason", None))

        if prov == "anthropic":
            # Prompt caching (same pattern as chat()): a breakpoint on the last message caches the
            # whole growing prefix (system + tools + prior turns). Next turn it is an interior
            # prefix -> served as a cache read (~10% of input price); we pay full price only for the
            # new turn. Without this a persistent tool session re-bills the entire transcript every
            # turn -> quadratic input cost (the N=80 stream blew past 2.5M spent tokens).
            anth_msgs = _oai_msgs_to_anthropic(messages)
            if anth_msgs:
                last = dict(anth_msgs[-1])
                blocks = ([{"type": "text", "text": last["content"]}]
                          if isinstance(last["content"], str) else list(last["content"]))
                blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
                last["content"] = blocks
                anth_msgs = anth_msgs[:-1] + [last]
            # BUG FIX (2026-07-09, found via the framing ladder's R2 rung): `tool_choice` was never
            # passed here, so a caller's "required" was silently dropped and Anthropic defaulted to
            # "auto" -- the model was always free to answer with plain text and no tool call at all.
            # This didn't surface on R1 (abstract balls: nothing to "solve," so the model always
            # chose to call a tool anyway even under unenforced "auto") but did on R2 (real numeric
            # problems tempt the model into hand-solving the arithmetic in free text instead of
            # deciding claim_solver/skip_solver, with nothing stopping it).
            resp = await self._anthropic().messages.create(
                model=model, max_tokens=max_tokens, system=system,
                messages=anth_msgs,
                tools=_oai_tools_to_anthropic(tools),
                tool_choice=_oai_tool_choice_to_anthropic(tool_choice),
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


def _coerce_tool_call_obj(obj, known_names: set[str]) -> dict | None:
    """Recognizes shapes seen from qwen2.5-coder via Ollama: (A) {"name": <tool>, "arguments": {...}}
    (the templated shape, sometimes emitted untagged); (B) {<tool>: {...args...}} (tool name used as
    the dict key directly, args as the value, no "name"/"arguments" wrapper at all); (C) a write_script
    call where the model puts the SCRIPT's name in the top-level "name" slot (where the tool name
    belongs) and leaves only "code" nested in "arguments" -- {"name": <script name>, "arguments":
    {"code": ...}} -- observed 2026-07-03: the model conflates the tool-call envelope with the tool's
    own (name, code) parameters. "code" is a write_script-only argument in this schema (run_script
    takes "inputs"), so its presence unambiguously identifies the intended tool."""
    if not isinstance(obj, dict):
        return None
    if obj.get("name") in known_names:                        # shape A
        name, args = obj["name"], obj.get("arguments", {})
    elif len(obj) == 1 and next(iter(obj)) in known_names and isinstance(next(iter(obj.values())), dict):
        name, args = next(iter(obj.items()))                  # shape B
    elif ("write_script" in known_names and isinstance(obj.get("name"), str)
          and isinstance(obj.get("arguments"), dict) and "code" in obj["arguments"]
          and "name" not in obj["arguments"]):
        name, args = "write_script", {"name": obj["name"], "code": obj["arguments"]["code"]}  # shape C
    else:
        return None
    if isinstance(args, str):                  # some templates put a JSON-encoded string here
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    return {"id": f"untagged_{uuid.uuid4().hex[:8]}", "name": name,
            "arguments": json.dumps(args), "args": args if isinstance(args, dict) else {}}


_TRIPLE_QUOTED = re.compile(r'"""(.*?)"""', re.DOTALL)


def _repair_triple_quoted_strings(content: str) -> str:
    """qwen2.5-coder sometimes writes a JSON string value as a Python triple-quoted literal (a
    'code' field opened and closed with three double-quote characters, wrapping raw unescaped
    newlines) instead of a properly escaped JSON string -- invalid JSON (an empty string
    immediately followed by a stray quote token), so every write_script call carrying real
    multi-line code fails to parse. Three consecutive double-quotes are never valid syntax inside
    well-formed JSON, so replacing each triple-quoted span with a correctly escaped JSON string
    literal (via json.dumps) can only turn a guaranteed decode failure into a valid parse -- never
    breaks an already-parseable document."""
    return _TRIPLE_QUOTED.sub(lambda m: json.dumps(m.group(1)), content)


def _extract_untagged_tool_call(content: str, known_names: set[str]) -> tuple[list[dict], str]:
    """Fallback for local models (observed: qwen2.5-coder via Ollama) whose chat template asks for
    a <tool_call>{"name":..,"arguments":..}</tool_call> block, but which instead emit the JSON with
    no tags, wrapped in a Markdown code fence (```json ... ```, sometimes with a stray trailing
    second fence), and/or with the tool name used as the top-level dict KEY instead of a "name"
    field -> Ollama's parser only recognizes the exact tagged shape-A form, so `message.tool_calls`
    comes back empty and the call silently looks like a plain text turn (all reproduced 2026-07-03;
    chat_tools() had never been exercised against Ollama before, so this is new territory, not a
    regression). Scans for every '{' and tries a real JSON decode from there (json.JSONDecoder.
    raw_decode -- correctly handles nested braces, unlike a brace-matching regex, which truncates at
    the first inner '}'), taking the first result matching a known tool call shape. Only fires for
    openai/ollama/vllm providers when the SDK found no tool_calls; a real tool_calls response is
    used as-is and this is never consulted."""
    content = _repair_triple_quoted_strings(content)
    dec = json.JSONDecoder()
    i = content.find("{")
    while i != -1:
        try:
            obj, end = dec.raw_decode(content, i)
        except json.JSONDecodeError:
            i = content.find("{", i + 1)
            continue
        tc = _coerce_tool_call_obj(obj, known_names)
        if tc:
            remaining = (content[:i] + content[end:]).strip()
            return [tc], remaining
        i = content.find("{", i + 1)
    return [], content


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


def _oai_tool_choice_to_anthropic(tool_choice: str) -> dict:
    """Translate this codebase's OpenAI-style `tool_choice` convention ("auto"/"required") into
    Anthropic's `{"type": ...}` shape. Anthropic has no "required" literal -- its equivalent is
    `{"type": "any"}` (force some tool call; the model may not respond with text-only)."""
    return {"type": "any"} if tool_choice == "required" else {"type": "auto"}


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
