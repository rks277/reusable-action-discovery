"""Minimal multi-provider raw-chat shim for the grammar-world runner.

Unlike lomekwi/providers/* (built for tool-use trials), the grammar world drives
a plain text conversation. This routes (model, system, messages) to the right
backend by model-name prefix and returns the assistant text.

messages: list of {"role": "user"|"assistant", "content": str}
"""

from __future__ import annotations

import os
from typing import Any


def _is_reasoning(model: str) -> bool:
    """OpenAI o-series / GPT-5 and Gemini 2.5 do hidden reasoning that consumes
    the token budget before any visible text. They return empty at low caps."""
    m = model.lower()
    return (m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3")
            or m.startswith("gemini-2.5"))


def _provider_for(model: str) -> str:
    m = model.lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    if m.startswith("gemini"):
        return "google"
    # When VLLM_BASE_URL is set we are serving open models via a local vLLM
    # OpenAI-compatible server (bf16/FP8/TP), so route qwen/llama/HF-repo names
    # there instead of Ollama. Unset -> fall back to the Ollama path.
    if os.environ.get("VLLM_BASE_URL") and (
            m.startswith("qwen") or m.startswith("llama") or "/" in m):
        return "vllm"
    if ":" in m or m.startswith("qwen") or m.startswith("llama"):
        return "ollama"
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
                api_key=os.environ.get("VLLM_API_KEY", "vllm"),
                base_url=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8001/v1"),
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

        if prov == "ollama":
            # Ollama "thinking" models (e.g. qwen3.x) burn the whole token cap on
            # hidden reasoning and return EMPTY visible content at these caps,
            # which the toolworld parser sees as NO-OPs. The OpenAI-compat /v1
            # endpoint ignores the `think` flag, so we hit the native /api/chat
            # endpoint with think=False to suppress reasoning entirely and emit
            # actions directly -- keeping these comparable to the no-extended-
            # thinking Anthropic runs.
            import asyncio
            import json as _json
            import urllib.request
            base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            api = base.rstrip("/")
            if api.endswith("/v1"):
                api = api[:-3]
            url = api.rstrip("/") + "/api/chat"
            body = _json.dumps({
                "model": model, "think": False, "stream": False,
                "messages": [{"role": "system", "content": system}] + messages,
                "options": {"num_predict": max_tokens},
            }).encode()

            def _post():
                req = urllib.request.Request(
                    url, data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req) as r:
                    return _json.load(r)

            resp = await asyncio.to_thread(_post)
            self._set_usage(
                input_tokens=resp.get("prompt_eval_count", 0) or 0,
                output_tokens=resp.get("eval_count", 0) or 0,
            )
            return (resp.get("message", {}).get("content") or "")

        if prov in ("openai", "vllm"):
            # vLLM exposes the OpenAI chat-completions API; same call path, just a
            # different client/base_url. No ollama-style `think` flag here.
            client = self._openai() if prov == "openai" else self._vllm()
            oai_msgs = [{"role": "system", "content": system}] + messages
            kwargs = dict(model=model, messages=oai_msgs)
            if reasoning:
                effort = os.environ.get("OPENAI_REASONING_EFFORT", "low")
                kwargs["reasoning_effort"] = effort
            if prov == "vllm" and "qwen3" in model.lower():
                # Qwen3/3.5 default to a long reasoning phase that fills max_tokens
                # before emitting the visible action. Disable via chat_template_kwargs.
                kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

            async def _create():
                # GPT-5 family uses max_completion_tokens; be tolerant.
                try:
                    return await client.chat.completions.create(
                        max_completion_tokens=max_tokens, **kwargs)
                except TypeError:
                    kwargs.pop("reasoning_effort", None)
                    return await client.chat.completions.create(
                        max_tokens=max_tokens, **kwargs)

            if prov == "vllm":
                # Local server can briefly be unreachable (restart/load). Retry a
                # few times with backoff so a transient blip is not a lost episode.
                import asyncio
                attempts = int(os.environ.get("VLLM_RETRIES", "5"))
                for a in range(attempts):
                    try:
                        resp = await _create()
                        break
                    except Exception:
                        if a == attempts - 1:
                            raise
                        await asyncio.sleep(min(2 ** a, 30))
            else:
                resp = await _create()
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
                # cap thinking so visible output isn't starved (and cost is bounded)
                try:
                    cfg_kw["thinking_config"] = types.ThinkingConfig(thinking_budget=512)
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

    # --- tool-use variants (for the oracle MCP environment) -------------------------
    # Unlike chat(), these pass a `tools` schema to the provider and return the RAW
    # response object so a caller can read tool_use / tool_call blocks and run a manual
    # agentic loop (needed to stamp a per-call token meter). Usage is recorded into
    # self.last_usage exactly like chat(), so the existing accounting keeps working.
    # `messages` must already be in the provider's native shape (the MCP host builds
    # assistant tool_use blocks and tool_result/role:tool turns itself).

    async def chat_tools_anthropic(self, model: str, system: str, messages: list[dict],
                                   tools: list[dict], max_tokens: int = 1500):
        self.last_usage = None
        self.last_debug = None
        resp = await self._anthropic().messages.create(
            model=model, max_tokens=max_tokens, system=system,
            messages=messages, tools=tools,
        )
        u = getattr(resp, "usage", None)
        self._set_usage(
            input_tokens=getattr(u, "input_tokens", 0),
            output_tokens=getattr(u, "output_tokens", 0),
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0),
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0),
        )
        return resp

    async def chat_tools_openai(self, model: str, system: str, messages: list[dict],
                                tools: list[dict], max_tokens: int = 1500,
                                provider: str | None = None):
        """OpenAI-compatible function-calling (used for Qwen via vLLM/Ollama and for
        OpenAI). `provider` selects the client; defaults to routing from the model."""
        prov = provider or _provider_for(model)
        client = {"openai": self._openai, "vllm": self._vllm,
                  "ollama": self._ollama}.get(prov, self._openai)()
        self.last_usage = None
        self.last_debug = None
        oai_msgs = [{"role": "system", "content": system}] + messages
        kwargs = dict(model=model, messages=oai_msgs, tools=tools, tool_choice="auto")
        try:
            resp = await client.chat.completions.create(
                max_completion_tokens=max_tokens, **kwargs)
        except TypeError:
            resp = await client.chat.completions.create(
                max_tokens=max_tokens, **kwargs)
        u = getattr(resp, "usage", None)
        ptd = getattr(u, "prompt_tokens_details", None)
        ctd = getattr(u, "completion_tokens_details", None)
        self._set_usage(
            input_tokens=getattr(u, "prompt_tokens", 0),
            output_tokens=getattr(u, "completion_tokens", 0),
            cache_read_tokens=getattr(ptd, "cached_tokens", 0),
            reasoning_tokens=getattr(ctd, "reasoning_tokens", 0),
        )
        return resp
