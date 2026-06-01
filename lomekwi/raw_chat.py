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
    if ":" in m or m.startswith("qwen") or m.startswith("llama"):
        return "ollama"
    raise ValueError(f"cannot route model {model!r}")


class RawChat:
    """Lazily constructs one client per provider; reusable across calls."""

    def __init__(self):
        self._clients: dict[str, Any] = {}

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

    def _google(self):
        if "google" not in self._clients:
            from google import genai
            self._clients["google"] = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            )
        return self._clients["google"]

    async def chat(self, model: str, system: str, messages: list[dict], max_tokens: int = 1200) -> str:
        prov = _provider_for(model)
        # Reasoning models burn the budget on hidden reasoning -> empty visible
        # text at low caps. Give them headroom; keep reasoning "low" so they stay
        # comparable to the no-extended-thinking Anthropic runs.
        reasoning = _is_reasoning(model)
        if reasoning:
            max_tokens = max(max_tokens, 4000)

        if prov == "anthropic":
            resp = await self._anthropic().messages.create(
                model=model, max_tokens=max_tokens, system=system, messages=messages,
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

        if prov in ("openai", "ollama"):
            client = self._openai() if prov == "openai" else self._ollama()
            oai_msgs = [{"role": "system", "content": system}] + messages
            kwargs = dict(model=model, messages=oai_msgs)
            if reasoning:
                kwargs["reasoning_effort"] = "low"
            # GPT-5 family uses max_completion_tokens; be tolerant.
            try:
                resp = await client.chat.completions.create(max_completion_tokens=max_tokens, **kwargs)
            except TypeError:
                kwargs.pop("reasoning_effort", None)
                resp = await client.chat.completions.create(max_tokens=max_tokens, **kwargs)
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
            cand = (resp.candidates or [None])[0]
            parts = getattr(getattr(cand, "content", None), "parts", []) or []
            return "".join(getattr(p, "text", "") or "" for p in parts)

        raise ValueError(prov)
