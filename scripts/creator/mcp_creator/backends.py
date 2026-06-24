"""Uniform tool-backend interface for the driver. Two implementations drive the SAME
EpisodeState logic:
  - InProcBackend: wraps an EpisodeState directly (no wire protocol, no `mcp` dep) — for
    fast local smoke tests.
  - MCPStdioBackend: spawns the real FastMCP server (server.py) as a stdio subprocess, one
    per episode (automatic state isolation), and proxies tool calls over MCP.

Both expose: briefing() (data for the system prompt), model_tools() (the 4 OpenAI tool
schemas), call(name, args), results() (final score), aclose().
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack

from scripts.creator.mcp_creator.episode_state import (TOOL_SCHEMAS, build_episode)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _briefing_from_es(es) -> dict:
    return {"template": es.template, "keys": es.keys, "n_test": es.n_test,
            "example_inputs": es.example_inputs, "example_reference": es.example_reference}


class InProcBackend:
    kind = "inproc"

    def __init__(self, es):
        self.es = es

    async def briefing(self) -> dict:
        return _briefing_from_es(self.es)

    def model_tools(self) -> list[dict]:
        return TOOL_SCHEMAS(self.es.keys)

    async def call(self, name: str, args: dict) -> dict:
        return self.es.call(name, args)

    async def results(self) -> dict:
        return self.es.score()

    async def aclose(self):
        pass


class MCPStdioBackend:
    kind = "mcp"

    def __init__(self, item_idx: int, seed: int, n_test: int):
        self.item_idx, self.seed, self.n_test = item_idx, seed, n_test
        self._stack: AsyncExitStack | None = None
        self.session = None
        self._tools: list = []

    async def setup(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        self._stack = AsyncExitStack()
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "scripts.creator.mcp_creator.server",
                  "--item-idx", str(self.item_idx), "--seed", str(self.seed),
                  "--n", str(self.n_test)],
            env={**os.environ, "PYTHONPATH": REPO_ROOT},
            cwd=REPO_ROOT,
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        self._tools = (await self.session.list_tools()).tools
        return self

    def model_tools(self) -> list[dict]:
        out = []
        for t in self._tools:
            if t.name.startswith("__"):
                continue
            out.append({"type": "function", "function": {
                "name": t.name, "description": t.description or "",
                "parameters": t.inputSchema or {"type": "object", "properties": {}}}})
        return out

    async def _call_raw(self, name: str, args: dict) -> dict:
        res = await self.session.call_tool(name, arguments=args or {})
        sc = getattr(res, "structuredContent", None)
        if isinstance(sc, dict) and sc:
            return sc.get("result", sc) if set(sc.keys()) == {"result"} else sc
        for block in (res.content or []):
            txt = getattr(block, "text", None)
            if txt:
                try:
                    return json.loads(txt)
                except json.JSONDecodeError:
                    return {"ok": False, "error": txt}
        return {"ok": False, "error": "empty tool result"}

    async def briefing(self) -> dict:
        return await self._call_raw("__briefing", {})

    async def call(self, name: str, args: dict) -> dict:
        return await self._call_raw(name, args)

    async def results(self) -> dict:
        return await self._call_raw("__results", {})

    async def aclose(self):
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:
                pass


async def make_backend(kind: str, item: dict, item_idx: int, seed: int, n_test: int):
    """item is required for inproc; for mcp the server rebuilds it from item_idx/seed."""
    if kind == "inproc":
        es = build_episode(item, item_idx, seed, n_test)
        if es is None:
            raise ValueError(f"item {item_idx} is infeasible (build_episode returned None)")
        return InProcBackend(es)
    if kind == "mcp":
        return await MCPStdioBackend(item_idx, seed, n_test).setup()
    raise ValueError(f"unknown backend kind {kind!r}")
