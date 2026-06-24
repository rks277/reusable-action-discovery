"""Real MCP server for one CREATOR episode (FastMCP, stdio). Spawned per-episode by
MCPStdioBackend so each process holds isolated state (scripts, the example + 20 hidden
test problems + private reference answers, the phase flag).

  python -m scripts.creator.mcp_creator.server --item-idx N --seed S --n 20

Exposes the 4 model-facing tools (write_script, run_script, begin_test, submit_answers)
plus two driver-only tools (__briefing, __results) filtered out of the model's tool list.
All tool logic lives in episode_state.EpisodeState (shared with the in-process backend).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from scripts.creator.mcp_creator.backends import _briefing_from_es
from scripts.creator.mcp_creator.episode_state import TOOL_SCHEMAS, build_episode

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CC = os.path.join(REPO_ROOT, "external", "CC.jsonl")


def _load_item(item_idx: int) -> dict:
    with open(CC) as f:
        for i, line in enumerate(f):
            if i == item_idx and line.strip():
                return json.loads(line)
    raise IndexError(f"item_idx {item_idx} not found in {CC}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item-idx", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    es = build_episode(_load_item(args.item_idx), args.item_idx, args.seed, args.n)
    if es is None:
        print(f"build_episode returned None for item {args.item_idx}", file=sys.stderr)
        sys.exit(1)

    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("creator-mcp")
    desc = {t["function"]["name"]: t["function"]["description"] for t in TOOL_SCHEMAS(es.keys)}

    @mcp.tool(description=desc["write_script"])
    def write_script(name: str, code: str) -> dict:
        return es.op_write_script(name, code)

    @mcp.tool(description=desc["run_script"])
    def run_script(name: str, inputs: dict | None = None) -> dict:
        return es.op_run_script(name, inputs)

    @mcp.tool(description=desc["begin_test"])
    def begin_test() -> dict:
        return es.op_begin_test()

    @mcp.tool(description=desc["submit_answers"])
    def submit_answers(answers: list[float]) -> dict:
        return es.op_submit_answers(answers)

    @mcp.tool(name="__briefing", description="(driver only) episode briefing for the prompt")
    def __briefing() -> dict:
        return _briefing_from_es(es)

    @mcp.tool(name="__results", description="(driver only) final score for this episode")
    def __results() -> dict:
        return es.score()

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
