"""Local stdio MCP server exposing the oracle environment's tools.

Every CANDIDATE tool is a decoy: it takes no arguments and returns the identical opaque
``DECOY_RESULT`` ("no result"). This behavioral identity is what makes the all-decoy
rollout post-processable into n oracle-assignments (see plan / config.py). One extra,
real tool ``submit_answer(answer)`` lets the model end the episode; the host detects it
and never treats it as a counterfactual candidate.

Run it (the host spawns this over stdio):
    python -m lomekwi.oracle.server <n> [--labels-json '{"0":"abc","1":"xyz",...}']

``--labels-json`` maps the canonical tool index (0..n-1) to the surface name to expose
(per-episode obfuscation). Omitted -> names come from config.TOOL_SPECS, falling back to
``tool_<i>`` when n exceeds the spec list.
"""

from __future__ import annotations

import argparse
import json

from mcp.server.fastmcp import FastMCP

from lomekwi.oracle.config import (DECOY_RESULT, SHARED_DESCRIPTION, SUBMIT_TOOL,
                                    tool_specs_name)


def tool_name(i: int, labels: dict | None) -> str:
    """Surface name for candidate index i: obfuscated label > config spec > tool_<i>."""
    if labels and str(i) in labels:
        return labels[str(i)]
    return tool_specs_name(i)


def build_server(n: int, labels: dict | None = None) -> FastMCP:
    """A FastMCP server with n identical decoy tools + a submit_answer tool."""
    mcp = FastMCP("oracle-decoys")

    for i in range(n):
        # Every decoy is behaviorally identical (no args, same opaque output); the only
        # differentiator is the registered name. add_tool's name= sets the tool name.
        def decoy() -> str:
            return DECOY_RESULT
        mcp.add_tool(decoy, name=tool_name(i, labels), description=SHARED_DESCRIPTION)

    def submit_answer(answer: str) -> str:
        """Submit your final answer. A single submission ends the task."""
        return "SUBMITTED"
    mcp.add_tool(submit_answer, name=SUBMIT_TOOL,
                 description="Submit your final answer to end the task.")
    return mcp


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=8)
    ap.add_argument("--labels-json", default=None)
    args = ap.parse_args()
    labels = json.loads(args.labels_json) if args.labels_json else None
    build_server(args.n, labels).run()   # stdio transport by default


if __name__ == "__main__":
    _main()
