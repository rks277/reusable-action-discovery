"""Editable knobs for the oracle-discovery environment.

This is the single source of truth the experimenter tweaks most: the (non-semantic)
tool identities, the opaque decoy return string, the per-call token surcharge, and the
default problem-difficulty / budget. Everything here is plain data so it can be edited
without touching the server, host, or sweep machinery.

Design recap (see plan): a model solves a hard problem under a TOKEN budget with `n`
MCP tools. In a real game exactly one tool is an ORACLE that returns the answer; the
rest are decoys. We make ALL n decoys (each returns DECOY_RESULT), run one rollout, and
post-process it into n scored episodes -- one per hypothetical "tool i is the oracle".
For that trick to be valid, every candidate tool must be BEHAVIORALLY IDENTICAL (same
opaque output, no args), so the only thing that varies across oracle-assignments is WHEN
the answer would have appeared.
"""

from __future__ import annotations

# What every decoy tool returns. Opaque + identical across tools (validity requirement).
DECOY_RESULT = "no result"

# Flat per-call token surcharge. NOTE: applied in POST-PROCESSING (the rollout stores a
# token-only meter), so this is a free re-scoring axis -- changing it never requires a
# re-run. This default is the convention the sweep/analysis use when not overridden.
SURCHARGE_TOKENS = 25

# Default number of candidate tools (a primary sweep axis; override per run).
DEFAULT_N = 8

# Default long-division difficulty (dividend digit count; see problems.py).
DEFAULT_DIFFICULTY = 6

# --- tool identities -------------------------------------------------------------------
# Distinguishable but NON-SEMANTIC: distinct names, one shared uninformative description.
# The description must differentiate tools from each other ONLY by name and reveal nothing
# about function. Edit freely. If `n` exceeds len(TOOL_SPECS), the server falls back to
# per-episode obfuscated labels (assign(..., scheme="alnum")); see server.py / mcp_host.py.
SHARED_DESCRIPTION = "An available operation. Calling it may or may not help."

TOOL_SPECS = [
    {"name": "tool_alpha",   "description": SHARED_DESCRIPTION},
    {"name": "tool_bravo",   "description": SHARED_DESCRIPTION},
    {"name": "tool_charlie", "description": SHARED_DESCRIPTION},
    {"name": "tool_delta",   "description": SHARED_DESCRIPTION},
    {"name": "tool_echo",    "description": SHARED_DESCRIPTION},
    {"name": "tool_foxtrot", "description": SHARED_DESCRIPTION},
    {"name": "tool_golf",    "description": SHARED_DESCRIPTION},
    {"name": "tool_hotel",   "description": SHARED_DESCRIPTION},
    {"name": "tool_india",   "description": SHARED_DESCRIPTION},
    {"name": "tool_juliet",  "description": SHARED_DESCRIPTION},
    {"name": "tool_kilo",    "description": SHARED_DESCRIPTION},
    {"name": "tool_lima",    "description": SHARED_DESCRIPTION},
    {"name": "tool_mike",    "description": SHARED_DESCRIPTION},
    {"name": "tool_november","description": SHARED_DESCRIPTION},
    {"name": "tool_oscar",   "description": SHARED_DESCRIPTION},
    {"name": "tool_papa",    "description": SHARED_DESCRIPTION},
    {"name": "tool_quebec",  "description": SHARED_DESCRIPTION},
    {"name": "tool_romeo",   "description": SHARED_DESCRIPTION},
    {"name": "tool_sierra",  "description": SHARED_DESCRIPTION},
    {"name": "tool_tango",   "description": SHARED_DESCRIPTION},
]

# Name of the (real, non-candidate) submission tool. Excluded from the n candidates and
# from counterfactual scoring; calling it ends the episode.
SUBMIT_TOOL = "submit_answer"


def tool_specs_name(i: int) -> str:
    """Default (non-obfuscated) surface name for candidate index i: from TOOL_SPECS,
    falling back to tool_<i> when i exceeds the spec list."""
    return TOOL_SPECS[i]["name"] if i < len(TOOL_SPECS) else f"tool_{i}"
