"""Shared configuration for the budget sweeps (run_budget_sweep*.py).

The world/budget/rep knobs live here so the headline Anthropic run and the
reasoning-model extension stay in lockstep: they must use the SAME per-rep
worlds (rep r => relabel_seed = drop_seed = r) for their episodes to be
paired-comparable and concatenable into one dataset.

BUDGET is DERIVED from N via the coupon-collector (grind) expectation, so
changing N here updates the budget everywhere. MAX_TURNS must stay strictly
above BUDGET, or the turn cap -- not the budget -- becomes the binding
constraint and the "build vs. grind" decision stops being economic.
"""

from __future__ import annotations

from math import comb

import lomekwi.obfuscation as _obfuscation

# --- obfuscation scheme ------------------------------------------------
# "tokens" = pronounceable nonsense tokens (default). "letter" = a single
# random letter per element. "alnum" = a random alphanumeric string (length
# 4-8) per element. Flipping this here propagates to every sweep that imports
# sweep_config (assign() reads the module default).
OBFUSCATION_SCHEME = "letter"
_obfuscation.DEFAULT_SCHEME = OBFUSCATION_SCHEME

# --- world / budget / reps (shared across all budget sweeps) -----------
N = 8              # locked doors per episode
T = 3              # distinct byproduct types (recipe-search hardness)
REPS = 10           # paired reps; rep r => relabel_seed = drop_seed = r
HINT = True        # subtle "feels active alongside a different kind" hint
MAX_TURNS = 300    # hard turn cap (keep > BUDGET so the budget binds first)
BUDGET_MULT = 1.2  # budget as a multiple of E[grind] (coupon-collector + n)


def _Hn(n: int) -> float:
    return sum(1.0 / k for k in range(1, n + 1))


def _grind_cost(n: int) -> float:
    """E[actions] to brute-force open all n doors: coupon-collector over keys
    (n*H_n) plus the n opens themselves."""
    return n * _Hn(n) + n


def _build_cost(n: int, t: int) -> float:
    """E[actions] for the build path: collect all T byproduct types (G), search
    the recipe (R), then exploit the machine over the doors.

    G = t*H_t examines (coupon-collector over types); R = (C(t,2)+1)/2 combines
    (one of C(t,2) candidate pairs is correct, searched uniformly).

    Key-aware exploit: each of the G examines ALSO drops a uniformly random
    door-key, so by the time the machine is built we already hold the keys to
    ~ n*(1 - (1-1/n)^G) DISTINCT doors. Those doors open in 1 action (use the
    held key); the rest cost 2 (operate machine -> key, then use). So the exploit
    is 2n minus one saved operate-machine action per already-keyed door."""
    g = t * _Hn(t)                              # E[G]: examines to collect T types
    r = (comb(t, 2) + 1) / 2                    # E[R]: recipe-search combines
    keyed = n * (1.0 - (1.0 - 1.0 / n) ** g)    # E[distinct doors keyed en route]
    exploit = 2 * n - keyed                     # 2/door, minus saved operate per keyed door
    return g + r + exploit


def budget_for(n: int = N) -> int:
    """Strict action budget = BUDGET_MULT x E[grind] (grind-calibrated). Brute
    force is always a viable escape hatch, so building stays OPTIONAL -- feasible
    only where it is genuinely cheaper than grinding (E[build] < E[grind]), which
    is exactly what makes built_tool a clean disposition signal rather than a
    feasibility outcome. _build_cost is retained for the region/boundary analyses."""
    return round(BUDGET_MULT * _grind_cost(n))


BUDGET = budget_for(N)  # ~36 at N=8

# --- model rosters + per-provider concurrency --------------------------
# Each sweep imports the roster it runs; rosters share the per-rep worlds, so
# their episodes.jsonl files concatenate into one paired dataset.
ANTHROPIC_MODELS = [
    #("anthropic", "claude-haiku-4-5-20251001"),
    ("anthropic", "claude-sonnet-4-6"),
    #("anthropic", "claude-opus-4-8"),
    #("anthropic", "claude-fable-5"),
]
NEWMODELS = [
    ("openai", "gpt-5"),
    ("google", "gemini-2.5-pro"),
]
# Open-weight Qwen2.5 ladder (known param counts) served locally via an
# OpenAI-compatible endpoint (Ollama on a GPU box, reached over an SSH tunnel).
# Ordered by parameter size: 0.5B < 1.5B < 3B < 7B -- a within-family capability
# axis below Haiku. The ":" in each tag routes them to the ollama backend.
# (Small hosted Gemma was dropped: the Gemini API only serves the 26B/31B Gemma 4
#  models, not the 1B/4B/12B sizes -- those are open-weights-only.)
OSS_MODELS = [
    ("ollama", "qwen2.5:1.5b"),
    ("ollama", "qwen2.5:3b"),
    ("ollama", "qwen2.5:7b"),
]

# Open-weight Gemma 4 ladder, also via Ollama on the box -- a SECOND family
# spanning the same below-Haiku range, so we can ask whether the build-vs-grind
# disposition tracks parameter count or is family-specific. Effective params:
# e2b ~2.3B < e4b ~4.5B < 12b. NB e2b/e4b are the edge (3n-lineage) architecture
# while 12b is dense -- a clean effective-param ladder, not a single-architecture
# scaling curve. The ":" in each tag routes them to the ollama backend.
GEMMA_MODELS = [
    ("ollama", "gemma4:e2b"),
    ("ollama", "gemma4:e4b"),
    ("ollama", "gemma4:12b"),
]

# Per-provider in-flight episode cap (bounds rate-limit / token bursts). A
# single shared dict is fine -- each sweep only indexes the providers it runs.
CONCURRENCY = {"anthropic": 6, "openai": 4, "google": 4, "ollama": 2}
