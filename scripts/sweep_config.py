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

# --- world / budget / reps (shared across all budget sweeps) -----------
N = 20              # locked doors per episode
T = 3              # distinct byproduct types (recipe-search hardness)
REPS = 1           # paired reps; rep r => relabel_seed = drop_seed = r
HINT = True        # subtle "feels active alongside a different kind" hint
MAX_TURNS = 300    # hard turn cap (keep > BUDGET so the budget binds first)
BUDGET_MULT = 1.2  # budget as a multiple of E[grind] (coupon-collector + n)


def _Hn(n: int) -> float:
    return sum(1.0 / k for k in range(1, n + 1))


def budget_for(n: int = N) -> int:
    """Strict action budget ~= BUDGET_MULT x E[grind] = BUDGET_MULT*(n*H_n + n)."""
    return round(BUDGET_MULT * (n * _Hn(n) + n))


BUDGET = budget_for(N)  # ~36 at N=8

# --- model rosters + per-provider concurrency --------------------------
# Each sweep imports the roster it runs; rosters share the per-rep worlds, so
# their episodes.jsonl files concatenate into one paired dataset.
ANTHROPIC_MODELS = [
    #("anthropic", "claude-haiku-4-5-20251001"),
    #("anthropic", "claude-sonnet-4-6"),
    #("anthropic", "claude-opus-4-8"),
    ("anthropic", "claude-fable-5"),
]
NEWMODELS = [
    ("openai", "gpt-5"),
    ("google", "gemini-2.5-pro"),
]

# Per-provider in-flight episode cap (bounds rate-limit / token bursts). A
# single shared dict is fine -- each sweep only indexes the providers it runs.
CONCURRENCY = {"anthropic": 6, "openai": 4, "google": 4}
