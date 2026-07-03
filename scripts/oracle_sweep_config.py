"""Shared knobs for the oracle-discovery sweep (mirrors scripts/sweep_config.py).

Models roster, per-provider concurrency, and a budget calibration helper. The budget
should sit where grinding the manual long-division solve is often OVER budget, so an
early oracle hit is the cheaper path -- that's the regime where win_i is an interesting
function of how soon tool i is called.

`budget_for` currently uses a coarse heuristic scaling with difficulty. Replace
`MANUAL_COST` with a measured pilot table (run each model with tools off, record the
token meter at correct submission) for principled calibration; see the plan.
"""

from __future__ import annotations

# Anthropic trio. Edit freely.
MODELS = [
    ("anthropic", "claude-haiku-4-5-20251001"),
    ("anthropic", "claude-sonnet-4-6"),
    ("anthropic", "claude-opus-4-8"),
]

# Qwen3.5 models via Ollama (or vLLM if VLLM_BASE_URL is set -- preferred for tool calling).
# Pull first:
#   ollama pull qwen3.5:2b && ollama pull qwen3.5:4b && ollama pull qwen3.5:9b
#   ollama pull qwen3.5:27b && ollama pull qwen3.5:35b
# VRAM: 2b~2.7GB, 4b~3.4GB, 9b~6.6GB, 27b~17GB, 35b~24GB -- all fit A100 40GB individually.
# vLLM HF model IDs for the A100 40GB job (run_qwen35_oracle_job.sh serves each
# model on port 8000 one at a time; VLLM_BASE_URL routes raw_chat here automatically).
# 27B uses GPTQ-Int4 because BF16 27B needs ~54GB > 40GB A100.
QWEN3_MODELS = [
    ("vllm", "Qwen/Qwen3.5-2B"),
    ("vllm", "Qwen/Qwen3.5-4B"),
    ("vllm", "Qwen/Qwen3.5-9B"),
    ("vllm", "Qwen/Qwen3.5-27B"),   # full BF16 (~54GB) -- fits on H100 80GB
]

CONCURRENCY = {"anthropic": 6, "openai": 4, "google": 4, "ollama": 2, "vllm": 32}

# Budget headroom over a correct manual solve. Manual long division turns out CHEAP
# (~2k tokens even at 20 digits -- see oracle_pilot.py findings), so budget mainly bounds
# how many tools the model can try, not "grind cost". ~2x gives room to attempt manual
# AND probe a handful of tools without making "try all n" free.
# Per-sequential-tool-call token overhead (measured from probe: growing context makes
# each call ~1600 tok for Haiku, ~1060 for Opus; use a conservative shared estimate).
PER_CALL_TOKENS = 1800

# Measured manual-solve token cost vs difficulty (median S among CORRECT solves in the
# pilot, tools off; roughly linear):
#   d8 ~ 950, d11 ~ 1250, d14 ~ 1450, d17 ~ 1900, d20 ~ 1950.
def _manual_cost(difficulty: int) -> int:
    return int(round(900 + 90 * max(0, difficulty - 8)))


def budget_for(difficulty: int, n: int = 8) -> int:
    """Budget covers searching all n tools sequentially PLUS a manual solve attempt.
    This makes the economics real: searching costs tokens, but so does not finding
    the oracle and having to solve manually anyway."""
    return _manual_cost(difficulty) + n * PER_CALL_TOKENS


# Per-model manual-reliability CLIFFS from the pilot (largest difficulty still solved
# reliably tools-off, and where accuracy starts dropping). The interesting tool regime is
# at/above each model's cliff, where manual is unreliable so hedging via tools can win.
#   haiku:  reliable <= 8;  cliff ~11;  hopeless >= 14
#   sonnet: reliable <= 17; cliff ~20
#   opus:   reliable <= 14; cliff ~17-20
CLIFF_NOTE = "haiku~11, opus~17-20, sonnet~20"
