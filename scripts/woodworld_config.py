"""Shared configuration for the woodworld sweeps (mirrors scripts/sweep_config.py).

Separate from sweep_config because woodworld's cost model (gather-rate / recipe
build) differs from the doors/keys coupon-collector, and we don't want to
cross-contaminate the toolworld budget. The editable MECHANICS (recipes, gather
prob, use yields) live in scripts/woodworld.py as the single source of truth; the
cost model / budget live in scripts/validate_woodworld.py. This file only holds
the sweep grid + obfuscation scheme + model rosters.

rep r => relabel_seed = gather_seed = r, shared across models, so each rep is the
same world (paired comparison). N_VALUES straddles N* (~13) so built_axe carries
calibration signal rather than being flat (the toolworld-v1 failure mode).
"""

from __future__ import annotations

import lomekwi.obfuscation as _obfuscation
from scripts.sweep_config import ANTHROPIC_MODELS, CONCURRENCY, NEWMODELS, OSS_MODELS  # noqa: F401
from scripts.validate_woodworld import budget_for, n_star  # noqa: F401

# --- obfuscation scheme ------------------------------------------------
# 3 latent items -> "letter" gives clean single-letter labels (a/b/c). Flipping
# this propagates to every woodworld run (assign() reads the module default).
OBFUSCATION_SCHEME = "letter"
_obfuscation.DEFAULT_SCHEME = OBFUSCATION_SCHEME

# --- sweep grid --------------------------------------------------------
N_VALUES = [8, 12, 16, 20, 30]   # straddle N* (~13); built_axe should rise across this
REPS = 10                         # paired reps; rep r => relabel_seed = gather_seed = r
HINT = True                       # subtle "items transform / tools persist" nudge
MAX_TURNS = 300                   # keep > max budget so the budget binds first
