"""Grind-calibrated budget B vs. number of doors N, with the build-path cost
overlaid for a few byproduct-type counts T.

B(N) = BUDGET_MULT * E[grind] is T-independent (sweep_config.budget_for): brute
force is the escape hatch the budget is sized to, so building is OPTIONAL. The
dashed curves are BUDGET_MULT * E[build] for each T (key-aware build path). Where
a build curve dips BELOW the solid budget line, building is genuinely cheaper
than grinding -- the region where electing to build is the smart, sub-budget
move and built_tool carries clean disposition signal.

Usage: PYTHONPATH=. python -m scripts.plot_budget_vs_n
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.sweep_config as cfg

N_VALUES = list(range(2, 21))
T_VALUES = [3, 6, 9]


def main():
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    # the actual (grind-calibrated, T-independent) budget the sweeps use
    budget = [cfg.budget_for(n) for n in N_VALUES]
    ax.plot(N_VALUES, budget, "-o", color="k", ms=3.5, lw=2.0,
            label="budget B (grind-calibrated)", zorder=3)

    # build-path cost at the same multiplier, one dashed line per T
    for t in T_VALUES:
        build = [cfg.BUDGET_MULT * cfg._build_cost(n, t) for n in N_VALUES]
        ax.plot(N_VALUES, build, "--", lw=1.5, label=f"build cost, T={t}")

    ax.set_xlabel("number of doors N")
    ax.set_ylabel("expected actions x BUDGET_MULT")
    ax.set_title(f"Grind-calibrated budget vs build cost "
                 f"(BUDGET_MULT={cfg.BUDGET_MULT})")
    ax.set_xticks(N_VALUES[::2])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = Path("figs/fig_budget_vs_n.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
