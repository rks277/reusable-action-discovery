"""Which path is cheaper over the (p, N) plane for woodworld.

The recipe is fixed (axe = 1 stick + 1 wood, i.e. 3 wood + 2 crafts to build).
For each (p, N) cell, color GREEN if E[build] < E[grind] (building the axe is
cheaper in expectation), else GREY. p (gather probability) on the x-axis, N
(wood to win) on the y-axis. Mirrors scripts/plot_build_vs_grind_region.py.

Cost models (y = use yield, read live from woodworld; the axe build cost is
derived from the current recipe table via validate_woodworld.axe_cost):
  E[grind](p, N)  = N / p                               # gather only
  E[build](p, N)  = wood/p + crafts + N/y               # build, then use the axe
Boundary where they are equal: N = (6 + 4p)/(2 - p) for the current recipe.

Usage: PYTHONPATH=. python -m scripts.plot_woodworld_build_region
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np

from scripts.validate_woodworld import axe_cost
from scripts.woodworld import GATHER_PROB, TOOL_ITEM, USE_YIELDS

P_VALUES = np.round(np.arange(0.05, 1.0001, 0.05), 2)   # x-axis: gather prob p
N_VALUES = np.arange(1, 61)                              # y-axis: wood to win N

USE_YIELD = USE_YIELDS[TOOL_ITEM][1]
AXE_WOOD, AXE_CRAFTS = axe_cost()                        # (3, 2) for the current recipe


def grind_cost(p: float, n: int) -> float:
    return n / p


def build_cost(p: float, n: int) -> float:
    return AXE_WOOD / p + AXE_CRAFTS + n / USE_YIELD


def main():
    # grid[i, j] = 1 if build cheaper at (N_VALUES[i], P_VALUES[j]) else 0
    grid = np.array([[1 if build_cost(p, int(n)) < grind_cost(p, int(n)) else 0
                      for p in P_VALUES] for n in N_VALUES])
    diff = np.array([[grind_cost(p, int(n)) - build_cost(p, int(n))
                      for p in P_VALUES] for n in N_VALUES])

    cmap = ListedColormap(["0.75", "#2ca02c"])  # grey, green

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.pcolormesh(P_VALUES, N_VALUES, grid, cmap=cmap, vmin=0, vmax=1,
                  shading="nearest")
    ax.contour(P_VALUES, N_VALUES, diff, levels=[0], colors="black",
               linewidths=1.2, linestyles="--")
    # operating point of the live env
    ax.axvline(GATHER_PROB, color="white", lw=1.0, alpha=0.8)
    ax.text(GATHER_PROB, N_VALUES[-1], f" p={GATHER_PROB}", va="top", ha="left",
            fontsize=8, color="white")

    ax.set_xlabel("gather probability p")
    ax.set_ylabel("wood to win N")
    ax.set_title(f"Cheaper path in expectation  (axe = 1 stick + 1 wood, "
                 f"use yield={USE_YIELD})\n"
                 "green: build < grind     grey: grind ≤ build", fontsize=10)

    fig.tight_layout()
    out = Path("figs/fig_woodworld_build_vs_grind_region.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
