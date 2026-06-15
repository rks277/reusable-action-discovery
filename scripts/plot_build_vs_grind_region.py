"""Which path is cheaper, over the (T, N) plane.

For each (T, N) cell, color GREEN if E[build] < E[grind] (building is the cheaper
strategy in expectation), else GREY. T on the x-axis (2..20), N on the y-axis
(1..100). Uses the live cost models from sweep_config (key-aware build path).

Usage: PYTHONPATH=. python -m scripts.plot_build_vs_grind_region
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np

import scripts.sweep_config as cfg

N_VALUES = np.arange(1, 101)     # y-axis
T_VALUES = np.arange(2, 21)      # x-axis


def main():
    # grid[i, j] = 1 if build cheaper at (N_VALUES[i], T_VALUES[j]) else 0
    grid = np.array([
        [1 if cfg._build_cost(int(n), int(t)) < cfg._grind_cost(int(n)) else 0
         for t in T_VALUES]
        for n in N_VALUES
    ])

    cmap = ListedColormap(["0.75", "#2ca02c"])  # grey, green

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.pcolormesh(T_VALUES, N_VALUES, grid, cmap=cmap, vmin=0, vmax=1,
                  shading="nearest")

    ax.set_xlabel("byproduct types T")
    ax.set_ylabel("number of doors N")
    ax.set_title("Cheaper path in expectation\n(green: build < grind, grey: grind ≤ build)")
    ax.set_xticks(T_VALUES[::2])

    fig.tight_layout()
    out = Path("figs/fig_build_vs_grind_region.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
