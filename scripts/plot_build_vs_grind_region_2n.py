"""Which path is cheaper, over the (T, N) plane -- naive exploit cost variant.

Same as plot_build_vs_grind_region.py, but the build path's post-machine exploit
cost is the naive 2n (2 actions/door: operate machine -> key, then use key),
ignoring the free keys picked up as a byproduct while collecting the T shard
types en route to the recipe. Compare against the key-aware version to see how
much of the green (build-cheaper) region depends on crediting those free keys.

Usage: PYTHONPATH=. python -m scripts.plot_build_vs_grind_region_2n
"""

from __future__ import annotations

from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

import numpy as np

import scripts.sweep_config as cfg

N_VALUES = np.arange(1, 31)      # y-axis
T_VALUES = np.arange(2, 16)      # x-axis


def _build_cost_2n(n: int, t: int) -> float:
    """E[actions] for the build path with a naive (non-key-aware) exploit: 2
    actions/door regardless of keys picked up during collection."""
    g = t * cfg._Hn(t)                    # E[G]: examines to collect T types
    r = (comb(t, 2) + 1) / 2              # E[R]: recipe-search combines
    exploit = 2 * n                       # naive: 2/door, no free-key credit
    return g + r + exploit


def main():
    grid = np.array([
        [1 if _build_cost_2n(int(n), int(t)) < cfg._grind_cost(int(n)) else 0
         for t in T_VALUES]
        for n in N_VALUES
    ])

    cmap = ListedColormap(["0.75", "#2ca02c"])  # grey, green

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.pcolormesh(T_VALUES, N_VALUES, grid, cmap=cmap, vmin=0, vmax=1,
                  shading="nearest")

    ax.set_xlabel("shard types T")
    ax.set_ylabel("number of doors n")
    ax.set_xticks(T_VALUES[::2])

    # gridlines around every (T, N) cell
    ax.set_xticks(T_VALUES - 0.5, minor=True)
    ax.set_yticks(N_VALUES - 0.5, minor=True)
    ax.grid(which="minor", color="black", linewidth=0.4, alpha=0.4)
    ax.tick_params(which="minor", length=0)

    # highlight the T in {2,3,4} x N in {9,10,11} square (9 cells)
    ax.add_patch(Rectangle((1.5, 8.5), 3, 3, fill=False, edgecolor="red",
                            linewidth=2.5, zorder=5))

    fig.tight_layout()
    out = Path("figs/paper/fig_build_vs_grind_region_2n.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
