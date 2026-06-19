"""Plot the beach v2 E[brute]=E[build] boundary + cheaper-strategy regime over an
arbitrary (n, m) grid. Data-free -- pure boundary geometry (see
plot_beach_v2_chain.build_brute_boundary for the cost model).

  n* (m) = (k + 1.5) / (rows * (0.5 - ROCK_DENSITY*k)),  k = m/(m+1)

For fixed rows the line is bounded: as m -> inf, n* -> (1+1.5)/(rows*(0.5-ROCK_DENSITY)).
With rows=5, ROCK_DENSITY=0.4 that asymptote is n=5, so on a large grid the brute-
cheaper region is just the bottom strip (n <~ 5) and build is cheaper everywhere above.

Usage: python -m scripts.plot_beach_v2_boundary [--nmax 100] [--mmax 100] [--rows 5]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

from scripts.plot_beach_v2_chain import ROCK_DENSITY, build_brute_boundary


def main():
    argv = sys.argv[1:]

    def opt(name, default):
        return type(default)(argv[argv.index(name) + 1]) if name in argv else default

    nmax = opt("--nmax", 100)
    mmax = opt("--mmax", 100)
    rows = opt("--rows", 5)
    n_lo = m_lo = 1

    nr = list(range(n_lo, nmax + 1))
    mr = list(range(m_lo, mmax + 1))
    grid = np.full((len(nr), len(mr)), np.nan)           # nan = infeasible (m>rocks)
    for i, n in enumerate(nr):
        rocks = round(ROCK_DENSITY * rows * n)           # rocks available at this n
        for j, m in enumerate(mr):
            if m > rocks:                                # can't collect m scraps
                continue                                 # -> infeasible, stays nan
            k = m / (m + 1)
            nstar = (k + 1.5) / (rows * (0.5 - ROCK_DENSITY * k))
            grid[i, j] = 1.0 if n > nstar else 0.0       # 1 = build cheaper
    feasible = ~np.isnan(grid)
    nb = int(np.nansum(grid)); tot = int(feasible.sum())
    asymptote = (1 + 1.5) / (rows * (0.5 - ROCK_DENSITY))

    extent = [m_lo - 0.5, mmax + 0.5, n_lo - 0.5, nmax + 0.5]
    cmap = ListedColormap(["#d1604f", "#4c78a8"]); cmap.set_bad("lightgrey")
    fig, ax = plt.subplots(figsize=(8, 7.6), constrained_layout=True)
    ax.imshow(np.ma.masked_invalid(grid), origin="lower", aspect="auto",
              interpolation="nearest", cmap=cmap, vmin=0, vmax=1, extent=extent,
              alpha=0.9)
    build_brute_boundary(ax, rows, n_lo, nmax, m_lo, mmax, legend=False)
    step_m = max(1, mmax // 10)
    step_n = max(1, nmax // 10)
    ax.set_xticks(range(0, mmax + 1, step_m))
    ax.set_yticks(range(0, nmax + 1, step_n))
    ax.set_xlabel("m = papers to form map", fontsize=10)
    ax.set_ylabel("n = strip width", fontsize=10)
    ax.set_title(f"beach v2 E[brute]=E[build] regime  (rows={rows}, area = n*{rows})\n"
                 f"of {tot} FEASIBLE cells: build cheaper {nb}, brute cheaper {tot-nb}; "
                 f"line ~ n={asymptote:.1f} = A*max/rows (critical AREA, not n)",
                 fontsize=10)
    ax.legend(handles=[
        mpatches.Patch(color="#4c78a8", label="build cheaper (above)"),
        mpatches.Patch(color="#d1604f", label="brute cheaper (below)"),
        mpatches.Patch(color="lightgrey", label="infeasible (m > rocks)"),
        Line2D([0], [0], color="k", ls="--", label="E[brute] = E[build]"),
    ], loc="center right", fontsize=9, framealpha=0.92)

    out = Path("runs") / "beach_v2" / f"fig_beach_v2_boundary_{nmax}x{mmax}_rows{rows}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out} (+ .pdf)  [build-cheaper {nb}/{tot}, line asymptote n={asymptote:.2f}]")


if __name__ == "__main__":
    main()
