"""Build-chain conditional-probability heatmaps for a beach v2 strip sweep, in the
style of plot_grid_sweep_v3_chain (fig_grid_v3_buildchain).

Three Gaussian-pooled (n x m) panels, x = papers m, y = width n:
  1. P(picked up all map pieces)         -- collected all m scraps. In beach,
     collecting all m scraps IS what forms the map, so this equals the map-build
     rate.
  2. P(built map | picked up all pieces) -- DEGENERATE: collecting all pieces is
     forming the map, so this is a solid 1.0 wherever defined. Kept for parity
     with the toolworld build chain (its analog, P(built | has both shards), is
     NOT degenerate because holding both shards still requires the combine).
  3. P(won | built map)                  -- among episodes that built the map, did
     the treasure get dug up? (How well the tool, once built, is exploited.)

Conditional panels are NaN (excluded from the Gaussian pooling) where the
condition isn't met; white dots mark cells where the metric is defined. One
figure per model. (No E[grind]=E[build] boundary line -- that's toolworld-specific.)

Usage: python -m scripts.plot_beach_v2_chain runs/beach_v2/<run>/episodes.jsonl [--sigma S]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plot_beach_v2_grid import gaussian_fill, short

ROCK_DENSITY = 0.4   # matches the sweep calibration: rocks = ROCK_DENSITY * area


def _astar(m):
    """Critical AREA where E[brute] == E[build]: A* = (k+1.5)/(0.5-ROCK_DENSITY*k),
    k = m/(m+1). Build is cheaper above this area, brute below."""
    k = m / (m + 1.0)
    return (k + 1.5) / (0.5 - ROCK_DENSITY * k)


def nstar_of_m(m, rows=None, square=False):
    """Boundary width n* for a given m: n* = sqrt(A*) for square (area=n^2), else
    A*/rows for an n x rows strip (area=n*rows)."""
    import numpy as np
    A = _astar(m)
    return np.sqrt(A) if square else A / rows


def build_brute_boundary(ax, rows, n_lo, n_hi, m_lo, m_hi, legend=False, square=False):
    """Curve where unconstrained E[brute-force] == E[build the map], in actions.

      E[brute] = (area+1)/2           -- dig cells until you hit the treasure
      E[build] = (R+1)*m/(m+1) + 2    -- inspect rocks to collect all m scraps,
                                         then read map + 1 dig;  R = ROCK_DENSITY*area

    The crossover is a critical AREA A*(m); converting to width n: n* = sqrt(A*)
    for a SQUARE grid (area=n^2), or A*/rows for an n x rows strip. ABOVE the line
    (larger n) building is the cheaper expected strategy; below it, brute force is.
    Unconstrained crossover -- ignores the durability/budget caps."""
    import numpy as np
    ms = np.linspace(m_lo, m_hi, 200)
    nstar = nstar_of_m(ms, rows=rows, square=square)
    nstar = np.clip(nstar, n_lo - 0.5, n_hi + 0.5)
    ax.plot(ms, nstar, "w-", lw=2.6)                       # halo for contrast
    ax.plot(ms, nstar, "k--", lw=1.4, label="E[brute]=E[build]")
    if legend:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)


def regime_map(run_dir, n_lo, n_hi, m_lo, m_hi, rows, sampled, square=False):
    """A per-cell map of which expected strategy is cheaper: blue = build cheaper
    (cell ABOVE the E[brute]=E[build] line, n > n*(m)), red = brute cheaper (BELOW).
    The continuous boundary is overlaid; white dots mark cells actually sampled."""
    import numpy as np
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    from matplotlib.colors import ListedColormap

    nrange = list(range(n_lo, n_hi + 1))
    mrange = list(range(m_lo, m_hi + 1))
    grid = np.zeros((len(nrange), len(mrange)))
    for i, n in enumerate(nrange):
        for j, m in enumerate(mrange):
            grid[i, j] = 1.0 if n > nstar_of_m(m, rows=rows, square=square) else 0.0
    nb = int(grid.sum()); tot = grid.size
    extent = [m_lo - 0.5, m_hi + 0.5, n_lo - 0.5, n_hi + 0.5]
    fig, ax = plt.subplots(figsize=(7.2, 6.6), constrained_layout=True)
    ax.imshow(grid, origin="lower", aspect="auto", interpolation="nearest",
              cmap=ListedColormap(["#d1604f", "#4c78a8"]), vmin=0, vmax=1,
              extent=extent, alpha=0.85)
    build_brute_boundary(ax, rows, n_lo, n_hi, m_lo, m_hi, legend=False, square=square)
    if sampled:
        ax.scatter([m for _, m in sampled], [n for n, _ in sampled], s=10,
                   c="white", edgecolors="black", linewidths=0.3, alpha=0.7)
    ax.set_xticks(range(m_lo, m_hi + 1))
    ax.set_yticks(range(n_lo, n_hi + 1))
    ax.set_xlabel("m = papers to form map", fontsize=10)
    ax.set_ylabel("n = grid width", fontsize=10)
    geom = "area = n^2" if square else f"area = n*{rows}"
    ax.set_title(f"Cheaper expected strategy per cell  ({geom})", fontsize=11)
    ax.legend(handles=[
        mpatches.Patch(color="#4c78a8", label=f"build cheaper (above): {nb} cells"),
        mpatches.Patch(color="#d1604f", label=f"brute cheaper (below): {tot - nb} cells"),
        Line2D([0], [0], color="k", ls="--", label="E[brute] = E[build]"),
    ], loc="lower right", fontsize=8, framealpha=0.92)
    out = run_dir / "fig_beach_v2_regime.png"
    fig.savefig(out, dpi=140); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out} (+ .pdf)  [build-cheaper cells: {nb}/{tot}]")


def _grid(mrows, fn, n_lo, n_hi, m_lo, m_hi):
    """n x m array of fn(row); NaN where fn returns None or no episode."""
    g = np.full((n_hi - n_lo + 1, m_hi - m_lo + 1), np.nan)
    for r in mrows:
        v = fn(r)
        if v is not None:
            g[r["grid_size"] - n_lo, r["papers_needed"] - m_lo] = float(v)
    return g


def _panel(ax, fig, gr, sigma, title, n_lo, n_hi, m_lo, m_hi, rows=None,
           legend=False, square=False):
    filled = gaussian_fill(gr, sigma)
    extent = [m_lo - 0.5, m_hi + 0.5, n_lo - 0.5, n_hi + 0.5]
    im = ax.imshow(filled, origin="lower", aspect="auto", cmap="viridis",
                   extent=extent, vmin=0, vmax=1, interpolation="nearest")
    ys, xs = np.where(~np.isnan(gr))                  # cells where metric defined
    ax.scatter(xs + m_lo, ys + n_lo, s=8, c="white", edgecolors="black",
               linewidths=0.3, alpha=0.7)
    if square or rows:
        build_brute_boundary(ax, rows, n_lo, n_hi, m_lo, m_hi, legend=legend,
                             square=square)
    m = np.nanmean(gr) if np.any(~np.isnan(gr)) else float("nan")
    ax.set_title(f"{title}  (mean {m:.2f})", fontsize=10.5)
    ax.set_xlabel("m = papers to form map", fontsize=9)
    ax.set_ylabel("n = grid width", fontsize=9)
    ax.set_xticks(range(m_lo, m_hi + 1))
    ax.set_yticks(range(n_lo, n_hi + 1, max(1, (n_hi - n_lo) // 10)))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _chain(model, mrows, sigma, n_lo, n_hi, m_lo, m_hi, rows_h, run_dir,
           square=False):
    # built_map == picked up all m scraps (collecting them all forms the map)
    g1 = _grid(mrows, lambda r: 1.0 if r["built_map"] else 0.0,
               n_lo, n_hi, m_lo, m_hi)
    g2 = _grid(mrows, lambda r: (None if not r["built_map"] else 1.0),
               n_lo, n_hi, m_lo, m_hi)          # degenerate: solid 1.0 where defined
    g3 = _grid(mrows, lambda r: (None if not r["built_map"]
                                 else (1.0 if r["won"] else 0.0)),
               n_lo, n_hi, m_lo, m_hi)
    g4 = _grid(mrows, lambda r: 1.0 if r["won"] else 0.0,   # unconditional success
               n_lo, n_hi, m_lo, m_hi)
    rh = rows_h[0] if len(rows_h) == 1 else rows_h
    bnd_rows = None if square else (rh if isinstance(rh, int) else None)
    geom = "n x n" if square else f"ROWS={rh}"
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.2), constrained_layout=True)
    _panel(axes[0], fig, g1, sigma, "P(picked up all map pieces)",
           n_lo, n_hi, m_lo, m_hi, rows=bnd_rows, square=square, legend=True)
    _panel(axes[1], fig, g2, sigma, "P(built map | picked up all pieces)",
           n_lo, n_hi, m_lo, m_hi, rows=bnd_rows, square=square)
    _panel(axes[2], fig, g3, sigma, "P(won | built map)",
           n_lo, n_hi, m_lo, m_hi, rows=bnd_rows, square=square)
    _panel(axes[3], fig, g4, sigma, "P(won) -- total success rate",
           n_lo, n_hi, m_lo, m_hi, rows=bnd_rows, square=square)
    fig.suptitle(f"beach v2 build chain -- {short(model)} ({geom}) -- "
                 f"gaussian-pooled (sigma={sigma}; dots = cells with the condition "
                 f"met)", fontsize=12)
    stem = f"fig_beach_v2_buildchain_{short(model)}"
    fig.savefig(run_dir / f"{stem}.png", dpi=130)
    fig.savefig(run_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"  wrote {run_dir / (stem + '.png')} (+ .pdf)")


def main():
    argv = sys.argv[1:]
    sigma = 1.2
    if "--sigma" in argv:
        i = argv.index("--sigma"); sigma = float(argv[i + 1]); del argv[i:i + 2]
    pos = [a for a in argv if not a.startswith("-")]
    if not pos:
        raise SystemExit("usage: python -m scripts.plot_beach_v2_chain "
                         "runs/beach_v2/<run>/episodes.jsonl [--sigma S]")
    path = Path(pos[0])
    run_dir = path.parent if path.suffix == ".jsonl" else path
    ep = path if path.suffix == ".jsonl" else path / "episodes.jsonl"
    rows = [json.loads(l) for l in ep.read_text().splitlines() if l.strip()]
    ok = [r for r in rows if not r.get("error")]
    ns = [r["grid_size"] for r in ok]
    ms = [r["papers_needed"] for r in ok]
    n_lo, n_hi, m_lo, m_hi = min(ns), max(ns), min(ms), max(ms)
    rows_h = sorted({r.get("rows") for r in ok})
    square = all(r.get("rows") == r["grid_size"] for r in ok)   # n x n grid?
    by_model = defaultdict(list)
    for r in ok:
        by_model[r["model"]].append(r)
    print(f"Loaded {len(ok)} episodes from {ep} ({len(by_model)} model(s)) "
          f"-- geometry: {'square (n x n)' if square else f'rows={rows_h}'}")
    for model, mrows in by_model.items():
        _chain(model, mrows, sigma, n_lo, n_hi, m_lo, m_hi, rows_h, run_dir,
               square=square)
    # model-independent regime map (boundary geometry vs cells)
    rh = None if square else (rows_h[0] if len(rows_h) == 1 else None)
    if square or isinstance(rh, int):
        sampled = {(r["grid_size"], r["papers_needed"]) for r in ok}
        regime_map(run_dir, n_lo, n_hi, m_lo, m_hi, rh, sampled, square=square)


if __name__ == "__main__":
    main()
