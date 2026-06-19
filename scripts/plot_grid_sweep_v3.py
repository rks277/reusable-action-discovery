"""(N, T) heatmaps for the v3 explicit-pickup grid sweep.

The grid sweep (run_grid_sweep_v3.py) runs ONE episode per randomly-sampled
(N, T) cell, so ~131 of the 231 cells are empty. Two render modes:

  raw (default)   -- one coloured cell per sampled point; unsampled cells grey.
  --gauss [SIGMA] -- fill the whole grid by GAUSSIAN POOLING: normalized
      Gaussian-weighted interpolation (Nadaraya-Watson). Each grid cell becomes
      sum_i w_i*v_i / sum_i w_i over sampled episodes i, with Gaussian weights
      w_i = exp(-dist^2 / 2 sigma^2). Implemented as normalized convolution:
      gaussian_filter(value*mask) / gaussian_filter(mask), which both fills gaps
      and smooths the single-rep 0/1 scatter into a propensity surface. SIGMA is
      in grid cells (default 1.2). Sampled-cell markers are overlaid as a '.'.

Renders three panels over N in [0,20] x T in [0,10]:
  built   -- did the agent construct the machine? (0/1)
  solved  -- were all doors opened within budget? (0/1)
  actions -- budgeted actions used (pickups are free, already excluded)

Usage: PYTHONPATH=. python -m scripts.plot_grid_sweep_v3 [runs/<dir>] [--gauss [SIGMA]]
       (run dir defaults to the most recent runs/grid_sweep_v3_* directory)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import scripts.sweep_config as cfg

N_LO, N_HI = 0, 20
T_LO, T_HI = 0, 10


def gaussian_fill(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Fill NaN cells via normalized Gaussian convolution (Nadaraya-Watson on the
    integer grid). Edge handling 'nearest' avoids border roll-off."""
    mask = ~np.isnan(grid)
    vals = np.where(mask, grid, 0.0)
    num = gaussian_filter(vals, sigma=sigma, mode="nearest")
    den = gaussian_filter(mask.astype(float), sigma=sigma, mode="nearest")
    return num / np.maximum(den, 1e-12)


def draw_boundary(ax, t_lo: int):
    """Overlay the E[grind]=E[build] strategy boundary (same definition as
    scripts.plot_haiku_region_heatmap): for each T, the smallest N at which the
    expected build cost drops below the expected grind cost. At/above the line
    building is the cheaper expected strategy; below it, grinding is. Only drawn
    for T>=2, where a recipe pair exists and building is actually possible."""
    ts = np.linspace(max(t_lo, 2), T_HI, 200)
    bnd = [next((n for n in range(1, N_HI + 1)
                 if cfg._build_cost(n, int(round(t))) < cfg._grind_cost(n)), np.nan)
           for t in ts]
    ax.plot(ts, bnd, "w--", lw=2.2)                       # halo for contrast
    ax.plot(ts, bnd, "k--", lw=1.3, label="E[grind]=E[build]")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)


def latest_run() -> Path:
    cands = sorted(p for p in Path("runs").rglob("grid_sweep_v3_*") if p.is_dir())
    if not cands:
        raise SystemExit("no runs/grid_sweep_v3_* directory found")
    return cands[-1]


def load(ep: Path) -> list[dict]:
    return [json.loads(l) for l in ep.read_text().splitlines() if l.strip()]


def cell_value(rows: list[dict], fn) -> np.ndarray:
    """N x T grid (rows=N, cols=T) of fn(row); NaN where no episode sampled."""
    g = np.full((N_HI - N_LO + 1, T_HI - T_LO + 1), np.nan)
    for r in rows:
        if r.get("error"):
            continue
        n, t = r["n"], r["n_types"]
        if N_LO <= n <= N_HI and T_LO <= t <= T_HI:
            g[n - N_LO, t - T_LO] = fn(r)
    return g


def built(r: dict) -> float:
    return 1.0 if any("fuse into" in o for o in r.get("obs", [])) else 0.0


def main():
    argv = sys.argv[1:]
    gauss = "--gauss" in argv
    sigma = 1.2
    if gauss:
        gi = argv.index("--gauss")
        if gi + 1 < len(argv) and not argv[gi + 1].startswith("-"):
            sigma = float(argv[gi + 1])
            argv.pop(gi + 1)
        argv.pop(gi)
    pos = [a for a in argv if not a.startswith("-")]
    run_dir = Path(pos[0]) if pos else latest_run()
    ep = run_dir if run_dir.suffix == ".jsonl" else run_dir / "episodes.jsonl"
    rows = load(ep)
    ok = [r for r in rows if not r.get("error")]
    print(f"Loaded {len(rows)} episodes ({len(rows)-len(ok)} errored) from {ep}")

    panels = [
        ("built tool", cell_value(ok, built), "viridis", (0, 1)),
        ("solved", cell_value(ok, lambda r: 1.0 if r.get("solved") else 0.0),
         "viridis", (0, 1)),
        ("budgeted actions", cell_value(ok, lambda r: r.get("total_actions", 0)),
         "magma", None),
    ]
    # sampled-cell coordinates (for overlay markers in gauss mode)
    sx = [r["n_types"] for r in ok]
    sy = [r["n"] for r in ok]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    # extent maps cell centres to integer (N, T); origin lower so T increases upward
    extent = [T_LO - 0.5, T_HI + 0.5, N_LO - 0.5, N_HI + 0.5]
    for ax, (title, grid, cmap, vlim) in zip(axes, panels):
        cm = plt.get_cmap(cmap).copy()
        if gauss:
            data = gaussian_fill(grid, sigma)        # dense, no NaNs
        else:
            cm.set_bad("lightgrey")                  # unsampled cells stay grey
            data = np.ma.masked_invalid(grid)
        interp = "nearest"                            # always render discrete cells
        vmin, vmax = (vlim if vlim else (np.nanmin(grid), np.nanmax(grid)))
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=cm,
                       extent=extent, vmin=vmin, vmax=vmax, interpolation=interp)
        if gauss:
            ax.scatter(sx, sy, s=8, c="white", edgecolors="black", linewidths=0.3,
                       alpha=0.7)               # mark which cells were actually sampled
        draw_boundary(ax, T_LO)
        ax.set_xlim(T_LO - 0.5, T_HI + 0.5)
        ax.set_ylim(N_LO - 0.5, N_HI + 0.5)
        # empirical mean over SAMPLED cells (one episode/cell), not the pooled grid
        mean = np.nanmean(grid)
        fmt = ".2f" if vlim == (0, 1) else ".1f"
        ax.set_title(f"{title}  (mean {mean:{fmt}})")
        ax.set_xlabel("T (byproduct types)")
        ax.set_ylabel("N (doors)")
        ax.set_xticks(range(T_LO, T_HI + 1))
        ax.set_yticks(range(N_LO, N_HI + 1))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    n_sampled = int(np.sum(~np.isnan(panels[0][1])))
    mode = (f"gaussian-pooled fill, sigma={sigma} cells (dots = sampled)" if gauss
            else "grey = not sampled")
    fig.suptitle(f"toolworld v3 (explicit pickup) grid sweep -- Haiku 4.5 -- "
                 f"{n_sampled} sampled (N,T) cells  ({mode})", fontsize=12)

    stem = "fig_grid_v3_heatmap_gauss" if gauss else "fig_grid_v3_heatmap"
    fig.savefig(run_dir / f"{stem}.png", dpi=130)
    fig.savefig(run_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"  wrote {run_dir / (stem + '.png')}")
    print(f"  wrote {run_dir / (stem + '.pdf')}")


if __name__ == "__main__":
    main()
