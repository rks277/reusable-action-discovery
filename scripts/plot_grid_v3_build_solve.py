"""Build (top) and solve (bottom) rate over the (N, T) surface for the 3 models
on the toolworld v3 explicit-pickup grid sweep -- one Gaussian-pooled heatmap per
(metric, model), in a 2x3 grid (rows = build / solve, columns = Haiku / Sonnet /
Opus).

The grid sweep (run_grid_sweep_v3.py) runs ONE episode per randomly-sampled
(N, T) cell, so the raw grid is a sparse 0/1 scatter. We fill + smooth it the
same way as plot_grid_sweep_v3 --gauss: normalized Gaussian convolution
(Nadaraya-Watson), value = sum_i w_i v_i / sum_i w_i with w_i = exp(-d^2/2σ^2),
σ in grid cells (default 1.2). Sampled cells are overlaid as dots and the
E[grind]=E[build] strategy boundary is drawn on every panel.

  build  = machine constructed ("fuse into" appears in the obs)
  solve  = all doors opened within budget (the logged `solved` flag)

Output: runs/grid_v3_summary/.

Usage: python -m scripts.plot_grid_v3_build_solve [--gauss SIGMA]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.sweep_config as cfg
from scripts.plot_grid_sweep_v3 import (
    N_LO, N_HI, T_LO, T_HI, gaussian_fill, cell_value, built, load,
)

MODELS = [("haiku", "Haiku"), ("sonnet", "Sonnet"), ("opus", "Opus")]
METRICS = [
    ("build rate", built),
    ("solve rate", lambda r: 1.0 if r.get("solved") else 0.0),
]


def latest_run(tag: str) -> Path:
    cands = sorted(p for p in Path("runs").glob(f"grid_sweep_v3_{tag}_*") if p.is_dir())
    if not cands:
        raise SystemExit(f"no runs/grid_sweep_v3_{tag}_* directory found")
    return cands[-1]


def boundary(ax, legend: bool = False):
    """E[grind]=E[build] strategy boundary (same definition as plot_grid_sweep_v3):
    smallest N at which building is the cheaper expected strategy, per T>=2."""
    ts = np.linspace(2, T_HI, 200)
    bnd = [next((n for n in range(1, N_HI + 1)
                 if cfg._build_cost(n, int(round(t))) < cfg._grind_cost(n)), np.nan)
           for t in ts]
    ax.plot(ts, bnd, "w--", lw=2.2)
    ax.plot(ts, bnd, "k--", lw=1.3, label="E[grind]=E[build]")
    if legend:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)


def main():
    argv = sys.argv[1:]
    sigma = 1.2
    if "--gauss" in argv:
        gi = argv.index("--gauss")
        if gi + 1 < len(argv) and not argv[gi + 1].startswith("-"):
            sigma = float(argv[gi + 1])

    # load each model's episodes once
    data = {}
    for tag, label in MODELS:
        run = latest_run(tag)
        rows = load(run / "episodes.jsonl")
        data[tag] = rows
        print(f"{label:7s} {run.name}: n={len(rows)}  "
              f"build={np.mean([built(r) for r in rows]):.2f}  "
              f"solve={np.mean([1.0 if r.get('solved') else 0.0 for r in rows]):.2f}")

    extent = [T_LO - 0.5, T_HI + 0.5, N_LO - 0.5, N_HI + 0.5]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), constrained_layout=True,
                             sharex=True, sharey=True)
    im = None
    for ri, (mlabel, fn) in enumerate(METRICS):
        for ci, (tag, label) in enumerate(MODELS):
            ax = axes[ri][ci]
            rows = data[tag]
            grid = cell_value(rows, fn)              # N x T, NaN where unsampled
            filled = gaussian_fill(grid, sigma)       # pooled + interpolated
            im = ax.imshow(filled, origin="lower", aspect="auto", cmap="viridis",
                           extent=extent, vmin=0, vmax=1, interpolation="nearest")
            ax.scatter([r["n_types"] for r in rows], [r["n"] for r in rows], s=7,
                       c="white", edgecolors="black", linewidths=0.3, alpha=0.6)
            boundary(ax, legend=(ri == 0 and ci == 0))
            ax.set_xlim(T_LO - 0.5, T_HI + 0.5)
            ax.set_ylim(N_LO - 0.5, N_HI + 0.5)
            ax.set_xticks(range(T_LO, T_HI + 1))
            ax.set_yticks(range(N_LO, N_HI + 1, 2))
            mean = float(np.nanmean(grid))            # empirical mean over sampled cells
            if ri == 0:
                ax.set_title(f"{label}   (build {mean:.2f})", fontsize=11)
            else:
                ax.set_title(f"solve {mean:.2f}", fontsize=9, loc="left")
            if ri == 1:
                ax.set_xlabel("T (byproduct types)", fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"{mlabel}\nN (doors)", fontsize=10)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("rate", fontsize=10)
    fig.suptitle("toolworld v3 grid sweep — build (top) & solve (bottom) over (N, T), "
                 f"Gaussian-pooled (σ={sigma})", fontsize=13)

    out_dir = Path("runs") / "grid_v3_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "fig_grid_v3_build_solve_heatmap.png"
    fig.savefig(out, dpi=140)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"\n  wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
