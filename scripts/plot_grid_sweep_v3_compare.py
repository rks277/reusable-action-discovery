"""3x3 cross-model comparison grid for the v3 build chain.

Rows = the three build-chain conditionals; columns = models (Haiku/Sonnet/Opus).
Each cell is the gaussian-pooled, cell-by-cell (N, T) heatmap (T>=2) with the
E[grind]=E[build] boundary overlaid and the empirical mean annotated. Shared
0..1 colour scale so cells are directly comparable across the whole grid.

Usage: PYTHONPATH=. python -m scripts.plot_grid_sweep_v3_compare \
           [haiku_dir sonnet_dir opus_dir]
       (defaults to the latest grid_sweep_v3_{haiku,sonnet,opus}_* run dirs)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plot_grid_sweep_v3 import load, gaussian_fill, draw_boundary
from scripts.plot_grid_sweep_v3_chain import episode_facts, grid, N_LO, N_HI, T_HI

SIGMA = 1.2
METRICS = [
    ("P(picked up both shards)", lambda f: 1.0 if f["has_both"] else 0.0),
    ("P(built | both shards)",
     lambda f: None if not f["has_both"] else (1.0 if f["built"] else 0.0)),
    ("P(tool used | built)",
     lambda f: None if not f["built"] else (1.0 if f["used_after"] else 0.0)),
    ("P(won | built)",
     lambda f: None if not f["built"] else (1.0 if f["solved"] else 0.0)),
]


def latest(tag: str) -> Path:
    cands = sorted(p for p in Path("runs").glob(f"grid_sweep_v3_{tag}_*") if p.is_dir())
    if not cands:
        raise SystemExit(f"no runs/grid_sweep_v3_{tag}_* dir found")
    return cands[-1]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    dirs = [Path(a) for a in args] if len(args) == 3 else \
        [latest("haiku"), latest("sonnet"), latest("opus")]
    models = []
    for d in dirs:
        ep = d if d.suffix == ".jsonl" else d / "episodes.jsonl"
        rows = load(ep)
        mid = rows[0].get("model", "") if rows else ""
        name = ("Haiku 4.5" if "haiku" in mid else "Sonnet 4.6" if "sonnet" in mid
                else "Opus 4.8" if "opus" in mid else mid)
        models.append((name, [f for f in (episode_facts(r) for r in rows) if f]))
    print("models:", [m[0] for m in models])

    nrow, ncol = len(METRICS), len(models)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.4 * nrow),
                             constrained_layout=True)
    extent = [2 - 0.5, T_HI + 0.5, N_LO - 0.5, N_HI + 0.5]
    im = None
    for r, (mname, mfn) in enumerate(METRICS):
        for c, (model, facts) in enumerate(models):
            ax = axes[r, c]
            gr = grid(facts, mfn, t_lo=2)
            im = ax.imshow(gaussian_fill(gr, SIGMA), origin="lower", aspect="auto",
                           cmap="viridis", extent=extent, vmin=0, vmax=1,
                           interpolation="nearest")
            ys, xs = np.where(~np.isnan(gr))
            ax.scatter(xs + 2, ys + N_LO, s=5, c="white", edgecolors="black",
                       linewidths=0.25, alpha=0.6)
            draw_boundary(ax, 2)
            ax.set_xlim(2 - 0.5, T_HI + 0.5); ax.set_ylim(N_LO - 0.5, N_HI + 0.5)
            ax.set_xticks(range(2, T_HI + 1, 2)); ax.set_yticks(range(0, N_HI + 1, 5))
            m = np.nanmean(gr) if np.any(~np.isnan(gr)) else float("nan")
            ax.text(0.04, 0.95, f"mean {m:.2f}", transform=ax.transAxes, fontsize=10,
                    va="top", ha="left", color="white",
                    bbox=dict(boxstyle="round", fc="black", alpha=0.45, lw=0))
            if r == 0:
                ax.set_title(model, fontsize=14, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{mname}\n\nN (doors)", fontsize=11)
            if r == nrow - 1:
                ax.set_xlabel("T (byproduct types)")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01, label="probability")
    fig.suptitle("toolworld v3 build chain -- cross-model comparison "
                 "(gaussian-pooled, T>=2; dashed = E[grind]=E[build])", fontsize=14)

    out = Path("runs") / "fig_grid_v3_compare.png"
    fig.savefig(out, dpi=130); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out}")
    print(f"  wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
