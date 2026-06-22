"""3x5 cross-model comparison grid for the v3 build chain (5 models).

Same three build-chain conditionals as fig_grid_v3_compare_3x3 (rows), but with
all five models as columns in the order: Qwen3.5-4B, Qwen3.5-9B, Haiku, Sonnet,
Opus. Each cell is the gaussian-pooled (N, T) heatmap (T>=2) with the
E[grind]=E[build] boundary overlaid and the empirical mean annotated.

Usage: PYTHONPATH=. python -m scripts.plot_grid_sweep_v3_compare5
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plot_grid_sweep_v3 import load, gaussian_fill, draw_boundary
from scripts.plot_grid_sweep_v3_chain import episode_facts, grid, N_LO, N_HI, T_HI

SIGMA = 1.2

# Column order requested: qwen 4b, qwen 9b, haiku, sonnet, opus.
MODELS = [
    ("qwen3-5-4b", "Qwen3.5-4B"),
    ("qwen3-5-9b", "Qwen3.5-9B"),
    ("haiku", "Haiku 4.5"),
    ("sonnet", "Sonnet 4.6"),
    ("opus", "Opus 4.8"),
]

# Same three rows as fig_grid_v3_compare_3x3 (drops "P(tool used | built)").
# Named: Curiosity = P(picked up both shards), Recognition = P(built | both
# shards), Efficiency = P(won | built).
METRICS = [
    ("Curiosity", lambda f: 1.0 if f["has_both"] else 0.0),
    ("Recognition",
     lambda f: None if not f["has_both"] else (1.0 if f["built"] else 0.0)),
    ("Efficiency",
     lambda f: None if not f["built"] else (1.0 if f["solved"] else 0.0)),
]


def latest(tag: str) -> Path:
    cands = sorted(p for p in Path("runs").rglob(f"grid_sweep_v3_{tag}_*") if p.is_dir())
    if not cands:
        raise SystemExit(f"no runs/grid_sweep_v3_{tag}_* dir found")
    return cands[-1]


def main():
    models = []
    for tag, name in MODELS:
        d = latest(tag)
        rows = load(d / "episodes.jsonl")
        facts = [f for f in (episode_facts(r) for r in rows) if f]
        models.append((name, facts))
        print(f"{name:12s} {d.name}: n={len(rows)}")

    nrow, ncol = len(METRICS), len(models)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.4 * nrow),
                             constrained_layout=True)
    extent = [2 - 0.5, T_HI + 0.5, N_LO - 0.5, N_HI + 0.5]
    im = None
    means = {}
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
            means[(model, mname)] = m
            ax.text(0.04, 0.95, f"mean {m:.2f}", transform=ax.transAxes, fontsize=10,
                    va="top", ha="left", color="white",
                    bbox=dict(boxstyle="round", fc="black", alpha=0.45, lw=0))
            if r == 0:
                ax.set_title(model, fontsize=14, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{mname}\n\nN (doors)", fontsize=11)
            if r == nrow - 1:
                ax.set_xlabel("T (byproduct types)")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="probability")
    fig.suptitle("toolworld v3 build chain -- cross-model comparison (5 models) "
                 "(gaussian-pooled, T>=2; dashed = E[grind]=E[build])", fontsize=14)

    out = Path("runs") / "grid_v3" / "fig_grid_v3_compare_3x5.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out}")
    print(f"  wrote {out.with_suffix('.pdf')}")
    print("\n=== per-cell means ===")
    for _, mname in [(0, m[0]) for m in METRICS]:
        line = "  ".join(f"{mod}={means[(mod, mname)]:.2f}" for mod, _ in models)
        print(f"{mname:28s} {line}")


if __name__ == "__main__":
    main()
