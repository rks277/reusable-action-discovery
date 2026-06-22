"""Cross-model success-rate (solved) heatmaps for the v3 grid sweep.

One panel per Anthropic model (Haiku / Sonnet / Opus). Each panel is the
gaussian-pooled P(solved) surface over the (N doors, T byproduct types) grid,
with the E[grind]=E[build] strategy boundary overlaid and the empirical solved
rate annotated. Shared 0..1 colour scale so panels are directly comparable.

Usage: PYTHONPATH=. python -m scripts.plot_grid_v3_success_heatmap \
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

from scripts.plot_grid_sweep_v3 import (
    load, cell_value, gaussian_fill, draw_boundary, N_LO, N_HI, T_LO, T_HI,
)

SIGMA = 1.2


def latest(tag: str) -> Path:
    cands = sorted(p for p in Path("runs").rglob(f"grid_sweep_v3_{tag}_*")
                   if p.is_dir())
    if not cands:
        raise SystemExit(f"no runs/grid_sweep_v3_{tag}_* dir found")
    return cands[-1]


def model_name(mid: str) -> str:
    return ("Haiku 4.5" if "haiku" in mid else "Sonnet 4.6" if "sonnet" in mid
            else "Opus 4.8" if "opus" in mid else mid)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    dirs = [Path(a) for a in args] if len(args) == 3 else \
        [latest("haiku"), latest("sonnet"), latest("opus")]

    models = []
    for d in dirs:
        ep = d if d.suffix == ".jsonl" else d / "episodes.jsonl"
        rows = load(ep)
        ok = [r for r in rows if not r.get("error")]
        mid = ok[0].get("model", "") if ok else ""
        models.append((model_name(mid), ok))
    print("models:", [(m, len(o)) for m, o in models])

    fig, axes = plt.subplots(1, len(models), figsize=(5.4 * len(models), 5.2),
                             constrained_layout=True)
    extent = [T_LO - 0.5, T_HI + 0.5, N_LO - 0.5, N_HI + 0.5]
    im = None
    for ax, (model, ok) in zip(axes, models):
        gr = cell_value(ok, lambda r: 1.0 if r.get("solved") else 0.0)
        im = ax.imshow(gaussian_fill(gr, SIGMA), origin="lower", aspect="auto",
                       cmap="viridis", extent=extent, vmin=0, vmax=1,
                       interpolation="nearest")
        sx = [r["n_types"] for r in ok]
        sy = [r["n"] for r in ok]
        ax.scatter(sx, sy, s=8, c="white", edgecolors="black", linewidths=0.3,
                   alpha=0.7)
        draw_boundary(ax, T_LO)
        ax.set_xlim(T_LO - 0.5, T_HI + 0.5)
        ax.set_ylim(N_LO - 0.5, N_HI + 0.5)
        ax.set_xticks(range(T_LO, T_HI + 1))
        ax.set_yticks(range(N_LO, N_HI + 1))
        rate = np.nanmean(gr) if np.any(~np.isnan(gr)) else float("nan")
        n = int(np.sum(~np.isnan(gr)))
        ax.set_title(f"{model}  (solved {rate:.2f}, n={n})", fontweight="bold")
        ax.set_xlabel("T (byproduct types)")
        ax.set_ylabel("N (doors)")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01, label="P(solved)")
    fig.suptitle("toolworld v3 (explicit pickup) -- success rate by (N, T) "
                 "(gaussian-pooled; dashed = E[grind]=E[build])", fontsize=13)

    out = Path("runs") / "grid_v3" / "fig_grid_v3_success_heatmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out}")
    print(f"  wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
