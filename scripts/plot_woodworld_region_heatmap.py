"""Gaussian-pooled (p, N) heatmaps of build rate and solve rate for a woodworld
region sweep, mirroring scripts/plot_recognition_panels_kde.py.

Reads episodes.jsonl from a run_woodworld_region_sweep.py output dir and renders
two panels -- P(built) and P(solved) -- pooled over the (gather probability p,
target wood N) plane with the Nadaraya-Watson conditional-expectation estimator:

    P(metric | p, N) = (n_pos / n_all) * KDE(positive cells) / KDE(all cells)

i.e. the ratio of a Gaussian KDE over the cells where the metric fired to a KDE
over all cells, rescaled by the base rate (a valid probability surface). With one
rep/cell each cell is a single 0/1 draw and the KDE smooths the scatter into a
propensity surface. The E[build]=E[grind] boundary is overlaid; below/left of it
building is the cheaper strategy.

Usage: PYTHONPATH=. python -m scripts.plot_woodworld_region_heatmap [runs/<dir>] \
           [--bandwidth 0.6] [--gridsize 200]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from scripts.validate_woodworld import brute_expected, build_expected

P_LO, P_HI = 0.2, 1.0
N_LO, N_HI = 2, 20
P_GRID = [round(0.2 + 0.1 * i, 1) for i in range(9)]   # 0.2 .. 1.0 lattice
N_GRID = list(range(N_LO, N_HI + 1))                    # 2 .. 20 lattice


def latest_run() -> Path:
    cands = sorted(Path("runs").glob("*woodworld_region_*"))
    if not cands:
        raise SystemExit("no runs/*woodworld_region_* directory found")
    return cands[-1]


def pooled(cells, bw):
    """Gaussian-pooled value on the discrete (p, N) lattice, rendered as boxes
    (same machinery as plot_efficiency_panels.py: gaussian_filter over the integer
    lattice, then shading='nearest'). cells: {(p, n): mean value}. Returns
    (gp, gn, Z) on the lattice; Z is NaN where no cell has support nearby."""
    gp, gn = np.meshgrid(P_GRID, N_GRID)            # shape (len N, len P)
    V = np.zeros_like(gp, float)
    M = np.zeros_like(gp, float)
    pidx = {p: j for j, p in enumerate(P_GRID)}
    for (p, n), v in cells.items():
        p = round(p, 1)
        if n in N_GRID and p in pidx:
            V[n - N_LO, pidx[p]] = v
            M[n - N_LO, pidx[p]] = 1.0
    num = gaussian_filter(V * M, bw, mode="nearest")
    den = gaussian_filter(M, bw, mode="nearest")
    Z = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-9)
    return gp, gn, np.clip(Z, 0, 1)


def boundary():
    """N where E[build] crosses E[grind], per p (build cheaper at/above the line)."""
    ps = np.linspace(P_LO, P_HI, 200)
    ns = [next((n for n in range(N_LO, N_HI + 1)
                if build_expected(n, p) < brute_expected(n, p)), np.nan) for p in ps]
    return ps, ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=None)
    ap.add_argument("--bandwidth", type=float, default=1.0,
                    help="gaussian_filter sigma in LATTICE CELLS (0 = raw boxes, "
                         "no pooling; like plot_efficiency_panels)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run()
    rows = [json.loads(l) for l in (run_dir / "episodes.jsonl").read_text().splitlines()
            if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    short = (rows[0]["model"].split("-")[1]
             if rows and rows[0].get("model", "").startswith("claude-") else "?")
    hint_on = bool(rows[0].get("hint")) if rows else False
    htag = "hint" if hint_on else "nohint"
    # infer the budget multiplier actually used (budget = round(mult * n / p)),
    # so figures from different budget regimes don't overwrite each other
    import statistics
    mult = (round(statistics.median(r["budget"] * r["gather_prob"] / r["n"]
                                    for r in rows), 1) if rows else 0)

    # per-cell mean over reps (1 rep -> raw 0/1)
    from collections import defaultdict
    agg_b, agg_s = defaultdict(list), defaultdict(list)
    for r in rows:
        key = (round(r["gather_prob"], 1), r["n"])
        agg_b[key].append(int(bool(r["built_axe"])))
        agg_s[key].append(int(bool(r["solved"])))
    cells_build = {k: np.mean(v) for k, v in agg_b.items()}
    cells_solve = {k: np.mean(v) for k, v in agg_s.items()}
    bmean = np.mean([int(bool(r["built_axe"])) for r in rows])
    smean = np.mean([int(bool(r["solved"])) for r in rows])
    print(f"{short}: {len(rows)} cells | build rate {bmean:.2f} | solve rate {smean:.2f}")

    bp, bn = boundary()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    mesh = None
    for ax, cells, mval, title in [(axes[0], cells_build, bmean, "build rate  P(built)"),
                                   (axes[1], cells_solve, smean, "solve rate  P(solved)")]:
        gp, gn, Z = pooled(cells, args.bandwidth)
        # ToolWorld style: discrete boxes (shading='nearest'), red (low) -> green (high)
        mesh = ax.pcolormesh(gp, gn, Z, cmap="RdYlGn", vmin=0, vmax=1, shading="nearest")
        arr = np.array([[p, n, v] for (p, n), v in cells.items()], float)
        ax.scatter(arr[:, 0], arr[:, 1], c=arr[:, 2], cmap="RdYlGn", vmin=0, vmax=1,
                   s=42, edgecolors="black", linewidths=0.6, zorder=3)
        ax.plot(bp, bn, "k--", lw=1.6, zorder=4, label="E[build]=E[grind]")
        ax.set_title(f"{title}   (mean {mval:.2f})", fontsize=13)
        ax.set_xlabel("gather probability p")
        ax.set_xlim(P_LO - 0.03, P_HI + 0.03); ax.set_ylim(N_LO - 0.5, N_HI + 0.5)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    axes[0].set_ylabel("target wood N")

    fig.suptitle(f"woodworld region ({short.capitalize()}, hint {'on' if hint_on else 'off'}, "
                 f"budget mult={mult:g}): Gaussian-pooled build & solve rate over (p, N)  "
                 f"[bw={args.bandwidth:g}, 1 rep/cell]", y=1.02, fontsize=13)
    cb = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("rate")

    out = Path(f"figs/fig_woodworld_region_build_solve_{short}_{htag}_mult{mult:g}"
               f"_Nle{N_HI}.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
