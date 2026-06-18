"""Interpolated (T, N) map for a Haiku region sweep, in three selectable metrics.

Reads episodes.jsonl from a run_haiku_region_sweep.py output dir, aggregates a
per-(T, N) cell value, then interpolates those scattered cell values onto the full
T x N grid with scipy.interpolate.griddata (linear inside the convex hull, nearest-
neighbour fill outside it) and renders a heatmap with the E[build]=E[grind] boundary
overlaid (below/left of it, building is the rational/cheaper strategy).

--metric:
  built_given_solved (default) -- P(built | solved); aggregated over SOLVED episodes
      only; cells that never solved are excluded (shown as 'x', denominator undefined).
  built              -- P(built); fraction of ALL episodes in the cell that built. With
      one rep/cell this is a single 0/1 draw per point; griddata smooths the scatter into
      a build-propensity DENSITY map. Every sampled cell contributes.
  built_and_solved   -- fraction of ALL episodes that both built AND solved.

Also writes the per-cell aggregation to a JSON file next to the figure.

Usage: PYTHONPATH=. python -m scripts.plot_haiku_region_heatmap [runs/<dir>] \
           [--metric built] [--n-hi 20]
       (run dir defaults to the most recent runs/haiku_region_sweep_* directory)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RBFInterpolator
from scipy.ndimage import gaussian_filter

import scripts.sweep_config as cfg

T_LO, T_HI = 2, 10

# per-metric: (figure stem, json name, colorbar label, title)
# per-metric: (figure base stem, json base name, colorbar label, title suffix). The
# model short name is prepended to the stem and the model display name to the title, so
# Haiku reproduces the original filenames exactly while Sonnet/Opus get distinct ones.
META = {
    "built_given_solved": ("region_build_rate", "region_build_rate",
                           "P(built | solved)", "P(built | solved) over (T, N)"),
    "built": ("density_built", "region_built_density",
              "P(built)", "build-propensity density over (T, N)"),
    "built_and_solved": ("density_builtsolved", "region_builtsolved_density",
                         "P(built & solved)", "P(built & solved) over (T, N)"),
}


def latest_run() -> Path:
    cands = sorted(Path("runs").glob("haiku_region_sweep_p*")) or \
        sorted(Path("runs").glob("haiku_region_sweep_*"))
    if not cands:
        raise SystemExit("no runs/haiku_region_sweep_* directory found")
    return cands[-1]


def built(row: dict) -> bool:
    # canonical predicate (matches analyze_budget.py): built_machine flag, or the
    # "fuse into" fusion message anywhere in the observations.
    return bool(row.get("built_machine")) or any("fuse into" in o for o in row.get("obs", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=None)
    ap.add_argument("--metric", choices=list(META), default="built_given_solved")
    ap.add_argument("--n-hi", type=int, default=40)
    ap.add_argument("--smooth", choices=["none", "gaussian", "rbf"], default="none",
                    help="neighbor-pooling: 'none' = griddata linear+nearest (interpolate); "
                         "'gaussian' = normalized Gaussian convolution (sigma=bandwidth, in "
                         "cells; handles gaps); 'rbf' = RBFInterpolator with smoothing=bandwidth.")
    ap.add_argument("--bandwidth", type=float, default=1.0,
                    help="gaussian: sigma in cells; rbf: smoothing strength.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run()
    metric = args.metric
    n_hi = args.n_hi

    rows = [json.loads(l) for l in (run_dir / "episodes.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    model = rows[0].get("model", "claude-haiku-4-5") if rows else "claude-haiku-4-5"
    short = model.split("-")[1] if model.startswith("claude-") else model  # haiku/sonnet/opus

    fig_base, jname, cblabel, title_suffix = META[metric]
    stem = f"fig_{short}_{fig_base}"
    title = f"{short.capitalize()} {title_suffix}"
    # tag the N-range unless it's the original default run (built_given_solved, N<=40),
    # so we never clobber the headline figs/toolworld/fig_haiku_region_build_rate.png
    if metric != "built_given_solved" or n_hi != 40:
        stem = f"{stem}_Nle{n_hi}"
        jname = f"{jname}_Nle{n_hi}"
    if args.smooth != "none":
        stem = f"{stem}_{args.smooth}{args.bandwidth:g}"   # distinct from the unsmoothed map
    print(f"{run_dir}: {len(rows)} non-error episodes | model={short} | metric={metric} "
          f"| smooth={args.smooth}({args.bandwidth:g})")

    # aggregate per (T, N) cell
    cells = defaultdict(lambda: {"n_eps": 0, "n_solved": 0, "n_built": 0, "n_built_solved": 0})
    for r in rows:
        c = cells[(r["n_types"], r["n"])]
        c["n_eps"] += 1
        b = built(r)
        c["n_built"] += b
        if r.get("solved"):
            c["n_solved"] += 1
            c["n_built_solved"] += b

    def cell_value(c):
        if metric == "built_given_solved":
            return (c["n_built_solved"] / c["n_solved"]) if c["n_solved"] else None
        if metric == "built":
            return c["n_built"] / c["n_eps"]
        return c["n_built_solved"] / c["n_eps"]            # built_and_solved

    table = [{"T": t, "N": n, "value": cell_value(c), **c} for (t, n), c in sorted(cells.items())]
    (run_dir / f"{jname}.json").write_text(json.dumps(table, indent=2))

    have = [d for d in table if d["value"] is not None]
    gaps = [d for d in table if d["value"] is None]
    print(f"cells sampled: {len(table)} | with a value: {len(have)} | undefined: {len(gaps)}")
    if not have:
        raise SystemExit("no cells with a value -- nothing to interpolate")

    pts = np.array([[d["T"], d["N"]] for d in have], float)
    vals = np.array([d["value"] for d in have], float)
    grid_t, grid_n = np.meshgrid(np.arange(T_LO, T_HI + 1), np.arange(1, n_hi + 1))

    if args.smooth == "none":
        # interpolate: linear inside the convex hull, nearest-neighbour fill outside.
        lin = griddata(pts, vals, (grid_t, grid_n), method="linear")
        near = griddata(pts, vals, (grid_t, grid_n), method="nearest")
        Z = np.where(np.isnan(lin), near, lin)
    elif args.smooth == "gaussian":
        # normalized Gaussian convolution: each cell becomes a Gaussian-weighted average
        # of its neighbours (pools across cells; naturally fills gaps weighted by distance).
        # grid index [i, j] <-> (N = 1 + i, T = T_LO + j).
        V = np.zeros_like(grid_t, float)
        M = np.zeros_like(grid_t, float)              # presence mask
        for d in have:
            V[d["N"] - 1, d["T"] - T_LO] = d["value"]
            M[d["N"] - 1, d["T"] - T_LO] = 1.0
        num = gaussian_filter(V * M, args.bandwidth, mode="nearest")
        den = gaussian_filter(M, args.bandwidth, mode="nearest")
        Z = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-9)
        if np.isnan(Z).any():                          # any cell with ~no weight -> nearest
            Z = np.where(np.isnan(Z), griddata(pts, vals, (grid_t, grid_n), method="nearest"), Z)
    else:  # rbf: smoothing > 0 turns exact interpolation into a pooling regression.
        rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline", smoothing=args.bandwidth)
        Z = rbf(np.column_stack([grid_t.ravel(), grid_n.ravel()])).reshape(grid_t.shape)
    Z = np.clip(Z, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    mesh = ax.pcolormesh(grid_t, grid_n, Z, cmap="RdYlGn", vmin=0, vmax=1, shading="nearest")
    fig.colorbar(mesh, ax=ax).set_label(cblabel)

    # E[build] = E[grind] boundary (below/left: building is the cheaper strategy)
    ts = np.linspace(T_LO, T_HI, 200)
    bnd = [next((n for n in range(1, n_hi + 1)
                 if cfg._build_cost(n, int(round(t))) < cfg._grind_cost(n)), np.nan) for t in ts]
    ax.plot(ts, bnd, "k--", lw=1.2, label="E[build]=E[grind]")

    # sampled cells: colored by value where defined, 'x' where undefined
    if have:
        hv = np.array([[d["T"], d["N"], d["value"]] for d in have], float)
        ax.scatter(hv[:, 0], hv[:, 1], c=hv[:, 2], cmap="RdYlGn", vmin=0, vmax=1,
                   s=42, edgecolors="black", linewidths=0.7, label="sampled point")
    if gaps:
        gv = np.array([[d["T"], d["N"]] for d in gaps], float)
        ax.scatter(gv[:, 0], gv[:, 1], marker="x", c="black", s=34, linewidths=1.0,
                   label="never solved (undefined)")

    note = "single-draw" if max(d["n_eps"] for d in table) == 1 else f"{len(rows)} eps"
    smooth_note = "interpolated" if args.smooth == "none" else \
        f"{args.smooth}-pooled (bw={args.bandwidth:g})"
    ax.set_xlabel("byproduct types T")
    ax.set_ylabel("number of doors N")
    ax.set_title(f"{title}\n({smooth_note}; grind-calibrated budget; {note})")
    ax.set_xlim(T_LO - 0.5, T_HI + 0.5)
    ax.set_ylim(0.5, n_hi + 0.5)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    out = Path(f"figs/toolworld/{stem}.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out} (+ .pdf) and {run_dir / (jname + '.json')}")


if __name__ == "__main__":
    main()
