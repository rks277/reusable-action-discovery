"""Gaussian-pooled (T, N) map of tool-EXPLOITATION efficiency for an efficiency
sweep (run_haiku_efficiency_sweep.py).

Once the tool is built and handed to the agent, what FRACTION of its remaining budget
does it burn on REDUNDANT actions? A redundant continuation action makes no progress --
neither opening a door nor operating the machine on a door. We plot, per (T, N) cell,
the mean of

    redundant_continuation / live_remaining          (live_remaining = budget - (ab + 2))

i.e. redundant actions as a proportion of the live budget the agent actually had after
the forced build. Normalizing by remaining budget makes cells comparable: a cell with 5
live actions and 5 redundant reads as fully wasteful (1.0), the same 5 redundant out of
60 reads as efficient (0.08) -- it removes the "small budget => no room to flounder"
confound of the raw count. Higher (red) = more of the post-build budget wasted.

The value is a proportion in [0, 1]. Degenerate runs (live_remaining <= 0: no budget
left after the context) have an undefined denominator and are shown as gaps ('x'). Pooled
across cells with a normalized Gaussian convolution (same pooling as the recognition maps).

Usage: PYTHONPATH=. python -m scripts.plot_haiku_efficiency_heatmap [runs/<dir>] \
           [--smooth gaussian] [--bandwidth 1.0] [--n-hi 20]
       (run dir defaults to the most recent runs/*_efficiency_sweep_* directory)
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


def latest_run() -> Path:
    cands = sorted(Path("runs").glob("*_efficiency_sweep_*"))
    if not cands:
        raise SystemExit("no runs/*_efficiency_sweep_* directory found")
    return cands[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=None)
    ap.add_argument("--n-hi", type=int, default=20)
    ap.add_argument("--smooth", choices=["none", "gaussian", "rbf"], default="gaussian",
                    help="neighbor-pooling: 'gaussian' = normalized Gaussian convolution "
                         "(sigma=bandwidth, in cells; handles gaps); 'none' = griddata "
                         "linear+nearest; 'rbf' = RBFInterpolator with smoothing=bandwidth.")
    ap.add_argument("--bandwidth", type=float, default=1.0,
                    help="gaussian: sigma in cells; rbf: smoothing strength.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run()
    n_hi = args.n_hi

    rows = [json.loads(l) for l in (run_dir / "episodes.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    model = rows[0].get("model", "claude-haiku-4-5") if rows else "claude-haiku-4-5"
    short = model.split("-")[1] if model.startswith("claude-") else model

    stem = f"fig_{short}_efficiency_redundprop_Nle{n_hi}"
    jname = f"region_efficiency_redundprop_Nle{n_hi}"
    if args.smooth != "none":
        stem = f"{stem}_{args.smooth}{args.bandwidth:g}"
    print(f"{run_dir}: {len(rows)} non-error runs | model={short} | "
          f"smooth={args.smooth}({args.bandwidth:g})")

    # aggregate per (T, N) cell: mean of redundant_continuation / live_remaining.
    # live_remaining <= 0 (degenerate) has no valid denominator -> excluded.
    cells = defaultdict(lambda: {"n_eps": 0, "n_valid": 0, "sum_prop": 0.0,
                                 "n_solved": 0, "n_degenerate": 0})
    for r in rows:
        c = cells[(r["n_types"], r["n"])]
        c["n_eps"] += 1
        c["n_solved"] += int(bool(r.get("live_solved")))
        c["n_degenerate"] += int(bool(r.get("degenerate")))
        lr = r.get("live_remaining", 0)
        if r.get("degenerate") or lr <= 0:
            continue
        c["n_valid"] += 1
        c["sum_prop"] += r.get("redundant_continuation", 0) / lr

    def cell_value(c):
        return c["sum_prop"] / c["n_valid"] if c["n_valid"] else None

    table = [{"T": t, "N": n, "value": cell_value(c), **c}
             for (t, n), c in sorted(cells.items())]
    (run_dir / f"{jname}.json").write_text(json.dumps(table, indent=2))

    have = [d for d in table if d["value"] is not None]
    gaps = [d for d in table if d["value"] is None]
    print(f"cells with a value: {len(have)} / {len(table)} | undefined (degenerate): {len(gaps)}")
    if not have:
        raise SystemExit("no cells with a value -- nothing to plot")

    pts = np.array([[d["T"], d["N"]] for d in have], float)
    vals = np.array([d["value"] for d in have], float)
    grid_t, grid_n = np.meshgrid(np.arange(T_LO, T_HI + 1), np.arange(1, n_hi + 1))

    if args.smooth == "none":
        lin = griddata(pts, vals, (grid_t, grid_n), method="linear")
        near = griddata(pts, vals, (grid_t, grid_n), method="nearest")
        Z = np.where(np.isnan(lin), near, lin)
    elif args.smooth == "gaussian":
        # normalized Gaussian convolution: each cell becomes a Gaussian-weighted average
        # of its neighbours (pools across cells; fills gaps weighted by distance).
        V = np.zeros_like(grid_t, float)
        M = np.zeros_like(grid_t, float)
        for d in have:
            V[d["N"] - 1, d["T"] - T_LO] = d["value"]
            M[d["N"] - 1, d["T"] - T_LO] = 1.0
        num = gaussian_filter(V * M, args.bandwidth, mode="nearest")
        den = gaussian_filter(M, args.bandwidth, mode="nearest")
        Z = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-9)
        if np.isnan(Z).any():
            Z = np.where(np.isnan(Z), griddata(pts, vals, (grid_t, grid_n), method="nearest"), Z)
    else:
        rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline", smoothing=args.bandwidth)
        Z = rbf(np.column_stack([grid_t.ravel(), grid_n.ravel()])).reshape(grid_t.shape)
    Z = np.clip(Z, 0.0, 1.0)   # proportion in [0, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    # RdYlGn_r: green = little of the budget wasted (efficient), red = most of it wasted.
    mesh = ax.pcolormesh(grid_t, grid_n, Z, cmap="RdYlGn_r", vmin=0, vmax=1,
                         shading="nearest")
    fig.colorbar(mesh, ax=ax).set_label("redundant actions / remaining budget")

    # E[build]=E[grind] reference line (where building first becomes the cheaper path)
    ts = np.linspace(T_LO, T_HI, 200)
    bnd = [next((n for n in range(1, n_hi + 1)
                 if cfg._build_cost(n, int(round(t))) < cfg._grind_cost(n)), np.nan) for t in ts]
    ax.plot(ts, bnd, "k--", lw=1.2, label="E[build]=E[grind] (ref)")

    # sampled cells, colored by value; degenerate cells as 'x'
    hv = np.array([[d["T"], d["N"], d["value"]] for d in have], float)
    ax.scatter(hv[:, 0], hv[:, 1], c=hv[:, 2], cmap="RdYlGn_r", vmin=0, vmax=1,
               s=42, edgecolors="black", linewidths=0.7, label="sampled run")
    if gaps:
        gv = np.array([[d["T"], d["N"]] for d in gaps], float)
        ax.scatter(gv[:, 0], gv[:, 1], marker="x", c="black", s=34, linewidths=1.0,
                   label="degenerate (no budget)")

    smooth_note = "interpolated" if args.smooth == "none" else \
        f"{args.smooth}-pooled (bw={args.bandwidth:g})"
    ax.set_xlabel("byproduct types T")
    ax.set_ylabel("number of doors N")
    ax.set_title(f"{short.capitalize()} tool-exploitation efficiency: redundant actions "
                 f"as a fraction of\nremaining budget after a forced build "
                 f"({smooth_note}; redundant = no door/machine progress)")
    ax.set_xlim(T_LO - 0.5, T_HI + 0.5)
    ax.set_ylim(0.5, n_hi + 0.5)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    out = Path(f"figs/{stem}.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out} (+ .pdf) and {run_dir / (jname + '.json')}")


if __name__ == "__main__":
    main()
