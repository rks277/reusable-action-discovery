"""3x3 cross-model comparison for the beach v2 obfuscated-square sweeps.

Rows = models (Haiku / Sonnet / Opus), columns = the three NON-degenerate panels
of the build chain:
  1. P(picked up all map pieces)  -- the map-build rate (collecting all m scraps
     IS forming the map).
  2. P(won | built map)           -- once built, was the tool exploited to win?
  3. P(won)                        -- unconditional success rate.

(The degenerate P(built | picked up all pieces) == 1.0 panel is omitted.)

Each panel is a Gaussian-pooled (n x m) heatmap, x = papers m, y = width n, shared
viridis 0..1 color scale, with the E[brute]=E[build] boundary overlaid and white
dots on cells where the metric is defined. One row per episode file passed.

Usage:
  python -m scripts.plot_beach_v2_compare RUN_HAIKU RUN_SONNET RUN_OPUS [--sigma S]
where each RUN is a run dir or its episodes.jsonl. Order = top-to-bottom rows.
If no runs are given, auto-picks the latest obfuscated-square run per model.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plot_beach_v2_grid import gaussian_fill, short
from scripts.plot_beach_v2_chain import _grid, build_brute_boundary

# desired row order
ORDER = ["haiku", "sonnet", "opus"]
COLS = [
    ("P(picked up all map pieces)", lambda r: 1.0 if r["built_map"] else 0.0),
    ("P(won | built map)",
     lambda r: None if not r["built_map"] else (1.0 if r["won"] else 0.0)),
    ("P(won) -- total success", lambda r: 1.0 if r["won"] else 0.0),
]


def _load(path):
    p = Path(path)
    ep = p if p.suffix == ".jsonl" else p / "episodes.jsonl"
    rows = [json.loads(l) for l in ep.read_text().splitlines() if l.strip()]
    return [r for r in rows if not r.get("error")]


def _auto_runs():
    """Latest obfuscated-square run per model, in ORDER."""
    by_model = {}
    for run in sorted(glob.glob("runs/beach_v2/beach_v2_sweep_*")):
        ok = _load(run)
        if not ok:
            continue
        if not all(r.get("obfuscate") for r in ok):
            continue
        if not all(r.get("rows") == r["grid_size"] for r in ok):
            continue
        by_model[short(ok[0]["model"])] = run            # later ts overwrites
    picked = [by_model[k] for k in ORDER if k in by_model]
    if len(picked) != len(ORDER):
        raise SystemExit(f"auto-pick found {sorted(by_model)}, need {ORDER}")
    return picked


def main():
    argv = sys.argv[1:]
    sigma = 1.2
    if "--sigma" in argv:
        i = argv.index("--sigma"); sigma = float(argv[i + 1]); del argv[i:i + 2]
    runs = [a for a in argv if not a.startswith("-")] or _auto_runs()

    datasets = []                                        # (model_short, rows)
    for run in runs:
        ok = _load(run)
        datasets.append((short(ok[0]["model"]), ok))
    # order rows by ORDER when names are known
    datasets.sort(key=lambda d: ORDER.index(d[0]) if d[0] in ORDER else 99)

    all_rows = [r for _, rs in datasets for r in rs]
    ns = [r["grid_size"] for r in all_rows]
    ms = [r["papers_needed"] for r in all_rows]
    n_lo, n_hi, m_lo, m_hi = min(ns), max(ns), min(ms), max(ms)
    square = all(r.get("rows") == r["grid_size"] for r in all_rows)
    rows_h = sorted({r.get("rows") for r in all_rows})
    bnd_rows = None if square else (rows_h[0] if len(rows_h) == 1 else None)

    nrow, ncol = len(datasets), len(COLS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 4.6 * nrow),
                             constrained_layout=True, squeeze=False)
    extent = [m_lo - 0.5, m_hi + 0.5, n_lo - 0.5, n_hi + 0.5]
    im = None
    for i, (mname, mrows) in enumerate(datasets):
        for j, (title, fn) in enumerate(COLS):
            ax = axes[i][j]
            gr = _grid(mrows, fn, n_lo, n_hi, m_lo, m_hi)
            filled = gaussian_fill(gr, sigma)
            im = ax.imshow(filled, origin="lower", aspect="auto", cmap="viridis",
                           extent=extent, vmin=0, vmax=1, interpolation="nearest")
            ys, xs = np.where(~np.isnan(gr))
            ax.scatter(xs + m_lo, ys + n_lo, s=7, c="white", edgecolors="black",
                       linewidths=0.3, alpha=0.7)
            if square or bnd_rows:
                build_brute_boundary(ax, bnd_rows, n_lo, n_hi, m_lo, m_hi,
                                     legend=(i == 0 and j == 0), square=square)
            mean = np.nanmean(gr) if np.any(~np.isnan(gr)) else float("nan")
            ax.set_title(f"{title}  (mean {mean:.2f})", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{mname}\n\nn = grid width", fontsize=10,
                              fontweight="bold")
            else:
                ax.set_ylabel("n = grid width", fontsize=8)
            if i == nrow - 1:
                ax.set_xlabel("m = papers to form map", fontsize=9)
            ax.set_xticks(range(m_lo, m_hi + 1))
            ax.set_yticks(range(n_lo, n_hi + 1, max(1, (n_hi - n_lo) // 10)))
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    geom = "square n x n" if square else f"rows={rows_h}"
    fig.suptitle(f"beach v2 (obfuscated, {geom}) -- model comparison -- "
                 f"gaussian-pooled (sigma={sigma}); dots = cells where metric defined",
                 fontsize=13, fontweight="bold")
    out = Path("runs") / "beach_v2" / "fig_beach_v2_compare_3x3.png"
    fig.savefig(out, dpi=130); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out} (+ .pdf)")
    print("  rows:", " / ".join(f"{m} (n={len(r)})" for m, r in datasets))


if __name__ == "__main__":
    main()
