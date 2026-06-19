"""Gaussian-pooled (n x m) heatmaps for a beach v2 strip sweep, in the style of
plot_grid_sweep_v3.

The strip sweep runs ~one episode per (width n, papers m) cell, so the raw grid
is a sparse 0/1 scatter. We fill + smooth it the same way plot_grid_sweep_v3
--gauss does: normalized Gaussian convolution (Nadaraya-Watson),
value = sum_i w_i v_i / sum_i w_i with Gaussian weights, sigma in grid cells
(default 1.2). One figure per model, three panels:

  built tool  -- map assembled? (built_map)
  solved      -- treasure dug up? (won)
  used map    -- read the map? (used_map)

x = papers m (scraps to form the map), y = width n (strip length).

Usage: python -m scripts.plot_beach_v2_grid runs/beach_v2_sweep_*/episodes.jsonl [--gauss SIGMA] [--raw]
       (default renders both a gaussian-pooled and a raw figure)
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
from scipy.ndimage import gaussian_filter

PANELS = [
    ("built tool", lambda r: 1.0 if r.get("built_map") else 0.0),
    ("solved", lambda r: 1.0 if r.get("won") else 0.0),
    ("used map", lambda r: 1.0 if r.get("used_map") else 0.0),
]


def short(model: str) -> str:
    s = model.lower()
    for k in ("haiku", "sonnet", "opus", "fable", "gpt", "gemini", "gemma"):
        if k in s:
            return k
    return model.split("/")[-1]


def gaussian_fill(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Fill NaN cells via normalized Gaussian convolution; edge mode 'nearest'."""
    mask = ~np.isnan(grid)
    vals = np.where(mask, grid, 0.0)
    num = gaussian_filter(vals, sigma=sigma, mode="nearest")
    den = gaussian_filter(mask.astype(float), sigma=sigma, mode="nearest")
    return num / np.maximum(den, 1e-12)


def main():
    argv = sys.argv[1:]
    sigma = 1.2
    gauss = "--gauss" in argv
    if gauss:
        gi = argv.index("--gauss")
        if gi + 1 < len(argv) and not argv[gi + 1].startswith("-"):
            sigma = float(argv[gi + 1]); argv.pop(gi + 1)
        argv.pop(gi)
    do_raw = "--raw" in argv
    argv = [a for a in argv if not a.startswith("-")]
    if not argv:
        raise SystemExit("usage: python -m scripts.plot_beach_v2_grid "
                         "runs/beach_v2_sweep_*/episodes.jsonl [--gauss SIGMA] [--raw]")
    path = Path(argv[0])
    run_dir = path.parent if path.suffix == ".jsonl" else path
    ep = run_dir / "episodes.jsonl" if path.suffix != ".jsonl" else path
    rows = [json.loads(l) for l in ep.read_text().splitlines() if l.strip()]
    ok = [r for r in rows if not r.get("error")]
    rows_h = sorted({r["rows"] for r in ok if "rows" in r}) or ["?"]

    ns = [r["grid_size"] for r in ok]
    ms = [r["papers_needed"] for r in ok]
    n_lo, n_hi = min(ns), max(ns)
    m_lo, m_hi = min(ms), max(ms)

    by_model = defaultdict(list)
    for r in ok:
        by_model[r["model"]].append(r)

    # render once per (model, mode); mode = pooled (default) and/or raw
    modes = [("gauss", True)] + ([("raw", True)] if do_raw else [])
    if not gauss and not do_raw:
        modes = [("gauss", True), ("raw", True)]   # default: emit both

    for model, mrows in by_model.items():
        for mode_name, _ in modes:
            _render(model, mrows, mode_name, sigma, n_lo, n_hi, m_lo, m_hi,
                    rows_h, run_dir)


def _grid(mrows, fn, n_lo, n_hi, m_lo, m_hi):
    g = np.full((n_hi - n_lo + 1, m_hi - m_lo + 1), np.nan)
    for r in mrows:
        g[r["grid_size"] - n_lo, r["papers_needed"] - m_lo] = fn(r)
    return g


def _render(model, mrows, mode, sigma, n_lo, n_hi, m_lo, m_hi, rows_h, run_dir):
    extent = [m_lo - 0.5, m_hi + 0.5, n_lo - 0.5, n_hi + 0.5]
    sx = [r["papers_needed"] for r in mrows]
    sy = [r["grid_size"] for r in mrows]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.4 * len(PANELS), 5.0),
                             constrained_layout=True)
    im = None
    for ax, (title, fn) in zip(axes, PANELS):
        grid = _grid(mrows, fn, n_lo, n_hi, m_lo, m_hi)
        cmap = plt.get_cmap("viridis").copy()
        if mode == "gauss":
            data = gaussian_fill(grid, sigma)
        else:
            cmap.set_bad("lightgrey")
            data = np.ma.masked_invalid(grid)
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap,
                       extent=extent, vmin=0, vmax=1, interpolation="nearest")
        if mode == "gauss":
            ax.scatter(sx, sy, s=10, c="white", edgecolors="black",
                       linewidths=0.3, alpha=0.65)
        ax.set_title(f"{title}  (mean {np.nanmean(grid):.2f})", fontsize=11)
        ax.set_xlabel("m = papers to form map", fontsize=9)
        ax.set_ylabel("n = strip width", fontsize=9)
        ax.set_xticks(range(m_lo, m_hi + 1))
        ax.set_yticks(range(n_lo, n_hi + 1, max(1, (n_hi - n_lo) // 10)))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    rh = rows_h[0] if len(rows_h) == 1 else rows_h
    n_sampled = len({(r["grid_size"], r["papers_needed"]) for r in mrows})
    mode_txt = (f"gaussian-pooled fill, sigma={sigma} cells (dots = sampled cells)"
                if mode == "gauss" else "raw (grey = not sampled)")
    fig.suptitle(f"beach v2 strip sweep ({short(model)}, ROWS={rh}) -- "
                 f"{n_sampled} sampled (n,m) cells -- {mode_txt}", fontsize=12)
    stem = f"fig_beach_v2_grid_{short(model)}" + ("" if mode == "gauss" else "_raw")
    fig.savefig(run_dir / f"{stem}.png", dpi=130)
    fig.savefig(run_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"  wrote {run_dir / (stem + '.png')} (+ .pdf)")


if __name__ == "__main__":
    main()
