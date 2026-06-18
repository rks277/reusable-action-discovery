"""Three-panel 'recognition' map: P(built | held both recipe ingredients) over (T, N),
for Haiku / Sonnet / Opus side by side — SEABORN KDE variant.

Same conditioning as scripts/plot_recognition_panels.py (cells where the model held BOTH
recipe byproducts; never-held-both cells are dropped), but the smooth recognition surface
is built from Gaussian KDEs instead of a normalized box convolution:

    P(built | T, N) = (n_built / n_all) * KDE(built cells) / KDE(all held-both cells)

i.e. the ratio of a filled KDE over the held-both cells that built to a KDE over all
held-both cells, rescaled by the base build rate so the surface is a valid probability.
This is the Nadaraya-Watson conditional-expectation estimator with a Gaussian kernel.
seaborn supplies the kernel/bandwidth machinery (sns.kdeplot grid) and figure styling.

Per-model headline P(built | held) is printed in each panel title.

Usage: PYTHONPATH=. python -m scripts.plot_recognition_panels_kde <haiku_dir> <sonnet_dir> \
           <opus_dir> [--n-hi 20] [--bandwidth 1] [--gridsize 200]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import seaborn as sns
from scipy.stats import gaussian_kde

import scripts.sweep_config as cfg

T_LO, T_HI = 2, 10


def built(row: dict) -> bool:
    return bool(row.get("built_machine")) or any("fuse into" in o for o in row.get("obs", []))


def collected(row: dict) -> set:
    """Byproduct letters the model ever held (pickup + holding messages)."""
    s = set()
    for o in row.get("obs", []):
        s.update(re.findall(r"and a (\w+)\.", o))
        s.update(re.findall(r"hold \d+ (\w+)\(s\)", o))
    return s


def held_both(row: dict) -> bool:
    return set(row["labels"]["recipe"]) <= collected(row)


def load_cells(run_dir: Path):
    """-> (short, {(T,N): built01 for held-both cells}, headline P(built|held))."""
    rows = [json.loads(l) for l in (run_dir / "episodes.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    short = (rows[0].get("model", "?").split("-")[1]
             if rows and rows[0].get("model", "").startswith("claude-") else "?")
    cells = {}
    nb = 0
    for r in rows:
        if held_both(r):
            b = int(built(r))
            cells[(r["n_types"], r["n"])] = b
            nb += b
    head = nb / len(cells) if cells else float("nan")
    return short, cells, head


def kde_surface(cells, n_hi, bw, gridsize):
    """Nadaraya-Watson P(built | T, N) via ratio of Gaussian KDEs over held-both cells.

    -> (gt, gn, Z) on a (gridsize x gridsize) mesh; Z is NaN where the held-both KDE has
    negligible support (so the surface only paints near observed cells).
    """
    pts = np.array([[t, n] for (t, n) in cells], float).T          # 2 x m
    vals = np.array([cells[(int(t), int(n))] for t, n in pts.T], float)
    n_all = pts.shape[1]
    n_built = int(vals.sum())

    gx = np.linspace(T_LO, T_HI, gridsize)
    gy = np.linspace(1, n_hi, gridsize)
    gt, gn = np.meshgrid(gx, gy)
    grid = np.vstack([gt.ravel(), gn.ravel()])

    kde_all = gaussian_kde(pts, bw_method=bw)
    den = kde_all(grid)
    if n_built > 0:
        kde_built = gaussian_kde(pts[:, vals > 0], bw_method=bw)
        num = (n_built / n_all) * kde_built(grid)
    else:
        num = np.zeros_like(den)

    # mask to the region with real support; rescale support cutoff to the densest cell
    support = den / den.max()
    Z = np.divide(num, den, out=np.full_like(den, np.nan), where=den > 1e-12)
    Z[support < 0.02] = np.nan
    Z = np.clip(Z, 0, 1)
    return gt, gn, Z.reshape(gt.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("haiku_dir"); ap.add_argument("sonnet_dir"); ap.add_argument("opus_dir")
    ap.add_argument("--n-hi", type=int, default=20)
    ap.add_argument("--bandwidth", type=float, default=1.0,
                    help="gaussian_kde bw_method (smaller = sharper)")
    ap.add_argument("--gridsize", type=int, default=200)
    ap.add_argument("--levels", type=int, default=11,
                    help="discrete color bands for a crisp, high-contrast surface")
    args = ap.parse_args()

    sns.set_theme(style="white", context="talk")

    # crisp banded map: discrete levels of viridis
    cmap = sns.color_palette("viridis", n_colors=args.levels, as_cmap=True)
    norm = BoundaryNorm(np.linspace(0, 1, args.levels + 1), cmap.N)

    dirs = [Path(args.haiku_dir), Path(args.sonnet_dir), Path(args.opus_dir)]
    loaded = [load_cells(d) for d in dirs]

    # boundary E[build] = E[grind]
    ts = np.linspace(T_LO, T_HI, 200)
    bnd = [next((n for n in range(1, args.n_hi + 1)
                 if cfg._build_cost(n, int(round(t))) < cfg._grind_cost(n)), np.nan) for t in ts]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    mesh = None
    for ax, (short, cells, head) in zip(axes, loaded):
        print(f"{short}: held-both cells {len(cells)} | recognition P(built|held)={head:.2f}")
        if cells:
            gt, gn, Z = kde_surface(cells, args.n_hi, args.bandwidth, args.gridsize)
            mesh = ax.pcolormesh(gt, gn, Z, cmap=cmap, norm=norm, shading="auto")
            # scatter the held-both cells, colored by built 0/1
            arr = np.array([[t, n, v] for (t, n), v in cells.items()], float)
            sc = ax.scatter(arr[:, 0], arr[:, 1], c=arr[:, 2], cmap=cmap, norm=norm,
                            s=46, edgecolors="black", linewidths=0.7, zorder=3)
            if mesh is None:
                mesh = sc
        # dropped cells (never held both) as small x
        held_keys = set(cells)
        dropped = [(t, n) for t in range(T_LO, T_HI + 1) for n in range(1, args.n_hi + 1)
                   if (t, n) not in held_keys]
        if dropped:
            dv = np.array(dropped, float)
            ax.scatter(dv[:, 0], dv[:, 1], marker="x", c="0.45", s=26, linewidths=0.8,
                       zorder=2, label="never held both (dropped)")
        ax.plot(ts, bnd, "k--", lw=1.2, label="E[build]=E[grind]")
        ax.set_title(f"{short.capitalize()}\nrecognition P(built | held) = {head:.2f}  "
                     f"(n={len(cells)})", fontsize=13)
        ax.set_xlabel("byproduct types T")
        ax.set_xlim(T_LO - 0.5, T_HI + 0.5); ax.set_ylim(0.5, args.n_hi + 0.5)
    axes[0].set_ylabel("number of doors N")
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle(f"Recognition: P(built | held both recipe ingredients) — held-both cells only "
                 f"(seaborn KDE ratio, bw={args.bandwidth:g}; grind-calibrated budget; single-draw)",
                 y=1.03, fontsize=13)
    cb = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("P(built | held both)")

    out = Path(f"figs/toolworld/fig_recognition_panels_kde_Nle{args.n_hi}.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
