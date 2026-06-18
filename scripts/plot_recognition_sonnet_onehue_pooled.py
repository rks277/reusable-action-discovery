"""Sonnet recognition panel under 9 single-hue gradients in a 3x3 grid (pooled surface).

Like scripts.plot_recognition_sonnet_cmaps_pooled, but every colormap is a one-color
gradient (sequential single hue), including a white->black (Greys) ramp. Surface only —
no scatter dots, no dropped-cell 'x' marks.

Usage: PYTHONPATH=. python -m scripts.plot_recognition_sonnet_onehue_pooled <sonnet_dir> \
           [--n-hi 20] [--bandwidth 1] [--levels 11]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import seaborn as sns

import scripts.sweep_config as cfg
from scripts.plot_recognition_panels import T_LO, T_HI, load_cells, pooled

# (title, spec): a matplotlib single-hue name, or ("light", color) for sns.light_palette.
CMAPS = [
    ("Greys (white→black)", "Greys"),
    ("Blues", "Blues"),
    ("Greens", "Greens"),
    ("Reds", "Reds"),
    ("Purples", "Purples"),
    ("Oranges", "Oranges"),
    ("Teal", ("light", "teal")),
    ("Crimson", ("light", "crimson")),
    ("Navy", ("light", "navy")),
]


def build_cmap(spec, levels):
    if isinstance(spec, tuple):  # ("light", color)
        return sns.light_palette(spec[1], n_colors=levels, as_cmap=True)
    return sns.color_palette(spec, n_colors=levels, as_cmap=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sonnet_dir")
    ap.add_argument("--n-hi", type=int, default=20)
    ap.add_argument("--bandwidth", type=float, default=1.0)
    ap.add_argument("--levels", type=int, default=11)
    args = ap.parse_args()

    sns.set_theme(style="white", context="talk")

    short, cells, head = load_cells(Path(args.sonnet_dir))
    print(f"{short}: held-both cells {len(cells)} | recognition P(built|held)={head:.2f}")
    gt, gn, Z = pooled(cells, args.n_hi, args.bandwidth)
    Z = np.clip(Z, 0, 1)

    # boundary E[build] = E[grind]
    ts = np.linspace(T_LO, T_HI, 200)
    bnd = [next((n for n in range(1, args.n_hi + 1)
                 if cfg._build_cost(n, int(round(t))) < cfg._grind_cost(n)), np.nan) for t in ts]

    fig, axes = plt.subplots(3, 3, figsize=(16, 15), sharex=True, sharey=True)
    for ax, (title, spec) in zip(axes.ravel(), CMAPS):
        cmap = build_cmap(spec, args.levels)
        norm = BoundaryNorm(np.linspace(0, 1, args.levels + 1), cmap.N)
        mesh = ax.pcolormesh(gt, gn, Z, cmap=cmap, norm=norm, shading="nearest")
        ax.plot(ts, bnd, "k--", lw=1.0, zorder=4)
        ax.set_title(title, fontsize=14)
        ax.set_xlim(T_LO - 0.5, T_HI + 0.5); ax.set_ylim(0.5, args.n_hi + 0.5)
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
    for ax in axes[-1]:
        ax.set_xlabel("byproduct types T")
    for ax in axes[:, 0]:
        ax.set_ylabel("number of doors N")

    fig.suptitle(f"Sonnet recognition P(built | held) = {head:.2f}  (n={len(cells)}) — "
                 f"9 single-hue gradients (gaussian-pooled bw={args.bandwidth:g})",
                 y=0.995, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    out = Path(f"figs/toolworld/fig_recognition_sonnet_onehue_pooled_Nle{args.n_hi}.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
