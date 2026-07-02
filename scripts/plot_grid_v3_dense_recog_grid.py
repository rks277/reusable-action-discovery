"""3x3 grid of recognition curves -- one subplot per (n, T) cell.

Each subplot plots R = P(built | holds both shards) against model size, with one
line per model family (same style as plot_grid_v3_dense_lines): x is size in B
params on a log axis, Anthropic sizes are supposed estimates. Rows are n in
{9,10,11}; columns are budget T in {2,3,4}. 95% Wilson CIs.

Usage: PYTHONPATH=. python -m scripts.plot_grid_v3_dense_recog_grid [dir ...]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plot_grid_sweep_v3 import load
from scripts.plot_grid_sweep_v3_chain import episode_facts
from scripts.plot_grid_v3_dense_bars import wilson
from scripts.plot_grid_v3_dense_lines import (
    DENSE_DIR, FAMILY_STYLE, classify, discover,
)

NS = [9, 10, 11]
TS = [2, 3, 4]


def main():
    args = [Path(a) for a in sys.argv[1:] if not a.startswith("-")]
    dirs = args if args else sorted(p for p in DENSE_DIR.iterdir() if p.is_dir())
    fams = discover(dirs)
    if not fams:
        raise SystemExit(f"no classifiable run dirs found under {DENSE_DIR}")
    for fam, members in fams.items():
        print(f"{fam}: " + ", ".join(f"{lab}(n={len(f)})" for _, lab, f in members))

    all_sizes = sorted({s for ms in fams.values() for s, _, _ in ms})

    fig, axes = plt.subplots(len(NS), len(TS), figsize=(15, 13),
                             constrained_layout=True, sharex=True, sharey=True)
    for i, n in enumerate(NS):
        for j, t in enumerate(TS):
            ax = axes[i][j]
            for fam, members in fams.items():
                style = FAMILY_STYLE.get(fam, dict(color="gray", marker="^"))
                xs, ys, los, his, labels = [], [], [], [], []
                for size, label, facts in members:
                    # R denominator: episodes in THIS cell that hold both shards.
                    sel = [f for f in facts
                           if f["n"] == n and f["t"] == t and f["has_both"] is True]
                    k = int(sum(1 for f in sel if f["built"]))
                    m = len(sel)
                    p = k / m if m else float("nan")
                    lo, hi = wilson(k, m)
                    xs.append(size); ys.append(p)
                    los.append(p - lo); his.append(hi - p); labels.append(label)
                ax.errorbar(xs, ys, yerr=[los, his], capsize=6, capthick=1.8,
                            elinewidth=1.8, lw=2, markersize=8,
                            markeredgecolor="black", markeredgewidth=0.6,
                            label=fam, **style)
                for x, y, lab in zip(xs, ys, labels):
                    if not math.isnan(y):
                        ax.annotate(lab, (x, y), textcoords="offset points",
                                    xytext=(0, 8), ha="center", fontsize=7,
                                    color=style["color"])
            ax.set_title(f"n={n}, T={t}", fontsize=11, fontweight="bold")
            ax.set_xscale("log")
            ax.set_xlim(5, 800); ax.set_ylim(0, 1.08)
            ax.set_xticks(all_sizes)
            ax.set_xticklabels([f"{int(s)}B" for s in all_sizes], fontsize=7)
            ax.tick_params(axis="x", which="minor", bottom=False)
            ax.grid(alpha=0.3, which="major")
            if i == len(NS) - 1:
                ax.set_xlabel("model size (B; Anthropic = supposed)", fontsize=9)
            if j == 0:
                ax.set_ylabel("P(built | both shards)", fontsize=9)
            if i == 0 and j == 0:
                ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("toolworld v3 dense -- recognition R = P(built | both shards) per (n, T) cell "
                 "(line = family, log x; Anthropic sizes supposed; 95% Wilson CI)",
                 fontsize=14)

    out = DENSE_DIR / "fig_grid_v3_dense_recog_grid.png"
    fig.savefig(out, dpi=130); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out}")
    print(f"  wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
