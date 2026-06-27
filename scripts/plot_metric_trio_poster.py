"""Poster trio: recognition / curiosity / efficiency vs. model parameter size,
Claude models only (Haiku, Sonnet, Opus), one panel each.

No titles. y = "probability", x = "model parameter size" (log). Model names are
annotated inside each panel rather than as x-axis tick labels.

Reuses the exact headline definitions from scripts.plot_metric_lines so each point
equals the corresponding number in the paper panels.

Usage:
  PYTHONPATH=. python -m scripts.plot_metric_trio_poster HAIKU_DIR SONNET_DIR OPUS_DIR \
      [--outdir figs/poster] [--out fig_metric_trio_poster.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plot_metric_lines import PARAM_B, _norm, _rows, metrics_toolworld

PANELS = ["curiosity", "recognition", "efficiency"]
COLORS = {"recognition": "#1f77b4", "curiosity": "#2ca02c", "efficiency": "#d62728"}
MARKERS = {"recognition": "o", "curiosity": "s", "efficiency": "^"}


def _disp(short: str) -> str:
    return short.capitalize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="Claude region run dirs (any order; sorted by size)")
    ap.add_argument("--outdir", default="figs/poster")
    ap.add_argument("--out", default="fig_metric_trio_poster.png")
    args = ap.parse_args()

    pts = []  # (x, short, {metric: (value, n)})
    for d in args.dirs:
        short, rows = _rows(Path(d))
        key = _norm(short)
        if key not in PARAM_B:
            print(f"WARN: no param size for {short!r} ({key}); skipping {d}")
            continue
        m = metrics_toolworld(rows)
        pts.append((PARAM_B[key], short, m))
        print(f"{short:8} (x={PARAM_B[key]:>7.1f}B)  " +
              "  ".join(f"{k}={m[k][0]:.2f}(n={m[k][1]})" for k in PANELS))
    pts.sort(key=lambda x: x[0])

    xs = [p for p, _, _ in pts]
    names = [_disp(s) for _, s, _ in pts]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, metric in zip(axes, PANELS):
        vals = [m[metric] for _, _, m in pts]
        ys = [v for v, n in vals]
        errs = [(v * (1 - v) / n) ** 0.5 if (n and v == v) else float("nan")
                for v, n in vals]
        ax.errorbar(xs, ys, yerr=errs, marker=MARKERS[metric], color=COLORS[metric],
                    lw=2, ms=10, capsize=4, elinewidth=1.3,
                    markeredgecolor="black", markeredgewidth=0.7)
        # label each model inside the panel, offset above its point
        for x, y, name in zip(xs, ys, names):
            if y != y:
                continue
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=10, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xlabel("model parameter size")
        ax.set_xticks([])
        ax.minorticks_off()
        ax.set_xlim(min(xs) / 2.2, max(xs) * 2.2)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, axis="y", ls=":", alpha=0.5)
        ax.set_title(metric.capitalize())
    axes[0].set_ylabel("probability")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / args.out
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
