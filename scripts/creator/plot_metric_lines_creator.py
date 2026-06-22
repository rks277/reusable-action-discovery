"""Line graph of the four C·R·E metrics vs. model size for the CREATOR fork,
visually mirroring figs/toolworld/fig_metric_lines_toolworld.png.

Reuses COLORS / MARKERS / CLAUDE_B / SERIES and the binomial-SE error-bar + shaded
"sizes estimated" band from scripts.plot_metric_lines, so the figure matches the
ToolWorld one. Reads the per-model creator_metrics.jsonl written by analyze_creator.

  PYTHONPATH=. python -m scripts.creator.plot_metric_lines_creator runs/creator_sweep_<ts>/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plot_metric_lines import CLAUDE_B, COLORS, MARKERS, SERIES

# Extend the ToolWorld series with grind (G = solve - C*R*E), the residual that closes
# Solve = C*R*E + G. Drawn dashed so it reads as a derived/decomposition term.
SERIES_C = SERIES + ["grind"]
COLORS_C = {**COLORS, "grind": "#ff7f0e"}
MARKERS_C = {**MARKERS, "grind": "v"}


def _disp(short: str) -> str:
    return short.capitalize()


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: plot_metric_lines_creator.py runs/creator_sweep_<ts>/")
    run_dir = Path(sys.argv[1])
    metrics_path = run_dir / "creator_metrics.jsonl"
    recs = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]

    pts = []  # (x, short, {metric: [value, n]})
    for rec in recs:
        model = rec["model"]
        short = model.split("-")[1] if model.startswith("claude-") else model
        if short not in CLAUDE_B:
            print(f"WARN: no size for {short!r}; skipping")
            continue
        pts.append((CLAUDE_B[short], short, rec))
    pts.sort(key=lambda x: x[0])

    xs = [p for p, _, _ in pts]
    labels = [f"{_disp(s)}\n~{p:g}B" for p, s, _ in pts]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    for metric in SERIES_C:
        vals = [rec[metric] for _, _, rec in pts]  # [value, n]
        ys = [v for v, n in vals]
        errs = [(v * (1 - v) / n) ** 0.5 if (n and v == v) else float("nan")
                for v, n in vals]
        ls = "--" if metric == "grind" else "-"
        ax.errorbar(xs, ys, yerr=errs, marker=MARKERS_C[metric], color=COLORS_C[metric],
                    lw=2, ls=ls, ms=8, capsize=3, elinewidth=1.2, label=metric,
                    markeredgecolor="black", markeredgewidth=0.6)

    ax.set_xscale("log")
    ax.set_xlabel("model parameter size (billions, log scale)")
    # All three are proprietary / size-estimated -> shade the whole span.
    ax.axvspan(min(xs) / 1.4, max(xs) * 1.4, color="0.85", alpha=0.4, zorder=0,
               label="proprietary (sizes estimated)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8, rotation=25, ha="right", rotation_mode="anchor")
    ax.minorticks_off()
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("metric value")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.set_title("CREATOR: C·R·E metrics vs. model size\n"
                 "(conditioning chain ask→build→correct; pooled headline ±1 binomial SE; "
                 "Claude sizes are rough estimates — see shaded band)", fontsize=12)

    outdir = Path("figs/creator")
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "fig_metric_lines_creator.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
