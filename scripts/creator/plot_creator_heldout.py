"""CREATOR v2 figure — held-out generalization vs model size, with the v1 overlay.

Plots v2 Recognition (P generalizes), Grind, and Solve(shown) across the Claude trio,
and overlays v1 Recognition (P built|ask, the un-gated regime) as a dashed reference so
the v1->v2 contrast is visible. Reuses the ToolWorld plot styling.

  PYTHONPATH=. python -m scripts.creator.plot_creator_heldout runs/creator_heldout_<ts>/ \
      [runs/creator_all/creator_metrics.jsonl]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plot_metric_lines import CLAUDE_B

COLORS = {"recognition": "#1f77b4", "grind": "#ff7f0e", "solve rate": "#000000"}
MARKERS = {"recognition": "o", "grind": "v", "solve rate": "D"}
V2_SERIES = ["recognition", "grind", "solve rate"]


def _short(model: str) -> str:
    return model.split("-")[1] if model.startswith("claude-") else model


def _load(metrics_path: Path) -> list[tuple[float, str, dict]]:
    pts = []
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        s = _short(rec["model"])
        if s in CLAUDE_B:
            pts.append((CLAUDE_B[s], s, rec))
    pts.sort(key=lambda x: x[0])
    return pts


def _bars(pts, metric):
    vals = [rec[metric] for _, _, rec in pts]
    ys = [v for v, n in vals]
    errs = [(v * (1 - v) / n) ** 0.5 if (n and v == v) else float("nan") for v, n in vals]
    return ys, errs


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: plot_creator_heldout.py runs/creator_heldout_<ts>/ [v1_metrics.jsonl]")
    run_dir = Path(sys.argv[1])
    v1_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("runs/creator_all/creator_metrics.jsonl")

    pts = _load(run_dir / "creator_heldout_metrics.jsonl")
    xs = [p for p, _, _ in pts]
    labels = [f"{s.capitalize()}\n~{p:g}B" for p, s, _ in pts]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    for metric in V2_SERIES:
        ys, errs = _bars(pts, metric)
        ls = "--" if metric == "grind" else "-"
        ax.errorbar(xs, ys, yerr=errs, marker=MARKERS[metric], color=COLORS[metric],
                    lw=2, ls=ls, ms=8, capsize=3, elinewidth=1.2,
                    label=f"{metric} (v2)" if metric != "grind" else "grind (v2)",
                    markeredgecolor="black", markeredgewidth=0.6)

    # Overlay v1 recognition (un-gated regime) as a dashed reference, if available.
    if v1_path.exists():
        v1 = {_short(json.loads(l)["model"]): json.loads(l)
              for l in v1_path.read_text().splitlines() if l.strip()}
        v1pts = [(CLAUDE_B[s], v1[s]["recognition"]) for _, s, _ in pts if s in v1]
        if v1pts:
            vx = [x for x, _ in v1pts]
            vy = [v for _, (v, n) in v1pts]
            ax.plot(vx, vy, ls=":", lw=2, marker="o", ms=7, color="#1f77b4",
                    alpha=0.55, markerfacecolor="white",
                    label="recognition (v1, un-gated)")

    ax.set_xscale("log")
    ax.set_xlabel("model parameter size (billions, log scale)")
    ax.axvspan(min(xs) / 1.4, max(xs) * 1.4, color="0.85", alpha=0.4, zorder=0,
               label="proprietary (sizes estimated)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8, rotation=25, ha="right", rotation_mode="anchor")
    ax.minorticks_off()
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("metric value")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.set_title("CREATOR v2 (announced held-out): generalization vs model size\n"
                 "Recognition = P(solve generalizes); Grind = P(shown-correct & not); "
                 "dashed = v1 recognition (un-gated)", fontsize=11)

    outdir = Path("figs/creator")
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "fig_creator_heldout.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
