"""CREATOR v4 (free evaluate-tool, gated) metric-lines figure: C·R·E·Solve vs model size,
one panel per N. Mirrors fig_metric_lines_toolworld styling.

  PYTHONPATH=. python -m scripts.creator.plot_creator_eval
  (defaults to the canonical gated-v4 trio runs; override with --runs haiku=DIR sonnet=DIR opus=DIR)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plot_metric_lines import CLAUDE_B

DEFAULT_RUNS = {
    "haiku": "runs/creator_eval_20260622_110648",
    "sonnet": "runs/creator_eval_20260622_115833",
    "opus": "runs/creator_eval_20260622_123750",
}
NS = [5, 20, 100]
MIN_EP = 100   # skip a model×N cell with fewer than this many episodes (e.g. Sonnet N=100)

COLORS = {"curiosity": "#2ca02c", "recognition": "#1f77b4",
          "efficiency": "#d62728", "solve rate": "#000000", "grind": "#ff7f0e"}
MARKERS = {"curiosity": "s", "recognition": "o", "efficiency": "^", "solve rate": "D", "grind": "v"}
SERIES = ["curiosity", "recognition", "efficiency", "solve rate", "grind"]


def chain(eps, N):
    g = [e for e in eps if e.get("N") == N and not e.get("error")]
    if len(g) < MIN_EP:
        return None
    ask = [e for e in g if e.get("asked")]
    tool = [e for e in ask if e.get("used_tool")]
    eff = [e for e in tool if e.get("all_correct")]
    solve = [e for e in g if e.get("all_correct")]
    n = len(g)
    C = len(ask) / n
    R = len(tool) / len(ask) if ask else float("nan")
    E = len(eff) / len(tool) if tool else float("nan")
    S = len(solve) / n
    cre = (C * R * E) if (R == R and E == E) else 0.0
    return {
        "curiosity": (C, n),
        "recognition": (R, len(ask)),
        "efficiency": (E, len(tool)),
        "solve rate": (S, n),
        "grind": (max(0.0, S - cre), n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", default="figs/creator/fig_creator_eval.png")
    args = ap.parse_args()
    runs = dict(DEFAULT_RUNS)
    for kv in args.runs:
        k, v = kv.split("=", 1)
        runs[k] = v

    data = {m: [json.loads(l) for l in (Path(d) / "episodes.jsonl").read_text().splitlines() if l.strip()]
            for m, d in runs.items()}
    models = [m for m in ["haiku", "sonnet", "opus"] if m in data and m in CLAUDE_B]

    fig, axes = plt.subplots(1, len(NS), figsize=(4.6 * len(NS), 5.0), sharey=True)
    for ax, N in zip(axes, NS):
        pts = []
        for m in models:
            c = chain(data[m], N)
            if c:
                pts.append((CLAUDE_B[m], m, c))
        pts.sort()
        xs = [p for p, _, _ in pts]
        for metric in SERIES:
            ys = [c[metric][0] for _, _, c in pts]
            if metric == "grind":
                errs = [float("nan")] * len(pts)
            else:
                errs = [(v * (1 - v) / nn) ** 0.5 if (nn and v == v) else float("nan")
                        for v, nn in (c[metric] for _, _, c in pts)]
            ls = "--" if metric == "grind" else "-"
            ax.errorbar(xs, ys, yerr=errs, marker=MARKERS[metric], color=COLORS[metric],
                        lw=2, ls=ls, ms=8, capsize=3, elinewidth=1.1, label=metric,
                        markeredgecolor="black", markeredgewidth=0.6)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{m.capitalize()}\n~{p:g}B" for p, m, _ in pts],
                           fontsize=8, rotation=25, ha="right", rotation_mode="anchor")
        ax.minorticks_off()
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, axis="y", ls=":", alpha=0.5)
        ax.set_title(f"N = {N}" + ("" if len(pts) == len(models) else "  (Sonnet partial — omitted)"),
                     fontsize=10)
        ax.set_xlabel("model size (B, log)")
    axes[0].set_ylabel("metric value")
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.95)
    fig.suptitle("CREATOR v4 — free evaluate-tool (gated): C·R·E·Solve vs model size\n"
                 "C=P(ask)  R=P(used tool|ask)  E=P(all correct|tool)  Solve=P(all correct)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
