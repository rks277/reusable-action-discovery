"""Recognition + Solve-rate vs model size for the UN-GATED Qwen ladder.

Un-gated regime (no Curiosity gate): Recognition = P(used_tool) over ALL episodes
(build-vs-grind), so we can't use plot_creator_eval_metrics (which keys R off the
asked subpopulation). Solve = P(all_correct).

  PYTHONPATH=. python -m scripts.creator.plot_qwen_ungated --runs <r1> <r2> ... \
      --outdir figs/creator/qwen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# substring -> (label, size in B). Order: longest/most-specific first. Note "1b"/"7b"
# don't false-match "1.5b"/"70b" (substring needs the digit adjacent to 'b').
SIZE_MAP = [
    ("0.5b", ("0.5B", 0.5)), ("1.5b", ("1.5B", 1.5)), ("1b", ("1B", 1)), ("3b", ("3B", 3)),
    ("7b", ("7B", 7)), ("8b", ("8B", 8)), ("14b", ("14B", 14)), ("32b", ("32B", 32)),
    ("70b", ("70B", 70)), ("72b", ("72B", 72)),
]


def resolve(model: str):
    m = model.lower()
    for key, lp in SIZE_MAP:
        if key in m:
            return lp
    return None


def load(path: str):
    return [json.loads(l) for l in Path(path).read_text().splitlines()
            if l.strip() and not json.loads(l).get("error")]


def metrics(g):
    n = len(g)
    tool = sum(1 for e in g if e.get("used_tool"))
    solve = sum(1 for e in g if e.get("all_correct"))
    R, S = tool / n, solve / n
    return {
        "recognition": (R, (R * (1 - R) / n) ** 0.5),
        "solve rate": (S, (S * (1 - S) / n) ** 0.5),
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--outdir", default="figs/creator/qwen")
    ap.add_argument("--prefix", default="qwen", help="output filename prefix: fig_<prefix>_<metric>.png")
    ap.add_argument("--title", default="Qwen2.5-Instruct un-gated (CREATOR v5 hard, costed, N=20)")
    args = ap.parse_args()

    pts = []  # (size, label, metrics)
    for r in args.runs:
        g = load(r)
        if not g:
            continue
        lp = resolve(g[0]["model"])
        if lp is None:
            print(f"  skip (unmapped model): {g[0]['model']}")
            continue
        pts.append((lp[1], lp[0], metrics(g)))
    pts.sort(key=lambda x: x[0])
    xs = [s for s, _, _ in pts]
    labels = [lbl for _, lbl, _ in pts]
    print("rungs:", ", ".join(f"{l}({s})" for s, l, _ in pts))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for metric in ("recognition", "solve rate"):
        ys = [m[metric][0] for _, _, m in pts]
        es = [m[metric][1] for _, _, m in pts]
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        ax.errorbar(xs, ys, yerr=es, marker="o", color="#1f77b4", lw=2, ms=8,
                    capsize=3, elinewidth=1.1, markeredgecolor="black", markeredgewidth=0.5)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9)
        ax.minorticks_off()
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(metric)
        ax.set_xlabel("model size (B params, log)")
        ax.grid(True, axis="y", ls=":", alpha=0.5)
        ax.set_title(f"{args.title}\n{metric} vs size", fontsize=10)
        out = outdir / f"fig_{args.prefix}_{metric.replace(' ', '_')}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
