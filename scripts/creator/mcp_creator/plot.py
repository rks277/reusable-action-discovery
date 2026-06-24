"""Plot MCP CREATOR results across the model-size ladder: (1) solve rate vs size, and
(2) the protocol-adherence funnel (wrote -> tested -> submitted -> solved) vs size.

  PYTHONPATH=. python -m scripts.creator.mcp_creator.plot --runs runs/mcp_creator_<ts>/ ... \
      --outdir figs/creator/mcp --prefix qwen_mcp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZE_MAP = [("0.5b", 0.5), ("1.5b", 1.5), ("1b", 1), ("3b", 3), ("7b", 7), ("8b", 8),
            ("14b", 14), ("32b", 32), ("70b", 70), ("72b", 72)]


def resolve_size(model: str):
    m = model.lower()
    for key, size in SIZE_MAP:
        if key in m:
            return size, key.upper()
    return None


def load(run):
    p = Path(run)
    f = p / "episodes.jsonl" if p.is_dir() else p
    return [json.loads(l) for l in f.read_text().splitlines()
            if l.strip() and not json.loads(l).get("error")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--outdir", default="figs/creator/mcp")
    ap.add_argument("--prefix", default="mcp")
    ap.add_argument("--title", default="MCP CREATOR (explore-then-test, N=20)")
    args = ap.parse_args()

    eps = []
    for r in args.runs:
        eps += load(r)
    by_size: dict = {}
    for e in eps:
        rs = resolve_size(e["model"])
        if rs:
            by_size.setdefault(rs, []).append(e)
    pts = sorted(by_size.items())
    xs = [s for (s, _), _ in pts]
    labels = [lab for (_, lab), _ in pts]
    print("rungs:", ", ".join(f"{lab}({s})" for (s, lab), _ in pts))

    def frac(rs, key):
        return sum(1 for r in rs if r[key]) / len(rs)

    solve = [frac(rs, "all_correct") for _, rs in pts]
    wrote = [frac(rs, "wrote_script") for _, rs in pts]
    tested = [frac(rs, "called_begin_test") for _, rs in pts]
    subm = [frac(rs, "submitted") for _, rs in pts]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # (1) solve rate
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    es = [(v * (1 - v) / len(rs)) ** 0.5 for v, (_, rs) in zip(solve, pts)]
    ax.errorbar(xs, solve, yerr=es, marker="o", lw=2, ms=8, capsize=3, color="#1f77b4",
                markeredgecolor="black", markeredgewidth=0.5)
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.minorticks_off(); ax.set_ylim(-0.02, 1.02); ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.set_ylabel("solve rate (all 20 correct)"); ax.set_xlabel("model size (B, log)")
    ax.set_title(f"{args.title}\nsolve rate vs size", fontsize=10)
    out = outdir / f"fig_{args.prefix}_solve_rate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print("wrote", out)

    # (2) funnel
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for ys, lab, mk in [(wrote, "wrote_script", "o"), (tested, "begin_test", "s"),
                        (subm, "submit_answers", "^"), (solve, "solved (all 20)", "D")]:
        ax.plot(xs, ys, marker=mk, lw=2, ms=7, label=lab, markeredgecolor="black",
                markeredgewidth=0.5)
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.minorticks_off(); ax.set_ylim(-0.02, 1.02); ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.set_ylabel("fraction of episodes"); ax.set_xlabel("model size (B, log)")
    ax.legend(loc="best", fontsize=8); ax.set_title(f"{args.title}\nprotocol funnel vs size", fontsize=10)
    out = outdir / f"fig_{args.prefix}_funnel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    main()
