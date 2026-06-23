"""TEMP variant of plot_creator_eval_metrics: x-axis = Artificial Analysis Intelligence
Index v4.1 (real verified capability) instead of the made-up ordinal SIZE_MAP.

Values from scripts/creator/aa_intelligence_index.json (scraped 2026-06-22). The 3 Qwen2.5
sizes AA doesn't track are placed by GPQA-Diamond scaled to 72B's anchor (GPQA-D 49 -> AA 10),
so they form a floor cluster below Haiku. Caption flags this.

  PYTHONPATH=. python -m scripts.creator.plot_creator_eval_metrics_aa --runs <f1> ... \
      --outdir figs/creator/v5_metrics_tmp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# substring -> (label, AA Intelligence Index v4.1). Order: specific GPT keys before bare gpt-5.
# Qwen 3B/7B/14B = GPQA-Diamond * (10/49) anchor (72B real AA=10); flag in caption.
AA_MAP = [
    ("3b", ("Qwen-3B", 6.2)), ("7b", ("Qwen-7B", 7.4)), ("14b", ("Qwen-14B", 9.3)),
    ("72b", ("Qwen-72B", 10.0)),
    ("haiku", ("Haiku", 30.0)), ("sonnet", ("Sonnet", 47.0)), ("opus", ("Opus", 56.0)),
    ("gpt-5.4-nano", ("GPT-5.4-nano", 38.0)), ("gpt-5.4-mini", ("GPT-5.4-mini", 40.0)),
    ("gpt-5-nano", ("GPT-5-nano", 30.0)), ("gpt-5-mini", ("GPT-5-mini", 33.0)),
    ("gpt-5.5", ("GPT-5.5", 53.0)), ("gpt-5.4", ("GPT-5.4", 51.0)), ("gpt-5", ("GPT-5", 36.0)),
    ("gemini-2.5-flash-lite", ("Gemini-2.5-flash-lite", 7.0)),
    ("gemini-2.5-flash", ("Gemini-2.5-flash", 14.0)), ("gemini-2.5-pro", ("Gemini-2.5-pro", 27.0)),
]
FAM_COLORS = {"Qwen": "#1f77b4", "Claude": "#d62728", "GPT-5": "#2ca02c",
              "GPT-5.4": "#9467bd", "Gemini": "#ff7f0e", "other": "#777777"}
FAM_MARKERS = {"Qwen": "o", "Claude": "s", "GPT-5": "^", "GPT-5.4": "D", "Gemini": "v", "other": "x"}
FAM_ORDER = ["Qwen", "Claude", "GPT-5", "GPT-5.4", "Gemini", "other"]
METRICS = ["curiosity", "recognition", "efficiency", "solve rate", "grind"]


def resolve(model):
    m = model.lower()
    for key, lp in AA_MAP:
        if key in m:
            return lp
    return None


def family(label):
    l = label.lower()
    if l.startswith("qwen"):
        return "Qwen"
    if any(k in l for k in ("haiku", "sonnet", "opus")):
        return "Claude"
    if "gemini" in l:
        return "Gemini"
    if "gpt" in l:
        return "GPT-5.4" if "5.4" in l else "GPT-5"
    return "other"


def chain(g):
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
    return {"curiosity": (C, n), "recognition": (R, len(ask)),
            "efficiency": (E, len(tool)), "solve rate": (S, n),
            "grind": (max(0.0, S - cre), n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--outdir", default="figs/creator/v5_metrics_tmp")
    ap.add_argument("--title", default="CREATOR v5 (costed+gated, hard N=20)")
    args = ap.parse_args()

    eps = []
    for f in args.runs:
        eps += [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
    me = [e for e in eps if not e.get("error")]

    groups = {}
    for e in me:
        lp = resolve(e["model"])
        if lp:
            groups.setdefault(lp, []).append(e)
    pts = sorted(((size, label, chain(g)) for (label, size), g in groups.items()), key=lambda x: x[0])
    all_xs = [p for p, _, _ in pts]
    labels = {p: lbl for p, lbl, _ in pts}
    fam_pts = {}
    for size, label, c in pts:
        fam_pts.setdefault(family(label), []).append((size, label, c))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        ls = "--" if metric == "grind" else "-"
        for fam in FAM_ORDER:
            if fam not in fam_pts:
                continue
            fp = sorted(fam_pts[fam])
            xs = [s for s, _, _ in fp]
            ys = [c[metric][0] for _, _, c in fp]
            errs = [0] * len(fp) if metric == "grind" else \
                [(v * (1 - v) / nn) ** 0.5 if (nn and v == v) else 0
                 for v, nn in (c[metric] for _, _, c in fp)]
            ax.errorbar(xs, ys, yerr=errs, marker=FAM_MARKERS[fam], color=FAM_COLORS[fam],
                        lw=2, ls=ls, ms=8, capsize=3, elinewidth=1.1, label=fam,
                        markeredgecolor="black", markeredgewidth=0.5)
        ax.set_xticks(all_xs)
        ax.set_xticklabels([f"{labels[x]}\n({x:g})" for x in all_xs], fontsize=6, rotation=40,
                           ha="right", rotation_mode="anchor")
        ax.minorticks_off()
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(metric)
        ax.set_xlabel("Artificial Analysis Intelligence Index v4.1 (linear; Qwen<=10 = GPQA-D anchored)")
        ax.grid(True, axis="y", ls=":", alpha=0.5)
        ax.legend(loc="best", fontsize=8, framealpha=0.95, title="family")
        ax.set_title(f"{args.title} — {metric} vs AA Intelligence Index", fontsize=10)
        out = outdir / f"fig_v5_aa_{metric.replace(' ', '_')}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
