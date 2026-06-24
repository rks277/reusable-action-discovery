"""One figure PER METRIC (C, R, E, Solve, Grind) for the v5 cross-family ladder.
Within a single-metric figure, color is free to encode FAMILY (Qwen / Claude / GPT-5 /
GPT-5.4), with within-family edges — so lines are distinguishable (unlike the combined fig).

  PYTHONPATH=. python -m scripts.creator.plot_creator_eval_metrics --runs <f1> <f2> ... \
      --outdir figs/creator/v5_metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# substring -> (label, ordinal size). Order matters (specific GPT keys before bare gpt-5).
SIZE_MAP = [
    ("3b", ("Qwen-3B", 3)), ("7b", ("Qwen-7B", 7)), ("14b", ("Qwen-14B", 14)), ("32b", ("Qwen-32B", 32)),
    ("72b", ("Qwen-72B", 72)),
    ("haiku", ("Haiku", 22)), ("sonnet", ("Sonnet", 600)), ("opus", ("Opus", 4000)),
    # ordinal placeholders, spaced out so labels don't pile up (real sizes unknown).
    ("gpt-5.4-nano", ("GPT-5.4-nano", 120)), ("gpt-5.4-mini", ("GPT-5.4-mini", 220)),
    ("gpt-5-nano", ("GPT-5-nano", 100)), ("gpt-5-mini", ("GPT-5-mini", 165)),
    ("gpt-5.5", ("GPT-5.5", 6500)), ("gpt-5.4", ("GPT-5.4", 2400)), ("gpt-5", ("GPT-5", 1500)),
    # Gemini at REAL estimated sizes (flash-lite 0.7-7B, flash 30-32B, pro 288B).
    # flash-lite key before flash. Gemini-2.5-pro placed just below GLM-5.2 (40) by request.
    ("gemini-2.5-flash-lite", ("Gemini-2.5-flash-lite", 4)),
    ("gemini-2.5-flash", ("Gemini-2.5-flash", 31)), ("gemini-2.5-pro", ("Gemini-2.5-pro", 38)),
    # Gemini 3.5-flash placed at ESTIMATED ACTIVE params (sparse MoE, <20B active) — its own
    # series so it isn't mis-read as a within-2.5 trend. Lands in the Qwen mid-cluster.
    ("gemini-3.5-flash", ("Gemini-3.5-flash", 17)),
    # GLM-5.2 (open MoE) at ~40B ACTIVE params; own series. Frontier-capability (AA ~51) open model.
    ("glm-5.2", ("GLM-5.2", 40)),
    # Mistral-Large-2: DENSE 123B (params = active) — a true large-dense anchor; own series.
    ("mistral-large", ("Mistral-Large-2", 123)),
]
# GPT-5 and GPT-5.4 are SEPARATE series with distinct colors.
FAM_COLORS = {"Qwen": "#1f77b4", "Claude": "#d62728", "GPT-5": "#2ca02c",
              "GPT-5.4": "#9467bd", "Gemini": "#ff7f0e", "Gemini-3.5": "#8c564b",
              "GLM": "#17becf", "Mistral": "#e377c2", "other": "#777777"}
FAM_MARKERS = {"Qwen": "o", "Claude": "s", "GPT-5": "^", "GPT-5.4": "D", "Gemini": "v",
               "Gemini-3.5": "*", "GLM": "h", "Mistral": "P", "other": "x"}
FAM_ORDER = ["Qwen", "Claude", "GPT-5", "GPT-5.4", "Gemini", "Gemini-3.5", "GLM", "Mistral", "other"]
METRICS = ["curiosity", "recognition", "efficiency", "solve rate", "grind"]


def resolve(model):
    m = model.lower()
    for key, lp in SIZE_MAP:
        if key in m:
            return lp
    return None


def family(label):
    l = label.lower()
    if l.startswith("qwen"):
        return "Qwen"
    if any(k in l for k in ("haiku", "sonnet", "opus")):
        return "Claude"
    if "glm" in l:
        return "GLM"
    if "mistral" in l or "mixtral" in l:
        return "Mistral"
    if "gemini" in l:
        return "Gemini-3.5" if "3.5" in l else "Gemini"
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
    ap.add_argument("--outdir", default="figs/creator/v5_metrics")
    ap.add_argument("--title", default="CREATOR v5 (costed+gated, hard N=20)")
    args = ap.parse_args()

    eps = []
    for f in args.runs:
        eps += [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
    me = [e for e in eps if not e.get("error")]

    EXCLUDE_LABELS = {"GPT-5-nano"}
    groups = {}
    for e in me:
        lp = resolve(e["model"])
        if lp and lp[0] not in EXCLUDE_LABELS:
            groups.setdefault(lp, []).append(e)
    pts = sorted(((size, label, chain(g)) for (label, size), g in groups.items()), key=lambda x: x[0])
    all_xs = [p for p, _, _ in pts]
    labels = {p: lbl for p, lbl, _ in pts}
    from matplotlib.lines import Line2D
    fam_pts = {}
    for size, label, c in pts:
        fam_pts.setdefault(family(label), []).append((size, label, c))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
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
            ms = 15 if fam == "Gemini-3.5" else (11 if fam == "GLM" else 8)  # enlarge sparse markers
            ax.errorbar(xs, ys, yerr=errs, marker=FAM_MARKERS[fam], color=FAM_COLORS[fam],
                        lw=2, ls=ls, ms=ms, capsize=3, elinewidth=1.1, label=fam,
                        markeredgecolor="black", markeredgewidth=0.5)
        ax.set_xscale("log")
        ax.set_xticks(all_xs)
        ax.set_xticklabels([labels[x] for x in all_xs], fontsize=7, rotation=40, ha="right",
                           rotation_mode="anchor")
        ax.minorticks_off()
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(metric)
        ax.set_xlabel("model size (B, log; Gemini-2.5=total est., 3.5-flash=ACTIVE est., "
                      "Claude/GPT=ordinal)")
        ax.grid(True, axis="y", ls=":", alpha=0.5)
        ax.legend(loc="best", fontsize=8, framealpha=0.95, title="family")
        ax.set_title(f"{args.title} — {metric} vs model size", fontsize=10)
        out = outdir / f"fig_v5_{metric.replace(' ', '_')}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
