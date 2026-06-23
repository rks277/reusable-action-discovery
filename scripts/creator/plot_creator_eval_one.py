"""C·R·E·Solve·Grind vs model size for one v5 condition, across one or more run files.
Resolves each episode's model to (label, ~param-size) so Qwen-7B/14B sit on the same
capability axis as the Claude trio.

  PYTHONPATH=. python -m scripts.creator.plot_creator_eval_one \
      --runs runs/creator_eval_hard_<claude>/episodes.jsonl \
             runs/creator_eval_hard_<qwen7b>/episodes.jsonl \
             runs/creator_eval_hard_<qwen14b>/episodes.jsonl \
      --title "v5 — costed tool + gated, hard N=20" --out figs/creator/fig_creator_v5.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# model-name substring -> (display label, approx params in B) for the size axis
SIZE_MAP = [
    ("3b", ("Qwen-3B", 3)), ("7b", ("Qwen-7B", 7)), ("14b", ("Qwen-14B", 14)), ("32b", ("Qwen-32B", 32)), ("72b", ("Qwen-72B", 72)),
    ("haiku", ("Haiku", 40)), ("sonnet", ("Sonnet", 300)), ("opus", ("Opus", 2000)),
    # GPT sizes are UNKNOWN (proprietary) — these are ordinal placeholders for x-ordering only.
    # Order matters: 'gpt-5-mini' and 'gpt-5.5' must precede the bare 'gpt-5' substring.
    ("gpt-5.4-nano", ("GPT-5.4-nano", 120)), ("gpt-5.4-mini", ("GPT-5.4-mini", 280)),
    ("gpt-5-nano", ("GPT-5-nano", 100)), ("gpt-5-mini", ("GPT-5-mini", 250)),
    ("gpt-5.5", ("GPT-5.5", 3000)), ("gpt-5.4", ("GPT-5.4", 1700)), ("gpt-5", ("GPT-5", 1500)),
]

COLORS = {"curiosity": "#2ca02c", "recognition": "#1f77b4",
          "efficiency": "#d62728", "solve rate": "#000000", "grind": "#ff7f0e"}
MARKERS = {"curiosity": "s", "recognition": "o", "efficiency": "^", "solve rate": "D", "grind": "v"}
SERIES = ["curiosity", "recognition", "efficiency", "solve rate", "grind"]


def resolve(model: str):
    m = model.lower()
    for key, lp in SIZE_MAP:
        if key in m:
            return lp
    return None


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
    ap.add_argument("--title", default="v5 — costed + gated, hard N=20")
    ap.add_argument("--out", default="figs/creator/fig_creator_v5.png")
    args = ap.parse_args()

    eps = []
    for f in args.runs:
        eps += [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
    me = [e for e in eps if not e.get("error")]

    groups = {}  # (label,size) -> episodes
    for e in me:
        lp = resolve(e["model"])
        if lp:
            groups.setdefault(lp, []).append(e)
    pts = sorted(((size, label, chain(g)) for (label, size), g in groups.items()),
                 key=lambda x: x[0])
    xs = [p for p, _, _ in pts]

    def family(label):
        l = label.lower()
        if l.startswith("qwen"):
            return "Qwen"
        if any(k in l for k in ("haiku", "sonnet", "opus")):
            return "Claude"
        if "gpt" in l:
            return "GPT-5.4" if "5.4" in l else "GPT-5"
        return "other"

    FAM_ORDER = ["Qwen", "Claude", "GPT-5", "GPT-5.4", "other"]
    fam_pts = {}  # family -> [(size,label,chain)] sorted by size
    for size, label, c in pts:
        fam_pts.setdefault(family(label), []).append((size, label, c))
    fams = [f for f in FAM_ORDER if f in fam_pts]

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for metric in SERIES:
        ls = "--" if metric == "grind" else "-"
        for fi, fam in enumerate(fams):  # one line segment per family (no cross-family edges)
            fp = fam_pts[fam]
            fxs = [p for p, _, _ in fp]
            ys = [c[metric][0] for _, _, c in fp]
            if metric == "grind":
                errs = [float("nan")] * len(fp)
            else:
                errs = [(v * (1 - v) / nn) ** 0.5 if (nn and v == v) else float("nan")
                        for v, nn in (c[metric] for _, _, c in fp)]
            ax.errorbar(fxs, ys, yerr=errs, marker=MARKERS[metric], color=COLORS[metric],
                        lw=2, ls=ls, ms=9, capsize=3, elinewidth=1.1,
                        label=metric if fi == 0 else None,   # one legend entry per metric
                        markeredgecolor="black", markeredgewidth=0.6)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{lbl}" for p, lbl, _ in pts], fontsize=8,
                       rotation=30, ha="right", rotation_mode="anchor")
    ax.minorticks_off()
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("metric value")
    ax.set_xlabel("model size (B, log)")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95, ncol=2)
    ax.set_title(f"CREATOR {args.title}: C·R·E·Solve·Grind vs model size\n"
                 "C=P(ask)  R=P(used tool|ask)  E=P(all correct|tool)  Solve=P(all correct)  Grind=Solve−C·R·E",
                 fontsize=9)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf); points: {[(l,p) for p,l,_ in pts]}")


if __name__ == "__main__":
    main()
