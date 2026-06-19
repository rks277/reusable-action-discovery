"""BFCL (Berkeley Function Calling Leaderboard) overall accuracy vs. model
parameter size, for the open-weight subset whose size is encoded in the model name.

The BFCL CSV (data/external/bfcl_v4_overall_*.csv, pulled from
gorilla.cs.berkeley.edu/data_overall.csv) carries no parameter column, so size is
parsed from the model name (e.g. "Qwen2.5-7B" -> 7, "Mixtral-8x7B" -> 56,
"Qwen3-30B-A3B" -> 30 total). Proprietary models (Claude/GPT/Gemini) publish no
size and are dropped -- the same reason Claude can't be placed on our own param
axis. This is the published-benchmark analogue of fig_metric_lines_*.

Usage: PYTHONPATH=. python -m scripts.plot_bfcl_scaling \
           [--csv data/external/bfcl_v4_overall_20260619.csv] [--outdir figs/bfcl]
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROP = {"Proprietary"}          # License value marking closed-weight (no public size)
SIZE_RE = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")
MOE_RE = re.compile(r"(\d+)\s*[xX]\s*(\d+(?:\.\d+)?)\s*[bB]")


def parse_size(name: str):
    """-> billions of (total) parameters parsed from a model name, or None.
    Handles 'NxMB' mixtures (N*M) and takes the first plain 'NB' otherwise (which
    is the TOTAL count for MoE names like 30B-A3B; active size is ignored)."""
    m = MOE_RE.search(name)
    if m:
        return int(m.group(1)) * float(m.group(2))
    m = SIZE_RE.search(name)
    return float(m.group(1)) if m else None


# family <- name prefix (order matters: first match wins). nemotron before llama so
# the NVIDIA tune isn't lumped into Meta's Llama ladder.
FAMILY_RULES = [
    ("nemotron", "Nemotron"), ("qwen", "Qwen3"), ("hammer", "Hammer"),
    ("xlam", "xLAM"), ("falcon", "Falcon"), ("gemma", "Gemma"),
    ("arch-agent", "Arch-Agent"), ("granite", "Granite"), ("coalm", "CoALM"),
    ("llama", "Llama"), ("minicpm", "MiniCPM"), ("ministral", "Mistral"),
    ("mistral", "Mistral"), ("mixtral", "Mistral"), ("toolace", "ToolACE"),
    ("bielik", "Bielik"), ("bitagent", "BitAgent"), ("nanbeige", "Nanbeige"),
]
FAMILY_COLORS = {
    "Qwen3": "#d62728", "Llama": "#1f77b4", "xLAM": "#2ca02c", "Hammer": "#ff7f0e",
    "Falcon": "#9467bd", "Gemma": "#8c564b", "Arch-Agent": "#e377c2",
    "Granite": "#17becf", "CoALM": "#bcbd22",
}


def family_of(name: str) -> str:
    low = name.lower()
    for key, fam in FAMILY_RULES:
        if key in low:
            return fam
    return "Other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/external/bfcl_v4_overall_20260619.csv")
    ap.add_argument("--outdir", default="figs/bfcl")
    args = ap.parse_args()

    rows = list(csv.DictReader(Path(args.csv).read_text().splitlines()))
    pts = []  # (params_B, acc_pct, name, is_open)
    for r in rows:
        name = r["Model"].strip()
        acc = float(r["Overall Acc"].rstrip("%"))
        lic = (r.get("License") or "").strip()
        is_open = lic not in PROP and lic != ""
        sz = parse_size(name)
        if sz is not None:
            pts.append((sz, acc, name, is_open))

    open_pts = [p for p in pts if p[3]]
    print(f"{len(rows)} leaderboard rows | {len(pts)} with a name-parsed size | "
          f"{len(open_pts)} open-weight")

    # group open models by family; within a family keep the best config per distinct
    # size (so FC/Prompt/thinking variants collapse to one point per rung)
    from collections import defaultdict
    fams = defaultdict(dict)        # family -> {size: best_acc}
    for sz, acc, name, _ in open_pts:
        fam = family_of(name)
        fams[fam][sz] = max(fams[fam].get(sz, 0.0), acc)

    fig, ax = plt.subplots(figsize=(10, 6.2))

    # faint all-open log-linear fit as background context
    xo = np.array([p[0] for p in open_pts], float)
    yo = np.array([p[1] for p in open_pts], float)
    lx = np.log10(xo)
    b, a = np.polyfit(lx, yo, 1)
    r = np.corrcoef(lx, yo)[0, 1]
    gx = np.linspace(lx.min(), lx.max(), 100)
    ax.plot(10 ** gx, a + b * gx, ls="--", color="0.6", lw=1.6, zorder=1,
            label=f"all-open fit (r={r:.2f}, +{b:.0f} pt/decade)")

    # multi-rung families -> connected ladder; single-size families -> pooled "Other"
    multi = {f: d for f, d in fams.items() if len(d) >= 2 and f != "Other"}
    singles = [(s, acc, f) for f, d in fams.items() if f not in multi
               for s, acc in d.items()]
    for fam in sorted(multi, key=lambda f: -max(multi[f])):   # biggest families first
        d = multi[fam]
        xs = sorted(d); ys = [d[s] for s in xs]
        ax.plot(xs, ys, "-o", color=FAMILY_COLORS.get(fam, "0.3"), lw=2, ms=7,
                markeredgecolor="black", markeredgewidth=0.6, zorder=4,
                label=f"{fam} ({len(xs)} sizes)")
    if singles:
        ax.scatter([s for s, _, _ in singles], [a for _, a, _ in singles],
                   s=42, facecolors="0.8", edgecolors="0.4", linewidths=0.7,
                   zorder=2, label=f"other single-size models (n={len(singles)})")

    ax.set_xscale("log")
    ax.set_xlabel("model parameter size (billions, log scale; parsed from model name)")
    ax.set_ylabel("BFCL V4 Overall Accuracy (%)")
    ax.set_ylim(0, 80)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95, ncol=2)
    ax.set_title("Berkeley Function Calling Leaderboard V4: tool-use accuracy vs. parameter size, by family\n"
                 "(open-weight only — proprietary models have no public size; best config per rung; "
                 "MoE rungs use total params. Source: gorilla.cs.berkeley.edu)", fontsize=10.5)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "fig_bfcl_accuracy_vs_params.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
