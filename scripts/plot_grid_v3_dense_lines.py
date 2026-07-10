"""Cross-model build-chain LINE graphs for the v3 dense sweep, split by family.

Same four statistics as plot_grid_v3_dense_bars, but drawn as lines with one
line per MODEL FAMILY (Anthropic: Haiku<Sonnet<Opus; Qwen2.5: 7B<14B<72B). The
x-axis is the within-family capability rank (1=smallest/weakest -> 3), so the two
families overlay and you can read each family's scaling trend per statistic.

  1. P(picked up both shards)   2. P(built | both shards)
  3. P(won | built)             4. P(won) overall (end-to-end)

Reads all run dirs under runs/grid_v3/dense/ (override with explicit dir args).
Usage: PYTHONPATH=. python -m scripts.plot_grid_v3_dense_lines [dir ...]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plot_grid_sweep_v3 import load
from scripts.plot_grid_sweep_v3_chain import episode_facts
from scripts.plot_grid_v3_dense_bars import PANELS, wilson

DENSE_DIR = Path("runs") / "grid_v3" / "dense"

# (family, size in B params, short label) keyed by substring of the dir name.
# Qwen sizes are exact; Anthropic sizes are SUPPOSED (unpublished) estimates used
# only to place the models on a common size axis.
CLASSIFY = [
    ("haiku",        ("Anthropic", 20.0, "Haiku 4.5")),
    ("sonnet",       ("Anthropic", 150.0, "Sonnet 4.6")),
    ("opus",         ("Anthropic", 600.0, "Opus 4.8")),
    ("Qwen2-5-7B",   ("Qwen2.5", 7.0, "7B")),
    ("Qwen2-5-14B",  ("Qwen2.5", 14.0, "14B")),
    ("Qwen2-5-72B",  ("Qwen2.5", 72.0, "72B")),
    ("Qwen3-5-9B",   ("Qwen3.5", 9.0, "9B")),
    ("Qwen3-5-27B",  ("Qwen3.5", 27.0, "27B")),
    ("gpt-5-mini",   ("OpenAI", 8.0, "gpt-5-mini")),
]
FAMILY_STYLE = {  # colour + marker per family
    "Anthropic": dict(color="#C44E52", marker="o"),
    "Qwen2.5":   dict(color="#4C72B0", marker="s"),
    "Qwen3.5":   dict(color="#9467BD", marker="^"),
    "OpenAI":    dict(color="#2CA02C", marker="D"),
}


def classify(name: str):
    for key, val in CLASSIFY:
        if key in name:
            return val
    return None


def discover(dirs: list[Path]) -> dict:
    """family -> list of (size_b, label, facts) sorted by size."""
    fams: dict[str, list] = {}
    for d in dirs:
        ep = d if d.suffix == ".jsonl" else d / "episodes.jsonl"
        if not ep.exists():
            continue
        info = classify(d.name)
        if not info:
            print(f"  (skipping unclassified {d.name})")
            continue
        fam, size, label = info
        facts = [f for f in (episode_facts(r) for r in load(ep)) if f]
        fams.setdefault(fam, []).append((size, label, facts))
    for fam in fams:
        fams[fam].sort(key=lambda x: x[0])
    return fams


def main():
    args = [Path(a) for a in sys.argv[1:] if not a.startswith("-")]
    dirs = args if args else sorted(p for p in DENSE_DIR.iterdir() if p.is_dir())
    fams = discover(dirs)
    if not fams:
        raise SystemExit(f"no classifiable run dirs found under {DENSE_DIR}")
    for fam, members in fams.items():
        print(f"{fam}: " + ", ".join(f"{lab}(n={len(f)})" for _, lab, f in members))

    # all model sizes present, for log-axis ticks
    all_sizes = sorted({s for ms in fams.values() for s, _, _ in ms})

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    for ax, (title, denom, outcome) in zip(axes, PANELS[:3]):
        for fam, members in fams.items():
            style = FAMILY_STYLE.get(fam, dict(color="gray", marker="^"))
            xs, ys, los, his, labels = [], [], [], [], []
            for size, label, facts in members:
                sel = [f for f in facts if denom(f)]
                k = int(sum(outcome(f) for f in sel)); n = len(sel)
                p = k / n if n else float("nan")
                lo, hi = wilson(k, n)
                xs.append(size); ys.append(p)
                los.append(p - lo); his.append(hi - p); labels.append(label)
            ax.errorbar(xs, ys, yerr=[los, his], capsize=7, capthick=2.0,
                        elinewidth=2.0, lw=2, markersize=9, markeredgecolor="black",
                        markeredgewidth=0.6, label=fam, **style)
            for x, y, lab in zip(xs, ys, labels):
                if not math.isnan(y):
                    ax.annotate(lab, (x, y), textcoords="offset points",
                                xytext=(0, 9), ha="center", fontsize=8,
                                color=style["color"])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xlim(5, 800); ax.set_ylim(0, 1.08)
        ax.set_xticks(all_sizes)
        ax.set_xticklabels([f"{int(s)}B" for s in all_sizes], fontsize=8)
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_xlabel("model size (B params; Anthropic = supposed)")
        ax.set_ylabel("probability")
        ax.grid(alpha=0.3, which="major")
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("toolworld v3 dense sweep -- build chain vs model size "
                 "(line = family, log x; Anthropic sizes are supposed; 95% Wilson CI)",
                 fontsize=14)

    out = DENSE_DIR / "fig_grid_v3_dense_chain_lines.png"
    fig.savefig(out, dpi=130); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out}")
    print(f"  wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
