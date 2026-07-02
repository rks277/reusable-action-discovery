"""CER decomposition for the v3 dense sweep (Haiku / Sonnet / Opus).

Decomposes the build chain into three conditional stages and shows that the
build-rate inversion lives entirely in RECOGNITION, and that it is a function
of the action budget T (n_types):

    build rate = C * R          success = C * R * E
    C (gather)    = P(picked up both shards)
    R (recognize) = P(built machine | holds both shards)
    E (execute)   = P(won | built)

Figure layout (1 x 4):
    A. C by model       (flat, saturated -> degenerate, no signal)
    B. R by model       (the inversion lives here)
    C. E by model       (rises with capability)
    D. R vs budget T    (one line per model -- the mechanism: capable models'
                         recognition collapses as the budget grows, Haiku's holds)

Usage: PYTHONPATH=. python -m scripts.plot_grid_v3_dense_cer \
           [haiku_dir sonnet_dir opus_dir]
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
from scripts.plot_grid_v3_dense_bars import latest, wilson, model_name

# capability-ordered models (weakest -> strongest)
ORDER = ["Haiku 4.5", "Sonnet 4.6", "Opus 4.8"]
MCOLOR = {"Haiku 4.5": "#4C72B0", "Sonnet 4.6": "#55A868", "Opus 4.8": "#C44E52"}
TS = [2, 3, 4]

# (label, denom predicate, outcome) for the three conditional stages.
STAGES = [
    ("C  gather\nP(both shards)",
     lambda f: f["has_both"] is not None,
     lambda f: f["has_both"]),
    ("R  recognize\nP(built | both)",
     lambda f: f["has_both"] is True,
     lambda f: f["built"]),
    ("E  execute\nP(won | built)",
     lambda f: f["built"],
     lambda f: f["solved"]),
]


def rate(facts, denom, outcome):
    sel = [f for f in facts if denom(f)]
    k = int(sum(1 for f in sel if outcome(f)))
    n = len(sel)
    p = k / n if n else float("nan")
    lo, hi = wilson(k, n)
    return p, lo, hi, n


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    dirs = [Path(a) for a in args] if len(args) == 3 else \
        [latest("haiku"), latest("sonnet"), latest("opus")]

    by_model = {}
    for d in dirs:
        ep = d if d.suffix == ".jsonl" else d / "episodes.jsonl"
        facts = [f for f in (episode_facts(r) for r in load(ep)) if f]
        mid = next((r.get("model", "") for r in load(ep) if r.get("model")), "")
        by_model[model_name(mid)] = facts
    models = [m for m in ORDER if m in by_model]
    print("models:", [(m, len(by_model[m])) for m in models])

    fig, axes = plt.subplots(1, 4, figsize=(19, 5), constrained_layout=True)
    x = np.arange(len(models))

    # Panels A-C: each stage by model.
    for ax, (label, denom, outcome) in zip(axes[:3], STAGES):
        rs, los, his = [], [], []
        for m in models:
            p, lo, hi, n = rate(by_model[m], denom, outcome)
            rs.append(p); los.append(p - lo); his.append(hi - p)
            ax.text(len(rs) - 1, (p if not math.isnan(p) else 0) + 0.02,
                    f"{p:.2f}\nn={n}", ha="center", va="bottom", fontsize=9)
        ax.bar(x, rs, color=[MCOLOR[m] for m in models], width=0.6,
               yerr=[los, his], capsize=5, ecolor="black")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10, rotation=12)
        ax.set_ylim(0, 1.12); ax.set_ylabel("probability")
        ax.grid(axis="y", alpha=0.3)

    # Panel D: R vs budget T, one line per model.
    axd = axes[3]
    for m in models:
        ys, los, his = [], [], []
        for T in TS:
            sub = [f for f in by_model[m] if f["t"] == T]
            p, lo, hi, n = rate(sub, lambda f: f["has_both"] is True,
                                lambda f: f["built"])
            ys.append(p); los.append(p - lo); his.append(hi - p)
        axd.errorbar(TS, ys, yerr=[los, his], capsize=5, capthick=1.8,
                     elinewidth=1.8, lw=2.2, marker="o", markersize=8,
                     markeredgecolor="black", markeredgewidth=0.6,
                     color=MCOLOR[m], label=m)
    axd.set_title("R  recognize  vs budget T", fontsize=12, fontweight="bold")
    axd.set_xticks(TS); axd.set_xlabel("budget  T  (n_types)")
    axd.set_ylim(0, 1.12); axd.set_ylabel("P(built | both shards)")
    axd.grid(alpha=0.3); axd.legend(loc="lower left", fontsize=9)

    fig.suptitle("toolworld v3 dense -- CER decomposition: build = C x R, success = C x R x E "
                 "(gathering C saturated; inversion lives in recognition R, and grows with budget T)",
                 fontsize=13)

    out = Path("runs") / "grid_v3" / "dense" / "fig_grid_v3_dense_cer.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out}")
    print(f"  wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
