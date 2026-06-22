"""Line graph: the three build-chain conditionals vs model size.

X axis = models ordered by size (Qwen3.5-4B, Qwen3.5-9B, Haiku, Sonnet, Opus);
ordinal because Anthropic parameter counts are not public. Y axis = probability
(the same per-(N,T)-cell empirical mean, T>=2, annotated on fig_grid_v3_compare).
One line per metric.

Usage: PYTHONPATH=. python -m scripts.plot_grid_v3_size_lines
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plot_grid_sweep_v3 import load
from scripts.plot_grid_sweep_v3_chain import episode_facts, grid

MODELS = [
    ("qwen3-5-4b", "Qwen3.5-4B"),
    ("qwen3-5-9b", "Qwen3.5-9B"),
    ("haiku", "Haiku 4.5"),
    ("sonnet", "Sonnet 4.6"),
    ("opus", "Opus 4.8"),
]

# Curiosity = P(picked up both shards), Recognition = P(built | both shards),
# Efficiency = P(won | built). Solve rate = unconditional P(solved).
METRICS = [
    ("Curiosity", lambda f: 1.0 if f["has_both"] else 0.0),
    ("Recognition",
     lambda f: None if not f["has_both"] else (1.0 if f["built"] else 0.0)),
    ("Efficiency",
     lambda f: None if not f["built"] else (1.0 if f["solved"] else 0.0)),
    ("Grind rate (solved | not built)",
     lambda f: None if f["built"] else (1.0 if f["solved"] else 0.0)),
    ("Solve rate", lambda f: 1.0 if f["solved"] else 0.0),
]


def latest(tag: str) -> Path:
    cands = sorted(p for p in Path("runs").rglob(f"grid_sweep_v3_{tag}_*") if p.is_dir())
    if not cands:
        raise SystemExit(f"no runs/grid_sweep_v3_{tag}_* dir found")
    return cands[-1]


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion. Returns (p, lo, hi).
    Each sampled (N,T) cell holds one episode, so each metric is a proportion
    over its conditional cells; Wilson stays sensible at p=0/1 where normal SE=0."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, center - half, center + half


# Left panel: build-chain conditionals. Right panel: outcomes (grind vs solve).
CHAIN = ["Curiosity", "Recognition", "Efficiency"]
OUTCOME = ["Grind rate (solved | not built)", "Solve rate"]
# Recognition uses the same default-blue (#1f77b4) as the Grind rate line.
COLORS = {"Curiosity": "green", "Recognition": "#1f77b4", "Efficiency": "red"}


def plot_panel(ax, names, facts_by_model, x, labels, title):
    by_name = dict(METRICS)
    for mname in names:
        mfn = by_name[mname]
        ys, lo_err, hi_err = [], [], []
        for _, facts in facts_by_model:
            gr = grid(facts, mfn, t_lo=2)
            n = int((~np.isnan(gr)).sum())
            k = int(np.nansum(gr))            # cells are 0/1 -> successes
            p, lo, hi = wilson(k, n)
            ys.append(p)
            lo_err.append(max(0.0, p - lo) if n else 0.0)
            hi_err.append(max(0.0, hi - p) if n else 0.0)
        line = ax.errorbar(x, ys, yerr=[lo_err, hi_err], marker="o", lw=2,
                           markersize=7, capsize=4, label=mname,
                           color=COLORS.get(mname))
        color = line[0].get_color()
        for xi, yi in zip(x, ys):
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8, color=color)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_xlabel("model (ordered by size; Anthropic params not public)")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title(title, fontsize=11)


def main():
    facts_by_model = []
    for tag, name in MODELS:
        d = latest(tag)
        rows = load(d / "episodes.jsonl")
        facts_by_model.append((name, [f for f in (episode_facts(r) for r in rows) if f]))

    x = list(range(len(MODELS)))
    labels = [n for _, n in MODELS]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True,
                                   sharey=True)
    plot_panel(axL, CHAIN, facts_by_model, x, labels, "build chain conditionals")
    plot_panel(axR, OUTCOME, facts_by_model, x, labels, "outcomes: grind vs solve")
    axL.set_ylabel("probability")

    # Secondary axis on the right panel: raw grind:build ratio of SOLVED episodes
    # (grind_solves / build_solves), counted on the same T>=2 cell basis.
    by_name = dict(METRICS)
    grind_fn, eff_fn = by_name["Grind rate (solved | not built)"], by_name["Efficiency"]
    ratios = []
    for _, facts in facts_by_model:
        g = float(np.nansum(grid(facts, grind_fn, t_lo=2)))   # solved & not built
        b = float(np.nansum(grid(facts, eff_fn, t_lo=2)))     # solved & built
        ratios.append(g / b if b > 0 else np.nan)
    ax2 = axR.twinx()
    ax2.plot(x, ratios, marker="s", ls="--", lw=2, color="black",
             label="grind:build ratio")
    for xi, ri in zip(x, ratios):
        if not np.isnan(ri):
            ax2.annotate(f"{ri:.2f}", (xi, ri), textcoords="offset points",
                         xytext=(0, -14), ha="center", fontsize=8, color="black")
    ax2.set_ylabel("grind:build ratio (solved episodes)")
    ax2.set_ylim(bottom=0)
    # merge legends from both axes
    h1, l1 = axR.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axR.legend().remove()
    axR.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, framealpha=0.9)
    fig.suptitle("toolworld v3 vs model size "
                 "(proportion over sampled N,T cells, T>=2; bars = Wilson 95% CI)",
                 fontsize=13)

    out = Path("runs") / "grid_v3" / "fig_grid_v3_size_lines.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out}\nwrote {out.with_suffix('.pdf')}")

    # Standalone: just the left panel (build-chain conditionals).
    figL, axL2 = plt.subplots(figsize=(9, 6), constrained_layout=True)
    plot_panel(axL2, CHAIN, facts_by_model, x, labels, "build chain conditionals")
    axL2.set_ylabel("probability")
    figL.suptitle("toolworld v3 build chain vs model size "
                  "(proportion over sampled N,T cells, T>=2; bars = Wilson 95% CI)",
                  fontsize=11)
    outL = Path("runs") / "grid_v3" / "fig_grid_v3_size_lines_chain.png"
    figL.savefig(outL, dpi=140); figL.savefig(outL.with_suffix(".pdf"))
    plt.close(figL)
    print(f"wrote {outL}\nwrote {outL.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
