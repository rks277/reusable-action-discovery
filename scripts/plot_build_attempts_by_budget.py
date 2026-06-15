"""Budget-faceted "when build attempts happen" histogram for a budget sweep.

The built-in fig_attempt_hist in analyze_episode_stats facets columns by n (the
axis the nt_sweep varies). A budget sweep instead holds n/T fixed and varies the
action budget, so this produces the SAME stacked-by-outcome histogram of build
attempts over normalized episode time, but with columns = budget (one row per
model). It reuses analyze_episode_stats' attempt extraction and styling verbatim
so the two figures stay definitionally identical apart from the facet axis.

Usage: PYTHONPATH=. python -m scripts.plot_build_attempts_by_budget runs/<sweep>/episodes.jsonl
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scripts.analyze_episode_stats import (
    FINAL_STRETCH, _OUTCOME_ORDER, _OUTCOME_STYLE, _short,
    build_attempts, load_episodes,
)


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        sys.exit("usage: plot_build_attempts_by_budget <episodes.jsonl|run_dir> ...")
    eps = load_episodes(paths)
    if not eps:
        sys.exit("No episodes loaded.")

    # rows = model, cols = budget (the swept axis); n/T are constant in a sweep.
    cells: dict[tuple, list] = defaultdict(list)
    for ep in eps:
        cells[(ep.model, ep.budget)].append(ep)
    models = sorted({m for m, _b in cells}, key=lambda m: (_short(m), m))
    budgets = sorted({b for _m, b in cells}, key=lambda b: (b is None, b))
    n = eps[0].n
    T = eps[0].n_types

    nr, nc = max(len(models), 1), max(len(budgets), 1)
    fig, axes = plt.subplots(nr, nc, squeeze=False,
                             figsize=(2.6 * nc + 0.6, 2.4 * nr + 0.9),
                             sharex=True, sharey="row")
    bins = [i / 10 for i in range(11)]

    for r, model in enumerate(models):
        for c, b in enumerate(budgets):
            ax = axes[r][c]
            grp = cells.get((model, b), [])
            if not grp:
                ax.set_axis_off()
                continue
            by_outcome: dict[str, list[float]] = {o: [] for o in _OUTCOME_ORDER}
            for ep in grp:
                # x = fraction of the BUDGET consumed at the attempt (not of the
                # episode's own length). Budget caps actions, so i < budget => x
                # in [0, 1); episodes that finish early pile up near 0.
                denom = ep.budget or ep.total_actions or 1
                for i, _x, _yb, o in build_attempts(ep):
                    by_outcome[o].append(i / denom)
            series = [by_outcome[o] for o in _OUTCOME_ORDER]
            ax.hist(series, bins=bins, stacked=True,
                    color=[_OUTCOME_STYLE[o]["color"] for o in _OUTCOME_ORDER])
            ax.axvline(FINAL_STRETCH, color="0.4", ls="--", lw=0.8)
            ax.set_title(f"{_short(model)}  b={b}", fontsize=9)
            if r == nr - 1:
                ax.set_xlabel("fraction of budget used", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{_short(model)}\n# attempts", fontsize=8)
            ax.tick_params(labelsize=7)

    handles = [Line2D([], [], color=_OUTCOME_STYLE[o]["color"], marker="s",
                      linestyle="none", markersize=8,
                      label=_OUTCOME_STYLE[o]["label"]) for o in _OUTCOME_ORDER]
    handles.append(Line2D([], [], color="0.4", ls="--",
                          label=f"{FINAL_STRETCH} of budget"))
    fig.suptitle(f"When build attempts happen (fraction of budget used)  ·  n={n}, T={T}",
                 y=1.0, fontsize=10)
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    first = Path(paths[0])
    outdir = first if first.is_dir() else first.parent
    out = outdir / "fig_build_attempts_hist_by_budget.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
