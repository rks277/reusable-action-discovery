"""Economic response surface figure (`docs/economic-response-surface-spec.md` §4, analysis phase).

Two side-by-side B x K heatmaps of the model's FIRST-SIGHT COMMITMENT HAZARD -- R0 (abstract
free-text) vs R2c (code-required claim) -- with the charge-aware EXACT HINDSIGHT NET-OPTIMUM's own
first-sight hazard overlaid in each cell (small "opt*=.." annotation; this is the prior-free
reference from the 2026-07-09 reference repair -- spec §2 -- not the old Dirichlet DP). The story is
the contrast: R0's hazard falls as the commitment charge K rises (it co-moves with the optimum),
while R2c stays pinned near 100% across the whole charge axis -- code-triggered commitment is
economically invariant.

Reads `runs/economic_surface_haiku/analysis.json` (produced by
`scripts.creator.tool_disposition_benchmark.analyze_economic_surface`).

  PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.analyze_economic_surface
  PYTHONPATH=. python -m scripts.plot_economic_response_surface
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FRAMINGS = ("R0", "R2c")
FRAME_TITLE = {"R0": "R0  (abstract free-text keep/pass)",
               "R2c": "R2c  (code-required claim/skip)"}
BUDGETS = (1, 3, 5)          # rows (top = smallest budget)
CHARGES = (0, 10, 24)        # cols: eager -> wait -> never (the repaired-reference figure axis, spec §7)
CHARGE_LABEL = {0: "K=0\n(eager)", 10: "K=10\n(wait)", 24: "K=24\n(never)"}

ANALYSIS = Path("runs/economic_surface_haiku/analysis.json")
OUT = Path("figs/paper/fig_economic_response_surface.png")


def main() -> None:
    summary = json.loads(ANALYSIS.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), constrained_layout=True)
    im = None
    for ax, framing in zip(axes, FRAMINGS):
        model = np.array([[summary[f"{framing}:B{B}:K{K}"]["model_fs_hazard"] or 0.0
                           for K in CHARGES] for B in BUDGETS])
        ref = [[summary[f"{framing}:B{B}:K{K}"]["reference_fs_hazard"] for K in CHARGES]
               for B in BUDGETS]

        im = ax.imshow(model, cmap="RdYlBu_r", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(FRAME_TITLE[framing], fontsize=11)
        ax.set_xticks(range(len(CHARGES)))
        ax.set_xticklabels([CHARGE_LABEL[k] for k in CHARGES], fontsize=9)
        ax.set_yticks(range(len(BUDGETS)))
        ax.set_yticklabels([f"B={b}" for b in BUDGETS], fontsize=10)
        ax.set_xlabel("commitment charge", fontsize=10)
        if framing == "R0":
            ax.set_ylabel("budget", fontsize=10)

        for i, B in enumerate(BUDGETS):
            for j, K in enumerate(CHARGES):
                m = model[i, j]
                r = ref[i][j]
                rtxt = "\u2013" if r is None else f"{r:.0%}"
                txt = f"model {m:.0%}\nopt* {rtxt}"
                # white text on the dark (high-hazard) end for legibility
                color = "white" if m > 0.62 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9.5, color=color)

    cbar = fig.colorbar(im, ax=axes, fraction=0.045, pad=0.02)
    cbar.set_label("model first-sight commitment hazard", fontsize=10)
    fig.suptitle("Economic response surface: does first-sight commitment track the charge-aware "
                 "hindsight optimum (opt*)?", fontsize=12.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    fig.savefig(OUT.with_suffix(".pdf"))
    print(f"wrote {OUT} and {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
