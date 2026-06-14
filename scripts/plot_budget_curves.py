"""Solve-rate-vs-budget and build-rate-vs-budget curves, one line per model.

Reads a multi-budget episodes.jsonl (e.g. from run_sonnet_opus_budget_sweep.py,
which holds n/T fixed and sweeps the action budget) and plots, against budget on a
log2 x-axis (budgets double): (1) solve rate and (2) build rate, with one line per
model. This is the across-budget view that analyze_budget -- single-budget by
design -- can't show in one figure.

Usage: PYTHONPATH=. python -m scripts.plot_budget_curves runs/<sweep_dir>/episodes.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def short(m): return m.replace("claude-", "").replace("-20251001", "")


def main():
    path = Path(sys.argv[1])
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]

    # (model, budget) -> list of per-episode (solved, built)
    cells = defaultdict(lambda: {"solve": [], "built": []})
    for r in rows:
        built = any("fuse into" in o for o in r["obs"])
        c = cells[(r["model"], r["budget"])]
        c["solve"].append(bool(r["solved"]))
        c["built"].append(built)

    models = sorted({m for m, _ in cells}, key=lambda m: ("opus" in m, m))
    budgets = sorted({b for _, b in cells})
    n = rows[0].get("n")
    T = rows[0].get("n_types")

    def curve(model, key):
        return [statistics.mean(cells[(model, b)][key]) for b in budgets]

    for kind, key, color_title in [("solve", "solve", "Solve"),
                                    ("build", "built", "Build")]:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for m in models:
            ax.plot(budgets, curve(m, key), marker="o", label=short(m))
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("action budget (log2)")
        ax.set_ylabel(f"{kind} rate")
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(f"{color_title} rate vs. budget (n={n}, T={T}, "
                     f"{len(cells[(models[0], budgets[0])]['solve'])} reps/cell)")
        ax.grid(True, alpha=0.3)
        ax.legend(title="model")
        fig.tight_layout()
        out = path.parent / f"fig_{kind}_rate_vs_budget.png"
        fig.savefig(out, dpi=150)
        fig.savefig(out.with_suffix(".pdf"))
        plt.close(fig)
        print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
