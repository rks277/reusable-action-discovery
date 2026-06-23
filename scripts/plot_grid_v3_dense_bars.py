"""Cross-model averaged build-chain bars for the v3 dense sweep.

Four panels, model on the x-axis (Haiku / Sonnet / Opus), each panel one
statistic averaged over all dense-sweep episodes for that model:

  1. P(picked up both shards)         -- over episodes where a recipe pair exists
  2. P(built | both shards)           -- conditioned on holding both shards
  3. P(won | built)                   -- conditioned on having built the machine
  4. P(won) overall (end-to-end)      -- unconditional success rate

Bars carry 95% Wilson confidence intervals and are annotated with the rate and
the denominator n (the conditioning population, which shrinks down the chain).

Usage: PYTHONPATH=. python -m scripts.plot_grid_v3_dense_bars \
           [haiku_dir sonnet_dir opus_dir]
       (defaults to the latest grid_sweep_v3_{haiku,sonnet,opus}_* run dirs)
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

# (title, predicate to select the denominator, outcome) -- each over episode facts.
# denom=None means "all episodes"; outcome maps a fact to 0/1.
PANELS = [
    ("P(picked up both shards)",
     lambda f: f["has_both"] is not None,
     lambda f: 1.0 if f["has_both"] else 0.0),
    ("P(built | both shards)",
     lambda f: f["has_both"] is True,
     lambda f: 1.0 if f["built"] else 0.0),
    ("P(won | built)",
     lambda f: f["built"],
     lambda f: 1.0 if f["solved"] else 0.0),
    ("P(won) overall (end-to-end)",
     lambda f: True,
     lambda f: 1.0 if f["solved"] else 0.0),
]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]


def latest(tag: str) -> Path:
    cands = sorted(p for p in Path("runs").rglob(f"grid_sweep_v3_{tag}_*")
                   if p.is_dir())
    if not cands:
        raise SystemExit(f"no runs/grid_sweep_v3_{tag}_* dir found")
    return cands[-1]


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - half) / d, (c + half) / d)


def model_name(mid: str) -> str:
    return ("Haiku 4.5" if "haiku" in mid else "Sonnet 4.6" if "sonnet" in mid
            else "Opus 4.8" if "opus" in mid else mid)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    dirs = [Path(a) for a in args] if len(args) == 3 else \
        [latest("haiku"), latest("sonnet"), latest("opus")]

    models = []
    for d in dirs:
        ep = d if d.suffix == ".jsonl" else d / "episodes.jsonl"
        facts = [f for f in (episode_facts(r) for r in load(ep)) if f]
        mid = next((r.get("model", "") for r in load(ep) if r.get("model")), "")
        models.append((model_name(mid), facts))
    print("models:", [(m, len(f)) for m, f in models])

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    x = np.arange(len(models))
    for ax, color, (title, denom, outcome) in zip(axes, COLORS, PANELS):
        rates, los, his, labels = [], [], [], []
        for name, facts in models:
            sel = [f for f in facts if denom(f)]
            k = int(sum(outcome(f) for f in sel))
            n = len(sel)
            p = k / n if n else float("nan")
            lo, hi = wilson(k, n)
            rates.append(p); los.append(p - lo); his.append(hi - p)
            labels.append(f"{p:.2f}\nn={n}" if n else "n=0")
        ax.bar(x, rates, color=color, width=0.6,
               yerr=[los, his], capsize=5, ecolor="black")
        for xi, (r, lab) in enumerate(zip(rates, labels)):
            y = (r if not math.isnan(r) else 0) + 0.02
            ax.text(xi, y, lab, ha="center", va="bottom", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m for m, _ in models], fontsize=11)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("probability")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("toolworld v3 dense sweep (N in {9,10,11} x T in {2,3,4}, 5 reps/cell) "
                 "-- build chain by model (95% Wilson CI)", fontsize=14)

    out = Path("runs") / "grid_v3" / "fig_grid_v3_dense_chain_bars.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out}")
    print(f"  wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
