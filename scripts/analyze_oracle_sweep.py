"""Analyze an oracle-discovery sweep: score every all-decoy rollout into its n
oracle-assignments and plot win-rate vs n and vs budget.

Two figures (one row per model):
  1. win-rate vs N (number of tools) at the recorded budget -- more decoys => lower
     chance of hitting the oracle in time (unless the model solves manually).
  2. win-rate vs BUDGET (re-scoring the same rollouts across a budget grid for free) --
     the whole point of storing a token-only meter.

Also prints a manual-vs-tool win breakdown and the run's $ cost (reusing
analyze_beach_sweep.cost_of).

Usage: python scripts/analyze_oracle_sweep.py runs/oracle/<run>/episodes.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.oracle_counterfactual import score_episode          # noqa: E402
from scripts.analyze_beach_sweep import cost_of                  # noqa: E402


def short(model: str) -> str:
    s = model.lower()
    for k in ("haiku", "sonnet", "opus", "qwen", "gpt", "gemini"):
        if k in s:
            return k
    return model


def _winrate(rows, budget=None):
    w = t = 0
    for r in rows:
        for s in score_episode(r, budget=budget):
            w += int(s["win_i"]); t += 1
    return (w / t) if t else float("nan")


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not pos:
        raise SystemExit("usage: python scripts/analyze_oracle_sweep.py "
                         "runs/oracle/<run>/episodes.jsonl")
    path = Path(pos[0])
    run_dir = path.parent
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    if not rows:
        raise SystemExit("no non-error rollouts found")

    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    models = sorted(by_model)
    Ns = sorted({r["n"] for r in rows})
    budgets = sorted({r["budget"] for r in rows})
    bgrid = np.linspace(min(budgets) * 0.4, max(budgets) * 1.8, 25)

    # ---- console summary ----
    cost = sum(cost_of(r.get("usage") or {}, r["model"]) for r in rows
               if cost_of(r.get("usage") or {}, r["model"]))
    print(f"{len(rows)} rollouts, {len(models)} model(s); est ${cost:.2f}\n")
    print(f"{'model':10s} {'N':>3s} {'win_rate':>9s} {'via_tool':>9s} {'via_manual':>11s}")
    for m in models:
        for n in Ns:
            sub = [r for r in by_model[m] if r["n"] == n]
            if not sub:
                continue
            scored = [s for r in sub for s in score_episode(r)]
            tot = len(scored)
            wins = [s for s in scored if s["win_i"]]
            via_tool = sum(1 for s in wins if s["win_via"] == "tool")
            via_man = sum(1 for s in wins if s["win_via"] == "manual")
            print(f"{short(m):10s} {n:>3d} {len(wins)/tot:>9.3f} "
                  f"{via_tool/tot:>9.3f} {via_man/tot:>11.3f}")

    # ---- figure 1: win-rate vs N ----
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.4),
                             squeeze=False, constrained_layout=True)
    for j, m in enumerate(models):
        ax = axes[0][j]
        ys = [_winrate([r for r in by_model[m] if r["n"] == n]) for n in Ns]
        ax.plot(Ns, ys, "o-", color="#4c78a8")
        ax.set_title(f"{short(m)} — win-rate vs N", fontsize=11)
        ax.set_xlabel("N (number of tools)"); ax.set_ylabel("counterfactual win-rate")
        ax.set_ylim(-0.02, 1.02); ax.set_xticks(Ns); ax.grid(alpha=0.3)
    out1 = run_dir / "fig_oracle_winrate_vs_n.png"
    fig.savefig(out1, dpi=140); plt.close(fig)

    # ---- figure 2: win-rate vs budget (free re-scoring) ----
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.4),
                             squeeze=False, constrained_layout=True)
    for j, m in enumerate(models):
        ax = axes[0][j]
        for n in Ns:
            sub = [r for r in by_model[m] if r["n"] == n]
            ys = [_winrate(sub, budget=int(b)) for b in bgrid]
            ax.plot(bgrid, ys, "-", label=f"N={n}")
        ax.set_title(f"{short(m)} — win-rate vs budget", fontsize=11)
        ax.set_xlabel("token budget"); ax.set_ylabel("counterfactual win-rate")
        ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    out2 = run_dir / "fig_oracle_winrate_vs_budget.png"
    fig.savefig(out2, dpi=140); plt.close(fig)

    print(f"\nwrote {out1}\nwrote {out2}")


if __name__ == "__main__":
    main()
