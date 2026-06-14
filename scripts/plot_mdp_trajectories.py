"""Overlay real episode trajectories on the MDP policy map (d vs b).

Reconstructs each episode's path through MDP state space: it starts at
(b=B0, d=D), every action spends 1 budget (move left), and every door that
opens drops d by 1 (move down). The action where the machine fuses is marked
(grind->build/exploit transition). Trajectories are drawn over the optimal
policy heatmap so you can see whether the agents follow the grind-early/
build-tail structure the MDP prescribes.

Usage:
  PYTHONPATH=. python -m scripts.plot_mdp_trajectories \
      runs/sonnet_opus_budget_sweep_*/episodes.jsonl --budget 80 --k 5 --out OUT.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from scripts.mdp_toolworld import solve

OPEN_RE = re.compile(r"it fits\.")        # door-open obs: "...: it fits. X opens."


def trajectory(row: dict, B0: int, D: int):
    """Return (xs, ys, build_idx, solved) where xs=budget-left, ys=doors-locked
    along the episode, build_idx = step index at which the machine fused (or
    None). Step i is AFTER the i-th action: b = B0-(i+1)."""
    xs, ys = [B0], [D]               # start state, before any action
    opened, m_idx = 0, None
    for i, o in enumerate(row["obs"]):
        if OPEN_RE.search(o):
            opened += 1
        if m_idx is None and "fuse into" in o:
            m_idx = i
        xs.append(B0 - (i + 1))
        ys.append(D - opened)
    return xs, ys, m_idx, row.get("solved", False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--budget", type=int, default=80)
    ap.add_argument("--k", type=int, default=5, help="episodes per model")
    ap.add_argument("--D", type=int, default=12)
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--out", default="fig_mdp_trajectories.png")
    args = ap.parse_args()

    path = Path(args.path)
    if path.is_dir():
        path = path / "episodes.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error") and r.get("budget") == args.budget]

    picks = {}
    for fam in ("sonnet", "opus"):
        fam_rows = sorted((r for r in rows if fam in r["model"]),
                          key=lambda r: r["relabel_seed"])[:args.k]
        picks[fam] = fam_rows

    # policy heatmap background
    res = solve(args.D, args.T, max(args.budget + 4, 100))
    pol = res["pol"]
    bmax = args.budget + 4
    code = {"fail": 0, "grind": 1, "build": 2, "win": 1, "-": 0}
    M = np.array([[code[pol[d][b]] for b in range(bmax + 1)]
                  for d in range(1, args.D + 1)])
    cmap = ListedColormap(["#e8e8e8", "#bfe3c8", "#f2c5c0"])  # muted grind/build

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.imshow(M, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=2,
              extent=[0, bmax, 0.5, args.D + 0.5], alpha=0.9, zorder=0)

    colors = {"sonnet": "#1f5fb4", "opus": "#e07b00"}
    for fam, frows in picks.items():
        for j, r in enumerate(frows):
            xs, ys, m_idx, solved = trajectory(r, args.budget, args.D)
            jit = (j - (args.k - 1) / 2) * 0.10        # spread overlapping lines
            ysj = [y + jit for y in ys]
            ax.plot(xs, ysj, color=colors[fam], lw=1.4, alpha=0.85, zorder=3,
                    solid_capstyle="round")
            if m_idx is not None:                      # build point (fuse)
                ax.scatter(args.budget - (m_idx + 1), (args.D - _opened_before(r, m_idx)) + jit,
                           marker="*", s=130, color=colors[fam],
                           edgecolors="white", linewidths=0.6, zorder=4)
            # end marker: filled circle if solved (reached d=0), x if not
            ax.scatter(xs[-1], ysj[-1], marker=("o" if solved else "x"), s=42,
                       color=colors[fam], zorder=4,
                       edgecolors="white" if solved else "none", linewidths=0.6)

    ax.set_xlim(0, bmax); ax.set_ylim(0, args.D + 0.6)
    ax.set_xlabel("budget left  b  (actions remaining)")
    ax.set_ylabel("doors still locked  d")
    ax.invert_xaxis()        # budget is spent over time -> time flows left->right
    ax.set_title(f"Episode trajectories on the MDP policy map "
                 f"(start b={args.budget}, d={args.D}; T={args.T})\n"
                 f"green=grind region, red=build region, grey=hopeless")

    handles = [
        Line2D([], [], color=colors["sonnet"], lw=2, label=f"Sonnet (n={len(picks['sonnet'])})"),
        Line2D([], [], color=colors["opus"], lw=2, label=f"Opus (n={len(picks['opus'])})"),
        Line2D([], [], color="0.3", marker="*", lw=0, markersize=11, label="machine built (fuse)"),
        Line2D([], [], color="0.3", marker="o", lw=0, markersize=7, label="solved (d=0)"),
        Line2D([], [], color="0.3", marker="x", lw=0, markersize=7, label="ran out (d>0)"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=150); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out} (+ .pdf)")
    # quick text summary
    for fam, frows in picks.items():
        for r in frows:
            xs, ys, m_idx, solved = trajectory(r, args.budget, args.D)
            print(f"  {fam:6} rep={r['relabel_seed']:>2} built@={m_idx} "
                  f"final_d={ys[-1]} actions={len(r['obs'])} solved={solved}")


def _opened_before(row: dict, idx: int) -> int:
    """Doors opened strictly before action idx (to place the build marker's y)."""
    return sum(1 for o in row["obs"][:idx + 1] if OPEN_RE.search(o))


if __name__ == "__main__":
    main()
