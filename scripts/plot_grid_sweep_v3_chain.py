"""Build-chain conditional-probability heatmaps for the v3 grid sweep.

Replays each episode (deterministic, no API) to recover three facts, then renders
Gaussian-pooled, cell-by-cell (N, T) heatmaps:

  1. P(picked up both required shards)  -- did the agent's INVENTORY ever hold
     both recipe types? combine() never consumes byproducts, so the final
     inventory == everything ever picked up; "has both" = final inventory holds
     recipe[0] and recipe[1]. Defined only where a recipe pair exists (T >= 2).
  2. P(built tool | has both shards)    -- among episodes that picked up both
     required shards, did the machine get built? (NaN where the agent never got
     both -> excluded from the pooling.)
  3. P(won | built tool)                -- among episodes that built, did all
     doors get opened within budget? (NaN where never built.)

A FOURTH plot is written to a SEPARATE file: P(solved) over the full grid.

All panels use normalized Gaussian pooling (Nadaraya-Watson): each cell is the
Gaussian-distance-weighted mean of the sampled episodes whose metric is defined;
NaN (condition-not-met) cells carry zero weight. Rendered nearest-neighbour so
cells stay discrete. White dots mark cells that actually contributed.

Usage: PYTHONPATH=. python -m scripts.plot_grid_sweep_v3_chain [runs/<dir>] [--sigma S]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from scripts.plot_grid_sweep_v3 import (load, latest_run, gaussian_fill,
                                        draw_boundary)
from scripts.replay_toolworld import replay

N_LO, N_HI = 0, 20
T_LO, T_HI = 0, 10


def episode_facts(row: dict) -> dict | None:
    """Replay one episode -> {n, t, has_both, built, solved}. None if errored."""
    if row.get("error"):
        return None
    s, _ = replay(row)
    recipe = (row.get("labels") or {}).get("recipe", []) or []
    built = bool(getattr(s, "has_machine", False))
    solved = bool(s.solved())
    # machine can only be operated AFTER it is built; machine_used_on records every
    # door the machine was successfully operated on (each yields that door's key).
    used_after = built and len(getattr(s, "machine_used_on", set())) >= 1
    has_both = None
    if len(recipe) >= 2:
        has_both = (s.byproducts.get(recipe[0], 0) >= 1
                    and s.byproducts.get(recipe[1], 0) >= 1)
    return {"n": row["n"], "t": row["n_types"], "has_both": has_both,
            "built": built, "solved": solved, "used_after": used_after}


def grid(facts: list[dict], fn, t_lo: int) -> np.ndarray:
    """N x T array of fn(fact) (float or np.nan). T columns start at t_lo."""
    g = np.full((N_HI - N_LO + 1, T_HI - t_lo + 1), np.nan)
    for f in facts:
        n, t = f["n"], f["t"]
        if N_LO <= n <= N_HI and t_lo <= t <= T_HI:
            v = fn(f)
            if v is not None:
                g[n - N_LO, t - t_lo] = float(v)
    return g


def panel(ax, fig, gr: np.ndarray, sigma: float, title: str, t_lo: int):
    cm = plt.get_cmap("viridis").copy()
    filled = gaussian_fill(gr, sigma)
    extent = [t_lo - 0.5, T_HI + 0.5, N_LO - 0.5, N_HI + 0.5]
    im = ax.imshow(filled, origin="lower", aspect="auto", cmap=cm, extent=extent,
                   vmin=0, vmax=1, interpolation="nearest")
    # dots where the metric was DEFINED (the evidence behind each conditional)
    ys, xs = np.where(~np.isnan(gr))
    ax.scatter(xs + t_lo, ys + N_LO, s=8, c="white", edgecolors="black",
               linewidths=0.3, alpha=0.7)
    draw_boundary(ax, t_lo)
    ax.set_xlim(t_lo - 0.5, T_HI + 0.5)
    ax.set_ylim(N_LO - 0.5, N_HI + 0.5)
    # empirical mean over the cells where the metric is DEFINED (conditional rate)
    m = np.nanmean(gr) if np.any(~np.isnan(gr)) else float("nan")
    ax.set_title(f"{title}  (mean {m:.2f})", fontsize=11)
    ax.set_xlabel("T (byproduct types)")
    ax.set_ylabel("N (doors)")
    ax.set_xticks(range(t_lo, T_HI + 1))
    ax.set_yticks(range(N_LO, N_HI + 1))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main():
    argv = sys.argv[1:]
    sigma = 1.2
    if "--sigma" in argv:
        i = argv.index("--sigma"); sigma = float(argv[i + 1])
        del argv[i:i + 2]
    pos = [a for a in argv if not a.startswith("-")]
    run_dir = Path(pos[0]) if pos else latest_run()
    ep = run_dir if run_dir.suffix == ".jsonl" else run_dir / "episodes.jsonl"
    rows = load(ep)
    mid = rows[0].get("model", "") if rows else ""
    model = ("Sonnet 4.6" if "sonnet" in mid else "Haiku 4.5" if "haiku" in mid
             else "Opus 4.8" if "opus" in mid else mid or "unknown")
    facts = [f for f in (episode_facts(r) for r in rows) if f]
    print(f"Replayed {len(facts)} episodes from {ep} (model: {model})")

    # --- Figure A: the 3 build-chain conditionals (T >= 2, where a recipe exists)
    g1 = grid(facts, lambda f: 1.0 if f["has_both"] else 0.0, t_lo=2)
    g2 = grid(facts, lambda f: (None if not f["has_both"]
                                else (1.0 if f["built"] else 0.0)), t_lo=2)
    g3 = grid(facts, lambda f: (None if not f["built"]
                                else (1.0 if f["solved"] else 0.0)), t_lo=2)
    figA, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    panel(axes[0], figA, g1, sigma, "P(picked up both required shards)", 2)
    panel(axes[1], figA, g2, sigma, "P(built tool | has both shards)", 2)
    panel(axes[2], figA, g3, sigma, "P(won | built tool)", 2)
    figA.suptitle(f"toolworld v3 build chain -- {model} -- gaussian-pooled "
                  f"(sigma={sigma}, T>=2; dots = cells with the condition met)",
                  fontsize=12)
    a_png = run_dir / "fig_grid_v3_buildchain.png"
    figA.savefig(a_png, dpi=130); figA.savefig(run_dir / "fig_grid_v3_buildchain.pdf")
    plt.close(figA)
    print(f"  wrote {a_png}")

    # --- Figure B (separate file): P(solved) over the full grid (T = 0..10) -----
    g4 = grid(facts, lambda f: 1.0 if f["solved"] else 0.0, t_lo=0)
    figB, ax = plt.subplots(1, 1, figsize=(6.2, 5.2), constrained_layout=True)
    panel(ax, figB, g4, sigma, "P(solved / won the game)", 0)
    figB.suptitle(f"toolworld v3 -- {model} -- P(solved), gaussian-pooled "
                  f"(sigma={sigma})", fontsize=11)
    b_png = run_dir / "fig_grid_v3_solved.png"
    figB.savefig(b_png, dpi=130); figB.savefig(run_dir / "fig_grid_v3_solved.pdf")
    plt.close(figB)
    print(f"  wrote {b_png}")


if __name__ == "__main__":
    main()
