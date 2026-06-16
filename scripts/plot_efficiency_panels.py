"""Three-panel tool-EXPLOITATION efficiency map for Haiku / Sonnet / Opus, one figure,
Gaussian-pooled. Mirrors plot_solve_panels.py / plot_recognition_panels.py.

Per (T, N) cell the value is the mean of

    redundant_continuation / live_remaining          (live_remaining = budget - (ab + 2))

i.e. redundant post-build actions as a FRACTION of the live budget the agent had after
the forced build (normalizing by remaining budget makes cells comparable and removes the
small-budget "no room to flounder" confound). Degenerate runs (live_remaining <= 0) have
an undefined denominator and are dropped (shown as 'x'). Green = efficient (little of the
budget wasted), red = most of it wasted.

Usage: PYTHONPATH=. python -m scripts.plot_efficiency_panels <haiku_dir> <sonnet_dir> \
           <opus_dir> [--n-hi 20] [--bandwidth 1]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import scripts.sweep_config as cfg

T_LO, T_HI = 2, 10


def load_cells(run_dir: Path):
    """Return (short_model, {(T,N): mean redundant/remaining}, headline_mean, gaps)."""
    rows = [json.loads(l) for l in (run_dir / "episodes.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    short = (rows[0].get("model", "?").split("-")[1]
             if rows and rows[0].get("model", "").startswith("claude-") else "?")
    agg = {}  # (T,N) -> [sum_prop, n_valid]
    gaps = []
    for r in rows:
        key = (r["n_types"], r["n"])
        lr = r.get("live_remaining", 0)
        if r.get("degenerate") or lr <= 0:
            gaps.append(key)
            continue
        a = agg.setdefault(key, [0.0, 0])
        a[0] += r.get("redundant_continuation", 0) / lr
        a[1] += 1
    cells = {k: s / n for k, (s, n) in agg.items() if n}
    head = (sum(cells.values()) / len(cells)) if cells else float("nan")
    # only keep gap cells that have no valid run at all
    gaps = [g for g in set(gaps) if g not in cells]
    return short, cells, head, gaps


def pooled(cells, n_hi, bw):
    gt, gn = np.meshgrid(np.arange(T_LO, T_HI + 1), np.arange(1, n_hi + 1))
    V = np.zeros_like(gt, float); M = np.zeros_like(gt, float)
    for (t, n), v in cells.items():
        if 1 <= n <= n_hi and T_LO <= t <= T_HI:
            V[n - 1, t - T_LO] = v; M[n - 1, t - T_LO] = 1.0
    num = gaussian_filter(V * M, bw, mode="nearest")
    den = gaussian_filter(M, bw, mode="nearest")
    Z = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-9)
    return gt, gn, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("haiku_dir"); ap.add_argument("sonnet_dir"); ap.add_argument("opus_dir")
    ap.add_argument("--n-hi", type=int, default=20)
    ap.add_argument("--bandwidth", type=float, default=1.0)
    args = ap.parse_args()

    loaded = [load_cells(Path(d)) for d in (args.haiku_dir, args.sonnet_dir, args.opus_dir)]

    ts = np.linspace(T_LO, T_HI, 200)
    bnd = [next((n for n in range(1, args.n_hi + 1)
                 if cfg._build_cost(n, int(round(t))) < cfg._grind_cost(n)), np.nan) for t in ts]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    mesh = None
    for ax, (short, cells, head, gaps) in zip(axes, loaded):
        print(f"{short}: mean redundant/remaining {head:.2f} over {len(cells)} cells "
              f"({len(gaps)} degenerate gaps)")
        gt, gn, Z = pooled(cells, args.n_hi, args.bandwidth)
        Z = np.clip(Z, 0, 1)
        # RdYlGn_r: green = little of the budget wasted (efficient), red = most wasted.
        mesh = ax.pcolormesh(gt, gn, Z, cmap="RdYlGn_r", vmin=0, vmax=1, shading="nearest")
        arr = np.array([[t, n, v] for (t, n), v in cells.items()], float)
        ax.scatter(arr[:, 0], arr[:, 1], c=arr[:, 2], cmap="RdYlGn_r", vmin=0, vmax=1,
                   s=46, edgecolors="black", linewidths=0.7, zorder=3)
        if gaps:
            gv = np.array(gaps, float)
            ax.scatter(gv[:, 0], gv[:, 1], marker="x", c="black", s=34, linewidths=1.0,
                       zorder=3)
        ax.plot(ts, bnd, "k--", lw=1.2, label="E[build]=E[grind]")
        ax.set_title(f"{short.capitalize()}\nmean waste = {head:.2f}")
        ax.set_xlabel("byproduct types T")
        ax.set_xlim(T_LO - 0.5, T_HI + 0.5); ax.set_ylim(0.5, args.n_hi + 0.5)
    axes[0].set_ylabel("number of doors N")
    axes[0].legend(loc="upper right", fontsize=7, framealpha=0.9)

    fig.suptitle("Tool-exploitation efficiency: redundant actions / remaining budget after a "
                 "forced build, over (T, N)\n"
                 f"(gaussian-pooled bw={args.bandwidth:g}; redundant = no door/machine progress; "
                 "green = efficient, red = budget wasted)",
                 y=1.04, fontsize=12)
    cb = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("redundant actions / remaining budget")

    out = Path(f"figs/fig_efficiency_panels_Nle{args.n_hi}.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
