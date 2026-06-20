"""(T, N)-plane heatmaps for the coupon-collector WoodWorld: RECOGNITION and SOLVE.

Two side-by-side Gaussian-pooled heatmaps (ToolWorld-style), x = tree-resource types T
(2..8), y = number of distinct wood kinds N (2..20), for a single model (Haiku):

  RECOGNITION = P(built axe | ever held >= 2 sticks)   -- held-sticks cells only
  SOLVE RATE  = P(solved = collected all N kinds)       -- all cells

Run from repo root:
  PYTHONPATH=. python tool-wood-discrepancy/plot_coupon_heatmaps.py tool-wood-discrepancy/runs/<dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget_coupon as bc                          # noqa: E402

T_LO, T_HI = 2, 8
N_LO, N_HI = 2, 20


def pooled(cells, bw):
    """Normalized Gaussian convolution of the (T, N) cells onto the integer lattice."""
    gt, gn = np.meshgrid(np.arange(T_LO, T_HI + 1), np.arange(N_LO, N_HI + 1))
    V = np.zeros_like(gt, float)
    M = np.zeros_like(gt, float)
    for (t, n), v in cells.items():
        if T_LO <= t <= T_HI and N_LO <= n <= N_HI:
            V[n - N_LO, t - T_LO] = v
            M[n - N_LO, t - T_LO] = 1.0
    num = gaussian_filter(V * M, bw, mode="nearest")
    den = gaussian_filter(M, bw, mode="nearest")
    Z = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-9)
    return gt, gn, Z


def load(run_dir: Path):
    rows = [json.loads(l) for l in (run_dir / "episodes.jsonl").read_text().splitlines()
            if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    m0 = rows[0].get("model", "?") if rows else "?"
    short = m0.split("-")[1] if m0.startswith("claude-") else m0.replace(":", "-")
    return short, rows


def recognition_cells(rows):
    """{(T,N): mean(built|held>=2 sticks)} over held-sticks cells; pooled headline."""
    acc = defaultdict(list)
    for r in rows:
        if r.get("held_ingredients"):
            acc[(r["n_types"], r["n"])].append(int(bool(r.get("built_axe"))))
    cells = {k: sum(v) / len(v) for k, v in acc.items()}
    nb = sum(sum(v) for v in acc.values())
    nd = sum(len(v) for v in acc.values())
    return cells, (nb / nd if nd else float("nan")), nd


def solve_cells(rows):
    acc = defaultdict(list)
    for r in rows:
        acc[(r["n_types"], r["n"])].append(int(bool(r.get("solved"))))
    cells = {k: sum(v) / len(v) for k, v in acc.items()}
    n = sum(len(v) for v in acc.values())
    head = sum(sum(v) for v in acc.values()) / n if n else float("nan")
    return cells, head, n


def draw(ax, cells, bw, title, all_cells):
    gt, gn, Z = pooled(cells, bw)
    Z = np.clip(Z, 0, 1)
    mesh = ax.pcolormesh(gt, gn, Z, cmap="RdYlGn", vmin=0, vmax=1, shading="nearest")
    if cells:
        arr = np.array([[t, n, v] for (t, n), v in cells.items()], float)
        ax.scatter(arr[:, 0], arr[:, 1], c=arr[:, 2], cmap="RdYlGn", vmin=0, vmax=1,
                   s=46, edgecolors="black", linewidths=0.7, zorder=3)
    # cells covered by the sweep but absent from THIS metric (e.g. never held sticks)
    held = set(cells)
    dropped = [(t, n) for t in range(T_LO, T_HI + 1) for n in range(N_LO, N_HI + 1)
               if (t, n) in all_cells and (t, n) not in held]
    if dropped:
        dv = np.array(dropped, float)
        ax.scatter(dv[:, 0], dv[:, 1], marker="x", c="0.45", s=24, linewidths=0.8,
                   zorder=2, label="no held-sticks episode")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("tree-resource types T")
    ax.set_xlim(T_LO - 0.5, T_HI + 0.5)
    ax.set_ylim(N_LO - 0.5, N_HI + 0.5)
    return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--bandwidth", type=float, default=1.0)
    ap.add_argument("--boundary", action="store_true",
                    help="overlay the (rough) E[build]=E[grind] coupon crossover")
    ap.add_argument("--prefix", default=None,
                    help="output filename/caption prefix; default auto from run dir "
                         "('gated' if the path contains it, else 'coupon')")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    short, rows = load(run_dir)
    prefix = args.prefix or ("gated" if "gated" in str(run_dir) else "coupon")
    gated = prefix == "gated"
    recipe = rows[0].get("recipe_mode", "same") if rows else "same"
    rdesc = ("axe = 2 DISTINCT resources" if recipe == "distinct"
             else "stick+stick→axe")
    if gated:
        rdesc += "; kinds GATED (locked containers, directed axe→key→open)"
    rcells, rhead, rn = recognition_cells(rows)
    scells, shead, sn = solve_cells(rows)
    all_cells = {(r["n_types"], r["n"]) for r in rows}
    print(f"{short}: recognition P(built|held sticks) = {rhead:.2f} (n={rn}) | "
          f"solve rate = {shead:.2f} (n={sn})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    mesh = draw(axes[0], rcells, args.bandwidth,
                f"RECOGNITION  P(built axe | held ≥2 sticks) = {rhead:.2f}", all_cells)
    draw(axes[1], scells, args.bandwidth,
         f"SOLVE RATE  P(collected all N kinds) = {shead:.2f}", all_cells)

    if args.boundary:
        ts = np.linspace(T_LO, T_HI, 200)
        bnd = [next((n for n in range(N_LO, N_HI + 1)
                     if bc.expected_build(n, int(round(t))) < bc.expected_grind(n)),
                    np.nan) for t in ts]
        for ax in axes:
            ax.plot(ts, bnd, "k--", lw=1.2, label="E[build]=E[grind]")

    axes[0].set_ylabel("number of distinct wood kinds N")
    axes[0].legend(loc="upper right", fontsize=7, framealpha=0.9)
    world_name = "GATED coupon WoodWorld" if gated else "Coupon-collector WoodWorld"
    tail = "" if gated else ", axe→new kind"
    fig.suptitle(f"{world_name} ({short}, recipe={recipe}) over (T, N): "
                 f"recognition & solve rate\n(gaussian-pooled bw={args.bandwidth:g}, "
                 f"1 rep/cell; goal = collect all N distinct kinds; {rdesc}{tail})",
                 y=1.02, fontsize=12)
    cb = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("probability")

    outdir = Path(args.outdir) if args.outdir else (Path(__file__).resolve().parent / "figs")
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"fig_{prefix}_{short}_{recipe}_T{T_LO}-{T_HI}_N{N_LO}-{N_HI}_recognition_solve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
