"""Difference map of build-propensity between two region sweeps (model A - model B).

Loads two run_haiku_region_sweep.py output dirs, computes each model's per-(T, N)
build rate (fraction of episodes in the cell that built), Gaussian-pools both onto
the full T x N grid (normalized convolution, same as plot_haiku_region_heatmap.py
--smooth gaussian), and renders A - B on a diverging colormap centered at 0. Positive
(red) = model A builds more; negative (blue) = model B builds more.

Both sweeps should cover the same (T, N) cells (e.g. full coverage) for a faithful diff;
cells present in only one are dropped from the comparison.

Usage: PYTHONPATH=. python -m scripts.plot_region_diff <dirA> <dirB> [--n-hi 20] [--bandwidth 1]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import scripts.sweep_config as cfg

T_LO, T_HI = 2, 10


def built(row):
    return bool(row.get("built_machine")) or any("fuse into" in o for o in row.get("obs", []))


def load_grid(run_dir: Path, n_hi: int):
    rows = [json.loads(l) for l in (run_dir / "episodes.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    model = rows[0].get("model", "?")
    short = model.split("-")[1] if model.startswith("claude-") else model
    agg = defaultdict(lambda: [0, 0])           # (T,N) -> [n_built, n_eps]
    for r in rows:
        a = agg[(r["n_types"], r["n"])]
        a[0] += built(r); a[1] += 1
    cells = {k: v[0] / v[1] for k, v in agg.items()}    # build rate per cell
    return short, cells


def pooled(cells, n_hi, bw):
    """Normalized Gaussian convolution of a scattered build-rate dict onto the grid."""
    gt, gn = np.meshgrid(np.arange(T_LO, T_HI + 1), np.arange(1, n_hi + 1))
    V = np.zeros_like(gt, float); M = np.zeros_like(gt, float)
    for (t, n), v in cells.items():
        if 1 <= n <= n_hi and T_LO <= t <= T_HI:
            V[n - 1, t - T_LO] = v; M[n - 1, t - T_LO] = 1.0
    num = gaussian_filter(V * M, bw, mode="nearest")
    den = gaussian_filter(M, bw, mode="nearest")
    return gt, gn, np.divide(num, den, out=np.zeros_like(num), where=den > 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a"); ap.add_argument("dir_b")
    ap.add_argument("--n-hi", type=int, default=20)
    ap.add_argument("--bandwidth", type=float, default=1.0)
    args = ap.parse_args()

    sa, ca = load_grid(Path(args.dir_a), args.n_hi)
    sb, cb = load_grid(Path(args.dir_b), args.n_hi)
    shared = sorted(set(ca) & set(cb))
    print(f"{sa}: {len(ca)} cells | {sb}: {len(cb)} cells | shared: {len(shared)}")

    gt, gn, Za = pooled({k: ca[k] for k in shared}, args.n_hi, args.bandwidth)
    _, _, Zb = pooled({k: cb[k] for k in shared}, args.n_hi, args.bandwidth)
    D = Za - Zb

    # raw per-cell diff for the scatter + summary
    raw = np.array([ca[k] - cb[k] for k in shared])
    print(f"raw cell diff ({sa}-{sb}): mean {raw.mean():+.3f}  "
          f"{sa}>{sb} in {int((raw>0).sum())} cells, {sb}>{sa} in {int((raw<0).sum())}, "
          f"tie {int((raw==0).sum())}")

    fig, ax = plt.subplots(figsize=(7, 6))
    mesh = ax.pcolormesh(gt, gn, D, cmap="RdBu_r", vmin=-1, vmax=1, shading="nearest")
    fig.colorbar(mesh, ax=ax).set_label(f"P(built): {sa} − {sb}")

    ts = np.linspace(T_LO, T_HI, 200)
    bnd = [next((n for n in range(1, args.n_hi + 1)
                 if cfg._build_cost(n, int(round(t))) < cfg._grind_cost(n)), np.nan) for t in ts]
    ax.plot(ts, bnd, "k--", lw=1.2, label="E[build]=E[grind]")

    sc = np.array([[k[0], k[1], ca[k] - cb[k]] for k in shared], float)
    ax.scatter(sc[:, 0], sc[:, 1], c=sc[:, 2], cmap="RdBu_r", vmin=-1, vmax=1,
               s=40, edgecolors="black", linewidths=0.6)

    ax.set_xlabel("byproduct types T"); ax.set_ylabel("number of doors N")
    ax.set_title(f"Build-propensity difference: {sa.capitalize()} − {sb.capitalize()}\n"
                 f"(gaussian-pooled bw={args.bandwidth:g}; red = {sa} builds more)")
    ax.set_xlim(T_LO - 0.5, T_HI + 0.5); ax.set_ylim(0.5, args.n_hi + 0.5)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    out = Path(f"figs/fig_{sa}_minus_{sb}_density_Nle{args.n_hi}.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150); fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
