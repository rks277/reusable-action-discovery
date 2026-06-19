"""Three-panel solve-rate map: P(solved) over (T, N) for Haiku / Sonnet / Opus, one
figure, Gaussian-pooled. Every cell contributes (solve is defined for all episodes;
nothing is dropped). Single rep/cell -> per-cell value is 0/1, pooled into a surface.

Usage: PYTHONPATH=. python -m scripts.plot_solve_panels <haiku_dir> <sonnet_dir> \
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
    """Reps per cell are AVERAGED into the cell value; headline is pooled over all
    episodes. Identical to the old 0/1 value for a 1-rep grid; tightens with reps."""
    from collections import defaultdict
    rows = [json.loads(l) for l in (run_dir / "episodes.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    m0 = rows[0].get("model", "?") if rows else "?"
    short = m0.split("-")[1] if m0.startswith("claude-") else m0
    acc = defaultdict(list)  # (T,N) -> solved 0/1 for each rep
    for r in rows:
        acc[(r["n_types"], r["n"])].append(int(bool(r.get("solved"))))
    cells = {k: sum(v) / len(v) for k, v in acc.items()}
    nd = sum(len(v) for v in acc.values())
    head = sum(sum(v) for v in acc.values()) / nd if nd else float("nan")
    return short, cells, head


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
    ap.add_argument("dirs", nargs="+", help="one or more region run dirs (one panel each)")
    ap.add_argument("--n-hi", type=int, default=20)
    ap.add_argument("--bandwidth", type=float, default=1.0)
    ap.add_argument("--outdir", default="figs/toolworld")
    args = ap.parse_args()

    loaded = [load_cells(Path(d)) for d in args.dirs]

    ts = np.linspace(T_LO, T_HI, 200)
    bnd = [next((n for n in range(1, args.n_hi + 1)
                 if cfg._build_cost(n, int(round(t))) < cfg._grind_cost(n)), np.nan) for t in ts]

    fig, axes = plt.subplots(1, len(loaded), figsize=(5 * len(loaded), 5.2), sharey=True)
    mesh = None
    for ax, (short, cells, head) in zip(axes, loaded):
        print(f"{short}: solve-rate {head:.2f} ({sum(cells.values())}/{len(cells)})")
        gt, gn, Z = pooled(cells, args.n_hi, args.bandwidth)
        Z = np.clip(Z, 0, 1)
        mesh = ax.pcolormesh(gt, gn, Z, cmap="RdYlGn", vmin=0, vmax=1, shading="nearest")
        arr = np.array([[t, n, v] for (t, n), v in cells.items()], float)
        ax.scatter(arr[:, 0], arr[:, 1], c=arr[:, 2], cmap="RdYlGn", vmin=0, vmax=1,
                   s=46, edgecolors="black", linewidths=0.7, zorder=3)
        ax.plot(ts, bnd, "k--", lw=1.2, label="E[build]=E[grind]")
        ax.set_title(f"{short.capitalize()}\nsolve rate = {head:.2f}")
        ax.set_xlabel("byproduct types T")
        ax.set_xlim(T_LO - 0.5, T_HI + 0.5); ax.set_ylim(0.5, args.n_hi + 0.5)
    axes[0].set_ylabel("number of doors N")
    axes[0].legend(loc="upper right", fontsize=7, framealpha=0.9)

    fig.suptitle(f"Solve rate P(solved) over (T, N) — full region sweep "
                 f"(gaussian-pooled bw={args.bandwidth:g}; grind-calibrated budget; single-draw)",
                 y=1.02, fontsize=12)
    cb = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("P(solved)")

    out = Path(args.outdir) / f"fig_solve_panels_Nle{args.n_hi}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
