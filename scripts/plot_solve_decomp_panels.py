"""Three-panel BUILD-PATH solve-probability map: per cell, Z = E * R * C, where each
factor is the Gaussian-pooled surface from an existing panel —

    C = P(held both ingredients)      (fig_held_both_panels;  gathering)
    R = P(built | held both)          (fig_recognition_panels; recognition)
    E = P(solved | tool built)        (fig_efficiency_solve_panels; exploitation)

so Z = C * R * E factorizes the end-to-end build-path solve probability:
gather the ingredients, recognize/commit to the build, then exploit the tool to finish.
Each factor is pooled SEPARATELY (normalized Gaussian convolution, the same pooled()
the source panels use), then multiplied pointwise AFTER pooling, exactly as requested.

C and R come from the original 180-run region square; E from the efficiency run. Pass
both dirs per model.

Usage: PYTHONPATH=. python -m scripts.plot_solve_decomp_panels \
           <haiku_region> <sonnet_region> <opus_region> \
           <haiku_eff> <sonnet_eff> <opus_eff> [--n-hi 20] [--bandwidth 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.sweep_config as cfg
from scripts.plot_held_both_panels import load_cells as load_held, pooled, T_LO, T_HI
from scripts.plot_recognition_panels import load_cells as load_recog
from scripts.plot_efficiency_solve_panels import load_cells as load_esolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("haiku_region"); ap.add_argument("sonnet_region"); ap.add_argument("opus_region")
    ap.add_argument("haiku_eff"); ap.add_argument("sonnet_eff"); ap.add_argument("opus_eff")
    ap.add_argument("--n-hi", type=int, default=20)
    ap.add_argument("--bandwidth", type=float, default=1.0)
    args = ap.parse_args()

    region = [args.haiku_region, args.sonnet_region, args.opus_region]
    eff = [args.haiku_eff, args.sonnet_eff, args.opus_eff]

    ts = np.linspace(T_LO, T_HI, 200)
    bnd = [next((n for n in range(1, args.n_hi + 1)
                 if cfg._build_cost(n, int(round(t))) < cfg._grind_cost(n)), np.nan) for t in ts]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    mesh = None
    for ax, rdir, edir in zip(axes, region, eff):
        short, C_cells, C_head = load_held(Path(rdir))
        _, R_cells, R_head = load_recog(Path(rdir))
        _, E_cells, E_head = load_esolve(Path(edir))
        gt, gn, ZC = pooled(C_cells, args.n_hi, args.bandwidth)
        _, _, ZR = pooled(R_cells, args.n_hi, args.bandwidth)
        _, _, ZE = pooled(E_cells, args.n_hi, args.bandwidth)
        ZC, ZR, ZE = (np.clip(z, 0, 1) for z in (ZC, ZR, ZE))
        Z = ZC * ZR * ZE                       # product AFTER pooling
        prod_head = C_head * R_head * E_head    # scalar gather x recognize x exploit
        print(f"{short}: C={C_head:.2f} R={R_head:.2f} E={E_head:.2f} -> "
              f"C*R*E(headline)={prod_head:.3f} | pooled-surface mean={np.nanmean(Z):.3f}")
        mesh = ax.pcolormesh(gt, gn, Z, cmap="RdYlGn", vmin=0, vmax=1, shading="nearest")
        ax.plot(ts, bnd, "k--", lw=1.2, label="E[build]=E[grind]")
        ax.set_title(f"{short.capitalize()}\nC·R·E headline = {prod_head:.3f}  "
                     f"(C={C_head:.2f}, R={R_head:.2f}, E={E_head:.2f})", fontsize=10)
        ax.set_xlabel("byproduct types T")
        ax.set_xlim(T_LO - 0.5, T_HI + 0.5); ax.set_ylim(0.5, args.n_hi + 0.5)
    axes[0].set_ylabel("number of doors N")
    axes[0].legend(loc="upper right", fontsize=7, framealpha=0.9)

    fig.suptitle("Build-path solve probability  Z = E·R·C  (pooled then multiplied), over (T, N)\n"
                 "C = P(held both) · R = P(built | held) · E = P(solved | tool built)   "
                 f"(gaussian-pooled bw={args.bandwidth:g})", y=1.04, fontsize=12)
    cb = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("E·R·C = P(solve via build path)")

    out = Path(f"figs/toolworld/fig_solve_decomp_panels_Nle{args.n_hi}.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
