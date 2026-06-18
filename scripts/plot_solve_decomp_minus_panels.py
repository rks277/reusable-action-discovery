"""Three-panel DIFFERENCE map: per cell, Z = (E·R·C) - S, where

    E·R·C = build-path solve probability (plot_solve_decomp_panels), with
       C = P(held both)      (fig_held_both_panels)
       R = P(built | held)   (fig_recognition_panels)
       E = P(solved | built) (fig_efficiency_solve_panels)
    S     = P(solved)        (fig_solve_panels; the ACTUAL overall solve rate of the
            original 180-run region square, build- AND grind-based solves together)

Every factor is Gaussian-pooled separately (the shared pooled()), C·R·E multiplied
AFTER pooling, then the pooled S surface subtracted. The result diverges about 0:
  > 0 (red):  the build-path estimate EXCEEDS actual solving — build-path "should" win
              here more than the model actually solves.
  < 0 (blue): actual solving EXCEEDS the build-path estimate — the model solves beyond
              what building predicts, i.e. it is grinding (solving without/over the tool).

Usage: PYTHONPATH=. python -m scripts.plot_solve_decomp_minus_panels \
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
from scripts.plot_solve_panels import load_cells as load_solve


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

    panels = []
    vmax = 0.0
    for rdir, edir in zip(region, eff):
        short, C_cells, C_head = load_held(Path(rdir))
        _, R_cells, R_head = load_recog(Path(rdir))
        _, E_cells, E_head = load_esolve(Path(edir))
        _, S_cells, S_head = load_solve(Path(rdir))
        gt, gn, ZC = pooled(C_cells, args.n_hi, args.bandwidth)
        _, _, ZR = pooled(R_cells, args.n_hi, args.bandwidth)
        _, _, ZE = pooled(E_cells, args.n_hi, args.bandwidth)
        _, _, ZS = pooled(S_cells, args.n_hi, args.bandwidth)
        ZC, ZR, ZE, ZS = (np.clip(z, 0, 1) for z in (ZC, ZR, ZE, ZS))
        Z = ZC * ZR * ZE - ZS                  # (E·R·C) - actual solve, after pooling
        head = C_head * R_head * E_head - S_head
        vmax = max(vmax, np.nanmax(np.abs(Z)))
        panels.append((short, gt, gn, Z, head, C_head, R_head, E_head, S_head))
        print(f"{short}: C*R*E_head={C_head*R_head*E_head:.3f} - S_head={S_head:.2f} "
              f"= {head:+.3f} | pooled-surface mean diff={np.nanmean(Z):+.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    mesh = None
    for ax, (short, gt, gn, Z, head, C, R, E, S) in zip(axes, panels):
        mesh = ax.pcolormesh(gt, gn, Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest")
        ax.plot(ts, bnd, "k--", lw=1.2, label="E[build]=E[grind]")
        ax.set_title(f"{short.capitalize()}\n(C·R·E)-S headline = {head:+.3f}  "
                     f"(C·R·E={C*R*E:.2f}, S={S:.2f})", fontsize=10)
        ax.set_xlabel("byproduct types T")
        ax.set_xlim(T_LO - 0.5, T_HI + 0.5); ax.set_ylim(0.5, args.n_hi + 0.5)
    axes[0].set_ylabel("number of doors N")
    axes[0].legend(loc="upper right", fontsize=7, framealpha=0.9)

    fig.suptitle("Build-path estimate minus actual solve rate:  Z = (E·R·C) - S,  over (T, N)\n"
                 "red = build-path over-predicts; blue = model solves beyond the build path "
                 f"(grinds)   (gaussian-pooled bw={args.bandwidth:g})", y=1.04, fontsize=12)
    cb = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("(E·R·C) − P(solved)")

    out = Path(f"figs/toolworld/fig_solve_decomp_minus_panels_Nle{args.n_hi}.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
