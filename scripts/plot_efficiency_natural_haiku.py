"""Single-panel 'natural-build efficiency' for Haiku -- pure BACK-ANALYSIS of the
region sweep (no new runs, no forced build).

The forced-build E (plot_efficiency_solve_panels.py) injects the machine into EVERY
episode to isolate pure exploitation. Here we instead condition on the self-selected
subset that built on its OWN: among the episodes that SPONTANEOUSLY built the machine
in each (T,N) cell, what fraction went on to solve?  E_natural = P(solved | built).

CAVEAT (why this differs from the forced-build E): the naturally-built episodes are not
a random sample -- they are exactly the trajectories where the model recognized building
was worthwhile, so the estimate is confounded with recognition (R) and is sparse/uneven
across cells. Cells that never built are DROPPED (small 'x'), mirroring the held-both
drop in the recognition panel.

Usage: PYTHONPATH=. python -m scripts.plot_efficiency_natural_haiku <haiku_region_dir> \
           [--n-hi 20] [--smooth gaussian] [--bandwidth 1] [--out PATH]
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

from scripts.plot_recognition_panels import built, pooled, T_LO, T_HI
import scripts.sweep_config as cfg


def load_built_cells(run_dir: Path):
    """-> (short, {(T,N): solve rate over naturally-built eps}, pooled P(solved|built), n_built)."""
    rows = [json.loads(l) for l in (run_dir / "episodes.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    short = (rows[0].get("model", "?").split("-")[1]
             if rows and rows[0].get("model", "").startswith("claude-") else "?")
    acc = defaultdict(list)  # (T,N) -> solved01 for each naturally-built episode
    for r in rows:
        if built(r):
            acc[(r["n_types"], r["n"])].append(int(bool(r.get("solved"))))
    cells = {k: sum(v) / len(v) for k, v in acc.items()}
    ns = sum(sum(v) for v in acc.values())
    nd = sum(len(v) for v in acc.values())
    head = ns / nd if nd else float("nan")
    return short, cells, head, ns, nd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("region_dir")
    ap.add_argument("--n-hi", type=int, default=20)
    ap.add_argument("--smooth", choices=["none", "gaussian"], default="gaussian")
    ap.add_argument("--bandwidth", type=float, default=1.0)
    ap.add_argument("--out", default="figs/toolworld/fig_efficiency_natural_haiku.png")
    args = ap.parse_args()

    short, cells, head, ns, nd = load_built_cells(Path(args.region_dir))
    se = (head * (1 - head) / nd) ** 0.5 if nd else float("nan")
    print(f"{short}: built cells {len(cells)} | naturally-built episodes {nd} "
          f"(solved {ns}) | E_natural=P(solved|built)={head:.3f} +/- {se:.3f}")

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    mesh = None
    if args.smooth == "gaussian":
        gt, gn, Z = pooled(cells, args.n_hi, args.bandwidth)
        Z = np.clip(Z, 0, 1)
        mesh = ax.pcolormesh(gt, gn, Z, cmap="RdYlGn", vmin=0, vmax=1, shading="nearest")
    if cells:
        arr = np.array([[t, n, v] for (t, n), v in cells.items()], float)
        sc = ax.scatter(arr[:, 0], arr[:, 1], c=arr[:, 2], cmap="RdYlGn", vmin=0, vmax=1,
                        s=46, edgecolors="black", linewidths=0.7, zorder=3)
        if mesh is None:
            mesh = sc
    # cells that never built -> dropped
    held_keys = set(cells)
    dropped = [(t, n) for t in range(T_LO, T_HI + 1) for n in range(1, args.n_hi + 1)
               if (t, n) not in held_keys]
    if dropped:
        dv = np.array(dropped, float)
        ax.scatter(dv[:, 0], dv[:, 1], marker="x", c="0.45", s=26, linewidths=0.8,
                   zorder=2, label="never built (dropped)")
    # E[build]=E[grind] boundary, same as the other region panels
    ts = np.linspace(T_LO, T_HI, 200)
    bnd = [next((n for n in range(1, args.n_hi + 1)
                 if cfg._build_cost(n, int(round(t))) < cfg._grind_cost(n)), np.nan) for t in ts]
    ax.plot(ts, bnd, "k--", lw=1.2, alpha=0.7, label="E[build]=E[grind]")

    ax.set_xlabel("T (byproduct types)")
    ax.set_ylabel("N (doors)")
    ax.set_xlim(T_LO - 0.5, T_HI + 0.5)
    ax.set_ylim(0.5, args.n_hi + 0.5)
    ax.set_title(f"{short} natural-build efficiency  E=P(solved|built)={head:.2f}"
                 f"  (n={nd} built eps)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.colorbar(mesh, ax=ax, label="P(solved | built) in cell")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
