"""Three-panel NATURAL-BUILD efficiency map for Haiku / Sonnet / Opus, one figure,
Gaussian-pooled. Pure back-analysis of the REGION sweep -- no forced build, no extra runs.

This is the natural-build analogue of plot_efficiency_solve_panels.py (the forced-build E).
Instead of injecting the machine into every episode, we condition on the self-selected
subset that built the machine on its OWN, and ask: among the episodes that SPONTANEOUSLY
built in each (T,N) cell, what fraction went on to solve?

    E_natural = P(solved | built)        (built episodes only; never-built cells dropped)

Versus the forced-build E this (a) charges the build's budget cost to the solve attempt and
(b) is confounded with recognition (built episodes are not a random sample) -- so it reads
LOWER and sparser. It is, however, fully self-consistent with C and R from the same region
run, so C*R*E_natural = P(solved & built) = the build-path solve probability exactly.

Usage: PYTHONPATH=. python -m scripts.plot_efficiency_natural_panels \
           <haiku_region> <sonnet_region> <opus_region> \
           [--n-hi 20] [--bandwidth 1] [--outdir figs/toolworld]
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

import scripts.sweep_config as cfg
from scripts.plot_recognition_panels import built, pooled, T_LO, T_HI


def load_built_cells(run_dir: Path):
    """-> (short, {(T,N): solve rate over naturally-built eps}, pooled P(solved|built)).

    Reps per cell are AVERAGED; the headline is pooled over all naturally-built episodes."""
    rows = [json.loads(l) for l in (run_dir / "episodes.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    m0 = rows[0].get("model", "?") if rows else "?"
    short = m0.split("-")[1] if m0.startswith("claude-") else m0
    acc = defaultdict(list)  # (T,N) -> solved01 for each naturally-built episode
    for r in rows:
        if built(r):
            acc[(r["n_types"], r["n"])].append(int(bool(r.get("solved"))))
    cells = {k: sum(v) / len(v) for k, v in acc.items()}
    nd = sum(len(v) for v in acc.values())
    head = sum(sum(v) for v in acc.values()) / nd if nd else float("nan")
    return short, cells, head, nd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="one or more region run dirs (one panel each)")
    ap.add_argument("--n-hi", type=int, default=20)
    ap.add_argument("--bandwidth", type=float, default=1.0)
    ap.add_argument("--outdir", default="figs/toolworld")
    args = ap.parse_args()

    loaded = [load_built_cells(Path(d)) for d in args.dirs]

    ts = np.linspace(T_LO, T_HI, 200)
    bnd = [next((n for n in range(1, args.n_hi + 1)
                 if cfg._build_cost(n, int(round(t))) < cfg._grind_cost(n)), np.nan) for t in ts]

    _CLAUDE = ("haiku", "sonnet", "opus")
    _rows = [r for r in ([e for e in loaded if e[0] not in _CLAUDE],
                         [e for e in loaded if e[0] in _CLAUDE]) if r] or [loaded]
    _ncol = max(len(r) for r in _rows)
    fig, _axg = plt.subplots(len(_rows), _ncol, figsize=(5 * _ncol, 5.2 * len(_rows)),
                             sharey=True, squeeze=False)
    pairs, row_lefts = [], []
    for _ri, _r in enumerate(_rows):
        row_lefts.append(_axg[_ri][0])
        for _ci in range(_ncol):
            _ax = _axg[_ri][_ci]
            (pairs.append((_ax, _r[_ci])) if _ci < len(_r) else _ax.axis("off"))
    mesh = None
    for ax, (short, cells, head, nd) in pairs:
        se = (head * (1 - head) / nd) ** 0.5 if nd else float("nan")
        print(f"{short}: built cells {len(cells)} | naturally-built eps {nd} | "
              f"E_natural=P(solved|built)={head:.3f} +/- {se:.3f}")
        gt, gn, Z = pooled(cells, args.n_hi, args.bandwidth)
        Z = np.clip(Z, 0, 1)
        mesh = ax.pcolormesh(gt, gn, Z, cmap="RdYlGn", vmin=0, vmax=1, shading="nearest")
        if cells:
            arr = np.array([[t, n, v] for (t, n), v in cells.items()], float)
            ax.scatter(arr[:, 0], arr[:, 1], c=arr[:, 2], cmap="RdYlGn", vmin=0, vmax=1,
                       s=46, edgecolors="black", linewidths=0.7, zorder=3)
        held_keys = set(cells)
        dropped = [(t, n) for t in range(T_LO, T_HI + 1) for n in range(1, args.n_hi + 1)
                   if (t, n) not in held_keys]
        if dropped:
            dv = np.array(dropped, float)
            ax.scatter(dv[:, 0], dv[:, 1], marker="x", c="0.45", s=22, linewidths=0.7, zorder=2)
        ax.plot(ts, bnd, "k--", lw=1.2, label="E[build]=E[grind]")
        ax.set_title(f"{short.capitalize()}\nE_nat = P(solved | built) = {head:.2f}  (n={nd})")
        ax.set_xlabel("byproduct types T")
        ax.set_xlim(T_LO - 0.5, T_HI + 0.5); ax.set_ylim(0.5, args.n_hi + 0.5)
    for _lax in row_lefts: _lax.set_ylabel("number of doors N")
    pairs[0][0].legend(loc="upper right", fontsize=7, framealpha=0.9)

    fig.suptitle("Natural-build efficiency  E = P(solved | spontaneously built), over (T, N)\n"
                 f"(region-sweep back-analysis; gaussian-pooled bw={args.bandwidth:g}; "
                 "x = never built in that cell; green = built and finished)",
                 y=1.04, fontsize=12)
    cb = fig.colorbar(mesh, ax=_axg, fraction=0.025, pad=0.02)
    cb.set_label("P(solved | naturally built)")

    out = Path(args.outdir) / f"fig_efficiency_natural_panels_Nle{args.n_hi}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
