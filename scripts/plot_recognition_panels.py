"""Three-panel 'recognition' map: P(built | held both recipe ingredients) over (T, N),
for Haiku / Sonnet / Opus side by side in one figure.

Recognition is conditioned on the model having held BOTH recipe byproducts at some
point in the episode (the necessary precondition to build). Cells where the model
never held both ingredients are DROPPED (shown as small 'x', not part of the surface).
With one rep/cell a held-both cell's value is 0/1 (built or not); we Gaussian-pool the
held-both cells (normalized convolution, --smooth gaussian, default) into a legible
build-recognition surface, or show raw with --smooth none.

Per-model headline P(built | held) is printed in each panel title.

Usage: PYTHONPATH=. python -m scripts.plot_recognition_panels <haiku_dir> <sonnet_dir> \
           <opus_dir> [--n-hi 20] [--smooth gaussian] [--bandwidth 1]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import scripts.sweep_config as cfg

T_LO, T_HI = 2, 10


def built(row: dict) -> bool:
    return bool(row.get("built_machine")) or any("fuse into" in o for o in row.get("obs", []))


def collected(row: dict) -> set:
    """Byproduct letters the model ever held (pickup + holding messages)."""
    s = set()
    for o in row.get("obs", []):
        s.update(re.findall(r"and a (\w+)\.", o))
        s.update(re.findall(r"hold \d+ (\w+)\(s\)", o))
    return s


def held_both(row: dict) -> bool:
    return set(row["labels"]["recipe"]) <= collected(row)


def load_cells(run_dir: Path):
    """-> (short, {(T,N): mean built over held-both reps}, headline pooled P(built|held)).

    Reps per cell are AVERAGED (the cell surface value is the per-cell built rate);
    the headline is pooled over all held-both episodes (numerator/denominator), so it
    is identical to the old 0/1 value for a 1-rep grid and tightens with more reps."""
    from collections import defaultdict
    rows = [json.loads(l) for l in (run_dir / "episodes.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    m0 = rows[0].get("model", "?") if rows else "?"
    short = m0.split("-")[1] if m0.startswith("claude-") else m0
    acc = defaultdict(list)  # (T,N) -> built01 for each held-both rep
    for r in rows:
        if held_both(r):
            acc[(r["n_types"], r["n"])].append(int(built(r)))
    cells = {k: sum(v) / len(v) for k, v in acc.items()}
    nb = sum(sum(v) for v in acc.values())
    nd = sum(len(v) for v in acc.values())
    head = nb / nd if nd else float("nan")
    return short, cells, head


def pooled(cells, n_hi, bw):
    """Normalized Gaussian convolution of the 0/1 held-both cells onto the grid."""
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
    ap.add_argument("--smooth", choices=["none", "gaussian"], default="gaussian")
    ap.add_argument("--bandwidth", type=float, default=1.0)
    ap.add_argument("--outdir", default="figs/toolworld")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.dirs]
    loaded = [load_cells(d) for d in dirs]

    # boundary E[build] = E[grind]
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
    for ax, (short, cells, head) in pairs:
        print(f"{short}: held-both cells {len(cells)} | recognition P(built|held)={head:.2f}")
        if args.smooth == "gaussian":
            gt, gn, Z = pooled(cells, args.n_hi, args.bandwidth)
            Z = np.clip(Z, 0, 1)
            mesh = ax.pcolormesh(gt, gn, Z, cmap="RdYlGn", vmin=0, vmax=1, shading="nearest")
        # scatter the held-both cells, colored by built 0/1
        if cells:
            arr = np.array([[t, n, v] for (t, n), v in cells.items()], float)
            sc = ax.scatter(arr[:, 0], arr[:, 1], c=arr[:, 2], cmap="RdYlGn", vmin=0, vmax=1,
                            s=46, edgecolors="black", linewidths=0.7, zorder=3)
            if mesh is None:
                mesh = sc
        # dropped cells (never held both) as small x
        held_keys = set(cells)
        dropped = [(t, n) for t in range(T_LO, T_HI + 1) for n in range(1, args.n_hi + 1)
                   if (t, n) not in held_keys]
        if dropped:
            dv = np.array(dropped, float)
            ax.scatter(dv[:, 0], dv[:, 1], marker="x", c="0.45", s=26, linewidths=0.8,
                       zorder=2, label="never held both (dropped)")
        ax.plot(ts, bnd, "k--", lw=1.2, label="E[build]=E[grind]")
        ax.set_title(f"{short.capitalize()}\nrecognition P(built | held) = {head:.2f}  "
                     f"(n={len(cells)})")
        ax.set_xlabel("byproduct types T")
        ax.set_xlim(T_LO - 0.5, T_HI + 0.5); ax.set_ylim(0.5, args.n_hi + 0.5)
    for _lax in row_lefts: _lax.set_ylabel("number of doors N")
    pairs[0][0].legend(loc="upper right", fontsize=7, framealpha=0.9)

    smooth_note = "gaussian-pooled bw=%g" % args.bandwidth if args.smooth == "gaussian" else "raw 0/1"
    fig.suptitle(f"Recognition: P(built | held both recipe ingredients) — held-both cells only "
                 f"({smooth_note}; grind-calibrated budget; single-draw)", y=1.02, fontsize=12)
    cb = fig.colorbar(mesh, ax=_axg, fraction=0.025, pad=0.02)
    cb.set_label("P(built | held both)")

    out = Path(args.outdir) / f"fig_recognition_panels_Nle{args.n_hi}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
