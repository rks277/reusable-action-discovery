"""Three-panel HELD-BOTH (gathering) map for Haiku / Sonnet / Opus, one figure,
Gaussian-pooled. Read from the ORIGINAL region-sweep squares (the 180-run recognition
sweeps), NOT the efficiency runs.

Each cell is 1 if that episode ever held BOTH recipe ingredients at once, 0 if not
(detected from the "hold N TYPE(s)" observations, == analyze_recognition_latency.
acquire_both_turn). This is the GATHERING rate -- the P(held both) denominator that the
build-rate factorizes through (build = gather x recognize) and that selected which runs
fed the efficiency sweep. Single rep/cell -> 0/1, pooled into a surface. Green = gathered
both ingredients, red = never held both.

Usage: PYTHONPATH=. python -m scripts.plot_held_both_panels <haiku_region_dir> \
           <sonnet_region_dir> <opus_region_dir> [--n-hi 20] [--bandwidth 1]
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
from scripts.analyze_recognition_latency import acquire_both_turn

T_LO, T_HI = 2, 10


def load_cells(run_dir: Path):
    """Reps per cell are AVERAGED into the cell value; headline is pooled over all
    episodes. Identical to the old 0/1 value for a 1-rep grid; tightens with reps."""
    from collections import defaultdict
    rows = [json.loads(l) for l in (run_dir / "episodes.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    m0 = rows[0].get("model", "?") if rows else "?"
    short = m0.split("-")[1] if m0.startswith("claude-") else m0
    acc = defaultdict(list)  # (T,N) -> held-both 0/1 for each rep
    for r in rows:
        acc[(r["n_types"], r["n"])].append(
            int(acquire_both_turn(r.get("obs"), set(r["labels"]["recipe"])) is not None))
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
    ap.add_argument("--poster", action="store_true",
                    help="poster mode: panel title = model name only, no suptitle")
    ap.add_argument("--outdir", default="figs/toolworld")
    args = ap.parse_args()

    loaded = [load_cells(Path(d)) for d in args.dirs]

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
        print(f"{short}: held-both rate {head:.2f} ({sum(cells.values())}/{len(cells)})")
        gt, gn, Z = pooled(cells, args.n_hi, args.bandwidth)
        Z = np.clip(Z, 0, 1)
        mesh = ax.pcolormesh(gt, gn, Z, cmap="RdYlGn", vmin=0, vmax=1, shading="nearest")
        arr = np.array([[t, n, v] for (t, n), v in cells.items()], float)
        ax.scatter(arr[:, 0], arr[:, 1], c=arr[:, 2], cmap="RdYlGn", vmin=0, vmax=1,
                   s=46, edgecolors="black", linewidths=0.7, zorder=3)
        ax.plot(ts, bnd, "k--", lw=1.2, label="E[build]=E[grind]")
        ax.set_title(short.capitalize() if args.poster
                     else f"{short.capitalize()}\nheld-both rate = {head:.2f}")
        ax.set_xlabel("byproduct types T")
        ax.set_xlim(T_LO - 0.5, T_HI + 0.5); ax.set_ylim(0.5, args.n_hi + 0.5)
    for _lax in row_lefts: _lax.set_ylabel("number of doors N")
    pairs[0][0].legend(loc="upper right", fontsize=7, framealpha=0.9)

    if not args.poster:
        fig.suptitle("Gathering: did the model ever hold BOTH recipe ingredients at once, over (T, N)\n"
                     f"(original 180-run region sweeps; gaussian-pooled bw={args.bandwidth:g}; "
                     "green = held both, red = never)",
                     y=1.04, fontsize=12)
    cb = fig.colorbar(mesh, ax=_axg, fraction=0.025, pad=0.02)
    cb.set_label("" if args.poster else "P(held both ingredients)")

    out = Path(args.outdir) / f"fig_industry_panels_Nle{args.n_hi}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
