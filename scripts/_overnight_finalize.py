"""Overnight finalize for the qwen3.5 grid_v3 sweep.

Computes a build/solve summary for qwen3.5:4b and :9b and renders heatmaps,
reusing the existing grid_v3 plotting helpers. Figure generation is wrapped so a
plotting failure never loses the numeric summary. Safe to run on partial data.
"""
import sys
from pathlib import Path
import numpy as np

from scripts.plot_grid_sweep_v3 import (
    N_LO, N_HI, T_LO, T_HI, gaussian_fill, cell_value, built, load,
)

TAGS = [("qwen3-5-4b", "Qwen3.5-4B"), ("qwen3-5-9b", "Qwen3.5-9B")]
OUTDIR = Path("runs/grid_v3_qwen_summary")


def latest_run(tag):
    cands = sorted(p for p in Path("runs").rglob(f"grid_sweep_v3_{tag}_*") if p.is_dir())
    return cands[-1] if cands else None


def solved(r):
    return 1.0 if r.get("solved") else 0.0


def summarize():
    data = {}
    lines = []
    for tag, label in TAGS:
        run = latest_run(tag)
        if run is None:
            lines.append(f"{label}: NO RUN DIR FOUND")
            continue
        rows = load(run / "episodes.jsonl")
        data[tag] = (label, rows)
        if not rows:
            lines.append(f"{label}: {run.name} (0 episodes)")
            continue
        b = np.mean([built(r) for r in rows])
        s = np.mean([solved(r) for r in rows])
        lines.append(f"{label:12s} {run.name}: n={len(rows):3d}  build={b:.3f}  solve={s:.3f}")
        # build/solve vs N (low/mid/high)
        for lo, hi in [(1, 7), (8, 14), (15, 20)]:
            sub = [r for r in rows if lo <= r["n"] <= hi]
            if sub:
                bb = np.mean([built(r) for r in sub])
                ss = np.mean([solved(r) for r in sub])
                lines.append(f"    N {lo:2d}-{hi:2d}: n={len(sub):3d}  build={bb:.3f}  solve={ss:.3f}")
    return data, lines


def make_fig(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [("build rate", built), ("solve rate", solved)]
    extent = [T_LO - 0.5, T_HI + 0.5, N_LO - 0.5, N_HI + 0.5]
    fig, axes = plt.subplots(2, len(TAGS), figsize=(5 * len(TAGS), 8.8),
                             constrained_layout=True, sharex=True, sharey=True)
    im = None
    for ri, (mlabel, fn) in enumerate(metrics):
        for ci, (tag, label) in enumerate(TAGS):
            ax = axes[ri][ci]
            if tag not in data:
                ax.set_axis_off(); continue
            _, rows = data[tag]
            grid = cell_value(rows, fn)
            im = ax.imshow(gaussian_fill(grid, 1.2), origin="lower", aspect="auto",
                           cmap="viridis", extent=extent, vmin=0, vmax=1, interpolation="nearest")
            ax.scatter([r["n_types"] for r in rows], [r["n"] for r in rows], s=7,
                       c="white", edgecolors="black", linewidths=0.3, alpha=0.6)
            mean = float(np.nanmean(grid))
            ax.set_title(f"{label}  ({mlabel} {mean:.2f})", fontsize=11)
            ax.set_xlabel("T (recipe types)"); ax.set_ylabel("N (doors)")
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"fig_grid_v3_qwen_build_solve.{ext}", dpi=140, bbox_inches="tight")
    return OUTDIR / "fig_grid_v3_qwen_build_solve.png"


def main():
    data, lines = summarize()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary = "\n".join(lines)
    (OUTDIR / "summary.txt").write_text(summary + "\n")
    print(summary)
    try:
        p = make_fig(data)
        print(f"\nFIGURE: {p}")
    except Exception as e:
        print(f"\nFIGURE FAILED (summary still saved): {e!r}")


if __name__ == "__main__":
    main()
