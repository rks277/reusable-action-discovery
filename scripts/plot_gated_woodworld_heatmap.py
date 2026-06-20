"""(T, N)-plane heatmaps for the gated WoodWorld sweep: RECOGNITION + SOLVE.

Two side-by-side Gaussian-pooled heatmaps (ToolWorld style), x = tree-resource types T
(2..6), y = distinct wood kinds N (10..30), for one model (Haiku):

  RECOGNITION = P(built axe | ever held the entry materials)  -- held_base cells only
  SOLVE RATE  = P(solved = opened all N containers)            -- all cells

Also prints the two-step decomposition for the two_layer recipe:
  step1 = P(built_part | held_base),  step2 = P(built_axe | built_part).

Writes to figs/gated-woodworld/. Run from repo root:
  PYTHONPATH=. python scripts/plot_gated_woodworld_heatmap.py runs/<dir>
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

T_LO, T_HI = 2, 6
N_LO, N_HI = 10, 30


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
    """{(T,N): mean(built_axe | held_base)} over held_base cells; pooled headline."""
    acc = defaultdict(list)
    for r in rows:
        if r.get("held_base"):
            acc[(r["n_types"], r["n"])].append(int(bool(r.get("built_axe"))))
    cells = {k: sum(v) / len(v) for k, v in acc.items()}
    nb = sum(sum(v) for v in acc.values())
    nd = sum(len(v) for v in acc.values())
    return cells, (nb / nd if nd else float("nan")), nd


def step1_cells(rows):
    """{(T,N): mean(built_part | held_base)} over held_base cells -- step-1 discovery."""
    acc = defaultdict(list)
    for r in rows:
        if r.get("held_base"):
            acc[(r["n_types"], r["n"])].append(int(bool(r.get("built_part"))))
    cells = {k: sum(v) / len(v) for k, v in acc.items()}
    nb = sum(sum(v) for v in acc.values())
    nd = sum(len(v) for v in acc.values())
    return cells, (nb / nd if nd else float("nan")), nd


def step2_cells(rows):
    """{(T,N): mean(built_axe | built_part)} over built_part cells -- step-2 follow-through."""
    acc = defaultdict(list)
    for r in rows:
        if r.get("built_part"):
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


def draw(ax, cells, bw, title, all_cells, drop_label="no episode in denominator"):
    gt, gn, Z = pooled(cells, bw)
    Z = np.clip(Z, 0, 1)
    mesh = ax.pcolormesh(gt, gn, Z, cmap="RdYlGn", vmin=0, vmax=1, shading="nearest")
    if cells:
        arr = np.array([[t, n, v] for (t, n), v in cells.items()], float)
        ax.scatter(arr[:, 0], arr[:, 1], c=arr[:, 2], cmap="RdYlGn", vmin=0, vmax=1,
                   s=46, edgecolors="black", linewidths=0.7, zorder=3)
    held = set(cells)
    dropped = [(t, n) for t in range(T_LO, T_HI + 1) for n in range(N_LO, N_HI + 1)
               if (t, n) in all_cells and (t, n) not in held]
    if dropped:
        dv = np.array(dropped, float)
        ax.scatter(dv[:, 0], dv[:, 1], marker="x", c="0.45", s=24, linewidths=0.8,
                   zorder=2, label=drop_label)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("tree-resource types T")
    ax.set_xlim(T_LO - 0.5, T_HI + 0.5)
    ax.set_ylim(N_LO - 0.5, N_HI + 0.5)
    return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--bandwidth", type=float, default=1.0)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    short, rows = load(run_dir)
    recipe = rows[0].get("recipe_mode", "two_layer") if rows else "two_layer"
    rdesc = {"two_layer": "r0+r1→part, part+part→axe",
             "distinct": "r0+r1→axe", "same": "stick+stick→axe"}.get(recipe, recipe)

    rcells, rhead, rn = recognition_cells(rows)
    scells, shead, sn = solve_cells(rows)
    s1cells, s1head, s1n = step1_cells(rows)
    s2cells, s2head, s2n = step2_cells(rows)
    all_cells = {(r["n_types"], r["n"]) for r in rows}

    two_layer = recipe == "two_layer"
    print(f"{short} ({recipe}): recognition P(axe|held_base) = {rhead:.2f} (n={rn}) | "
          f"solve = {shead:.2f} (n={sn})")
    print(f"  decomposition: step1 P(part|held_base) = {s1head:.2f} (n={s1n}) | "
          f"step2 P(axe|built_part) = {s2head:.2f} (n={s2n})")

    if two_layer:
        # split the recognition panel into step-1 (discovery) and step-2 (follow-through)
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharey=True)
        mesh = draw(axes[0], s1cells, args.bandwidth,
                    f"STEP 1  P(built part | held base) = {s1head:.2f}", all_cells,
                    drop_label="no held-base episode")
        draw(axes[1], s2cells, args.bandwidth,
             f"STEP 2  P(built axe | built part) = {s2head:.2f}", all_cells,
             drop_label="never built a part")
        draw(axes[2], scells, args.bandwidth,
             f"SOLVE RATE  P(opened all N) = {shead:.2f}", all_cells,
             drop_label="(all cells present)")
        sub = (f"step-1 discovery (r0+r1→part) vs step-2 follow-through (part+part→axe) "
               f"vs solve\noverall recognition P(axe|held_base) = {rhead:.2f}")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
        mesh = draw(axes[0], rcells, args.bandwidth,
                    f"RECOGNITION  P(built axe | held base materials) = {rhead:.2f}", all_cells,
                    drop_label="no held-base episode")
        draw(axes[1], scells, args.bandwidth,
             f"SOLVE RATE  P(opened all N) = {shead:.2f}", all_cells,
             drop_label="(all cells present)")
        sub = "recognition & solve"
    axes[0].set_ylabel("number of distinct wood kinds N")
    for ax in axes:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.suptitle(f"GATED WoodWorld ({short}, recipe={recipe}) over (T, N): {sub}\n"
                 f"(gaussian-pooled bw={args.bandwidth:g}, 1 rep/cell; goal = collect all N "
                 f"distinct kinds, GATED; {rdesc}; directed axe→key→open)",
                 y=1.04, fontsize=11)
    cb = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("probability")

    outdir = Path(args.outdir) if args.outdir else Path("figs/gated-woodworld")
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"fig_gatedwood_{short}_{recipe}_T{T_LO}-{T_HI}_N{N_LO}-{N_HI}_recognition_solve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
