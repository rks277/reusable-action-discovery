"""ToolWorld-style EFFICIENCY / RECOGNITION / CURIOSITY panels for woodworld.

Three figures, each a Haiku / Sonnet / Opus row of Gaussian-pooled (gather prob p,
target wood N) heatmaps in the woodworld region style (gaussian_filter over the
integer lattice + shading="nearest", RdYlGn, E[build]=E[grind] boundary overlaid).
All read the same hint-on region sweeps (1 rep/cell, 171 cells/model); every metric
is recomputed from the recorded actions/obs, so no re-run is needed.

These mirror ToolWorld's build = C·R·E solve-rate decomposition (docs/6-17-summary):

  CURIOSITY    C = P(hold ingredients) = P(reached the recipe ingredients / got
               sticks) over ALL cells. The acquisition stage (ToolWorld's "industry"
               panel = P(held both)). green = acquired the ingredients.

  RECOGNITION  R = P(built axe | got sticks) over GOT-STICKS cells only (ToolWorld's
               P(built | held both): sticks are the discovered intermediate, wood is
               freely gathered). Never-got-sticks cells dropped (grey x). green =
               follows through to the tool.

  EFFICIENCY   E = P(solved | tool built) over BUILT-AXE cells only (ToolWorld's
               exploitation stage). Given the axe, did it reach the goal within
               budget? Never-built cells dropped (grey x). green = the tool paid off.

  (SOLVE = P(solved) over all cells; PERSISTENCE = how far the build chain got,
   0 neither / 1 sticks / 2 axe.)

Usage: PYTHONPATH=. python -m scripts.plot_woodworld_panels [--panel all] \
           [--haiku DIR --sonnet DIR --opus DIR] [--bandwidth 1.0]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plot_woodworld_region_heatmap import (
    P_GRID, N_GRID, P_LO, P_HI, N_LO, N_HI, pooled, boundary)

DEFAULT_DIRS = {
    "haiku": "runs/haiku_woodworld_region_r1_hint_N2-20_20260617_134321",
    "sonnet": "runs/sonnet_woodworld_region_r1_hint_N2-20_20260617_135245",
    "opus": "runs/opus_woodworld_region_r1_hint_N2-20_20260617_141212",
}
OUT_DIR = Path("figs/woodworld/panels")


# --- per-episode metric helpers (recomputed from recorded actions/obs) ----------
def got_sticks(r: dict) -> bool:
    """Did the episode ever produce the stick intermediate? (robust to obfuscation:
    read the recorded stick label and scan combine outputs). Returns False for
    variants with no stick (e.g. the iso single-step recipe)."""
    stk = r.get("labels", {}).get("stick")
    if stk is None:
        return False
    for o in r["obs"]:
        if o.startswith("You combine") and "(new)" in o:
            out = o.split(" into ", 1)[1].split(" (new)")[0]
            if stk in out.split():
                return True
    return False


def held_ingredients(r: dict) -> bool:
    """Recipe-agnostic acquisition signal C = P(hold the tool recipe's ingredients).
    Prefers the recorded `held_ingredients` flag (iso + new base runs); falls back
    to got_sticks for legacy base runs (where the stick intermediate ~= held both
    axe ingredients, since wood is freely gathered)."""
    if "held_ingredients" in r:
        return bool(r["held_ingredients"])
    return got_sticks(r)


def build_stage(r: dict) -> float:
    """Build persistence: 2 if the episode built the axe, 1 if it only acquired the
    recipe ingredients, 0 if neither. axe implies ingredients, so axe is first."""
    if r.get("built_axe"):
        return 2.0
    if held_ingredients(r):
        return 1.0
    return 0.0


def load(run_dir: str) -> tuple[str, list[dict]]:
    rows = [json.loads(l) for l in (Path(run_dir) / "episodes.jsonl").read_text().splitlines()
            if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    m0 = rows[0].get("model", "?") if rows else "?"
    short = m0.split("-")[1] if m0.startswith("claude-") else m0.replace(":", "-")
    return short, rows


def cell_key(r: dict) -> tuple[float, int]:
    return (round(r["gather_prob"], 1), r["n"])


# --- generic three-panel renderer ----------------------------------------------
def _render(loaded, value_fn, keep_fn, title, cb_label, fname, bw,
            vmin=0.0, vmax=1.0, cmap="RdYlGn", headline_fn=None,
            cb_ticks=None, cb_ticklabels=None, outdir=None):
    """value_fn(r)->float|None (cell value, None to skip), keep_fn(r)->bool (does the
    episode contribute / is its cell 'live'). Cells with no live episode are drawn as
    grey x (coverage shown, not painted). headline_fn(rows)->float for panel title.
    outdir overrides where the figure is written (else region/<model> for single-model
    runs, panels/ for the trio)."""
    # infer the (p, N) lattice from the data so non-default grids (e.g. N=10..90
    # step 10) render correctly; fall back to the module grids if empty.
    n_grid = sorted({r["n"] for _, rows in loaded for r in rows}) or N_GRID
    p_grid = sorted({round(r["gather_prob"], 1) for _, rows in loaded for r in rows}) or P_GRID
    n_lo, n_hi = n_grid[0], n_grid[-1]
    nstep = (n_grid[1] - n_grid[0]) if len(n_grid) > 1 else 1
    # boundary uses THIS run's recipe economics (recorded axe_cost) when present
    axe = next((r.get("axe_cost") for _, rows in loaded for r in rows
                if r.get("axe_cost")), None)
    bp, bn = boundary(axe_cost_override=tuple(axe) if axe else None, n_hi=max(n_hi, 20))

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
    for ax, (short, rows) in pairs:
        agg = defaultdict(list)
        for r in rows:
            if keep_fn(r):
                v = value_fn(r)
                if v is not None:
                    agg[cell_key(r)].append(v)
        cells = {k: float(np.mean(v)) for k, v in agg.items() if v}
        head = headline_fn(rows) if headline_fn else (np.mean(list(cells.values()))
                                                      if cells else float("nan"))
        print(f"{short}: {len(cells)} live cells | headline {head:.2f}")

        gp, gn, Z = pooled(cells, bw, clip=(vmin, vmax), p_grid=p_grid, n_grid=n_grid)
        mesh = ax.pcolormesh(gp, gn, np.clip(Z, vmin, vmax), cmap=cmap,
                             vmin=vmin, vmax=vmax, shading="nearest")
        if cells:
            arr = np.array([[p, n, v] for (p, n), v in cells.items()], float)
            ax.scatter(arr[:, 0], arr[:, 1], c=np.clip(arr[:, 2], vmin, vmax), cmap=cmap,
                       vmin=vmin, vmax=vmax, s=44, edgecolors="black", linewidths=0.6,
                       zorder=3)
        # cells with no live episode -> grey x (coverage)
        live = set(cells)
        dropped = [(p, n) for p in p_grid for n in n_grid if (p, n) not in live]
        if dropped:
            dv = np.array(dropped, float)
            ax.scatter(dv[:, 0], dv[:, 1], marker="x", c="0.5", s=24, linewidths=0.8,
                       zorder=2, label="no live episode")
        ax.plot(bp, bn, "k--", lw=1.4, zorder=4, label="E[build]=E[grind]")
        ax.set_title(f"{short.capitalize()}   ({head:.2f})", fontsize=13)
        ax.set_xlabel("gather probability p")
        ax.set_xlim(P_LO - 0.03, P_HI + 0.03)
        ax.set_ylim(n_lo - nstep / 2, n_hi + nstep / 2)
    for _lax in row_lefts: _lax.set_ylabel("target wood N")
    pairs[0][0].legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.suptitle(title, y=1.02, fontsize=13)
    cb = fig.colorbar(mesh, ax=_axg, fraction=0.025, pad=0.02)
    cb.set_label(cb_label)
    if cb_ticks is not None:
        cb.set_ticks(cb_ticks)
        if cb_ticklabels is not None:
            cb.set_ticklabels(cb_ticklabels)
    # explicit outdir wins; else single-model renders group under region/<model>/ and
    # the multi-model trio stays in panels/
    out_dir = (Path(outdir) if outdir
               else Path("figs/woodworld/region") / loaded[0][0] if len(loaded) == 1
               else OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} (+ .pdf)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel",
                    choices=["efficiency", "recognition", "curiosity", "solve",
                             "persistence", "all"],
                    default="all")
    ap.add_argument("--haiku", default=DEFAULT_DIRS["haiku"])
    ap.add_argument("--sonnet", default=DEFAULT_DIRS["sonnet"])
    ap.add_argument("--opus", default=DEFAULT_DIRS["opus"])
    ap.add_argument("--dirs", default=None,
                    help="comma-separated run dirs (overrides --haiku/--sonnet/--opus); "
                         "renders one panel per dir, e.g. a single Haiku N=10-90 run")
    ap.add_argument("--outdir", default=None,
                    help="explicit output folder for the figures (e.g. figs/woodworld/iso); "
                         "default routes single-model->region/<model>/, trio->panels/")
    ap.add_argument("--bandwidth", type=float, default=1.0)
    args = ap.parse_args()

    run_dirs = (args.dirs.split(",") if args.dirs
                else [args.haiku, args.sonnet, args.opus])
    loaded = [load(d) for d in run_dirs]
    bw = args.bandwidth
    render = partial(_render, outdir=args.outdir)   # thread outdir into every panel
    # N-range tag for filenames (so single / large-N runs don't clobber the 2-20 panels)
    n_all = sorted({r["n"] for _, rows in loaded for r in rows})
    ntag = f"N{n_all[0]}-{n_all[-1]}" if n_all else f"Nle{N_HI}"
    models = "_".join(s for s, _ in loaded)
    default_trio = run_dirs == [DEFAULT_DIRS["haiku"], DEFAULT_DIRS["sonnet"],
                                DEFAULT_DIRS["opus"]]
    htag = "hint" if (loaded and loaded[0][1] and loaded[0][1][0].get("hint")) else "nohint"

    def fnm(panel: str) -> str:
        # keep the documented 2-20 trio names stable; tag everything else by
        # models + hint state + N range (so hint/nohint runs don't collide)
        return (f"fig_woodworld_{panel}_panels_Nle{N_HI}.png" if default_trio
                else f"fig_woodworld_{panel}_{models}_{htag}_{ntag}.png")

    if args.panel in ("efficiency", "all"):
        render(
            loaded, value_fn=lambda r: float(bool(r["solved"])),
            keep_fn=lambda r: bool(r["built_axe"]),
            title=("woodworld EFFICIENCY: E = P(solved | tool built) over (p, N), "
                   "built-axe cells only\n"
                   f"[gaussian-pooled bw={bw:g}, 1 rep/cell; the exploitation stage E in "
                   "build = curiosity × recognition × efficiency; given the axe, did it "
                   "reach the goal in budget? green = yes]"),
            cb_label="P(solved | built)", fname=fnm("efficiency"), bw=bw,
            headline_fn=lambda rows: (
                sum(1 for r in rows if r["built_axe"] and r["solved"]) /
                max(1, sum(1 for r in rows if r["built_axe"]))))

    if args.panel in ("recognition", "all"):
        render(
            loaded, value_fn=lambda r: float(bool(r["built_axe"])),
            keep_fn=held_ingredients,
            title=("woodworld RECOGNITION: R = P(built axe | held ingredients) over "
                   "(p, N), held-ingredients cells only\n"
                   f"[gaussian-pooled bw={bw:g}, 1 rep/cell; given the recipe inputs, "
                   "did it build the tool? green = follows through]"),
            cb_label="P(built | held ingredients)",
            fname=fnm("recognition"), bw=bw,
            headline_fn=lambda rows: (
                sum(1 for r in rows if held_ingredients(r) and r["built_axe"]) /
                max(1, sum(1 for r in rows if held_ingredients(r)))))

    if args.panel in ("solve", "all"):
        render(
            loaded, value_fn=lambda r: float(bool(r["solved"])), keep_fn=lambda r: True,
            title=("woodworld SOLVE RATE: P(held N wood within budget) over (p, N), "
                   "all cells\n"
                   f"[gaussian-pooled bw={bw:g}, 1 rep/cell; green = solved; brute "
                   "grind is a viable escape hatch at this budget]"),
            cb_label="P(solved)", fname=fnm("solve"), bw=bw)

    if args.panel in ("persistence", "all"):
        render(
            loaded, value_fn=build_stage, keep_fn=lambda r: True,
            title=("woodworld BUILD PERSISTENCE: how far the build chain got over "
                   "(p, N), all cells\n"
                   f"[gaussian-pooled bw={bw:g}, 1 rep/cell; 2 = built axe, "
                   "1 = held ingredients only, 0 = neither; green = furthest]"),
            cb_label="build stage reached", vmin=0.0, vmax=2.0,
            fname=fnm("persistence"), bw=bw,
            cb_ticks=[0, 1, 2],
            cb_ticklabels=["0 neither", "1 ingredients", "2 axe"])

    if args.panel in ("curiosity", "all"):
        render(
            loaded, value_fn=lambda r: float(held_ingredients(r)), keep_fn=lambda r: True,
            title=("woodworld CURIOSITY: C = P(hold ingredients) = P(reached the tool "
                   "recipe's ingredients) over (p, N), all cells\n"
                   f"[gaussian-pooled bw={bw:g}, 1 rep/cell; the acquisition stage C in "
                   "build = curiosity × recognition × efficiency; green = acquired "
                   "ingredients]"),
            cb_label="P(hold ingredients)", fname=fnm("curiosity"), bw=bw)


if __name__ == "__main__":
    main()
