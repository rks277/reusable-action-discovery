"""Analysis for run_beach_sweep.py (the build-vs-grind durability surface).

Groups episodes by (model, shovel_durability, grid_size) and reads the surface
along the durability axis -- the beach's "build margin" lever, analogous to n in
analyze_nt_sweep.py:
  - tiny durability  -> blind digging is hopeless, building the map is forced
  - large durability -> grinding is a viable (often cheaper) escape hatch, so
                        building the map becomes an economic CHOICE

Per cell it reports win / built / used-map rates with WILSON score intervals
(robust at 0/k and k/k, where bootstrap degenerates), plus mean digs, wasted
digs, actions, and build-turn. The headline disposition signal is built_map:
where it falls as durability rises is where the model flips grind->build.

Each beach episode already logs its own instrumentation (won / built_map /
used_map / digs / ...), so unlike the toolworld analyzers there is no replay/
verify step -- we read the logged fields directly.

Usage: python -m scripts.analyze_beach_sweep runs/beach_sweep_*/episodes.jsonl
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-8",
         "claude-fable-5", "gpt-5", "gemini-2.5-pro"]
COLORS = {"haiku": "seagreen", "sonnet": "steelblue", "opus": "indianred",
          "fable": "mediumvioletred", "gpt": "darkorange", "gemini": "purple"}
FAM = {"haiku": "Haiku", "sonnet": "Sonnet", "opus": "Opus", "fable": "Fable",
       "gpt": "GPT-5", "gemini": "Gemini"}

# 2x2 outcome: did it build the map x did it win.
OC_CATS = ["built+won", "built+lost", "brute+won", "brute+lost"]
OC_LEG = {"built+won": "built + won", "built+lost": "built, lost",
          "brute+won": "brute + won", "brute+lost": "brute, lost"}
OC_COLORS = {"built+won": "seagreen", "built+lost": "goldenrod",
             "brute+won": "steelblue", "brute+lost": "indianred"}

# USD per 1M tokens (input, output), Anthropic list pricing (cached 2026-05).
# Cache reads bill at 0.1x input, writes at 1.25x input (5-min ephemeral).
PRICING = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-fable-5": (10.00, 50.00),
}


def short(model: str) -> str:
    """Family-ish short label from a model id (matches the toolworld analyzers'
    naming so tables read the same)."""
    s = model.lower()
    for k in ("haiku", "sonnet", "opus", "fable", "gpt", "gemini"):
        if k in s:
            return "gpt-5" if k == "gpt" else k
    return model


def fam_label(m: str) -> str:
    s = short(m)
    return next((v for k, v in FAM.items() if k in s), s)


def model_color(m: str) -> str:
    s = short(m)
    return next((c for k, c in COLORS.items() if k in s), "gray")


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion. Returns (p, lo, hi). Stays sane at
    k=0 and k=n, unlike the bootstrap/Wald interval."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def outcome(row: dict) -> str:
    return ("built" if row.get("built_map") else "brute") + \
           ("+won" if row.get("won") else "+lost")


def cost_of(usage: dict, model: str):
    """Episode cost in USD, or None if the model is unpriced (non-Anthropic:
    tokens are logged but we don't guess rates)."""
    rate = PRICING.get(model)
    if not rate:
        return None
    ri, ro = rate
    u = usage or {}
    return (u.get("input_tokens", 0) * ri
            + u.get("output_tokens", 0) * ro
            + u.get("cache_read_tokens", 0) * ri * 0.10
            + u.get("cache_write_tokens", 0) * ri * 1.25) / 1e6


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float("nan")


def main():
    if len(sys.argv) < 2:
        print("usage: python -m scripts.analyze_beach_sweep "
              "runs/beach_sweep_*/episodes.jsonl")
        sys.exit(1)
    path = Path(sys.argv[1])
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    nerr = sum(1 for r in rows if r.get("error"))
    rows = [r for r in rows if not r.get("error")]
    if nerr:
        print(f"({nerr} episode(s) errored and were dropped)\n")

    # group by (model, durability, grid)
    cells = defaultdict(list)
    for r in rows:
        cells[(r["model"], r["shovel_durability"], r["grid_size"])].append(r)

    models = ([m for m in ORDER if any(k[0] == m for k in cells)]
              + sorted({k[0] for k in cells} - set(ORDER)))
    durs = sorted({k[1] for k in cells})
    grids = sorted({k[2] for k in cells})

    summ = {}
    for key, g in cells.items():
        n = len(g)
        k_won = sum(bool(e["won"]) for e in g)
        k_built = sum(bool(e["built_map"]) for e in g)
        k_used = sum(bool(e["used_map"]) for e in g)
        bts = [e["build_turn"] for e in g if e.get("build_turn") is not None]
        oc = defaultdict(int)
        for e in g:
            oc[outcome(e)] += 1
        wins = [e for e in g if e["won"]]
        brute_win_share = (sum(1 for e in wins if not e["built_map"]) / len(wins)
                           if wins else float("nan"))
        summ[key] = {
            "n_eps": n,
            "win": wilson(k_won, n),
            "built": wilson(k_built, n),
            "used": wilson(k_used, n),
            "digs": _mean([e["digs"] for e in g]),
            "wasted": _mean([e["wasted_digs"] for e in g]),
            "actions": _mean([e["total_actions"] for e in g]),
            "papers": _mean([e["papers_collected"] for e in g]),
            "bt": _mean(bts),
            "brute_win_share": brute_win_share,
            "oc": oc,
        }

    print(f"Build-vs-grind surface: {len(models)} model(s) x durability in "
          f"{durs} x grid in {grids}  ({len(rows)} episodes)\n")
    hdr = (f"{'model':14s} {'dur':>4} {'grid':>4} {'eps':>4} "
           f"{'win [95% CI]':>22} {'built [95% CI]':>22} {'usedmap':>8} "
           f"{'digs':>5} {'wast':>5} {'acts':>6} {'build@':>7} {'bruteW%':>8}")
    print(hdr)
    print("-" * len(hdr))
    for m in models:
        for d in durs:
            for grid in grids:
                s = summ.get((m, d, grid))
                if not s:
                    continue
                wp, wlo, whi = s["win"]
                bp, blo, bhi = s["built"]
                up = s["used"][0]
                bws = (f"{s['brute_win_share']*100:.0f}%"
                       if s["brute_win_share"] == s["brute_win_share"] else "-")
                bt = f"{s['bt']:.1f}" if s["bt"] == s["bt"] else "-"
                print(f"{short(m):14s} {d:>4} {grid:>4} {s['n_eps']:>4} "
                      f"{f'{wp:.2f} [{wlo:.2f}-{whi:.2f}]':>22} "
                      f"{f'{bp:.2f} [{blo:.2f}-{bhi:.2f}]':>22} "
                      f"{up:>8.2f} {s['digs']:>5.1f} {s['wasted']:>5.1f} "
                      f"{s['actions']:>6.1f} {bt:>7} {bws:>8}")
    print("\n  usedmap = fraction that read the map; digs/wast/acts = mean per "
          "episode; build@ = mean turn the map formed (built episodes only);\n"
          "  bruteW% = share of WINS achieved by brute force (no map) -- the "
          "grind-over-build signal.")

    print_token_usage(rows, models, durs)
    print_costs(rows, models, durs)
    fig_build_rate(rows, models, path.parent / "fig_beach_build_rate.png")
    fig_funnel(rows, models, path.parent / "fig_beach_funnel.png")
    fig_win_heatmap(rows, models, path.parent / "fig_beach_win_heatmap.png")
    fig_win_heatmap_smoothed(rows, models,
                             path.parent / "fig_beach_win_heatmap_smoothed.png")
    fig_win_given_built_heatmap(rows, models,
                                path.parent / "fig_beach_win_given_built.png")
    fig_built_heatmap(rows, models, path.parent / "fig_beach_built_heatmap.png")
    fig_win_via_built_heatmap(rows, models,
                              path.parent / "fig_beach_win_via_built.png")
    fig_win_via_built_minus_win_heatmap(
        rows, models, path.parent / "fig_beach_win_via_built_minus_win.png")
    fig_win_given_notbuilt_heatmap(
        rows, models, path.parent / "fig_beach_win_given_notbuilt.png")
    fig_build_win_by_grid(rows, models,
                          path.parent / "fig_beach_build_win_by_grid.png")
    fig_rates(summ, models, durs, grids, path.parent / "fig_beach_rates.png")
    fig_outcomes(summ, models, durs, grids, path.parent / "fig_beach_outcomes.png")
    fig_efficiency(summ, models, durs, grids, path.parent / "fig_beach_efficiency.png")


def print_token_usage(rows, models, durs):
    FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens",
              "cache_write_tokens")
    agg = defaultdict(lambda: {**{k: 0 for k in FIELDS}, "calls": 0, "eps": 0})
    for r in rows:
        a = agg[(r["model"], r["shovel_durability"])]
        u = r.get("usage") or {}
        for k in FIELDS:
            a[k] += int(u.get(k, 0) or 0)
        a["calls"] += int(u.get("calls", 0) or 0)
        a["eps"] += 1

    def cache_pct(a):
        denom = a["input_tokens"] + a["cache_read_tokens"] + a["cache_write_tokens"]
        return f"{a['cache_read_tokens'] / denom * 100:>5.0f}%" if denom else f"{'-':>6}"

    print("\nToken usage by (model, durability) (raw counts; "
          "cache% = read / (in+read+write)):\n")
    hdr = (f"{'model':14s} {'dur':>4} {'eps':>4} {'calls':>6} {'in':>10} "
           f"{'out':>9} {'cache_rd':>10} {'cache_wr':>10} {'cache%':>6}")
    print(hdr)
    print("-" * len(hdr))
    tot = {**{k: 0 for k in FIELDS}, "calls": 0, "eps": 0}
    for m in models:
        for d in durs:
            a = agg.get((m, d))
            if not a:
                continue
            for k in tot:
                tot[k] += a[k]
            print(f"{short(m):14s} {d:>4} {a['eps']:>4} {a['calls']:>6} "
                  f"{a['input_tokens']:>10} {a['output_tokens']:>9} "
                  f"{a['cache_read_tokens']:>10} {a['cache_write_tokens']:>10} "
                  f"{cache_pct(a)}")
    print("-" * len(hdr))
    print(f"{'TOTAL':14s} {'':>4} {tot['eps']:>4} {tot['calls']:>6} "
          f"{tot['input_tokens']:>10} {tot['output_tokens']:>9} "
          f"{tot['cache_read_tokens']:>10} {tot['cache_write_tokens']:>10} "
          f"{cache_pct(tot)}")


def print_costs(rows, models, durs):
    by = defaultdict(lambda: {"eps": 0, "cost": 0.0})
    unpriced = set()
    for r in rows:
        c = cost_of(r.get("usage") or {}, r["model"])
        if c is None:
            unpriced.add(r["model"])
            continue
        a = by[(r["model"], r["shovel_durability"])]
        a["eps"] += 1
        a["cost"] += c

    print("\nEstimated cost (USD; Anthropic list pricing 2026-05; "
          "cache rd=0.1x in, wr=1.25x in):\n")
    print(f"{'model':14s} {'dur':>4} {'eps':>4} {'total_$':>10} {'$/ep':>9}")
    print("-" * 45)
    grand, geps = 0.0, 0
    for m in models:
        for d in durs:
            a = by.get((m, d))
            if not a:
                continue
            grand += a["cost"]
            geps += a["eps"]
            print(f"{short(m):14s} {d:>4} {a['eps']:>4} "
                  f"{a['cost']:>10.4f} {a['cost']/a['eps']:>9.4f}")
    print("-" * 45)
    print(f"{'TOTAL':14s} {'':>4} {geps:>4} {grand:>10.4f}")
    if unpriced:
        print(f"\n  (no list price for "
              f"{', '.join(sorted(short(m) for m in unpriced))}"
              f" -- tokens logged, cost omitted)")


def _dodge(idx: int, i: int, m_count: int, width: float = 0.7) -> float:
    """Horizontal position for model i (of m_count) at categorical slot idx, so
    same-x points fan out instead of overplotting. Single model -> no offset."""
    if m_count <= 1:
        return float(idx)
    return idx + (i - (m_count - 1) / 2) * (width / m_count)


def fig_win_heatmap(rows, models, out: Path):
    """Win rate over the (papers x grid) world surface, one heatmap per model
    (1x3, ordered Haiku/Sonnet/Opus). x = papers_needed, y = grid_size, cell
    color = fraction of that cell's episodes that won. Colormap is seaborn 'vlag'
    reversed so high win rate is BLUE (success) and low is RED (failure),
    centered at 0.5. Cells with no episodes (skipped combos) are left blank."""
    import numpy as np
    import seaborn as sns
    from collections import defaultdict
    from matplotlib.colors import TwoSlopeNorm

    cmap = sns.color_palette("vlag_r", as_cmap=True)   # high=blue, low=red
    cmap.set_bad("0.85")                                 # missing cells -> light gray
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)

    by = defaultdict(lambda: [0, 0])  # (model, grid, papers) -> [wins, N]
    for r in rows:
        a = by[(r["model"], r["grid_size"], r["papers_needed"])]
        a[0] += bool(r["won"]); a[1] += 1
    grids = sorted({g for _, g, _ in by})
    paps = sorted({p for _, _, p in by})
    present = [m for m in models if any(k[0] == m for k in by)]

    nc = len(present)
    fig, axes = plt.subplots(1, nc, figsize=(3.1 * nc + 0.8, 3.4),
                             squeeze=False, sharey=True)
    im = None
    for ci, m in enumerate(present):
        ax = axes[0][ci]
        M = np.full((len(grids), len(paps)), np.nan)
        for gi, g in enumerate(grids):
            for pj, p in enumerate(paps):
                k, N = by.get((m, g, p), [0, 0])
                if N:
                    M[gi, pj] = k / N
        im = ax.imshow(M, cmap=cmap, norm=norm, origin="lower", aspect="auto")
        for gi in range(len(grids)):
            for pj in range(len(paps)):
                v = M[gi, pj]
                if not np.isnan(v):
                    ax.text(pj, gi, f"{v:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if (v < 0.28 or v > 0.72) else "black")
        ax.set_xticks(range(len(paps))); ax.set_xticklabels(paps)
        ax.set_yticks(range(len(grids))); ax.set_yticklabels(grids)
        ax.set_xlabel("papers needed", fontsize=9)
        ax.set_title(fam_label(m), fontsize=11)
        if ci == 0:
            ax.set_ylabel("grid size", fontsize=9)
    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.046, pad=0.04,
                        ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label("win rate  (blue = success, red = failure)", fontsize=9)
    fig.suptitle("Win rate over the (papers x grid) surface", fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} (+ .pdf)")


def _winrate_matrices(rows, models):
    """Per (model, grid, papers) build the three component surfaces used by the
    heatmaps: P(built), P(win|built), P(win). Returns (present, grids, paps,
    pbuilt, pwgb, pwin) with each p* a dict model -> 2D array (np.nan where a
    cell has no episodes, or no built episodes for pwgb)."""
    import numpy as np
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0, 0])  # -> [built, wins_built, wins_total, total]
    for r in rows:
        a = agg[(r["model"], r["grid_size"], r["papers_needed"])]
        b, w = bool(r["built_map"]), bool(r["won"])
        a[0] += b; a[1] += (b and w); a[2] += w; a[3] += 1
    grids = sorted({g for _, g, _ in agg})
    paps = sorted({p for _, _, p in agg})
    present = [m for m in models if any(k[0] == m for k in agg)]
    mk = lambda: {m: np.full((len(grids), len(paps)), np.nan) for m in present}
    pbuilt, pwgb, pwin = mk(), mk(), mk()
    for (m, g, p), (nb, wb, wt, nt) in agg.items():
        gi, pj = grids.index(g), paps.index(p)
        pbuilt[m][gi, pj] = nb / nt
        pwin[m][gi, pj] = wt / nt
        if nb:
            pwgb[m][gi, pj] = wb / nb
    return present, grids, paps, pbuilt, pwgb, pwin


def _render_winmap_1x3(present, grids, paps, mats, fmt, norm, cbar_label, suptitle, out):
    """Shared 1x3 (per-model) heatmap renderer with the vlag_r colormap and the
    given norm. `mats` is model -> 2D array; `fmt` formats a cell's annotation."""
    import numpy as np
    import seaborn as sns
    cmap = sns.color_palette("vlag_r", as_cmap=True)
    cmap.set_bad("0.85")
    nc = len(present)
    fig, axes = plt.subplots(1, nc, figsize=(3.1 * nc + 0.8, 3.4),
                             squeeze=False, sharey=True)
    im = None
    for ci, m in enumerate(present):
        ax = axes[0][ci]
        M = mats[m]
        im = ax.imshow(M, cmap=cmap, norm=norm, origin="lower", aspect="auto")
        for gi in range(len(grids)):
            for pj in range(len(paps)):
                v = M[gi, pj]
                if not np.isnan(v):
                    nv = float(norm(v))  # position in [0,1] along the cmap
                    ax.text(pj, gi, fmt(v), ha="center", va="center", fontsize=8,
                            color="white" if (nv < 0.25 or nv > 0.75) else "black")
        ax.set_xticks(range(len(paps))); ax.set_xticklabels(paps)
        ax.set_yticks(range(len(grids))); ax.set_yticklabels(grids)
        ax.set_xlabel("papers needed", fontsize=9)
        ax.set_title(fam_label(m), fontsize=11)
        if ci == 0:
            ax.set_ylabel("grid size", fontsize=9)
    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=8.5)
    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} (+ .pdf)")


def fig_build_win_by_grid(rows, models, out: Path):
    """Build rate (top row) and solve/win rate (bottom row) with the 3 models
    side by side, one column per grid size. Within each panel the x-axis is the
    models (Haiku/Sonnet/Opus); episodes are pooled over papers_needed (and reps)
    for the (model, grid) cell. Bars are model-colored with Wilson 95% CIs."""
    import numpy as np
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0])  # (model, grid) -> [built, won, n]
    for r in rows:
        a = agg[(r["model"], r["grid_size"])]
        a[0] += bool(r["built_map"]); a[1] += bool(r["won"]); a[2] += 1
    grids = sorted({g for _, g in agg})
    present = [m for m in models if any(k[0] == m for k in agg)]

    metrics = [(0, "build rate"), (1, "solve rate")]  # index into [built, won, n]
    nc = len(grids)
    fig, axes = plt.subplots(2, nc, figsize=(2.6 * nc + 0.5, 5.4),
                             squeeze=False, sharey=True)
    for ri, (idx, mlabel) in enumerate(metrics):
        for ci, g in enumerate(grids):
            ax = axes[ri][ci]
            x = np.arange(len(present))
            ps, los, his = [], [], []
            for m in present:
                k = agg[(m, g)][idx]; n = agg[(m, g)][2]
                p, lo, hi = wilson(k, n)
                ps.append(p); los.append(p - lo); his.append(hi - p)
            ax.bar(x, ps, 0.66, yerr=[los, his], capsize=4,
                   color=[model_color(m) for m in present], edgecolor="black", lw=0.4)
            for xi, p in enumerate(ps):
                ax.text(xi, min(p + his[xi] + 0.02, 1.04), f"{p:.2f}",
                        ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels([fam_label(m) for m in present], fontsize=8.5)
            ax.set_ylim(0, 1.12)
            ax.grid(axis="y", alpha=0.25, lw=0.6)
            if ri == 0:
                ax.set_title(f"grid {g}x{g}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(mlabel, fontsize=10)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    fig.suptitle("Build rate (top) and solve rate (bottom) by grid  "
                 "(pooled over papers)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} (+ .pdf)")


def fig_win_given_notbuilt_heatmap(rows, models, out: Path):
    """Win rate CONDITIONAL on NOT building the map: P(win | not built) per
    (grid, papers) cell, one heatmap per model (1x3). Among only the episodes
    that never assembled the map, what fraction still won by brute-force digging.
    Values are NEGATED so they render RED on the vlag_r colormap (centered at 0):
    a deeper red = a higher map-less win rate. Each cell annotates the negated
    rate over wins/not-built. Cells where every episode built the map (no
    not-built episodes) are blank (gray)."""
    import numpy as np
    import seaborn as sns
    from collections import defaultdict
    from matplotlib.colors import TwoSlopeNorm

    cmap = sns.color_palette("vlag_r", as_cmap=True)
    cmap.set_bad("0.85")

    by = defaultdict(lambda: [0, 0])  # (model, grid, papers) -> [wins_notbuilt, n_notbuilt]
    for r in rows:
        if not r.get("built_map"):
            a = by[(r["model"], r["grid_size"], r["papers_needed"])]
            a[0] += bool(r["won"]); a[1] += 1
    grids = sorted({g for _, g, _ in by})
    paps = sorted({p for _, _, p in by})
    present = [m for m in models if any(k[0] == m for k in by)]

    # build matrices first to fix a symmetric, data-scaled diverging norm
    mats, anns = {}, {}
    for m in present:
        M = np.full((len(grids), len(paps)), np.nan)
        A = {}
        for gi, g in enumerate(grids):
            for pj, p in enumerate(paps):
                k, n = by.get((m, g, p), [0, 0])
                if n:
                    M[gi, pj] = -(k / n)            # negate -> red
                    A[(gi, pj)] = (M[gi, pj], k, n)
        mats[m] = M; anns[m] = A
    mx = max(0.05, max((np.nanmax(np.abs(mats[m])) for m in present
                        if not np.all(np.isnan(mats[m]))), default=0.05))
    norm = TwoSlopeNorm(vmin=-mx, vcenter=0.0, vmax=mx)

    nc = len(present)
    fig, axes = plt.subplots(1, nc, figsize=(3.1 * nc + 0.8, 3.4),
                             squeeze=False, sharey=True)
    im = None
    for ci, m in enumerate(present):
        ax = axes[0][ci]
        im = ax.imshow(mats[m], cmap=cmap, norm=norm, origin="lower", aspect="auto")
        for (gi, pj), (v, k, n) in anns[m].items():
            nv = float(norm(v))
            ax.text(pj, gi, f"{v:+.2f}\n{k}/{n}", ha="center", va="center",
                    fontsize=7.5, color="white" if (nv < 0.25 or nv > 0.75) else "black")
        ax.set_xticks(range(len(paps))); ax.set_xticklabels(paps)
        ax.set_yticks(range(len(grids))); ax.set_yticklabels(grids)
        ax.set_xlabel("papers needed", fontsize=9)
        ax.set_title(fam_label(m), fontsize=11)
        if ci == 0:
            ax.set_ylabel("grid size", fontsize=9)
    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.046, pad=0.04)
    cbar.set_label("-P(win | map NOT built)   (red = brute-force/grind wins)", fontsize=8.5)
    fig.suptitle("Win rate given the map was NOT built  (negated -> red), over wins/not-built",
                 fontsize=11.5)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} (+ .pdf)")


def fig_win_via_built_heatmap(rows, models, out: Path):
    """Cell-wise product of the P(win | built) heatmap and the P(built) heatmap:
    P(win|built)·P(built) = P(win AND map built) -- the probability of winning
    THROUGH the built map (the tool-path win rate). 1x3 per model, x = papers,
    y = grid. vlag_r (blue = success, red = failure), centered at 0.5."""
    from matplotlib.colors import TwoSlopeNorm
    present, grids, paps, pbuilt, pwgb, pwin = _winrate_matrices(rows, models)
    mats = {m: pwgb[m] * pbuilt[m] for m in present}   # cell-wise product
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    _render_winmap_1x3(
        present, grids, paps, mats, lambda v: f"{v:.2f}", norm,
        "P(win and map built) = P(win|built) x P(built)   (blue=success, red=failure)",
        "Win via the map  =  P(win | built) x P(built)", out)


def fig_win_via_built_minus_win_heatmap(rows, models, out: Path):
    """The win-via-built product MINUS the overall P(win) heatmap:
    P(win|built)·P(built) - P(win) = P(win∩built) - P(win) = -P(win without the
    map). So this is the (negated) brute-force win rate: 0 (white) where every
    win came through the map, negative (red) where the model won some games by
    grinding instead. 1x3 per model; diverging vlag_r centered at 0."""
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    present, grids, paps, pbuilt, pwgb, pwin = _winrate_matrices(rows, models)
    mats = {m: pwgb[m] * pbuilt[m] - pwin[m] for m in present}
    mx = max(0.05, max(np.nanmax(np.abs(mats[m])) for m in present))
    norm = TwoSlopeNorm(vmin=-mx, vcenter=0.0, vmax=mx)
    _render_winmap_1x3(
        present, grids, paps, mats, lambda v: f"{v:+.2f}", norm,
        "P(win∩built) - P(win) = -P(win without map)   (red = grind/brute wins)",
        "P(win | built) x P(built)  -  P(win)", out)


def fig_built_heatmap(rows, models, out: Path):
    """Probability that ALL scraps were collected -- i.e. the map was built -- per
    (grid, papers) cell, one heatmap per model (1x3, Haiku/Sonnet/Opus). Since
    collecting all P paper scraps is exactly what forms the map, this is
    P(built_map): how often a model does the full collect-the-tool work,
    independent of whether it then exploits it. x = papers_needed, y = grid_size;
    each cell annotates the rate over built/total. Same vlag_r colormap
    (blue = success, red = failure), centered at 0.5."""
    import numpy as np
    import seaborn as sns
    from collections import defaultdict
    from matplotlib.colors import TwoSlopeNorm

    cmap = sns.color_palette("vlag_r", as_cmap=True)
    cmap.set_bad("0.85")
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)

    by = defaultdict(lambda: [0, 0])  # (model, grid, papers) -> [built, total]
    for r in rows:
        a = by[(r["model"], r["grid_size"], r["papers_needed"])]
        a[0] += bool(r["built_map"]); a[1] += 1
    grids = sorted({g for _, g, _ in by})
    paps = sorted({p for _, _, p in by})
    present = [m for m in models if any(k[0] == m for k in by)]

    nc = len(present)
    fig, axes = plt.subplots(1, nc, figsize=(3.1 * nc + 0.8, 3.4),
                             squeeze=False, sharey=True)
    im = None
    for ci, m in enumerate(present):
        ax = axes[0][ci]
        M = np.full((len(grids), len(paps)), np.nan)
        ann = {}
        for gi, g in enumerate(grids):
            for pj, p in enumerate(paps):
                k, n = by.get((m, g, p), [0, 0])
                if n:
                    M[gi, pj] = k / n
                    ann[(gi, pj)] = (k / n, k, n)
        im = ax.imshow(M, cmap=cmap, norm=norm, origin="lower", aspect="auto")
        for (gi, pj), (v, k, n) in ann.items():
            ax.text(pj, gi, f"{v:.2f}\n{k}/{n}", ha="center", va="center",
                    fontsize=7.5, color="white" if (v < 0.28 or v > 0.72) else "black")
        ax.set_xticks(range(len(paps))); ax.set_xticklabels(paps)
        ax.set_yticks(range(len(grids))); ax.set_yticklabels(grids)
        ax.set_xlabel("papers needed", fontsize=9)
        ax.set_title(fam_label(m), fontsize=11)
        if ci == 0:
            ax.set_ylabel("grid size", fontsize=9)
    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.046, pad=0.04,
                        ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label("P(all scraps collected = map built)  (blue = success, red = failure)",
                   fontsize=8.5)
    fig.suptitle("Probability all scraps were collected (map built), over built/total",
                 fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} (+ .pdf)")


def fig_win_given_built_heatmap(rows, models, out: Path):
    """Win rate CONDITIONAL on building the map: P(win | built) per (grid, papers)
    cell, one heatmap per model (1x3, Haiku/Sonnet/Opus). Among only the episodes
    that assembled the map, what fraction went on to win -- i.e. how well a model
    EXPLOITS the tool once it has it (isolates the use->win step from whether it
    bothered to build). x = papers_needed, y = grid_size; each cell annotates the
    rate over wins/built. Same vlag_r colormap (blue = success, red = failure),
    centered at 0.5. Cells where no episode built the map are blank (gray)."""
    import numpy as np
    import seaborn as sns
    from collections import defaultdict
    from matplotlib.colors import TwoSlopeNorm

    cmap = sns.color_palette("vlag_r", as_cmap=True)
    cmap.set_bad("0.85")
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)

    by = defaultdict(lambda: [0, 0])  # (model, grid, papers) -> [wins_among_built, built]
    for r in rows:
        if r.get("built_map"):
            a = by[(r["model"], r["grid_size"], r["papers_needed"])]
            a[0] += bool(r["won"]); a[1] += 1
    grids = sorted({g for _, g, _ in by})
    paps = sorted({p for _, _, p in by})
    present = [m for m in models if any(k[0] == m for k in by)]

    nc = len(present)
    fig, axes = plt.subplots(1, nc, figsize=(3.1 * nc + 0.8, 3.4),
                             squeeze=False, sharey=True)
    im = None
    for ci, m in enumerate(present):
        ax = axes[0][ci]
        M = np.full((len(grids), len(paps)), np.nan)
        ann = {}
        for gi, g in enumerate(grids):
            for pj, p in enumerate(paps):
                k, n = by.get((m, g, p), [0, 0])
                if n:
                    M[gi, pj] = k / n
                    ann[(gi, pj)] = (k / n, k, n)
        im = ax.imshow(M, cmap=cmap, norm=norm, origin="lower", aspect="auto")
        for (gi, pj), (v, k, n) in ann.items():
            ax.text(pj, gi, f"{v:.2f}\n{k}/{n}", ha="center", va="center",
                    fontsize=7.5, color="white" if (v < 0.28 or v > 0.72) else "black")
        ax.set_xticks(range(len(paps))); ax.set_xticklabels(paps)
        ax.set_yticks(range(len(grids))); ax.set_yticklabels(grids)
        ax.set_xlabel("papers needed", fontsize=9)
        ax.set_title(fam_label(m), fontsize=11)
        if ci == 0:
            ax.set_ylabel("grid size", fontsize=9)
    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.046, pad=0.04,
                        ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label("P(win | built map)  (blue = success, red = failure)", fontsize=9)
    fig.suptitle("Win rate given the map was built  —  P(win | built), over wins/built",
                 fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} (+ .pdf)")


def fig_win_heatmap_smoothed(rows, models, out: Path, radius: int = 1, B: int = 2000):
    """Win-rate surface like fig_win_heatmap, but each (grid, papers) cell BORROWS
    STRENGTH from its neighbors: it pools every episode in the cell PLUS all cells
    within Chebyshev distance `radius` (self + the up-to-8 adjacent cells), then
    bootstraps that pool (episode-level resampling, B reps) for a stabilized win
    rate. With only ~10 episodes/cell the raw surface is noisy; neighbor-pooling
    smooths it and the bootstrap supplies the +/- 95% half-width annotated under
    each value. Same vlag_r colormap (blue = success, red = failure), centered
    at 0.5. Edge/corner cells simply pool fewer neighbors."""
    import numpy as np
    import seaborn as sns
    from collections import defaultdict
    from matplotlib.colors import TwoSlopeNorm

    rng = np.random.default_rng(0)
    cmap = sns.color_palette("vlag_r", as_cmap=True)
    cmap.set_bad("0.85")
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)

    won = defaultdict(list)  # (model, grid, papers) -> [0/1, ...]
    for r in rows:
        won[(r["model"], r["grid_size"], r["papers_needed"])].append(int(bool(r["won"])))
    grids = sorted({g for _, g, _ in won})
    paps = sorted({p for _, _, p in won})
    present = [m for m in models if any(k[0] == m for k in won)]

    nc = len(present)
    fig, axes = plt.subplots(1, nc, figsize=(3.1 * nc + 0.8, 3.4),
                             squeeze=False, sharey=True)
    im = None
    for ci, m in enumerate(present):
        ax = axes[0][ci]
        M = np.full((len(grids), len(paps)), np.nan)
        HW = np.full((len(grids), len(paps)), np.nan)
        for gi in range(len(grids)):
            for pj in range(len(paps)):
                pool = []
                for dg in range(-radius, radius + 1):
                    for dp in range(-radius, radius + 1):
                        gg, pp = gi + dg, pj + dp
                        if 0 <= gg < len(grids) and 0 <= pp < len(paps):
                            pool += won.get((m, grids[gg], paps[pp]), [])
                if not pool:
                    continue
                pool = np.asarray(pool)
                n = len(pool)
                reps = pool[rng.integers(0, n, size=(B, n))].mean(axis=1)
                M[gi, pj] = reps.mean()
                lo, hi = np.percentile(reps, [2.5, 97.5])
                HW[gi, pj] = (hi - lo) / 2
        im = ax.imshow(M, cmap=cmap, norm=norm, origin="lower", aspect="auto")
        for gi in range(len(grids)):
            for pj in range(len(paps)):
                v = M[gi, pj]
                if not np.isnan(v):
                    ax.text(pj, gi, f"{v:.2f}\n±{HW[gi, pj]:.2f}", ha="center",
                            va="center", fontsize=7.5,
                            color="white" if (v < 0.28 or v > 0.72) else "black")
        ax.set_xticks(range(len(paps))); ax.set_xticklabels(paps)
        ax.set_yticks(range(len(grids))); ax.set_yticklabels(grids)
        ax.set_xlabel("papers needed", fontsize=9)
        ax.set_title(fam_label(m), fontsize=11)
        if ci == 0:
            ax.set_ylabel("grid size", fontsize=9)
    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.046, pad=0.04,
                        ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label("win rate  (blue = success, red = failure)", fontsize=9)
    fig.suptitle("Win rate, neighbor-bootstrapped (self + adjacent cells pooled)",
                 fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} (+ .pdf)")


def fig_build_rate(rows, models, out: Path):
    """Map-build rate per model -- the headline disposition signal: of all the
    episodes in each (grid, papers) world, what fraction did the model actually
    assemble the reusable tool (the map). One grouped panel per (grid, papers)
    cell; bars are per model with Wilson 95% intervals and the n=k/N annotated.

    Derived straight from the rows (grid, papers) so it tracks the sweep's real
    axes regardless of how the durability-keyed summary groups things."""
    import numpy as np
    from collections import defaultdict
    by = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # cell -> model -> [k,N]
    for r in rows:
        cell = (r["grid_size"], r["papers_needed"])
        a = by[cell][r["model"]]
        a[0] += bool(r["built_map"]); a[1] += 1
    grids = sorted({g for g, _ in by})
    paps = sorted({p for _, p in by})
    present_models = [m for m in models if any(m in by[c] for c in by)]

    nr, nc = len(grids), len(paps)         # rows = grid, cols = papers
    fig, axes = plt.subplots(nr, nc, squeeze=False, sharey=True,
                             figsize=(max(2.6, 0.9 * len(present_models) + 1.0) * nc,
                                      2.7 * nr))
    for gi, grid in enumerate(grids):
        for pj, papers in enumerate(paps):
            ax = axes[gi][pj]
            cell = (grid, papers)
            present = [m for m in present_models if m in by.get(cell, {})]
            if not present:
                ax.set_axis_off(); continue
            x = np.arange(len(present))
            ps, los, his, ann = [], [], [], []
            for m in present:
                k, N = by[cell][m]
                p, lo, hi = wilson(k, N)
                ps.append(p); los.append(p - lo); his.append(hi - p); ann.append((p, k, N))
            ax.bar(x, ps, 0.66, yerr=[los, his], capsize=3,
                   color=[model_color(m) for m in present], edgecolor="black", lw=0.4)
            for xi, (p, k, N) in enumerate(ann):
                ax.text(xi, min(p + his[xi] + 0.03, 1.02), f"{p:.2f}",
                        ha="center", va="bottom", fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels([fam_label(m) for m in present], fontsize=7.5)
            ax.set_ylim(0, 1.14)
            ax.set_title(f"grid {grid}x{grid}, P={papers}", fontsize=8.5)
            ax.grid(axis="y", alpha=0.25, lw=0.6)
            if pj == 0:
                ax.set_ylabel(f"build rate", fontsize=9)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    fig.suptitle("Map-build rate by world (rows = grid, cols = papers)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out} (+ .pdf)")


def fig_funnel(rows, models, out: Path):
    """The reusable-tool pipeline per model: build -> use -> win. Three bars per
    model -- P(built map), P(used map), P(won) -- with Wilson 95% intervals,
    one panel per (grid, papers) cell.

    Reading the gaps:
      - build > use  : assembled the map but never read it (the tool sits idle)
      - use   < build: e.g. Haiku building then not exploiting -- the failure the
                       build-rate bar alone hides
      - win   > build: won WITHOUT the map (brute-force grind), so win is not a
                       strict child of build -- these are unconditional rates,
                       not a nested funnel, which is exactly why the crossings
                       are informative."""
    import numpy as np
    from collections import defaultdict
    STAGES = [("built_map", "build", "#4C72B0"),
              ("used_map", "use", "#DD8452"),
              ("won", "win", "#55A868")]
    by = defaultdict(lambda: defaultdict(list))  # cell -> model -> rows
    for r in rows:
        by[(r["grid_size"], r["papers_needed"])][r["model"]].append(r)
    grids = sorted({g for g, _ in by})
    paps = sorted({p for _, p in by})
    present_models = [m for m in models if any(m in by[c] for c in by)]

    nr, nc = len(grids), len(paps)         # rows = grid, cols = papers
    fig, axes = plt.subplots(nr, nc, squeeze=False, sharey=True,
                             figsize=(max(2.8, 1.1 * len(present_models) + 0.8) * nc,
                                      2.8 * nr))
    w = 0.26
    ref_ax = None
    for gi, grid in enumerate(grids):
        for pj, papers in enumerate(paps):
            ax = axes[gi][pj]
            cell = (grid, papers)
            present = [m for m in present_models if m in by.get(cell, {})]
            if not present:
                ax.set_axis_off(); continue
            ref_ax = ref_ax or ax
            x = np.arange(len(present))
            for si, (field, label, color) in enumerate(STAGES):
                ps, los, his = [], [], []
                for m in present:
                    g = by[cell][m]
                    k = sum(bool(e[field]) for e in g)
                    p, lo, hi = wilson(k, len(g))
                    ps.append(p); los.append(p - lo); his.append(hi - p)
                off = (si - 1) * w
                ax.bar(x + off, ps, w, yerr=[los, his], capsize=2, color=color,
                       edgecolor="black", lw=0.3, label=label)
            ax.set_xticks(x)
            ax.set_xticklabels([fam_label(m) for m in present], fontsize=7.5)
            ax.set_ylim(0, 1.14)
            ax.set_title(f"grid {grid}x{grid}, P={papers}", fontsize=8.5)
            ax.grid(axis="y", alpha=0.25, lw=0.6)
            if pj == 0:
                ax.set_ylabel("fraction", fontsize=9)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    handles, labels = ref_ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, fontsize=9, loc="lower center", ncol=3,
                     frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Build → use → win pipeline", fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150, bbox_inches="tight", bbox_extra_artists=(leg,))
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", bbox_extra_artists=(leg,))
    plt.close(fig)
    print(f"  wrote {out} (+ .pdf)")


def fig_rates(summ, models, durs, grids, out: Path):
    """Win rate and build rate vs durability, one series per model, faceted by
    grid -- the build-vs-grind flip curve (built_map should fall as durability
    rises and grinding becomes affordable). Models are dodged horizontally at
    each durability so their markers/error bars don't overlap; durability sits
    on categorical x slots so a single-value sweep still reads cleanly."""
    metrics = [("win", "win rate"), ("built", "build rate")]
    xidx = list(range(len(durs)))
    fig, axes = plt.subplots(len(metrics), len(grids),
                             figsize=(max(4.8, 1.3 * len(durs) + 2.6) * len(grids),
                                      3.2 * len(metrics)),
                             squeeze=False, sharex=True, sharey=True)
    for ri, (mkey, mlabel) in enumerate(metrics):
        for ci, grid in enumerate(grids):
            ax = axes[ri][ci]
            present = [m for m in models
                       if any((m, d, grid) in summ for d in durs)]
            line = "-" if len(durs) > 1 else "none"
            for i, m in enumerate(present):
                xs, ys, lo, hi = [], [], [], []
                for di, d in enumerate(durs):
                    s = summ.get((m, d, grid))
                    if not s:
                        continue
                    p, l, h = s[mkey]
                    xs.append(_dodge(di, i, len(present)))
                    ys.append(p); lo.append(p - l); hi.append(h - p)
                if not xs:
                    continue
                ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3,
                            color=model_color(m), label=fam_label(m), lw=1.5,
                            ms=6, ls=line)
            ax.set_xlim(-0.5, len(durs) - 0.5)
            ax.set_xticks(xidx)
            ax.set_xticklabels(durs)
            ax.set_ylim(-0.05, 1.08)
            ax.grid(axis="y", alpha=0.25, lw=0.6)
            if ri == 0:
                ax.set_title(f"grid {grid}x{grid}", fontsize=10)
            if ri == len(metrics) - 1:
                ax.set_xlabel("shovel durability (digs)", fontsize=9)
            if ci == 0:
                ax.set_ylabel(mlabel, fontsize=9)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, fontsize=9, loc="lower center",
                     ncol=max(1, len(labels)), frameon=False,
                     bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.8)
    extra = (leg,)
    fig.savefig(out, dpi=150, bbox_inches="tight", bbox_extra_artists=extra)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", bbox_extra_artists=extra)
    plt.close(fig)
    print(f"\n  wrote {out} (+ .pdf)")


def fig_outcomes(summ, models, durs, grids, out: Path):
    """Stacked 2x2-outcome bars (built+won / built+lost / brute+won /
    brute+lost) per model, tiled per (grid, durability) cell."""
    import numpy as np
    nr, nc = len(grids), len(durs)
    fig, axes = plt.subplots(nr, nc, figsize=(max(3.4, 2.9 * nc), 3.0 * nr),
                             squeeze=False, sharey=True)
    cap = max((summ[k]["n_eps"] for k in summ), default=8)
    cap = max(2, cap + (cap % 2))
    for ri, grid in enumerate(grids):
        for ci, d in enumerate(durs):
            ax = axes[ri][ci]
            present = [m for m in models if (m, d, grid) in summ]
            x = np.arange(len(present))
            bottom = np.zeros(len(present))
            for c in OC_CATS:
                vals = [summ[(m, d, grid)]["oc"].get(c, 0) for m in present]
                ax.bar(x, vals, 0.78, bottom=bottom, color=OC_COLORS[c],
                       label=OC_LEG[c])
                bottom += np.array(vals)
            ax.set_xticks(x)
            ax.set_xticklabels([fam_label(m) for m in present], fontsize=7)
            ax.set_ylim(0, cap)
            ax.set_yticks(list(range(0, cap + 1, max(2, cap // 4))))
            ax.set_title(f"grid {grid}, dur {d}", fontsize=8)
            if ci == 0:
                ax.set_ylabel("episodes", fontsize=9)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    ncol = 4 if nc >= 2 else 2
    leg = fig.legend(handles, labels, fontsize=8, loc="lower center",
                     ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    extra = (leg,)
    fig.savefig(out, dpi=200, bbox_inches="tight", bbox_extra_artists=extra)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", bbox_extra_artists=extra)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", bbox_extra_artists=extra)
    plt.close(fig)
    print(f"  wrote {out} (+ .pdf, .svg)")


def fig_efficiency(summ, models, durs, grids, out: Path):
    """Mean total actions and mean digs vs durability -- the cost view. The map
    path trades many exploration actions to save digs; grinding inverts that, so
    where the action/dig curves cross tells the same build-vs-grind story as the
    build-rate curve, in cost terms."""
    metrics = [("actions", "mean total actions"), ("digs", "mean digs")]
    xidx = list(range(len(durs)))
    fig, axes = plt.subplots(len(metrics), len(grids),
                             figsize=(max(4.8, 1.3 * len(durs) + 2.6) * len(grids),
                                      3.2 * len(metrics)),
                             squeeze=False, sharex=True)
    for ri, (mkey, mlabel) in enumerate(metrics):
        for ci, grid in enumerate(grids):
            ax = axes[ri][ci]
            present = [m for m in models
                       if any((m, d, grid) in summ for d in durs)]
            line = "-" if len(durs) > 1 else "none"
            for i, m in enumerate(present):
                xs, ys = [], []
                for di, d in enumerate(durs):
                    s = summ.get((m, d, grid))
                    if not s or s[mkey] != s[mkey]:
                        continue
                    xs.append(_dodge(di, i, len(present))); ys.append(s[mkey])
                if not xs:
                    continue
                ax.plot(xs, ys, marker="o", color=model_color(m),
                        label=fam_label(m), lw=1.5, ms=6, ls=line)
            ax.set_xlim(-0.5, len(durs) - 0.5)
            ax.set_xticks(xidx)
            ax.set_xticklabels(durs)
            ax.grid(axis="y", alpha=0.25, lw=0.6)
            if ri == 0:
                ax.set_title(f"grid {grid}x{grid}", fontsize=10)
            if ri == len(metrics) - 1:
                ax.set_xlabel("shovel durability (digs)", fontsize=9)
            if ci == 0:
                ax.set_ylabel(mlabel, fontsize=9)
            ax.set_ylim(bottom=0)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, fontsize=9, loc="lower center",
                     ncol=max(1, len(labels)), frameon=False,
                     bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.8)
    extra = (leg,)
    fig.savefig(out, dpi=150, bbox_inches="tight", bbox_extra_artists=extra)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", bbox_extra_artists=extra)
    plt.close(fig)
    print(f"  wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
