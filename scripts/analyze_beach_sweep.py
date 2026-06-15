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


def fig_rates(summ, models, durs, grids, out: Path):
    """Win rate and build rate vs durability, one line per model, faceted by
    grid -- the build-vs-grind flip curve (built_map should fall as durability
    rises and grinding becomes affordable)."""
    metrics = [("win", "win rate"), ("built", "build rate")]
    fig, axes = plt.subplots(len(metrics), len(grids),
                             figsize=(3.6 * len(grids), 3.0 * len(metrics)),
                             squeeze=False, sharex=True, sharey=True)
    for ri, (mkey, mlabel) in enumerate(metrics):
        for ci, grid in enumerate(grids):
            ax = axes[ri][ci]
            for m in models:
                xs, ys, lo, hi = [], [], [], []
                for d in durs:
                    s = summ.get((m, d, grid))
                    if not s:
                        continue
                    p, l, h = s[mkey]
                    xs.append(d); ys.append(p); lo.append(p - l); hi.append(h - p)
                if not xs:
                    continue
                ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3,
                            color=model_color(m), label=fam_label(m), lw=1.5, ms=4)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks(durs)
            if ri == 0:
                ax.set_title(f"grid {grid}x{grid}", fontsize=10)
            if ri == len(metrics) - 1:
                ax.set_xlabel("shovel durability (digs)", fontsize=9)
            if ci == 0:
                ax.set_ylabel(mlabel, fontsize=9)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="upper center",
               ncol=max(1, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"\n  wrote {out} (+ .pdf)")


def fig_outcomes(summ, models, durs, grids, out: Path):
    """Stacked 2x2-outcome bars (built+won / built+lost / brute+won /
    brute+lost) per model, tiled per (grid, durability) cell."""
    import numpy as np
    nr, nc = len(grids), len(durs)
    fig, axes = plt.subplots(nr, nc, figsize=(2.7 * nc, 2.7 * nr),
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
    fig.legend(handles, labels, fontsize=8, loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    print(f"  wrote {out} (+ .pdf, .svg)")


def fig_efficiency(summ, models, durs, grids, out: Path):
    """Mean total actions and mean digs vs durability -- the cost view. The map
    path trades many exploration actions to save digs; grinding inverts that, so
    where the action/dig curves cross tells the same build-vs-grind story as the
    build-rate curve, in cost terms."""
    metrics = [("actions", "mean total actions"), ("digs", "mean digs")]
    fig, axes = plt.subplots(len(metrics), len(grids),
                             figsize=(3.6 * len(grids), 3.0 * len(metrics)),
                             squeeze=False, sharex=True)
    for ri, (mkey, mlabel) in enumerate(metrics):
        for ci, grid in enumerate(grids):
            ax = axes[ri][ci]
            for m in models:
                xs, ys = [], []
                for d in durs:
                    s = summ.get((m, d, grid))
                    if not s or s[mkey] != s[mkey]:
                        continue
                    xs.append(d); ys.append(s[mkey])
                if not xs:
                    continue
                ax.plot(xs, ys, marker="o", color=model_color(m),
                        label=fam_label(m), lw=1.5, ms=4)
            ax.set_xticks(durs)
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
    fig.legend(handles, labels, fontsize=8, loc="upper center",
               ncol=max(1, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
