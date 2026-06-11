"""Grid-aware analysis for run_nt_sweep.py (the n x T design surface).

Unlike analyze_budget.py (which groups by model only, assuming one n/T/budget),
this groups by (model, n, T) and reads the surface along the two axes:
  - n  -> the build-margin lever (grind cost grows ~n*H_n; building amortizes)
  - T  -> the discoverability lever (recipe search is C(T,2)+T candidates)

Per cell it reports solve_rate and built_rate with WILSON score intervals (not
bootstrap -- these are proportions from few reps, where bootstrap degenerates at
0/k and k/k; see the rep-count discussion). Figure: solve-rate and built-rate
vs n, one line per model, faceted by T, with Wilson error bars.

Usage: python -m scripts.analyze_nt_sweep runs/nt_sweep_*/episodes.jsonl
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

from scripts.replay_toolworld import verify
from scripts.analyze_budget import per_episode, short

ORDER = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-8",
         "gpt-5", "gemini-2.5-pro"]
COLORS = {"haiku": "seagreen", "sonnet": "steelblue", "opus": "indianred",
          "gpt": "darkorange", "gemini": "purple"}


def model_color(m):
    s = short(m)
    return next((c for k, c in COLORS.items() if k in s), "gray")


# match analyze_budget's family labels and outcome-category styling exactly
FAM = {"haiku": "Haiku", "sonnet": "Sonnet", "opus": "Opus",
       "gemini": "Gemini", "gpt": "GPT-5"}
OC_CATS = ["built+solved", "built+ranout", "brute+solved", "neverbuilt"]
OC_LEG = {"built+solved": "built + solved", "built+ranout": "built, ran out",
          "brute+solved": "brute + solved", "neverbuilt": "never built"}
OC_COLORS = {"built+solved": "seagreen", "built+ranout": "goldenrod",
             "brute+solved": "steelblue", "neverbuilt": "indianred"}


def fam_label(m):
    s = short(m)
    return next((v for k, v in FAM.items() if k in s), s)


# USD per 1M tokens (input, output), Anthropic list pricing (cached 2026-05).
# Cache reads bill at 0.1x input; cache writes at 1.25x input (5-min ephemeral,
# which is what RawChat uses) -- derived from the input rate, not hardcoded.
PRICING = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
}


def cost_of(usage: dict, model: str):
    """Episode cost in USD from a usage dict, or None if the model is unpriced
    (e.g. non-Anthropic models -- tokens are logged but we don't guess rates)."""
    rate = PRICING.get(model)
    if not rate:
        return None
    ri, ro = rate
    u = usage or {}
    return (u.get("input_tokens", 0) * ri
            + u.get("output_tokens", 0) * ro
            + u.get("cache_read_tokens", 0) * ri * 0.10
            + u.get("cache_write_tokens", 0) * ri * 1.25) / 1e6


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


def main():
    path = Path(sys.argv[1])
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    nfail = sum(1 for r in rows if not verify(r)[0])
    print(f"Replay verification: {len(rows)-nfail}/{len(rows)} ok\n")

    # group by (model, n, T)
    cells = defaultdict(list)
    budget_of = {}
    for r in rows:
        eps = per_episode(r)
        key = (r["model"], r["n"], r["n_types"])
        cells[key].append(eps)
        budget_of[r["n"]] = r["budget"]

    models = ([m for m in ORDER if any(k[0] == m for k in cells)]
              + sorted({k[0] for k in cells} - set(ORDER)))
    ns = sorted({k[1] for k in cells})
    Ts = sorted({k[2] for k in cells})

    # per-cell summary
    summ = {}
    for key, g in cells.items():
        k_solve = sum(e["solved"] for e in g)
        k_built = sum(e["built"] for e in g)
        bts = [e["build_turn"] for e in g if e["build_turn"] is not None]
        oc = defaultdict(int)
        for e in g:
            oc[e["outcome"]] += 1
        summ[key] = {
            "n_eps": len(g),
            "solve": wilson(k_solve, len(g)),
            "built": wilson(k_built, len(g)),
            "waste": statistics.mean(e["waste_frac"] for e in g),
            "bt": statistics.mean(bts) if bts else float("nan"),
            "oc": oc,
        }

    print(f"Design surface: {len(models)} models x n in {ns} x T in {Ts}  "
          f"(budgets: {', '.join(f'n{n}=b{budget_of[n]}' for n in ns)})\n")
    print(f"{'model':14s} {'n':>3} {'T':>2} {'eps':>4} "
          f"{'solve [95% CI]':>22} {'built [95% CI]':>22} {'waste%':>7} {'build@':>7}")
    print("-" * 92)
    for m in models:
        for n in ns:
            for T in Ts:
                s = summ.get((m, n, T))
                if not s:
                    continue
                sp, slo, shi = s["solve"]
                bp, blo, bhi = s["built"]
                print(f"{short(m):14s} {n:>3} {T:>2} {s['n_eps']:>4} "
                      f"{f'{sp:.2f} [{slo:.2f}-{shi:.2f}]':>22} "
                      f"{f'{bp:.2f} [{blo:.2f}-{bhi:.2f}]':>22} "
                      f"{s['waste']*100:>6.0f}% {s['bt']:>7.1f}")

    print_token_usage(rows, models, ns)
    print_costs(rows, models, ns)
    fig_surface(summ, models, ns, Ts, path.parent / "fig_nt_surface.png")
    fig_nt_headline(summ, models, ns, Ts, budget_of, path.parent / "fig_nt_headline.png")
    fig_nt_outcomes(summ, models, ns, Ts, path.parent / "fig_nt_outcomes.png")


def print_token_usage(rows, models, ns):
    """Per-(model, n) token totals, same columns as analyze_budget's table.
    Grouped by n because cache% is n-dependent here: transcripts only cross the
    model's cacheable-prefix floor (4096 for Haiku/Opus, 2048 for Sonnet) once n
    is large enough, so cache_rd / cache% climb with n."""
    FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens",
              "cache_write_tokens")
    agg = defaultdict(lambda: {**{k: 0 for k in FIELDS}, "calls": 0, "eps": 0})
    for r in rows:
        a = agg[(r["model"], r["n"])]
        u = r.get("usage") or {}
        for k in FIELDS:
            a[k] += int(u.get(k, 0) or 0)
        a["calls"] += int(u.get("calls", 0) or 0)
        a["eps"] += 1

    def cache_pct(a):
        denom = a["input_tokens"] + a["cache_read_tokens"] + a["cache_write_tokens"]
        return f"{a['cache_read_tokens'] / denom * 100:>5.0f}%" if denom else f"{'-':>6}"

    print("\nToken usage by (model, n) (raw counts; cache% = read / (in+read+write)):\n")
    hdr = (f"{'model':14s} {'n':>3} {'eps':>4} {'calls':>6} {'in':>10} {'out':>9} "
           f"{'cache_rd':>10} {'cache_wr':>10} {'cache%':>6}")
    print(hdr)
    print("-" * len(hdr))
    tot = {**{k: 0 for k in FIELDS}, "calls": 0, "eps": 0}
    for m in models:
        for n in ns:
            a = agg.get((m, n))
            if not a:
                continue
            for k in tot:
                tot[k] += a[k]
            print(f"{short(m):14s} {n:>3} {a['eps']:>4} {a['calls']:>6} "
                  f"{a['input_tokens']:>10} {a['output_tokens']:>9} "
                  f"{a['cache_read_tokens']:>10} {a['cache_write_tokens']:>10} "
                  f"{cache_pct(a)}")
    print("-" * len(hdr))
    print(f"{'TOTAL':14s} {'':>3} {tot['eps']:>4} {tot['calls']:>6} "
          f"{tot['input_tokens']:>10} {tot['output_tokens']:>9} "
          f"{tot['cache_read_tokens']:>10} {tot['cache_write_tokens']:>10} "
          f"{cache_pct(tot)}")


def print_costs(rows, models, ns):
    """Estimated USD cost grouped by (model, n) -- the cost-vs-n view, since n
    drives cost super-linearly (transcript regrows each turn -> input tokens
    scale ~quadratically in turns, and larger n means more turns)."""
    by = defaultdict(lambda: {"eps": 0, "cost": 0.0})
    unpriced = set()
    for r in rows:
        c = cost_of(r.get("usage") or {}, r["model"])
        if c is None:
            unpriced.add(r["model"])
            continue
        a = by[(r["model"], r["n"])]
        a["eps"] += 1
        a["cost"] += c

    print("\nEstimated cost (USD; Anthropic list pricing 2026-05; "
          "cache rd=0.1x in, wr=1.25x in):\n")
    print(f"{'model':14s} {'n':>3} {'eps':>4} {'total_$':>10} {'$/ep':>9}")
    print("-" * 44)
    grand, geps = 0.0, 0
    for m in models:
        for n in ns:
            a = by.get((m, n))
            if not a:
                continue
            grand += a["cost"]
            geps += a["eps"]
            print(f"{short(m):14s} {n:>3} {a['eps']:>4} "
                  f"{a['cost']:>10.4f} {a['cost']/a['eps']:>9.4f}")
    print("-" * 44)
    print(f"{'TOTAL':14s} {'':>3} {geps:>4} {grand:>10.4f}")
    if unpriced:
        print(f"\n  (no list price for {', '.join(sorted(short(m) for m in unpriced))}"
              f" -- tokens logged, cost omitted)")


def fig_nt_headline(summ, models, ns, Ts, budget_of, out: Path):
    """analyze_budget's headline (solve rate + wasted-action fraction grouped
    bars per model), tiled across the n x T grid: one panel per (n, T) cell."""
    import numpy as np
    nr, nc = len(Ts), len(ns)
    fig, axes = plt.subplots(nr, nc, figsize=(2.7 * nc, 2.7 * nr),
                             squeeze=False, sharey=True)
    w = 0.38
    for ri, T in enumerate(Ts):
        for ci, n in enumerate(ns):
            ax = axes[ri][ci]
            present = [m for m in models if (m, n, T) in summ]
            x = np.arange(len(present))
            solve = [summ[(m, n, T)]["solve"][0] for m in present]
            waste = [summ[(m, n, T)]["waste"] for m in present]
            ax.bar(x - w/2, solve, w, color="seagreen", label="solve rate")
            ax.bar(x + w/2, waste, w, color="indianred",
                   label="wasted-action fraction")
            for xi, (s, wv) in enumerate(zip(solve, waste)):
                ax.text(xi - w/2, s + 0.02, f"{s:.2f}", ha="center", fontsize=6)
                ax.text(xi + w/2, wv + 0.02, f"{wv:.0%}", ha="center", fontsize=6)
            ax.set_xticks(x)
            ax.set_xticklabels([fam_label(m) for m in present], fontsize=7)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"n={n}, T={T}  (b={budget_of[n]})", fontsize=8)
            if ci == 0:
                ax.set_ylabel("fraction", fontsize=9)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out} (+ .pdf)")


def fig_nt_outcomes(summ, models, ns, Ts, out: Path):
    """analyze_budget's stacked outcome bars (built+solved / built+ranout /
    brute+solved / neverbuilt per model), tiled per (n, T) cell. Same colors."""
    import numpy as np
    nr, nc = len(Ts), len(ns)
    fig, axes = plt.subplots(nr, nc, figsize=(2.7 * nc, 2.7 * nr),
                             squeeze=False, sharey=True)
    cap = max((summ[k]["n_eps"] for k in summ), default=8)
    cap = max(2, cap + (cap % 2))
    for ri, T in enumerate(Ts):
        for ci, n in enumerate(ns):
            ax = axes[ri][ci]
            present = [m for m in models if (m, n, T) in summ]
            x = np.arange(len(present))
            bottom = np.zeros(len(present))
            for c in OC_CATS:
                vals = [summ[(m, n, T)]["oc"].get(c, 0) for m in present]
                ax.bar(x, vals, 0.78, bottom=bottom, color=OC_COLORS[c],
                       label=OC_LEG[c])
                bottom += np.array(vals)
            ax.set_xticks(x)
            ax.set_xticklabels([fam_label(m) for m in present], fontsize=7)
            ax.set_ylim(0, cap)
            ax.set_yticks(list(range(0, cap + 1, max(2, cap // 4))))
            ax.set_title(f"n={n}, T={T}", fontsize=8)
            if ci == 0:
                ax.set_ylabel("episodes", fontsize=9)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    print(f"  wrote {out} (+ .pdf, .svg)")


def fig_surface(summ, models, ns, Ts, out: Path):
    import numpy as np
    metrics = [("solve", "solve rate"), ("built", "build rate")]
    fig, axes = plt.subplots(len(metrics), len(Ts),
                             figsize=(3.6 * len(Ts), 3.0 * len(metrics)),
                             squeeze=False, sharex=True, sharey=True)
    for ri, (mkey, mlabel) in enumerate(metrics):
        for ci, T in enumerate(Ts):
            ax = axes[ri][ci]
            for m in models:
                xs, ys, lo, hi = [], [], [], []
                for n in ns:
                    s = summ.get((m, n, T))
                    if not s:
                        continue
                    p, l, h = s[mkey]
                    xs.append(n); ys.append(p); lo.append(p - l); hi.append(h - p)
                if not xs:
                    continue
                ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3,
                            color=model_color(m), label=short(m), lw=1.5, ms=4)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks(ns)
            if ri == 0:
                ax.set_title(f"T = {T}", fontsize=10)
            if ri == len(metrics) - 1:
                ax.set_xlabel("n (doors)", fontsize=9)
            if ci == 0:
                ax.set_ylabel(mlabel, fontsize=9)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    # one shared legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="upper center",
               ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"\n  wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
