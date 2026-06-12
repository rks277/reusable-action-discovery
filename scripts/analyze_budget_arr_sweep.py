"""Budget-axis analysis for run_budget_arr_sweep.py (fixed n, T; swept budget).

The mirror of analyze_nt_sweep.py: there the swept axis is n (with budget DERIVED
per n), here n and T are held constant and the BUDGET is the independent variable.
So this groups by (model, budget) and reads every metric ALONG the budget axis --
the curve we actually built that sweep to see (e.g. the budget at which the
"build the tool vs. brute-force" trade-off flips, where solve-rate takes off).

Per cell it reports solve_rate and built_rate with WILSON score intervals (same
as analyze_nt_sweep -- proportions from few reps, where bootstrap degenerates at
0/k and k/k). Figures: solve/built rate vs budget (one line per model, Wilson
error bars), plus analyze_budget's headline + outcome bars faceted per budget.

Assumes a single (n, T) across the file (what run_budget_arr_sweep produces). If
the input mixes (n, T), it warns and still groups by (model, budget) -- pooling
across n/T, which is only meaningful if they're constant.

Usage: python -m scripts.analyze_budget_arr_sweep runs/budget_arr_sweep_*/episodes.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.replay_toolworld import verify
from scripts.analyze_budget import per_episode, short
# reuse the shared stats/plot vocabulary rather than redefining it
from scripts.analyze_nt_sweep import (
    ORDER, wilson, n_refusals, cost_of, model_color, fam_label,
    OC_CATS, OC_LEG, OC_COLORS,
)


def main():
    path = Path(sys.argv[1])
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    nfail = sum(1 for r in rows if not verify(r)[0])
    print(f"Replay verification: {len(rows)-nfail}/{len(rows)} ok\n")

    # this analysis assumes a single (n, T); pooling across them is only valid
    # if they're constant, so warn loudly if the file mixes them.
    nts = {(r["n"], r["n_types"]) for r in rows}
    if len(nts) > 1:
        print(f"WARNING: input mixes {len(nts)} (n, T) combos {sorted(nts)}; "
              f"grouping by (model, budget) pools across them.\n")
    n_val = sorted({r["n"] for r in rows})
    t_val = sorted({r["n_types"] for r in rows})

    # group by (model, budget)
    cells = defaultdict(list)
    for r in rows:
        eps = per_episode(r)
        eps["refused"] = n_refusals(r) >= 1   # episode hit >=1 safety refusal
        cells[(r["model"], r["budget"])].append(eps)

    models = ([m for m in ORDER if any(k[0] == m for k in cells)]
              + sorted({k[0] for k in cells} - set(ORDER)))
    budgets = sorted({k[1] for k in cells})

    # per-cell summary (same fields as analyze_nt_sweep)
    summ = {}
    for key, g in cells.items():
        k_solve = sum(e["solved"] for e in g)
        k_built = sum(e["built"] for e in g)
        bts = [e["build_turn"] for e in g if e["build_turn"] is not None]
        oc = defaultdict(int)
        for e in g:
            oc[e["outcome"]] += 1
        # "clean" solve rate excludes episodes a refusal killed (refused & not
        # solved) -- those never really attempted the task.
        clean = [e for e in g if not (e["refused"] and not e["solved"])]
        clean_solve = statistics.mean(e["solved"] for e in clean) if clean else float("nan")
        summ[key] = {
            "n_eps": len(g),
            "solve": wilson(k_solve, len(g)),
            "built": wilson(k_built, len(g)),
            "waste": statistics.mean(e["waste_frac"] for e in g),
            "bt": statistics.mean(bts) if bts else float("nan"),
            "oc": oc,
            "refused": sum(1 for e in g if e["refused"]),
            "clean": clean_solve,
        }

    print(f"Budget sweep: {len(models)} models x budget in {budgets}  "
          f"(n={n_val}, T={t_val})\n")
    print(f"{'model':14s} {'budget':>6} {'eps':>4} "
          f"{'solve [95% CI]':>22} {'built [95% CI]':>22} {'waste%':>7} {'build@':>7} "
          f"{'refd':>5} {'clean':>6}")
    print("-" * 102)
    for m in models:
        for b in budgets:
            s = summ.get((m, b))
            if not s:
                continue
            sp, slo, shi = s["solve"]
            bp, blo, bhi = s["built"]
            clean = f"{s['clean']:.2f}" if s["clean"] == s["clean"] else "-"
            print(f"{short(m):14s} {b:>6} {s['n_eps']:>4} "
                  f"{f'{sp:.2f} [{slo:.2f}-{shi:.2f}]':>22} "
                  f"{f'{bp:.2f} [{blo:.2f}-{bhi:.2f}]':>22} "
                  f"{s['waste']*100:>6.0f}% {s['bt']:>7.1f} "
                  f"{s['refused']:>5} {clean:>6}")
    print("\n  refd = episodes with >=1 safety refusal; clean = solve rate "
          "excluding refusal-killed episodes")

    print_token_usage(rows, models, budgets)
    print_costs(rows, models, budgets)
    fig_budget_curve(summ, models, budgets, path.parent / "fig_budget_curve.png")
    fig_budget_headline(summ, models, budgets, path.parent / "fig_budget_headline.png")
    fig_budget_outcomes(summ, models, budgets, path.parent / "fig_budget_outcomes.png")


def print_token_usage(rows, models, budgets):
    """Per-(model, budget) token totals -- grouped by budget because longer
    budgets mean more turns, so input tokens (transcript regrows each turn) and
    cache reads climb with budget."""
    FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens",
              "cache_write_tokens")
    agg = defaultdict(lambda: {**{k: 0 for k in FIELDS}, "calls": 0, "eps": 0})
    for r in rows:
        a = agg[(r["model"], r["budget"])]
        u = r.get("usage") or {}
        for k in FIELDS:
            a[k] += int(u.get(k, 0) or 0)
        a["calls"] += int(u.get("calls", 0) or 0)
        a["eps"] += 1

    def cache_pct(a):
        denom = a["input_tokens"] + a["cache_read_tokens"] + a["cache_write_tokens"]
        return f"{a['cache_read_tokens'] / denom * 100:>5.0f}%" if denom else f"{'-':>6}"

    print("\nToken usage by (model, budget) (raw counts; "
          "cache% = read / (in+read+write)):\n")
    hdr = (f"{'model':14s} {'budget':>6} {'eps':>4} {'calls':>6} {'in':>10} {'out':>9} "
           f"{'cache_rd':>10} {'cache_wr':>10} {'cache%':>6}")
    print(hdr)
    print("-" * len(hdr))
    tot = {**{k: 0 for k in FIELDS}, "calls": 0, "eps": 0}
    for m in models:
        for b in budgets:
            a = agg.get((m, b))
            if not a:
                continue
            for k in tot:
                tot[k] += a[k]
            print(f"{short(m):14s} {b:>6} {a['eps']:>4} {a['calls']:>6} "
                  f"{a['input_tokens']:>10} {a['output_tokens']:>9} "
                  f"{a['cache_read_tokens']:>10} {a['cache_write_tokens']:>10} "
                  f"{cache_pct(a)}")
    print("-" * len(hdr))
    print(f"{'TOTAL':14s} {'':>6} {tot['eps']:>4} {tot['calls']:>6} "
          f"{tot['input_tokens']:>10} {tot['output_tokens']:>9} "
          f"{tot['cache_read_tokens']:>10} {tot['cache_write_tokens']:>10} "
          f"{cache_pct(tot)}")


def print_costs(rows, models, budgets):
    """Estimated USD cost grouped by (model, budget) -- larger budgets cost more
    (more turns -> transcript regrows -> input tokens scale super-linearly)."""
    by = defaultdict(lambda: {"eps": 0, "cost": 0.0})
    unpriced = set()
    for r in rows:
        c = cost_of(r.get("usage") or {}, r["model"])
        if c is None:
            unpriced.add(r["model"])
            continue
        a = by[(r["model"], r["budget"])]
        a["eps"] += 1
        a["cost"] += c

    print("\nEstimated cost (USD; Anthropic list pricing 2026-05; "
          "cache rd=0.1x in, wr=1.25x in):\n")
    print(f"{'model':14s} {'budget':>6} {'eps':>4} {'total_$':>10} {'$/ep':>9}")
    print("-" * 47)
    grand, geps = 0.0, 0
    for m in models:
        for b in budgets:
            a = by.get((m, b))
            if not a:
                continue
            grand += a["cost"]
            geps += a["eps"]
            print(f"{short(m):14s} {b:>6} {a['eps']:>4} "
                  f"{a['cost']:>10.4f} {a['cost']/a['eps']:>9.4f}")
    print("-" * 47)
    print(f"{'TOTAL':14s} {'':>6} {geps:>4} {grand:>10.4f}")
    if unpriced:
        print(f"\n  (no list price for {', '.join(sorted(short(m) for m in unpriced))}"
              f" -- tokens logged, cost omitted)")


def fig_budget_curve(summ, models, budgets, out: Path):
    """The headline figure for this sweep: solve rate and build rate vs BUDGET,
    one line per model, Wilson error bars. The mirror of analyze_nt_sweep's
    fig_surface with budget on x instead of n."""
    metrics = [("solve", "solve rate"), ("built", "build rate")]
    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(4.2 * len(metrics), 3.4),
                             squeeze=False, sharex=True, sharey=True)
    for ci, (mkey, mlabel) in enumerate(metrics):
        ax = axes[0][ci]
        for m in models:
            xs, ys, lo, hi = [], [], [], []
            for b in budgets:
                s = summ.get((m, b))
                if not s:
                    continue
                p, l, h = s[mkey]
                xs.append(b); ys.append(p); lo.append(p - l); hi.append(h - p)
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3,
                        color=model_color(m), label=short(m), lw=1.5, ms=4)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(budgets)
        ax.set_xlabel("budget (actions)", fontsize=9)
        ax.set_ylabel(mlabel, fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="upper center",
               ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"\n  wrote {out} (+ .pdf)")


def fig_budget_headline(summ, models, budgets, out: Path):
    """analyze_budget's headline (solve rate + wasted-action fraction grouped
    bars per model), one panel per budget."""
    import numpy as np
    nc = len(budgets)
    fig, axes = plt.subplots(1, nc, figsize=(2.7 * nc, 2.9),
                             squeeze=False, sharey=True)
    w = 0.38
    for ci, b in enumerate(budgets):
        ax = axes[0][ci]
        present = [m for m in models if (m, b) in summ]
        x = np.arange(len(present))
        solve = [summ[(m, b)]["solve"][0] for m in present]
        waste = [summ[(m, b)]["waste"] for m in present]
        ax.bar(x - w/2, solve, w, color="seagreen", label="solve rate")
        ax.bar(x + w/2, waste, w, color="indianred",
               label="wasted-action fraction")
        for xi, (s, wv) in enumerate(zip(solve, waste)):
            ax.text(xi - w/2, s + 0.02, f"{s:.2f}", ha="center", fontsize=6)
            ax.text(xi + w/2, wv + 0.02, f"{wv:.0%}", ha="center", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels([fam_label(m) for m in present], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"budget = {b}", fontsize=8)
        if ci == 0:
            ax.set_ylabel("fraction", fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out} (+ .pdf)")


def fig_budget_outcomes(summ, models, budgets, out: Path):
    """analyze_budget's stacked outcome bars (built+solved / built+ranout /
    brute+solved / neverbuilt per model), one panel per budget. Same colors."""
    import numpy as np
    nc = len(budgets)
    fig, axes = plt.subplots(1, nc, figsize=(2.7 * nc, 2.9),
                             squeeze=False, sharey=True)
    cap = max((summ[k]["n_eps"] for k in summ), default=8)
    cap = max(2, cap + (cap % 2))
    for ci, b in enumerate(budgets):
        ax = axes[0][ci]
        present = [m for m in models if (m, b) in summ]
        x = np.arange(len(present))
        bottom = np.zeros(len(present))
        for c in OC_CATS:
            vals = [summ[(m, b)]["oc"].get(c, 0) for m in present]
            ax.bar(x, vals, 0.78, bottom=bottom, color=OC_COLORS[c],
                   label=OC_LEG[c])
            bottom += np.array(vals)
        ax.set_xticks(x)
        ax.set_xticklabels([fam_label(m) for m in present], fontsize=7)
        ax.set_ylim(0, cap)
        ax.set_yticks(list(range(0, cap + 1, max(2, cap // 4))))
        ax.set_title(f"budget = {b}", fontsize=8)
        if ci == 0:
            ax.set_ylabel("episodes", fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    print(f"  wrote {out} (+ .pdf, .svg)")


if __name__ == "__main__":
    main()
