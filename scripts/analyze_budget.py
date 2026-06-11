"""Consolidated analysis for the cross-model BUDGET condition (the headline).

Replays each budgeted episode (verifying determinism), then reports per model:
  - outcome breakdown: BUILT+SOLVED / BUILT-ran-out / BRUTE+SOLVED / never-built
  - solve_rate, built_rate
  - wasted_action_fraction: no-op actions (failed combines, wrong keys, redundant
    ops) / total actions -- the inefficiency that the budget makes decisive
  - build_turn (actions before the machine was built)

Headline figure: solve-rate and wasted-action fraction per model.

Usage: python -m scripts.analyze_budget runs/budget_sweep_*/episodes.jsonl
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

WASTE = ("nothing happens", "nothing.", "already hold", "already open",
         "does not fit", "don't have", "nothing of note")


def short(m): return m.replace("claude-", "").replace("-20251001", "")


def per_episode(row: dict) -> dict:
    obs = row["obs"]
    built = any("fuse into" in o for o in obs)
    solved = row["solved"]
    waste = sum(1 for o in obs if any(w in o for w in WASTE))
    total = len(obs)
    build_turn = next((i for i, o in enumerate(obs) if "fuse into" in o), None)
    if built and solved:
        outcome = "built+solved"
    elif solved:
        outcome = "brute+solved"
    elif built:
        outcome = "built+ranout"
    else:
        outcome = "neverbuilt"
    return {"model": row["model"], "built": built, "solved": solved,
            "waste_frac": waste / total if total else 0.0,
            "build_turn": build_turn, "outcome": outcome,
            "total_actions": total}


def print_token_usage(rows: list[dict], models: list[str]) -> None:
    """Per-model token totals (raw counts, no pricing). cache% is the cache-read
    share of input-side tokens (Anthropic-style: in + read + write) and doubles
    as a check that prompt caching is biting -- ~0 on pre-caching runs, high after.
    Rows from older runs have no 'usage' key; those render as 0 / '-'."""
    FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens",
              "cache_write_tokens")
    agg = defaultdict(lambda: {**{k: 0 for k in FIELDS}, "calls": 0, "eps": 0})
    for r in rows:
        a = agg[r["model"]]
        u = r.get("usage") or {}
        for k in FIELDS:
            a[k] += int(u.get(k, 0) or 0)
        a["calls"] += int(u.get("calls", 0) or 0)
        a["eps"] += 1

    def cache_pct(a):
        denom = a["input_tokens"] + a["cache_read_tokens"] + a["cache_write_tokens"]
        return f"{a['cache_read_tokens'] / denom * 100:>5.0f}%" if denom else f"{'-':>6}"

    print("\nToken usage per model (raw counts; cache% = read / (in+read+write)):\n")
    print(f"{'model':14s} {'eps':>3} {'calls':>6} {'in':>10} {'out':>10} "
          f"{'cache_rd':>10} {'cache_wr':>10} {'cache%':>6}")
    print("-" * 79)
    tot = {**{k: 0 for k in FIELDS}, "calls": 0, "eps": 0}
    for m in models:
        a = agg[m]
        for k in tot:
            tot[k] += a[k]
        print(f"{short(m):14s} {a['eps']:>3} {a['calls']:>6} "
              f"{a['input_tokens']:>10} {a['output_tokens']:>10} "
              f"{a['cache_read_tokens']:>10} {a['cache_write_tokens']:>10} "
              f"{cache_pct(a)}")
    print("-" * 79)
    print(f"{'TOTAL':14s} {tot['eps']:>3} {tot['calls']:>6} "
          f"{tot['input_tokens']:>10} {tot['output_tokens']:>10} "
          f"{tot['cache_read_tokens']:>10} {tot['cache_write_tokens']:>10} "
          f"{cache_pct(tot)}")


def main():
    path = Path(sys.argv[1])
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]

    # replay verification (the post-hoc determinism guarantee)
    nfail = sum(1 for r in rows if not verify(r)[0])
    print(f"Replay verification: {len(rows)-nfail}/{len(rows)} ok\n")

    eps = [per_episode(r) for r in rows]
    budget = rows[0].get("budget")
    by = defaultdict(list)
    for e in eps:
        by[e["model"]].append(e)

    order = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-8"]
    models = [m for m in order if m in by] + [m for m in by if m not in order]

    print(f"Budget condition (budget={budget}); outcomes per model:\n")
    print(f"{'model':14s} {'eps':>3} {'solve':>6} {'built':>6} {'waste%':>7} "
          f"{'build@':>7} | built+solved  built+ranout  brute+solved  neverbuilt")
    print("-" * 104)
    summary = {}
    for m in models:
        g = by[m]
        solve = statistics.mean(e["solved"] for e in g)
        built = statistics.mean(e["built"] for e in g)
        waste = statistics.mean(e["waste_frac"] for e in g)
        bts = [e["build_turn"] for e in g if e["build_turn"] is not None]
        bt = statistics.mean(bts) if bts else float("nan")
        oc = defaultdict(int)
        for e in g:
            oc[e["outcome"]] += 1
        summary[m] = {"solve": solve, "built": built, "waste": waste, "bt": bt,
                      "oc": oc, "n": len(g)}
        print(f"{short(m):14s} {len(g):>3} {solve:>6.2f} {built:>6.2f} "
              f"{waste*100:>6.0f}% {bt:>7.1f} | "
              f"{oc['built+solved']:>11}  {oc['built+ranout']:>12}  "
              f"{oc['brute+solved']:>12}  {oc['neverbuilt']:>10}")

    print_token_usage(rows, models)

    fig_headline(summary, models, budget, path.parent / "fig_budget_headline.png")
    fig_outcomes(summary, models, path.parent / "fig_budget_outcomes.png")


def fig_headline(summary, models, budget, out: Path):
    import numpy as np
    labels = [short(m) for m in models]
    x = np.arange(len(models))
    solve = [summary[m]["solve"] for m in models]
    waste = [summary[m]["waste"] for m in models]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.38
    ax.bar(x - w/2, solve, w, label="solve rate", color="seagreen")
    ax.bar(x + w/2, waste, w, label="wasted-action fraction", color="indianred")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05); ax.set_ylabel("fraction")
    ax.set_title(f"Cross-model under action budget = {budget} (T=3, n=8)")
    for xi, (s, wv) in enumerate(zip(solve, waste)):
        ax.text(xi - w/2, s + 0.02, f"{s:.2f}", ha="center", fontsize=8)
        ax.text(xi + w/2, wv + 0.02, f"{wv:.0%}", ha="center", fontsize=8)
    ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(out, dpi=130)
    plt.close(fig); print(f"\n  wrote {out}")


def fig_outcomes(summary, models, out: Path):
    import numpy as np
    cats = ["built+solved", "built+ranout", "brute+solved", "neverbuilt"]
    leg = {"built+solved": "built + solved", "built+ranout": "built, ran out",
           "brute+solved": "brute + solved", "neverbuilt": "never built"}
    colors = {"built+solved": "seagreen", "built+ranout": "goldenrod",
              "brute+solved": "steelblue", "neverbuilt": "indianred"}
    fam = {"haiku": "Haiku", "sonnet": "Sonnet", "opus": "Opus",
           "gemini": "Gemini", "gpt": "GPT-5"}
    def label(m):
        s = short(m)
        for k, v in fam.items():
            if k in s:
                return v
        return s
    labels = [label(m) for m in models]
    x = np.arange(len(models))
    # short and wide: vertical space is the constraint, so keep it low and put
    # the legend to the right of the axes rather than above the bars.
    fig, ax = plt.subplots(figsize=(6.2, 2.1))
    bottom = np.zeros(len(models))
    for c in cats:
        vals = [summary[m]["oc"][c] for m in models]
        ax.bar(x, vals, 0.78, bottom=bottom, label=leg[c], color=colors[c])
        bottom += np.array(vals)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    # cap = max episodes any single model has (bars stack to per-model count);
    # round up to an even number so the ticks land cleanly.
    cap = max((summary[m]["n"] for m in models), default=8)
    cap = max(8, cap + (cap % 2))
    ax.set_ylim(0, cap); ax.set_yticks(list(range(0, cap + 1, max(2, cap // 4))))
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel("episodes", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, handlelength=1.1, handletextpad=0.5,
              labelspacing=0.4, borderaxespad=0.0)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")          # vector, for LaTeX
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")          # vector, for web
    plt.close(fig); print(f"  wrote {out} (+ .pdf, .svg)")


if __name__ == "__main__":
    main()
