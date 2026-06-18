"""Post-hoc metrics + figures for tool-world episodes (04_metrics, 05_experiments).

Metrics are recomputed by REPLAYING each logged trace through a fresh world
(scripts.replay_toolworld), so they never depended on what the runner logged
live. The four target metrics (04_metrics.md):

  built_tool          binary: did the agent ever combine 2 byproducts -> machine?
  grind_before_build  # examine actions before the machine was built (opt ~2).
  action_efficiency   (brute_expected - actual) / (brute_expected - optimal),
                      brute_expected = n*H_n + n, optimal = 2n+1. 0=grind, 1=opt.
  rational_use        of doors the agent could NOT open with a free drop-key,
                      the fraction it opened via the machine. (Understanding.)

We deliberately IGNORE the old reuse_fraction (04_metrics: it penalized the
rational use of free drop-keys).

Figures (05_experiments.md): calibration curve (built_tool vs n, n*=3 marked),
cross-model action_efficiency bars at n=8, grind_before_build distribution.

Usage:
  python -m scripts.analyze_toolworld runs/toolworld_sweep_*/episodes.jsonl
"""

from __future__ import annotations

import json
import random
import re
import statistics
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.replay_toolworld import replay
from scripts.validate_toolworld_v2 import simulate_machine_informed


def Hn(n: int) -> float:
    return sum(1.0 / k for k in range(1, n + 1))


def brute_expected(n: int) -> float:
    return n * Hn(n) + n


@lru_cache(maxsize=None)
def optimal_tool(n: int, n_types: int = 1) -> float:
    """Informed-optimal machine cost. v1 (n_types<=1): analytic 2n+1. v2: the
    Monte-Carlo informed-optimal from validate_toolworld_v2 (the re-derived
    optimum that accounts for gathering the right T-type recipe)."""
    if n_types <= 1:
        return 2 * n + 1
    return statistics.mean(simulate_machine_informed(n, n_types, random.Random(7000 + s))
                           for s in range(5000))


def n_star(n_types: int) -> int:
    """Rational threshold: v1 single-type = 3; T-type from validate_toolworld_v2."""
    return 3 if n_types <= 1 else 4


def metrics(row: dict) -> dict | None:
    """Replay one episode and compute the four metrics. None for errored rows."""
    if row.get("error"):
        return None
    n = row["n"]
    n_types = row.get("n_types", 1)
    s, obs = replay(row)
    actions = row["actions"]
    # v3 records free `pickup` actions in the trace (so replay reconstructs the
    # inventory exactly), but they cost no budget -- exclude them from any
    # action-cost metric. v1/v2 traces have no pickups, so this is a no-op there.
    budgeted = [a for a in actions if a[0] != "pickup"]

    # --- built_tool & build index (first combine that fused the machine) ---
    build_idx = next((i for i, o in enumerate(obs) if "fuse into" in o), None)
    built_tool = build_idx is not None

    # --- grind_before_build: # examines before the build (None if never built) ---
    if built_tool:
        grind = sum(1 for a in actions[:build_idx] if a[0] == "examine")
    else:
        grind = None

    # --- action_efficiency (solved only; degenerate at/below threshold where
    #     brute_expected <= optimal_tool, i.e. the tool can't help) ---
    be, opt = brute_expected(n), optimal_tool(n, n_types)
    if row["solved"] and be > opt:
        action_efficiency = (be - len(budgeted)) / (be - opt)
    else:
        action_efficiency = None

    # --- rational_use: of doors with no free drop-key, fraction machine-opened ---
    key_base = row["labels"]["key_base"]
    drop_re = re.compile(rf"Out falls {re.escape(key_base)}_(\d+) ")
    drop_supplied = set()
    for o in obs:
        m = drop_re.search(o)
        if m:
            drop_supplied.add(int(m.group(1)))
    # s.opened: door_idx -> (method, turn); method 'machine' or 'drop'
    needed_tool = [d for d in s.opened if d not in drop_supplied]
    if needed_tool:
        via_machine = sum(1 for d in needed_tool if s.opened[d][0] == "machine")
        rational_use = via_machine / len(needed_tool)
    else:
        rational_use = None  # no door required the tool

    return {
        "model": row["model"], "n": n, "n_types": n_types,
        "solved": row["solved"], "total_actions": len(budgeted),
        "built_tool": built_tool, "grind_before_build": grind,
        "action_efficiency": action_efficiency, "rational_use": rational_use,
        "open_methods": [s.opened[d][0] for d in sorted(s.opened)],
        "n_needed_tool": len(needed_tool),
    }


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, sd, len(vals)


def summarize(rows_metrics: list[dict]):
    """Group by (model, n); print a table of the four metrics."""
    cells = defaultdict(list)
    for m in rows_metrics:
        cells[(m["model"], m["n"])].append(m)

    print(f"\n{'model':30s} {'n':>2} {'eps':>3} {'solved':>6} "
          f"{'built':>6} {'grind':>11} {'act_eff':>13} {'rat_use':>10}")
    print("-" * 96)
    summary = {}
    for (model, n), ms in sorted(cells.items()):
        solved_rate = statistics.mean([m["solved"] for m in ms])
        built_rate = statistics.mean([m["built_tool"] for m in ms])
        gm, gs, gn = _agg([m["grind_before_build"] for m in ms])
        em, es, en = _agg([m["action_efficiency"] for m in ms])
        rm, rs, rn = _agg([m["rational_use"] for m in ms])
        summary[(model, n)] = {
            "eps": len(ms), "solved_rate": solved_rate, "built_rate": built_rate,
            "grind": (gm, gs, gn), "act_eff": (em, es, en),
            "rational_use": (rm, rs, rn),
        }
        g = f"{gm:.1f}±{gs:.1f}(n{gn})" if gm is not None else "  -"
        e = f"{em:.2f}±{es:.2f}(n{en})" if em is not None else "   -"
        r = f"{rm:.2f}(n{rn})" if rm is not None else "  -"
        print(f"{model[:30]:30s} {n:>2} {len(ms):>3} {solved_rate:>6.2f} "
              f"{built_rate:>6.2f} {g:>11} {e:>13} {r:>10}")
    return summary


# --- Figures (05_experiments.md) ---------------------------------------

def fig_calibration(summary, out: Path, nstar: int):
    """built_tool rate vs n, per model, with n* marked."""
    models = sorted({m for m, _ in summary})
    ns = sorted({n for _, n in summary})
    fig, ax = plt.subplots(figsize=(6, 4))
    for model in models:
        xs = [n for n in ns if (model, n) in summary]
        ys = [summary[(model, n)]["built_rate"] for n in xs]
        ax.plot(xs, ys, "o-", label=model.replace("claude-", "").replace("-20251001", ""))
    ax.axvline(nstar, ls="--", color="grey", label=f"n*={nstar} (rational threshold)")
    ax.set_xlabel("n (number of doors)"); ax.set_ylabel("built_tool rate")
    ax.set_ylim(-0.05, 1.05); ax.set_title("Calibration: tool construction vs n")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out, dpi=130)
    plt.close(fig); print(f"  wrote {out}")


def fig_efficiency_bars(summary, out: Path, n: int = 8):
    """action_efficiency per model at n, with brute(0) and optimal(1) refs."""
    items = [(m, summary[(m, n)]["act_eff"]) for m, nn in summary if nn == n]
    items = [(m, a) for m, a in items if a[0] is not None]
    items.sort()
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [m.replace("claude-", "").replace("-20251001", "") for m, _ in items]
    means = [a[0] for _, a in items]
    sds = [a[1] for _, a in items]
    ax.bar(labels, means, yerr=sds, capsize=4, color="steelblue")
    ax.axhline(0, ls="--", color="firebrick", label="brute (0)")
    ax.axhline(1, ls="--", color="green", label="optimal tool (1)")
    ax.set_ylabel("action_efficiency")
    lo = min([m - s for m, s in zip(means, sds)] + [0]) - 0.2
    ax.set_ylim(lo, 1.1)
    ax.set_title(f"Cross-model action efficiency (n={n})")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out, dpi=130)
    plt.close(fig); print(f"  wrote {out}")


def fig_grind(rows_metrics, out: Path, n: int = 8):
    """Distribution of grind_before_build per model at n (built episodes)."""
    by_model = defaultdict(list)
    n_types = 1
    for m in rows_metrics:
        if m["n"] == n and m["grind_before_build"] is not None:
            by_model[m["model"]].append(m["grind_before_build"])
            n_types = max(n_types, m["n_types"])
    # optimal grind: examines to gather the components before building.
    # v1 single-type -> 2; v2 T-type -> ~3T/2 (gather both recipe types).
    opt_grind = 2 if n_types <= 1 else 1.5 * n_types
    models = sorted(by_model)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, model in enumerate(models):
        ys = by_model[model]
        xs = [i + (hash(str(j)) % 100 - 50) / 400 for j in range(len(ys))]
        ax.scatter(xs, ys, alpha=0.6, label=f"{model.replace('claude-', '').replace('-20251001','')} (n={len(ys)})")
    ax.axhline(opt_grind, ls="--", color="green",
               label=f"optimal grind ≈ {opt_grind:g}")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace("claude-", "").replace("-20251001", "") for m in models],
                       rotation=15, fontsize=8)
    ax.set_ylabel("grind_before_build (examines pre-build)")
    ax.set_title(f"Build timing per model (n={n})")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out, dpi=130)
    plt.close(fig); print(f"  wrote {out}")


def main():
    path = Path(sys.argv[1])
    out_dir = path.parent
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows_metrics = [m for m in (metrics(r) for r in rows) if m is not None]
    n_err = sum(1 for r in rows if r.get("error"))
    print(f"Loaded {len(rows)} rows ({n_err} errored) from {path}")

    summary = summarize(rows_metrics)

    # per-episode metrics dump (for transparency / reuse)
    (out_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(m) for m in rows_metrics) + "\n")
    print(f"\nPer-episode metrics: {out_dir/'metrics.jsonl'}")

    nstar = n_star(max((m["n_types"] for m in rows_metrics), default=1))
    print("\nFigures:")
    fig_calibration(summary, out_dir / "fig_calibration.png", nstar)
    fig_efficiency_bars(summary, out_dir / "fig_efficiency_bars.png", n=8)
    fig_grind(rows_metrics, out_dir / "fig_grind.png", n=8)


if __name__ == "__main__":
    main()
