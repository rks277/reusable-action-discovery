"""Post-hoc metrics + figures for woodworld episodes (mirrors analyze_toolworld.py).

Metrics are recomputed by REPLAYING each logged trace through a fresh world
(scripts.replay_woodworld), so they never depended on what the runner logged
live. Target metrics:

  built_axe            binary: did the agent ever craft the persistent tool?
  gather_before_build  # gather actions before the axe was built (opt ~5/0.8).
  action_efficiency    (brute_expected - actual)/(brute_expected - oracle_min),
                       0 = brute (N/0.8), 1 = oracle. Solved episodes only.
  tool_share           of all wood PRODUCED, the fraction from using the axe
                       vs. gathering (the exploitation analog of rational_use).

Figures: calibration (built_axe vs N, N* marked), cross-model action_efficiency
bars at the top N, build-timing distribution.

Usage:
  python -m scripts.analyze_woodworld runs/woodworld_sweep_*/episodes.jsonl
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

from scripts.replay_woodworld import replay
from scripts.validate_woodworld import axe_cost, brute_expected, n_star, oracle_min
from scripts.woodworld import GATHER_PROB


def _short(model: str) -> str:
    return model.replace("claude-", "").replace("-20251001", "")


def metrics(row: dict) -> dict | None:
    """Replay one episode and compute the metrics. None for errored rows."""
    if row.get("error"):
        return None
    n = row["n"]
    s, _ = replay(row)
    actions = row["actions"]

    built = s.built_axe
    if built and s.build_turn is not None:
        gather_before_build = sum(1 for a in actions[:s.build_turn] if a[0] == "gather")
    else:
        gather_before_build = None

    be, om = brute_expected(n), oracle_min(n)
    if row["solved"] and be > om:
        action_efficiency = (be - len(actions)) / (be - om)
    else:
        action_efficiency = None

    produced = s.wood_from_gather + s.wood_from_use
    tool_share = (s.wood_from_use / produced) if (built and produced > 0) else None

    return {
        "model": row["model"], "n": n, "solved": row["solved"],
        "total_actions": len(actions), "built_axe": built,
        "gather_before_build": gather_before_build,
        "action_efficiency": action_efficiency, "tool_share": tool_share,
        "use_axe_count": s.use_axe_count,
    }


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, sd, len(vals)


def summarize(rows_metrics: list[dict]):
    cells = defaultdict(list)
    for m in rows_metrics:
        cells[(m["model"], m["n"])].append(m)
    print(f"\n{'model':22s} {'N':>3} {'eps':>3} {'solved':>6} {'built':>6} "
          f"{'gather_pre':>12} {'act_eff':>13} {'tool_share':>12}")
    print("-" * 90)
    summary = {}
    for (model, n), ms in sorted(cells.items()):
        solved_rate = statistics.mean([m["solved"] for m in ms])
        built_rate = statistics.mean([m["built_axe"] for m in ms])
        gm, gs, gn = _agg([m["gather_before_build"] for m in ms])
        em, es, en = _agg([m["action_efficiency"] for m in ms])
        tm, tsd, tn = _agg([m["tool_share"] for m in ms])
        summary[(model, n)] = {"eps": len(ms), "solved_rate": solved_rate,
                               "built_rate": built_rate, "gather": (gm, gs, gn),
                               "act_eff": (em, es, en), "tool_share": (tm, tsd, tn)}
        g = f"{gm:.1f}±{gs:.1f}(n{gn})" if gm is not None else "  -"
        e = f"{em:.2f}±{es:.2f}(n{en})" if em is not None else "   -"
        t = f"{tm:.2f}(n{tn})" if tm is not None else "  -"
        print(f"{_short(model)[:22]:22s} {n:>3} {len(ms):>3} {solved_rate:>6.2f} "
              f"{built_rate:>6.2f} {g:>12} {e:>13} {t:>12}")
    return summary


def fig_calibration(summary, out: Path, nstar: int):
    """built_axe rate vs N, per model, with N* marked."""
    models = sorted({m for m, _ in summary})
    ns = sorted({n for _, n in summary})
    fig, ax = plt.subplots(figsize=(6, 4))
    for model in models:
        xs = [n for n in ns if (model, n) in summary]
        ys = [summary[(model, n)]["built_rate"] for n in xs]
        ax.plot(xs, ys, "o-", label=_short(model))
    ax.axvline(nstar, ls="--", color="grey", label=f"N*={nstar} (rational threshold)")
    ax.set_xlabel("N (target wood)"); ax.set_ylabel("built_axe rate")
    ax.set_ylim(-0.05, 1.05); ax.set_title("Calibration: tool construction vs N")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out, dpi=130)
    plt.close(fig); print(f"  wrote {out}")


def fig_efficiency_bars(summary, out: Path, n: int):
    """action_efficiency per model at N, with brute(0) and oracle(1) refs."""
    items = [(m, summary[(m, nn)]["act_eff"]) for m, nn in summary if nn == n]
    items = [(m, a) for m, a in items if a[0] is not None]
    items.sort()
    if not items:
        print(f"  (no solved episodes at N={n}; skipping efficiency bars)")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [_short(m) for m, _ in items]
    means = [a[0] for _, a in items]
    sds = [a[1] for _, a in items]
    ax.bar(labels, means, yerr=sds, capsize=4, color="steelblue")
    ax.axhline(0, ls="--", color="firebrick", label="brute (0)")
    ax.axhline(1, ls="--", color="green", label="oracle (1)")
    ax.set_ylabel("action_efficiency")
    ax.set_ylim(min(means + [0]) - 0.2, 1.1)
    ax.set_title(f"Cross-model action efficiency (N={n})")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out, dpi=130)
    plt.close(fig); print(f"  wrote {out}")


def fig_build_timing(rows_metrics, out: Path, n: int):
    """Distribution of gather_before_build per model at N (built episodes)."""
    by_model = defaultdict(list)
    for m in rows_metrics:
        if m["n"] == n and m["gather_before_build"] is not None:
            by_model[m["model"]].append(m["gather_before_build"])
    if not by_model:
        print(f"  (no built episodes at N={n}; skipping build-timing)")
        return
    opt = axe_cost()[0] / GATHER_PROB  # E[gathers] to collect the build wood
    models = sorted(by_model)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, model in enumerate(models):
        ys = by_model[model]
        xs = [i + (hash(str(j)) % 100 - 50) / 400 for j in range(len(ys))]
        ax.scatter(xs, ys, alpha=0.6, label=f"{_short(model)} (n={len(ys)})")
    ax.axhline(opt, ls="--", color="green", label=f"optimal ≈ {opt:.1f}")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([_short(m) for m in models], rotation=15, fontsize=8)
    ax.set_ylabel("gather_before_build")
    ax.set_title(f"Build timing per model (N={n})")
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
    (out_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(m) for m in rows_metrics) + "\n")
    print(f"\nPer-episode metrics: {out_dir/'metrics.jsonl'}")

    top_n = max((m["n"] for m in rows_metrics), default=20)
    print("\nFigures:")
    fig_calibration(summary, out_dir / "fig_wood_calibration.png", n_star())
    fig_efficiency_bars(summary, out_dir / "fig_wood_efficiency.png", n=top_n)
    fig_build_timing(rows_metrics, out_dir / "fig_wood_build_timing.png", n=top_n)


if __name__ == "__main__":
    main()
