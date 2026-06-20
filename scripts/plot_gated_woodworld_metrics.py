"""Capability-axis metric panels for the gated WoodWorld (two-layer recipe), 3 Claude models.

C-R-E decomposition + solve, one panel per metric, models as a connected series ordered by
capability (Haiku < Sonnet < Opus). Recognition is split into its two recipe subparts:

  Curiosity   C  = P(held the base ingredients)              [held_base]
  Recognition R1 = P(built part | held base)                [step-1 discovery]
  Recognition R2 = P(built axe  | built part)               [step-2 follow-through]
  Efficiency  E  = P(solved | built axe)                    [exploit the built tool]
  Solve          = P(solved)                                [overall]

Computed on the FULL grid T2-6 x N10-30 (all three models now fully cover it).

Run from repo root:
  PYTHONPATH=. python scripts/plot_gated_woodworld_metrics.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# capability-axis x positions (rough param-size estimates, log scale; same convention as
# scripts/plot_metric_lines.py). Used only for ordering/spacing; labels are the model names.
MODELS = [("Haiku", "haiku_gatedwood_two_layer_r1_nohint_T2-6_N10-30_*", 40.0),
          ("Sonnet", "sonnet_gatedwood_two_layer_r1_nohint_T2-6_N10-30_*", 300.0),
          ("Opus", "opus_gatedwood_two_layer_r1_nohint_T2-6_N10-30_*", 2000.0)]

T_LO, T_HI = 2, 6
N_LO, N_HI = 10, 30          # full grid (all three models now fully cover it)


def load(glob):
    d = sorted(Path("runs").glob(glob))[-1]
    rows = [json.loads(l) for l in (d / "episodes.jsonl").read_text().splitlines()
            if l.strip()]
    return [r for r in rows if not r.get("error")
            and T_LO <= r["n_types"] <= T_HI and N_LO <= r["n"] <= N_HI]


def metrics(rows):
    """Return {metric: (value, numerator, denominator)} for one model."""
    n = len(rows)
    base = [r for r in rows if r.get("held_base")]
    part = [r for r in rows if r.get("built_part")]
    axe = [r for r in rows if r.get("built_axe")]
    def frac(num, den):
        return (num / den if den else float("nan"), num, den)
    return {
        "Curiosity\nP(held base)":           frac(len(base), n),
        "Recognition step 1\nP(part | base)": frac(sum(r.get("built_part") for r in base), len(base)),
        "Recognition step 2\nP(axe | part)":  frac(sum(r.get("built_axe") for r in part), len(part)),
        "Efficiency\nP(solved | axe)":        frac(sum(r.get("solved") for r in axe), len(axe)),
        "Solve rate\nP(solved)":              frac(sum(r.get("solved") for r in rows), n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figs/gated-woodworld")
    args = ap.parse_args()

    data = {name: metrics(load(glob)) for name, glob, _ in MODELS}
    xs = [b for _, _, b in MODELS]
    names = [n for n, _, _ in MODELS]
    panels = list(next(iter(data.values())).keys())

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.2))
    color = "#1f77b4"
    for ax, metric in zip(axes, panels):
        ys = [data[nm][metric][0] for nm in names]
        ax.plot(xs, ys, "-o", color=color, lw=2, ms=8, zorder=3)
        for x, nm in zip(xs, names):
            v, num, den = data[nm][metric]
            txt = f"{v:.2f}\n({num}/{den})" if den else "n/a"
            ax.annotate(txt, (x, data[nm][metric][0]), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylim(-0.05, 1.08)
        ax.set_title(metric, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("model (→ capability)")
    axes[0].set_ylabel("probability")
    fig.suptitle("Gated WoodWorld (two-layer recipe r0+r1→part, part+part→axe) — "
                 "C·R·E + solve across capability\n(full grid T2-6 × N10-30; "
                 "recognition split into its two recipe steps; nohint, 1 rep/cell)",
                 y=1.02, fontsize=11)
    fig.tight_layout()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "fig_gatedwood_two_layer_metric_panels.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")

    # also print the table
    print(f"\n{'metric':32}" + "".join(f"{n:>14}" for n in names))
    for metric in panels:
        label = metric.replace("\n", " ")
        cells = "".join(f"{data[nm][metric][0]:>8.2f}{('('+str(data[nm][metric][1])+'/'+str(data[nm][metric][2])+')'):>6}" for nm in names)
        print(f"{label:32}{cells}")


if __name__ == "__main__":
    main()
