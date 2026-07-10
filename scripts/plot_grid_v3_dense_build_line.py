"""2×1 summary figure: P(won) overall + P(built) by model family.

Left panel: end-to-end win rate P(won), unconditional.
Right panel: unconditional build rate P(built the machine).

Both use the same style as plot_grid_v3_dense_lines: x = model size (log;
Anthropic = supposed), one line per family (Anthropic / Qwen2.5), 95% Wilson CI.

Usage: PYTHONPATH=. python -m scripts.plot_grid_v3_dense_build_line [dir ...]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plot_grid_v3_dense_lines import DENSE_DIR, FAMILY_STYLE, discover
from scripts.plot_grid_v3_dense_bars import PANELS, wilson


def _draw_panel(ax, fams, title, denom, outcome, all_sizes):
    for fam, members in fams.items():
        style = FAMILY_STYLE.get(fam, dict(color="gray", marker="^"))
        xs, ys, los, his, labels = [], [], [], [], []
        for size, label, facts in members:
            sel = [f for f in facts if denom(f)]
            k = int(sum(outcome(f) for f in sel)); n = len(sel)
            p = k / n if n else float("nan")
            lo, hi = wilson(k, n)
            xs.append(size); ys.append(p)
            los.append(p - lo); his.append(hi - p); labels.append(label)
            print(f"  {fam:<10} {label:<11} {title[:12]}: {k:>3}/{n:<3} = {p:.3f}")
        # small-capped whiskers: readable but lighter than the line so CIs don't shout
        ax.errorbar(xs, ys, yerr=[los, his], capsize=3, capthick=1.0, elinewidth=1.2,
                    ecolor=style["color"], alpha=0.6, zorder=1, lw=0, marker="none")
        ax.plot(xs, ys, lw=2, markersize=9, markeredgecolor="black",
                markeredgewidth=0.6, label=fam, zorder=2, **style)
        for x, y, lab in zip(xs, ys, labels):
            if not math.isnan(y):
                ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 9),
                            ha="center", fontsize=8, color=style["color"])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(1.5, 800); ax.set_ylim(0, 1.0)
    ax.set_xticks(all_sizes)
    ax.set_xticklabels([f"{int(s)}B" for s in all_sizes], fontsize=8,
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlabel("model size (B params; Anthropic = supposed)")
    ax.set_ylabel("probability")
    ax.grid(alpha=0.3, which="major")
    ax.legend(loc="upper left", fontsize=10)


def main():
    args = [Path(a) for a in sys.argv[1:] if not a.startswith("-")]
    dirs = args if args else sorted(p for p in DENSE_DIR.iterdir() if p.is_dir())
    fams = discover(dirs)
    if not fams:
        raise SystemExit(f"no classifiable run dirs found under {DENSE_DIR}")

    all_sizes = sorted({s for ms in fams.values() for s, _, _ in ms})

    # PANELS[3] = P(won) overall, PANELS[0..2] are the chain panels.
    # Build an unconditional-built panel: denom=all, outcome=built.
    won_title, won_denom, won_outcome = PANELS[3]
    built_title = "P(built machine), all episodes"
    built_denom = lambda f: True
    built_outcome = lambda f: 1.0 if f["built"] else 0.0

    fig, (ax_won, ax_built) = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    _draw_panel(ax_won,   fams, won_title,   won_denom,   won_outcome,   all_sizes)
    _draw_panel(ax_built, fams, built_title, built_denom, built_outcome, all_sizes)

    fig.suptitle("toolworld v3 dense sweep -- summary rates by model size\n"
                 "(line = family, log x; Anthropic sizes are supposed; 95% Wilson CI)",
                 fontsize=13, fontweight="bold")

    out = DENSE_DIR / "fig_grid_v3_dense_summary_lines.png"
    fig.savefig(out, dpi=130); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out}")
    print(f"  wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
