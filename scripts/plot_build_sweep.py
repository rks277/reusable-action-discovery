"""Build-rate and build-timing vs. budget, for run_sonnet_build_sweep.py output.

Two panels, budget on a log2 x-axis (budgets double):
  (1) build rate (fraction of reps that built the machine) with Wilson 95% CIs.
  (2) actions-to-build among builders (build_turn+1): median line, with the
      min–max range as a shaded band and the 25-75th percentile as a darker band.

Usage: PYTHONPATH=. python -m scripts.plot_build_sweep runs/<sonnet_build_sweep_dir>/episodes.jsonl
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
import matplotlib.ticker as mticker


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a proportion (point, lo, hi). Stable at 0/n
    and n/n where the normal approximation degenerates."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def pctl(xs: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0,1]) on a sorted-able list."""
    s = sorted(xs)
    if not s:
        return float("nan")
    i = q * (len(s) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (i - lo)


def main() -> None:
    path = Path(sys.argv[1])
    if path.is_dir():
        path = path / "episodes.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]

    by = defaultdict(list)
    for r in rows:
        by[r["budget"]].append(r)
    budgets = sorted(by)
    n = rows[0].get("n")
    T = rows[0].get("n_types")
    rmin = min(len(g) for g in by.values())
    rmax = max(len(g) for g in by.values())
    reps = f"{rmin}" if rmin == rmax else f"{rmin}–{rmax}"
    mid = (rows[0].get("model") or "").lower()
    label = ("Sonnet" if "sonnet" in mid else "Opus" if "opus" in mid
             else "Haiku" if "haiku" in mid else "Fable" if "fable" in mid
             else (rows[0].get("model") or "model"))

    rate, rlo, rhi = [], [], []
    med, p25, p75, lo, hi = [], [], [], [], []
    for b in budgets:
        g = by[b]
        k = sum(1 for r in g if r.get("built_machine"))
        p, l, h = wilson(k, len(g))
        rate.append(p); rlo.append(p - l); rhi.append(h - p)
        atb = [r["build_turn"] + 1 for r in g
               if r.get("built_machine") and r.get("build_turn") is not None]
        if atb:
            med.append(statistics.median(atb)); p25.append(pctl(atb, .25))
            p75.append(pctl(atb, .75)); lo.append(min(atb)); hi.append(max(atb))
        else:
            med.append(float("nan")); p25.append(float("nan"))
            p75.append(float("nan")); lo.append(float("nan")); hi.append(float("nan"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # panel 1: build rate
    ax1.errorbar(budgets, rate, yerr=[rlo, rhi], marker="o", capsize=3,
                 color="seagreen", lw=1.6, ms=5)
    ax1.set_ylim(-0.03, 1.05)
    ax1.set_ylabel("build rate (fraction of reps that built)")
    ax1.set_title(f"Did {label} build the machine?")
    for x, y in zip(budgets, rate):
        ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=7)

    # panel 2: build timing among builders
    ax2.fill_between(budgets, lo, hi, color="steelblue", alpha=0.15, label="min–max")
    ax2.fill_between(budgets, p25, p75, color="steelblue", alpha=0.35, label="25–75th pct")
    ax2.plot(budgets, med, marker="o", color="steelblue", lw=1.6, ms=5, label="median")
    ax2.set_ylabel("actions to build (builders only)")
    ax2.set_title("When did it build?")
    ax2.legend(fontsize=8, frameon=False)
    ax2.set_ylim(bottom=0)

    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel("announced action budget (log2)")
        ax.grid(True, alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle(f"{label} build behavior vs. budget  ·  n={n}, T={T}, {reps} reps/budget",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = path.parent / "fig_build_rate_and_timing_vs_budget.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
