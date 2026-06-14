"""Overlay build-rate and build-timing vs. budget for MULTIPLE build sweeps.

Same two panels as plot_build_sweep.py, but plots every run passed on the command
line on shared axes so models can be compared directly (e.g. Sonnet vs. Opus):
  (1) build rate (fraction of reps that built) with Wilson 95% CIs, one line/model.
  (2) actions-to-build among builders (build_turn+1): median line + 25-75th band.

Model labels and per-model rep counts are read from the data. Budgets need not be
identical across runs; the union is used for the (log2) x-axis.

Usage:
  PYTHONPATH=. python -m scripts.plot_build_sweep_compare \
      runs/<sonnet_build_sweep_dir>/episodes.jsonl \
      runs/<opus_build_sweep_dir>/episodes.jsonl \
      [out.png]
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
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def pctl(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    i = q * (len(s) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (i - lo)


def label_for(model: str) -> str:
    m = (model or "").lower()
    return ("Sonnet" if "sonnet" in m else "Opus" if "opus" in m
            else "Haiku" if "haiku" in m else "Fable" if "fable" in m
            else (model or "model"))


# distinct (color, marker) per series, in order
STYLES = [("seagreen", "o"), ("firebrick", "s"), ("steelblue", "^"),
          ("darkorange", "D"), ("purple", "v")]


def load(path: Path) -> dict:
    if path.is_dir():
        path = path / "episodes.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    by = defaultdict(list)
    for r in rows:
        by[r["budget"]].append(r)
    rmin = min(len(g) for g in by.values())
    rmax = max(len(g) for g in by.values())
    return {
        "label": label_for(rows[0].get("model")),
        "n": rows[0].get("n"), "T": rows[0].get("n_types"),
        "reps": f"{rmin}" if rmin == rmax else f"{rmin}–{rmax}",
        "by": by, "budgets": sorted(by),
    }


def main() -> None:
    args = [a for a in sys.argv[1:]]
    out = None
    if args and args[-1].lower().endswith((".png", ".pdf")):
        out = Path(args.pop())
    paths = [Path(a) for a in args]
    if len(paths) < 2:
        sys.exit("need >=2 episodes.jsonl paths (plus optional out.png)")

    runs = [load(p) for p in paths]
    all_budgets = sorted({b for r in runs for b in r["budgets"]})
    n = runs[0]["n"]
    T = runs[0]["T"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for run, (color, marker) in zip(runs, STYLES):
        by = run["by"]
        budgets = run["budgets"]
        rate, rlo, rhi = [], [], []
        med, p25, p75 = [], [], []
        for b in budgets:
            g = by[b]
            k = sum(1 for r in g if r.get("built_machine"))
            p, lo, hi = wilson(k, len(g))
            rate.append(p); rlo.append(p - lo); rhi.append(hi - p)
            atb = [r["build_turn"] + 1 for r in g
                   if r.get("built_machine") and r.get("build_turn") is not None]
            if atb:
                med.append(statistics.median(atb))
                p25.append(pctl(atb, .25)); p75.append(pctl(atb, .75))
            else:
                med.append(float("nan")); p25.append(float("nan"))
                p75.append(float("nan"))
        leg = f"{run['label']} ({run['reps']} reps)"
        ax1.errorbar(budgets, rate, yerr=[rlo, rhi], marker=marker, capsize=3,
                     color=color, lw=1.6, ms=5, label=leg)
        ax2.fill_between(budgets, p25, p75, color=color, alpha=0.18)
        ax2.plot(budgets, med, marker=marker, color=color, lw=1.6, ms=5, label=leg)

    ax1.set_ylim(-0.03, 1.05)
    ax1.set_ylabel("build rate (fraction of reps that built)")
    ax1.set_title("Did it build the machine?")
    ax1.legend(fontsize=8, frameon=False, loc="lower right")

    ax2.set_ylabel("actions to build (builders only)")
    ax2.set_title("When did it build?  (median, 25–75th band)")
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    ax2.set_ylim(bottom=0)

    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(all_budgets)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel("announced action budget (log2)")
        ax.grid(True, alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    names = " vs. ".join(r["label"] for r in runs)
    fig.suptitle(f"{names} build behavior vs. budget  ·  n={n}, T={T}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if out is None:
        slug = "_".join(r["label"].lower() for r in runs)
        out = Path("runs") / f"fig_build_compare_{slug}.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
