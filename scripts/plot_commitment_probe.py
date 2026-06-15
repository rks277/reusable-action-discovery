"""Figure for experiment (J), the commitment probe.

Two panels (house style):
  (1) recognition rate P(built | held both) per condition, with Wilson 95% CIs,
      plus build rate P(built) as a lighter bar; dashed reference at Haiku's
      no-nudge recognition (0.98) and Opus's original base rate (0.77).
  (2) recognition rate vs. budget, one line per condition.

Usage: PYTHONPATH=. python -m scripts.plot_commitment_probe \
           runs/commitment_probe_T3_n12_<ts>/episodes.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from scripts.analyze_recognition_latency import per_episode
from scripts.plot_build_sweep import wilson

ORDER = ["baseline", "commit", "neutral"]
COLOR = {"baseline": "dimgray", "commit": "seagreen", "neutral": "indianred"}
MARK = {"baseline": "o", "commit": "s", "neutral": "^"}
PRETTY = {"baseline": "baseline\n(no nudge)", "commit": "commit\nnudge",
          "neutral": "neutral\nnudge"}
HAIKU_RECOG = 0.98   # no-nudge recognition rates from experiment (E)
SONNET_RECOG = 0.81
OPUS_BASE = 0.77     # original Opus build-sweep recognition rate


def main() -> None:
    path = Path(sys.argv[1])
    if path.is_dir():
        path = path / "episodes.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]

    by = defaultdict(list)
    for r in rows:
        by[r.get("condition", "?")].append(r)
    conds = [c for c in ORDER if c in by]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # panel 1: recognition (with Wilson CI) + build rate, per condition
    recog, rlo, rhi, build, hb_n = [], [], [], [], []
    for c in conds:
        pe = [per_episode(r) for r in by[c]]
        hb = [e for e in pe if e["held_both"]]
        k = sum(e["built"] for e in hb)
        p, lo, hi = wilson(k, len(hb))
        recog.append(p); rlo.append(p - lo); rhi.append(hi - p); hb_n.append(len(hb))
        build.append(sum(e["built"] for e in pe) / len(pe))

    x = range(len(conds))
    w = 0.38
    bars_r = ax1.bar([i - w / 2 for i in x], recog, w,
                     yerr=[rlo, rhi], capsize=4,
                     color=[COLOR[c] for c in conds],
                     label="recognition  P(built | held both)")
    ax1.bar([i + w / 2 for i in x], build, w,
            color=[COLOR[c] for c in conds], alpha=0.4,
            label="build  P(built)")
    ax1.axhline(HAIKU_RECOG, ls="--", lw=1, color="seagreen", alpha=0.7)
    ax1.text(len(conds) - 0.5, HAIKU_RECOG + 0.005, "Haiku recognition (0.98)",
             ha="right", va="bottom", fontsize=7, color="seagreen")
    ax1.axhline(SONNET_RECOG, ls="--", lw=1, color="firebrick", alpha=0.7)
    ax1.text(-0.45, SONNET_RECOG + 0.006, "Sonnet recognition (0.81)",
             ha="left", va="bottom", fontsize=7, color="firebrick")
    ax1.axhline(OPUS_BASE, ls=":", lw=1, color="black", alpha=0.5)
    ax1.text(len(conds) - 0.5, OPUS_BASE - 0.035, "Opus base rate (0.77)",
             ha="right", va="top", fontsize=7, color="black", alpha=0.7)
    for i, c in enumerate(conds):
        ax1.annotate(f"{recog[i]:.2f}", (i - w / 2, recog[i] + rhi[i]),
                     textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
        ax1.annotate(f"{build[i]:.2f}", (i + w / 2, build[i]),
                     textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7,
                     alpha=0.8)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([PRETTY[c] for c in conds])
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("rate")
    ax1.set_title("Recognition & build rate by condition\n(error bars: Wilson 95% CI)")
    ax1.legend(fontsize=8, frameon=False, loc="lower left")

    # panel 2: recognition rate vs budget, per condition
    budgets = sorted({r["budget"] for r in rows})
    for c in conds:
        rate = []
        for b in budgets:
            hb = [e for e in (per_episode(r) for r in by[c] if r["budget"] == b)
                  if e["held_both"]]
            k = sum(e["built"] for e in hb)
            rate.append(k / len(hb) if hb else float("nan"))
        ax2.plot(budgets, rate, marker=MARK[c], color=COLOR[c], lw=1.6, ms=5,
                 label=c)
    ax2.axhline(HAIKU_RECOG, ls="--", lw=1, color="seagreen", alpha=0.5)
    ax2.axhline(SONNET_RECOG, ls="--", lw=1, color="firebrick", alpha=0.45)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(budgets)
    ax2.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax2.set_ylim(-0.03, 1.05)
    ax2.set_xlabel("announced action budget (log2)")
    ax2.set_ylabel("recognition rate  P(built | held both)")
    ax2.set_title("Recognition rate vs. budget")
    ax2.legend(fontsize=8, frameon=False, loc="lower right")

    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    n_cells = len(by[conds[0]])
    fig.suptitle("Commitment probe (J): does a nudge close Opus's recognition gap?"
                 f"  ·  {n_cells} b≤640 paired-loss seeds × 3 conditions, n=12 T=3",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = path.parent / "fig_commitment_probe.png"  # alongside episodes.jsonl
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
