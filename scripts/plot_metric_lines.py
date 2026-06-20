"""Line graph of the four panel metrics vs. model parameter size, one figure per
world (toolworld | woodworld). Each metric (recognition, curiosity, efficiency,
solve rate) is a separate series, its per-model pooled headline connected by a line.

x = parameter size (log scale, billions). Open-weight sizes are exact; Claude sizes
are ROUGH public-estimate placeholders (no official numbers exist) -- edit CLAUDE_B
below and the on-figure caveat stands. y = metric in [0, 1].

The headlines are computed identically to the panel scripts (same predicates, pooled
over all contributing episodes), so each point equals the number in its panel title.

Usage:
  PYTHONPATH=. python -m scripts.plot_metric_lines --world toolworld DIR DIR ... \
      [--outdir figs/toolworld] [--out fig_metric_lines_toolworld.png]
  PYTHONPATH=. python -m scripts.plot_metric_lines --world woodworld DIR DIR ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plot_recognition_panels import built, held_both
from scripts.analyze_recognition_latency import acquire_both_turn
from scripts.plot_woodworld_panels import held_ingredients

# --- parameter-size estimates (billions) ----------------------------------------
# Open-weight: exact / effective params. Claude: ROUGH order-of-magnitude guesses --
# Anthropic publishes no sizes; these only fix the left->right ordering and decade.
OSS_B = {
    "qwen2.5:1.5b": 1.5, "qwen2.5:3b": 3.0, "qwen2.5:7b": 7.0,
    "gemma4:e2b": 2.3, "gemma4:e4b": 4.5, "gemma4:12b": 12.0,
}
CLAUDE_B = {"haiku": 40.0, "sonnet": 300.0, "opus": 2000.0}   # <-- estimates, editable
# Proprietary frontier models also have no public size; key by the _norm() form
# ("gpt-5.5" -> "gpt:5.5"). Pure guess, placed at the frontier near Opus.
GPT_B = {"gpt:5.5": 1500.0, "gpt:5.4:mini": 100.0}  # "mini" -> guessed mid-tier
ESTIMATED_B = {**CLAUDE_B, **GPT_B}                            # all size-guessed (shaded)
PARAM_B = {**OSS_B, **ESTIMATED_B}

# BFCL V4 Overall Accuracy (%) as an alternative x-axis (capability proxy). FC scores
# from data/external/bfcl_v4_overall_*. Qwen2.5 is NOT on BFCL V4, so each qwen2.5 rung
# uses its same-size Qwen3 score as a PROXY; Claude entries are the 4-5 versions (our
# runs are 4-8/4-6/4-5 -- Haiku matches exactly, Sonnet/Opus are one minor version off).
BFCL_SCORE = {
    "qwen2.5:1.5b": 28.41, "qwen2.5:3b": 35.68, "qwen2.5:7b": 42.57,
    "haiku": 68.70, "sonnet": 73.24, "opus": 77.47,
}
BFCL_PROXY = {"qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b"}   # same-size Qwen3 stand-in


def _norm(short: str) -> str:
    """woodworld load() turns 'qwen2.5:7b' into 'qwen2.5-7b'; toolworld keeps the
    colon. Canonicalize to the colon form so one PARAM_B dict serves both."""
    return short.replace("-", ":")


def _rows(run_dir: Path) -> tuple[str, list[dict]]:
    rs = [json.loads(l) for l in (run_dir / "episodes.jsonl").read_text().splitlines() if l.strip()]
    rs = [r for r in rs if not r.get("error")]
    m0 = rs[0].get("model", "?") if rs else "?"
    short = m0.split("-")[1] if m0.startswith("claude-") else m0
    return short, rs


def _ratio(num: int, den: int) -> tuple[float, int]:
    """Return (proportion, denominator n). n feeds the binomial standard error
    sqrt(p(1-p)/n) used for error bars; denominators differ per metric."""
    return (num / den if den else float("nan"), den)


def metrics_toolworld(rows: list[dict]) -> dict:
    solved = [r for r in rows if r.get("solved")]
    held = [r for r in rows if held_both(r)]                     # recognition denom
    blt = [r for r in rows if built(r)]                          # efficiency denom
    cur = [r for r in rows                                       # industry panel def
           if acquire_both_turn(r.get("obs"), set(r["labels"]["recipe"])) is not None]
    return {
        "recognition": _ratio(sum(built(r) for r in held), len(held)),
        "curiosity": _ratio(len(cur), len(rows)),
        "efficiency": _ratio(sum(bool(r.get("solved")) for r in blt), len(blt)),
        "solve rate": _ratio(len(solved), len(rows)),
    }


def metrics_woodworld(rows: list[dict]) -> dict:
    held = [r for r in rows if held_ingredients(r)]              # recognition denom
    axe = [r for r in rows if r.get("built_axe")]                # efficiency denom
    return {
        "recognition": _ratio(sum(bool(r.get("built_axe")) for r in held), len(held)),
        "curiosity": _ratio(len(held), len(rows)),
        "efficiency": _ratio(sum(bool(r.get("solved")) for r in axe), len(axe)),
        "solve rate": _ratio(sum(bool(r.get("solved")) for r in rows), len(rows)),
    }


SERIES = ["recognition", "curiosity", "efficiency", "solve rate"]
COLORS = {"recognition": "#1f77b4", "curiosity": "#2ca02c",
          "efficiency": "#d62728", "solve rate": "#000000"}
MARKERS = {"recognition": "o", "curiosity": "s", "efficiency": "^", "solve rate": "D"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=["toolworld", "woodworld"], required=True)
    ap.add_argument("dirs", nargs="+", help="region run dirs (any order; sorted by x)")
    ap.add_argument("--xaxis", choices=["params", "bfcl"], default="params",
                    help="x-axis: estimated parameter size (log) or BFCL V4 score (linear)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    xmap = PARAM_B if args.xaxis == "params" else BFCL_SCORE
    mfn = metrics_toolworld if args.world == "toolworld" else metrics_woodworld
    pts = []  # (x, short, {metric: value})
    for d in args.dirs:
        short, rows = _rows(Path(d))
        key = _norm(short)
        if key not in xmap:
            print(f"WARN: no {args.xaxis} value for {short!r} ({key}); skipping {d}")
            continue
        m = mfn(rows)
        pts.append((xmap[key], short, m))
        print(f"{short:14} (x={xmap[key]:>7.2f})  " +
              "  ".join(f"{k}={m[k][0]:.2f}(n={m[k][1]})" for k in SERIES))
    pts.sort(key=lambda x: x[0])

    def _disp(s: str) -> str:
        k = _norm(s)
        if k.startswith("qwen2.5:"):
            return "Qwen " + k.split(":", 1)[1]
        if k.startswith("gemma4:"):
            return "Gemma " + k.split(":", 1)[1]
        if k.startswith("gpt"):
            return "GPT-" + k.split(":", 1)[1].replace(":", "-")
        return s.capitalize()

    xs = [p for p, _, _ in pts]
    if args.xaxis == "params":
        labels = [f"{_disp(s)}\n~{p:g}B" for p, s, _ in pts]
    else:
        labels = [f"{_disp(s)}{'*' if _norm(s) in BFCL_PROXY else ''}\n{p:.0f}%"
                  for p, s, _ in pts]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    for metric in SERIES:
        vals = [m[metric] for _, _, m in pts]
        ys = [v for v, n in vals]
        # binomial standard error sqrt(p(1-p)/n); nan where the metric is undefined
        errs = [(v * (1 - v) / n) ** 0.5 if (n and v == v) else float("nan")
                for v, n in vals]
        ax.errorbar(xs, ys, yerr=errs, marker=MARKERS[metric], color=COLORS[metric],
                    lw=2, ms=8, capsize=3, elinewidth=1.2, label=metric,
                    markeredgecolor="black", markeredgewidth=0.6)
    if args.xaxis == "params":
        ax.set_xscale("log")
        ax.set_xlabel("model parameter size (billions, log scale)")
        est_xs = [p for p, s, _ in pts if _norm(s) in ESTIMATED_B]
        if est_xs:
            ax.axvspan(min(est_xs) / 1.4, max(est_xs) * 1.4, color="0.85",
                       alpha=0.4, zorder=0, label="proprietary (sizes estimated)")
        title = (f"{args.world.capitalize()}: panel metrics vs. model parameter size\n"
                 "(pooled headline ±1 binomial SE; proprietary Claude/GPT sizes are rough estimates — see shaded band)")
    else:
        ax.set_xlim(min(xs) - 6, max(xs) + 6)
        ax.set_xlabel("BFCL V4 Overall Accuracy, FC (%)  —  capability proxy")
        title = (f"{args.world.capitalize()}: panel metrics vs. BFCL V4 tool-use score\n"
                 "(pooled headline per model; * = Qwen2.5 rung shown at its same-size Qwen3 "
                 "score — Qwen2.5 not on BFCL V4)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8, rotation=25, ha="right", rotation_mode="anchor")
    ax.minorticks_off()
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("metric value")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.set_title(title, fontsize=12)

    outdir = Path(args.outdir or f"figs/{args.world}")
    outdir.mkdir(parents=True, exist_ok=True)
    default_out = f"fig_metric_lines_{args.world}" + ("_bfcl" if args.xaxis == "bfcl" else "") + ".png"
    out = outdir / (args.out or default_out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
