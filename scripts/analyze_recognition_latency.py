"""Experiment (E): recognition vs gathering -- when (if ever) does a model that
HOLDS both winning ingredients commit to building the tool?

Reframes the build gap around "discovery = recognizing the tool can be made."
Build rate factorizes exactly:

    build_rate  =  gathering_rate  ×  recognition_rate
    (built/total) = (held_both/total) × (built/held_both)

because combine() builds iff you hold both recipe types (toolworld_v2), so every
builder held both, and among ingredient-holders the only outcomes are "built"
(recognized + committed) or "held both, never tried the winning combine". This
script computes that decomposition per model and the RECOGNITION LATENCY -- the
number of actions between first holding both ingredients and issuing the winning
combine -- and writes two figures.

Usage:
  PYTHONPATH=. python -m scripts.analyze_recognition_latency \
      runs/haiku_build_sweep_T3_n12_20260614_190439/episodes.jsonl \
      runs/sonnet_build_sweep_T3_n12_20260613_003611/episodes.jsonl \
      runs/opus_build_sweep_T3_n12_20260613_212256/episodes.jsonl
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HOLD_RE = re.compile(r"hold (\d+) (\S+?)\(s\)")
# capability order -> consistent colors/markers across both figures
STYLE = {"Haiku": ("seagreen", "o"), "Sonnet": ("firebrick", "s"),
         "Opus": ("steelblue", "^"), "Fable": ("darkorange", "D")}


def label_for(model: str) -> str:
    m = (model or "").lower()
    return ("Sonnet" if "sonnet" in m else "Opus" if "opus" in m
            else "Haiku" if "haiku" in m else "Fable" if "fable" in m
            else (model or "model"))


def load(path: Path) -> list[dict]:
    if path.is_dir():
        path = path / "episodes.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [r for r in rows if not r.get("error")]


def acquire_both_turn(obs_list: list[str], recipe: set[str]) -> int | None:
    """First turn index by which BOTH recipe types have appeared in a hold line.
    Holdings are monotonic, so from this turn on the agent holds both."""
    seen = set()
    for t, o in enumerate(obs_list or []):
        for _, ty in HOLD_RE.findall(o or ""):
            seen.add(ty)
        if recipe <= seen:
            return t
    return None


def per_episode(r: dict) -> dict:
    recipe = set(r["labels"]["recipe"])
    ab = acquire_both_turn(r.get("obs"), recipe)
    actions = r.get("actions") or []
    first_combine = next((i for i, a in enumerate(actions)
                          if a and a[0] == "combine"), None)
    built = bool(r.get("built_machine"))
    bt = r.get("build_turn")
    return {
        "budget": r["budget"],
        "held_both": ab is not None,
        "acquire_both_turn": ab,
        "built": built,
        # recognition latency = actions from holding both -> winning combine
        "recog_latency": (bt - ab) if (built and bt is not None and ab is not None) else None,
        "first_combine": first_combine,
    }


def ecdf(ax, xs, color, marker, label):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return
    ys = [(i + 1) / len(xs) for i in range(len(xs))]
    ax.step([0] + xs, [0] + ys, where="post", color=color, lw=1.8, label=label)
    ax.plot(xs, ys, marker, color=color, ms=4, alpha=0.6)


def main() -> None:
    paths = [Path(a) for a in sys.argv[1:]]
    if len(paths) < 2:
        sys.exit("need >=2 episodes.jsonl paths")

    models = {}  # label -> list[per_episode dict]
    order = []
    for p in paths:
        rows = load(p)
        lbl = label_for(rows[0].get("model"))
        order.append(lbl)
        models[lbl] = [per_episode(r) for r in rows]

    # --- headline decomposition table -----------------------------------
    print("== Experiment E: gathering × recognition decomposition ==\n")
    print(f"{'model':>7} {'eps':>4} {'gather':>7} {'recognize':>10} "
          f"{'build':>7}   (build = gather × recognize)")
    print("-" * 60)
    summ = {}
    for lbl in order:
        eps = models[lbl]
        n = len(eps)
        hb = sum(e["held_both"] for e in eps)
        bu = sum(e["built"] for e in eps)
        gather = hb / n
        recog = bu / hb if hb else float("nan")
        build = bu / n
        summ[lbl] = dict(n=n, hb=hb, bu=bu, gather=gather, recog=recog, build=build)
        print(f"{lbl:>7} {n:>4} {gather:>7.2f} {recog:>10.2f} {build:>7.2f}")
    print("\nrecognition latency (actions from holding both -> winning combine, "
          "builders only):")
    for lbl in order:
        lat = [e["recog_latency"] for e in models[lbl] if e["recog_latency"] is not None]
        if lat:
            print(f"  {lbl:>7}: median {statistics.median(lat):.0f}  "
                  f"mean {statistics.mean(lat):.1f}  max {max(lat)}  (n={len(lat)})")

    # ===================================================================
    # Figure 1: decomposition bars + recognition-rate vs budget
    # ===================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    labels = order
    x = range(len(labels))
    width = 0.26
    gthr = [summ[l]["gather"] for l in labels]
    rcg = [summ[l]["recog"] for l in labels]
    bld = [summ[l]["build"] for l in labels]
    ax1.bar([i - width for i in x], gthr, width, label="gathering  P(held both)",
            color="lightsteelblue")
    ax1.bar([i for i in x], rcg, width, label="recognition  P(built | held both)",
            color="indianred")
    ax1.bar([i + width for i in x], bld, width, label="build  P(built)",
            color="dimgray")
    for i, l in enumerate(labels):
        for off, v in ((-width, gthr[i]), (0, rcg[i]), (width, bld[i])):
            ax1.annotate(f"{v:.2f}", (i + off, v), textcoords="offset points",
                         xytext=(0, 3), ha="center", fontsize=7)
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("rate")
    ax1.set_title("Build rate = gathering × recognition")
    ax1.legend(fontsize=8, frameon=False, loc="lower center")

    # recognition rate vs budget, per model
    for lbl in labels:
        color, marker = STYLE.get(lbl, ("gray", "o"))
        by = defaultdict(lambda: [0, 0])  # budget -> [built, held_both]
        for e in models[lbl]:
            if e["held_both"]:
                by[e["budget"]][1] += 1
                by[e["budget"]][0] += int(e["built"])
        budgets = sorted(by)
        rate = [by[b][0] / by[b][1] if by[b][1] else float("nan") for b in budgets]
        ax2.plot(budgets, rate, marker=marker, color=color, lw=1.6, ms=5,
                 label=lbl)
    ax2.set_xscale("log", base=2)
    all_b = sorted({e["budget"] for lbl in labels for e in models[lbl]})
    ax2.set_xticks(all_b)
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
    fig.suptitle("Holding the ingredients, does the model recognize it can build? "
                 " ·  n=12, T=3", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out1 = Path("runs") / "fig_recognition_decomposition.png"
    fig.savefig(out1, dpi=150); fig.savefig(out1.with_suffix(".pdf"))
    plt.close(fig)
    print(f"\nwrote {out1} (+ .pdf)")

    # ===================================================================
    # Figure 2: timing -- recognition latency + time-to-first-combine ECDFs
    # ===================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for lbl in labels:
        color, marker = STYLE.get(lbl, ("gray", "o"))
        lat = [e["recog_latency"] for e in models[lbl]]
        ecdf(ax1, lat, color, marker, lbl)
        fc = [e["first_combine"] for e in models[lbl] if e["held_both"]]
        ecdf(ax2, fc, color, marker, lbl)
    ax1.set_title("Recognition latency (builders)")
    ax1.set_xlabel("actions from holding both → winning combine")
    ax1.set_ylabel("cumulative fraction of builders")
    ax1.legend(fontsize=8, frameon=False, loc="lower right")
    ax2.set_title("First combine attempt (ingredient-holders)")
    ax2.set_xlabel("action index of first combine of any kind")
    ax2.set_ylabel("cumulative fraction")
    ax2.legend(fontsize=8, frameon=False, loc="lower right")
    for ax in (ax1, ax2):
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("How fast do ingredient-holders commit to the build?  ·  n=12, T=3",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out2 = Path("runs") / "fig_recognition_latency.png"
    fig.savefig(out2, dpi=150); fig.savefig(out2.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out2} (+ .pdf)")


if __name__ == "__main__":
    main()
